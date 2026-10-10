"""Runtime hooks for MiniMax H3 two-phase generation: choose the LoRA used in phase 2.

H3's two-phase mode denoises at half resolution (phase 1), upscales the latent, then refines it
at full resolution (phase 2). In WanGP, phase 2 always uses the FL2V Turbo v0.1 LoRA:
`get_loras_transformer` adds it with multiplier "0;1" (off in phase 1, on in phase 2), and a policy
inside `generate` forces it to 1.0 and switches off every other LoRA with "turbo" in its name.

This plugin, per generation and without editing WanGP's files:
* wraps `get_loras_transformer` to add the chosen LoRA for phase 2 (and, on its own, replace or
  drop the automatic Turbo LoRA);
* wraps `update_loras_slists` for the duration of one `generate` call so that, at the switch into
  phase 2 and after WanGP's own policy has run, the phase-2 multipliers are the ones chosen here.

"Replace the built-in Turbo LoRA" off keeps that LoRA and adds the chosen one alongside it. Without 2 Phases
Plus, the number of phase-2 steps can also be set (`H3_PHASE_2_SIGMAS` is swapped for one `generate` call).
2 Phases Plus compatibility: see compat.py.
"""

import contextlib
import functools
import os
import sys
import threading
import time

from . import compat
from . import settings
from .settings import DEFAULT_LORA, NO_LORA

_WRAPPED_FLAG = "_h3_phase2_control_wrapped"
_ACTIVE = threading.local()
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_control.log")
_LOG_MAX_BYTES = 256 * 1024


def _log(message):
    print(f"[H3 Phase 2 Control] {message}")
    try:
        if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _LOG_MAX_BYTES:
            os.replace(_LOG_PATH, _LOG_PATH + ".old")
        with open(_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


def lora_basename(entry):
    return os.path.basename(str(entry).split("|", 1)[0]).lower()


def effective_lora_choice(cfg):
    """The phase-2 LoRA choice actually applied: a missing file falls back to WanGP's default."""
    choice = cfg["phase2_lora"]
    if choice not in (DEFAULT_LORA, NO_LORA) and not os.path.isfile(str(choice)):
        return DEFAULT_LORA
    return choice


def lora_choice_label(choice):
    if choice == DEFAULT_LORA:
        return "WanGP default (FL2V Turbo v0.1)"
    if choice == NO_LORA:
        return "no LoRA"
    return os.path.basename(str(choice))


def _guide_phases(value):
    try:
        return int(str(value if value is not None else 1).strip("~") or 1)
    except ValueError:
        return 1


def phase2_sigmas(native, steps, start):
    """Phase-2 sigma tuple for `steps` steps.

    Wan2GP's native 3-step schedule is (start, 0.6316, 0.3158, 0): after the start level, evenly spaced steps
    from 0.9474 down to 0. Other step counts keep that spacing, so the native count gives back the native tuple
    exactly. If the first spaced value would not be below the start level (high step counts), the steps are
    spaced evenly from the start level instead.
    """
    native = tuple(float(v) for v in native)
    native_steps = len(native) - 1
    steps = max(1, int(steps))
    if steps == native_steps:
        return native
    base = native[1] * native_steps / (native_steps - 1) if native_steps > 1 else float(start)
    if steps > 1 and base * (steps - 1) / steps >= float(start):
        base = float(start)
    return (native[0], *[base * (steps - index) / steps for index in range(1, steps + 1)])


# --------------------------------------------------------------------------- wrapping helpers

def _chain_contains_wrapper(function, limit=64):
    """True if our wrapper is already in a chain of other plugins' wrappers."""
    pending, seen = [function], set()
    while pending and len(seen) < limit:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        value = getattr(value, "__func__", value)
        if getattr(value, _WRAPPED_FLAG, False):
            return True
        pending.append(getattr(value, "__wrapped__", None))
        for cell in getattr(value, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            if callable(content):
                pending.append(content)
    return False


def _mark(function):
    setattr(function, _WRAPPED_FLAG, True)
    return function


# --------------------------------------------------------------------------- LoRA selection

def _wrap_get_loras_transformer(cls, module):
    current = cls.__dict__.get("get_loras_transformer")
    if current is None or _chain_contains_wrapper(current):
        return False
    original = current
    is_required = getattr(module, "_is_required_h3_turbo", None)
    if is_required is None:
        _log(f"{cls.__module__} has no _is_required_h3_turbo; phase-2 LoRA choice unavailable there")
        return False

    @functools.wraps(original)
    def get_loras_transformer(self, *args, **kwargs):
        loras, multipliers = original(self, *args, **kwargs)
        cfg = settings.get()
        model_def = kwargs.get("model_def") or {}
        if (not cfg["enabled"] or _guide_phases(kwargs.get("guidance_phases")) <= 1
                or model_def.get("pdd", False) or model_def.get("vdn", False)):
            return loras, multipliers
        choice = effective_lora_choice(cfg)
        if choice != cfg["phase2_lora"]:
            _log(f"phase-2 LoRA file not found, keeping WanGP's default: {cfg['phase2_lora']}")
        if choice == DEFAULT_LORA:
            return loras, multipliers
        two_phases_plus = compat.two_phases_plus()
        if (two_phases_plus is not None and not cfg["apply_to_upscaler"]
                and compat.is_upscaler_task(two_phases_plus, kwargs)):
            return loras, multipliers  # H3 Upscaler tasks left to 2 Phases Plus
        if not cfg["replace_turbo"]:
            kept = list(zip(loras, multipliers))  # the built-in Turbo LoRA stays (2 Phases Plus, if active, sets its strength)
            if choice == NO_LORA:
                _log("'No LoRA' is not applied while 'Replace the built-in Turbo LoRA' is off")
                return loras, multipliers
        else:
            kept = [(lora, mult) for lora, mult in zip(loras, multipliers) if not is_required(lora_basename(lora))]
            if len(kept) != len(loras):
                _log(f"WanGP's automatic phase-2 Turbo LoRA replaced by: {lora_choice_label(choice)}"
                     + (" (at 2 Phases Plus' Turbo LoRA multiplier)" if two_phases_plus is not None and choice != NO_LORA else ""))
        if choice != NO_LORA:
            activated = {lora_basename(lora) for lora in kwargs.get("activated_loras") or ()}
            if lora_basename(choice) in activated:
                _log(f"{os.path.basename(choice)} is already in your LoRA list; its phase-2 multiplier will be set at the phase switch")
            else:
                kept.append((str(choice), f"0;{float(cfg['phase2_lora_multiplier']):g}"))
        return [lora for lora, _ in kept], [mult for _, mult in kept]

    cls.get_loras_transformer = _mark(get_loras_transformer)
    return True


def _phase2_multiplier_hook(original_update, module, cfg, loras_selected, defer_turbo, two_phases_plus=None):
    """update_loras_slists wrapper that rewrites the phase-2 multipliers at the phase switch.

    defer_turbo: the built-in Turbo LoRA is kept, so it is left to WanGP's policy or to 2 Phases Plus.
    two_phases_plus: its patch module when active. If the Turbo LoRA was replaced, the chosen LoRA takes
    2 Phases Plus' Turbo LoRA multiplier, and 2 Phases Plus is told there is no Turbo LoRA to change.
    """
    is_required = module._is_required_h3_turbo
    choice = effective_lora_choice(cfg)
    choice_name = lora_basename(choice) if choice not in (DEFAULT_LORA, NO_LORA) else None
    turbo_present = any(is_required(lora_basename(lora)) for lora in loras_selected or ())
    tied = two_phases_plus is not None and not turbo_present and choice_name is not None
    state = {"done": False}

    def update_loras_slists(trans, slists_dict, num_inference_steps, *args, **kwargs):
        phase_switch_step = kwargs.get("phase_switch_step", args[0] if args else None)
        replaced = contextlib.nullcontext()
        if phase_switch_step == 0 and not state["done"] and isinstance(slists_dict, dict):
            state["done"] = True
            multiplier = float(cfg["phase2_lora_multiplier"])
            if tied:
                pp_multiplier = compat.turbo_multiplier(two_phases_plus)
                if pp_multiplier is not None:
                    multiplier = pp_multiplier
                    _log(f"phase 2: {os.path.basename(str(choice))} replaces the built-in Turbo LoRA at 2 Phases Plus' "
                         f"Turbo LoRA multiplier {multiplier:g}")
            if two_phases_plus is not None and not turbo_present:
                replaced = compat.turbo_replaced(two_phases_plus)
            changes = []
            for index, lora in enumerate(loras_selected or ()):
                if index >= len(slists_dict.get("phase2", ())):
                    break
                name = lora_basename(lora)
                if choice_name is not None and name == choice_name:
                    value = multiplier
                elif is_required(name):
                    if defer_turbo or choice == DEFAULT_LORA:
                        continue  # kept: WanGP's own policy or 2 Phases Plus decides its strength
                    value = 0.0
                elif cfg["other_loras_off"]:
                    value = 0.0
                else:
                    continue
                slists_dict["phase2"][index] = value
                slists_dict["shared"][index] = False
                changes.append(f"{os.path.basename(str(lora).split('|', 1)[0])}={value:g}")
            if changes:
                _log("phase-2 LoRA multipliers: " + ", ".join(changes))
        with replaced:
            return original_update(trans, slists_dict, num_inference_steps, *args, **kwargs)

    # Lets other plugins that walk __wrapped__ chains (2 Phases Plus re-checks its own patch while
    # tasks are queued) see their wrapper underneath this temporary one instead of wrapping again.
    update_loras_slists.__wrapped__ = original_update
    update_loras_slists.expire = lambda: state.update(done=True)  # inert once its generation has finished
    return update_loras_slists


# --------------------------------------------------------------------------- generate

def _wrap_generate(cls, module):
    current = cls.__dict__.get("generate")
    if current is None or _chain_contains_wrapper(current):
        return False
    original = current

    @functools.wraps(original)
    def generate(self, *args, **kwargs):
        cfg = settings.get()
        if not cfg["enabled"] or getattr(_ACTIVE, "on", False):
            return original(self, *args, **kwargs)
        pdd = getattr(getattr(self, "transformer", None), "pdd_num_steps", None) is not None
        if (_guide_phases(kwargs.get("guide_phases")) <= 1 or pdd or kwargs.get("refinement_mode")
                or getattr(self, "audio_only", False)):
            return original(self, *args, **kwargs)
        two_phases_plus = compat.two_phases_plus()
        if two_phases_plus is not None and compat.is_upscaler_task(two_phases_plus, kwargs):
            if not cfg["apply_to_upscaler"]:
                _log("H3 Upscaler task from 2 Phases Plus: left to that plugin")
                return original(self, *args, **kwargs)
            _log("H3 Upscaler task from 2 Phases Plus: phase 2 settings applied")
        patches = {}
        native = getattr(module, "H3_PHASE_2_SIGMAS", None)
        if native is not None and int(cfg["phase2_steps"]) != len(native) - 1:
            if compat.two_phases_plus(require_active=False) is not None:
                _log(f"phase 2 steps ({int(cfg['phase2_steps'])}) not applied: 2 Phases Plus is loaded and sets them")
            else:
                start = float(kwargs.get("switch_threshold") or native[0])
                patches["H3_PHASE_2_SIGMAS"] = phase2_sigmas(native, cfg["phase2_steps"], start)
                shown = ", ".join(f"{v:.4f}" for v in (start, *patches["H3_PHASE_2_SIGMAS"][1:]))
                _log(f"phase 2: {int(cfg['phase2_steps'])} steps (noise levels {shown})")
        choice = effective_lora_choice(cfg)
        keep_turbo = not cfg["replace_turbo"]
        if keep_turbo and choice == NO_LORA:
            choice_applies = False
        else:
            choice_applies = choice != DEFAULT_LORA
        update = getattr(module, "update_loras_slists", None)
        if (choice_applies or cfg["other_loras_off"]) and update is not None and kwargs.get("loras_slists") is not None:
            tied = two_phases_plus is not None and not keep_turbo and choice not in (DEFAULT_LORA, NO_LORA)
            _log(f"phase 2 LoRA: {lora_choice_label(choice) if choice_applies else lora_choice_label(DEFAULT_LORA)}"
                 + ((" at 2 Phases Plus' Turbo LoRA multiplier" if tied else f" x{float(cfg['phase2_lora_multiplier']):g}")
                    if choice_applies and choice != NO_LORA else "")
                 + ("; other LoRAs off" if cfg["other_loras_off"] else "")
                 + (("; replaces the built-in Turbo LoRA" if not keep_turbo else "; alongside the built-in Turbo LoRA")
                    if choice_applies and choice != NO_LORA else ""))
            patches["update_loras_slists"] = _phase2_multiplier_hook(update, module, cfg, kwargs.get("loras_selected"),
                                                                     keep_turbo, two_phases_plus)
        if not patches:
            return original(self, *args, **kwargs)

        saved = {name: getattr(module, name) for name in patches}
        _ACTIVE.on = True
        for name, value in patches.items():
            setattr(module, name, value)
        try:
            return original(self, *args, **kwargs)
        finally:
            hook = patches.get("update_loras_slists")
            if hook is not None:
                hook.expire()  # if another plugin wrapped over the hook meanwhile, it stays in place but does nothing
            for name, value in saved.items():
                if getattr(module, name, None) is value or getattr(module, name, None) is patches[name]:
                    setattr(module, name, value)
            _ACTIVE.on = False

    cls.generate = _mark(generate)
    return True


def install():
    """Wrap WanGP's H3 pipeline (and any class from a module with the same phase-2 globals)."""
    from models.minimax_h3 import pipeline as native

    classes = [native.MiniMaxH3Pipeline]
    for module in list(sys.modules.values()):
        try:
            cls = module.__dict__.get("MiniMaxH3Pipeline") if hasattr(module, "__dict__") else None
        except Exception:
            continue
        if isinstance(cls, type) and all(cls is not c for c in classes):
            classes.append(cls)
    hooked = []
    for cls in classes:
        module = sys.modules.get(cls.__module__)
        if module is None or not hasattr(module, "H3_PHASE_2_SIGMAS"):
            continue  # e.g. H3 Latent Continue's copy: its models are single-phase only
        changed = _wrap_generate(cls, module)
        changed = _wrap_get_loras_transformer(cls, module) or changed
        if changed:
            hooked.append(f"{cls.__module__}.{cls.__qualname__}")
    for label in hooked:
        _log(f"hooked {label}")
    return hooked
