"""Compatibility with the "2 Phases Plus" plugin (github.com/g3n3rativ3/wan2gp-2phases-plus).

Both plugins act on MiniMax H3 phase 2:

* Phase 2 steps: while 2 Phases Plus is loaded, its Phase 2 Noise Level Start decides them and this
  plugin's step count is not applied.

* Turbo LoRA: 2 Phases Plus sets the strength of WanGP's built-in phase-2 Turbo LoRA. With "Replace the
  built-in Turbo LoRA" on, a LoRA chosen here replaces it and takes that strength; with it off, this
  plugin never touches that LoRA and the chosen one is added alongside it at this plugin's multiplier.
* H3 Upscaler tab: its tasks are left entirely to 2 Phases Plus unless "Also apply to H3 Upscaler tasks" is on.
"""

import contextlib
import sys

_PATCH_MARKER = "_two_phases_plus_patch"


def two_phases_plus(require_active=True):
    """The 2 Phases Plus patch module if it is loaded (and, by default, has patched the H3 pipeline)."""
    for name, module in list(sys.modules.items()):
        if not name.endswith("h3_phase2_patch") or module is None:
            continue
        if getattr(module, "PATCH_MARKER", None) != _PATCH_MARKER:
            continue
        if require_active and not getattr(module, "_installed", False):
            continue
        return module
    return None


def is_upscaler_task(module, generate_kwargs):
    custom_settings = generate_kwargs.get("custom_settings")
    key = getattr(module, "UPSCALE_SOURCE_KEY", "two_phases_plus_upscale_source")
    return isinstance(custom_settings, dict) and bool(custom_settings.get(key))


def _current_run(module):
    try:
        run = module._current()
    except Exception:
        return None
    return run if isinstance(run, dict) else None


def turbo_multiplier(module):
    """2 Phases Plus' phase-2 Turbo LoRA multiplier for the generation running in this thread (or None)."""
    run = _current_run(module)
    try:
        return float(run["multiplier"]) if run is not None and "multiplier" in run else None
    except (TypeError, ValueError):
        return None


@contextlib.contextmanager
def turbo_replaced(module):
    """While the built-in Turbo LoRA has been replaced, tell 2 Phases Plus there is nothing to change, so it does
    not report a missing Turbo LoRA (this plugin logs what is used instead). Its own code is not modified."""
    run = _current_run(module)
    if run is None or "multiplier" not in run:
        yield
        return
    saved = run["multiplier"]
    run["multiplier"] = getattr(module, "WANGP_MULTIPLIER", 1.0)
    try:
        yield
    finally:
        run["multiplier"] = saved
