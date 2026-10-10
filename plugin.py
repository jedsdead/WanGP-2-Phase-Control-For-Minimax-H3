import os

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import compat
from . import settings
from .settings import DEFAULT_LORA, NO_LORA

_LORA_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth")


class H3Phase2ControlPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "H3 Phase 2 Control"
        self.version = "0.1"
        self.description = "MiniMax H3 two-phase generation: choose the LoRA and the number of steps used in phase 2."
        settings.load()

    def setup_ui(self):
        self.request_global("get_lora_dir")
        self.add_tab(tab_id="h3_phase2_control", label="H3 Phase 2", component_constructor=self.create_ui)
        self._install()

    def post_ui_setup(self, components):
        self._install()
        return {}

    def on_model_change(self, state, model_type):
        self._install()

    def _install(self):
        from . import runtime
        try:
            runtime.install()
        except Exception as error:
            runtime._log(f"could not hook MiniMax H3, plugin inactive: {error!r}")

    # ------------------------------------------------------------------ LoRA list
    def _h3_lora_dir(self):
        get_lora_dir = getattr(self, "get_lora_dir", None)
        if get_lora_dir is None:
            return None
        try:
            from models.minimax_h3.minimax_h3_handler import family_handler
            model_types = family_handler.query_supported_types()
        except Exception:
            model_types = ["minimax_h3_fl2va", "minimax_h3_fl2va_pruned"]
        for model_type in model_types:
            try:
                folder = get_lora_dir(model_type)
            except Exception:
                continue
            if folder and os.path.isdir(folder):
                return folder
        return None

    def _lora_choices(self, current=None):
        choices = [("Wan2GP default: FL2V Turbo v0.1 at 1.0", DEFAULT_LORA), ("No LoRA in phase 2", NO_LORA)]
        folder = self._h3_lora_dir()
        found = []
        if folder:
            for root, dirs, files in os.walk(folder):
                if os.path.relpath(root, folder).count(os.sep) >= 2:
                    dirs[:] = []
                for name in files:
                    if name.lower().endswith(_LORA_EXTENSIONS):
                        path = os.path.join(root, name)
                        found.append((os.path.relpath(path, folder), path))
        choices += sorted(found, key=lambda item: item[0].lower())
        if current and current not in (DEFAULT_LORA, NO_LORA) and all(value != current for _, value in choices):
            choices.append((f"{os.path.basename(current)} (not in the H3 LoRA folder)", current))
        return choices

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _pp_loaded():
        return compat.two_phases_plus(require_active=False) is not None

    @classmethod
    def _multiplier_tied(cls, cfg):
        return cls._pp_loaded() and cfg["replace_turbo"]

    @classmethod
    def _multiplier_update(cls, cfg):
        tied = cls._multiplier_tied(cfg)
        return gr.update(interactive=not tied,
                         info=("Tied to 2 Phases Plus' Phase 2 Turbo LoRA Multiplier while the Turbo LoRA is replaced"
                               if tied else "Ignored for the WanGP default"))

    def create_ui(self):
        current = settings.get()
        pp_loaded = self._pp_loaded()
        gr.Markdown(
            "## H3 Phase 2 Control\n"
            "For MiniMax H3 **two-phase** generation (Guidance phases = 2). WanGP always runs phase 2 for **3 steps** "
            "with the **FL2V Turbo v0.1** LoRA and switches off any other LoRA with \"turbo\" in its name there. "
            "Choose the LoRA phase 2 uses instead. Settings apply to the next generation."
        )
        if pp_loaded:
            gr.Markdown(
                "*2 Phases Plus is enabled: its Phase 2 Noise Level Start sets the phase 2 steps, so the steps slider is "
                "disabled. With \"Replace the built-in Turbo LoRA\" on, your LoRA replaces the Turbo LoRA at 2 Phases Plus' "
                "Phase 2 Turbo LoRA Multiplier; turn it off to keep the Turbo LoRA and add your LoRA alongside at the "
                "multiplier set here. Its H3 Upscaler tasks are only affected with \"Also apply to H3 Upscaler tasks\" on.*"
            )
        enabled = gr.Checkbox(label="Enable", value=current["enabled"])
        steps = gr.Slider(1, 12, step=1, value=current["phase2_steps"], label="Phase 2 steps (3 = Wan2GP default)",
                          interactive=not pp_loaded,
                          info="Disabled: set by 2 Phases Plus (Phase 2 Noise Level Start)" if pp_loaded else None)
        with gr.Row():
            lora = gr.Dropdown(choices=self._lora_choices(current["phase2_lora"]), value=current["phase2_lora"],
                               label="Phase 2 LoRA", allow_custom_value=True, scale=4,
                               info="Any H3 LoRA file. It runs only in phase 2. You can also paste a full path.")
            refresh = gr.Button("Refresh list", scale=1)
        tied = self._multiplier_tied(current)
        multiplier = gr.Slider(0.0, 2.0, step=0.05, value=current["phase2_lora_multiplier"],
                               label="Phase 2 LoRA multiplier", interactive=not tied,
                               info=("Tied to 2 Phases Plus' Phase 2 Turbo LoRA Multiplier while the Turbo LoRA is replaced"
                                     if tied else "Ignored for the WanGP default"))
        replace = gr.Checkbox(value=current["replace_turbo"],
                              label="Replace the built-in Turbo LoRA (off: the chosen LoRA is added alongside it; \"No LoRA\" then does nothing)")
        others_off = gr.Checkbox(value=current["other_loras_off"],
                                 label="Switch off all other LoRAs in phase 2 (otherwise they follow their own phase-2 multipliers, e.g. \"1;0.5\")")
        upscaler = gr.Checkbox(value=current["apply_to_upscaler"],
                               label="Also apply to H3 Upscaler tasks (2 Phases Plus; off: they use 2 Phases Plus' settings only)",
                               visible=pp_loaded)
        status = gr.Markdown(self._summary(current))

        def save(key):
            def handler(value):
                settings.set_value(key, value)
                return self._summary(settings.get())
            return handler

        def save_replace(value):
            settings.set_value("replace_turbo", value)
            cfg = settings.get()
            return self._summary(cfg), self._multiplier_update(cfg)

        enabled.change(save("enabled"), inputs=enabled, outputs=status, show_progress="hidden")
        steps.change(save("phase2_steps"), inputs=steps, outputs=status, show_progress="hidden")
        lora.change(save("phase2_lora"), inputs=lora, outputs=status, show_progress="hidden")
        multiplier.change(save("phase2_lora_multiplier"), inputs=multiplier, outputs=status, show_progress="hidden")
        replace.change(save_replace, inputs=replace, outputs=[status, multiplier], show_progress="hidden")
        upscaler.change(save("apply_to_upscaler"), inputs=upscaler, outputs=status, show_progress="hidden")
        others_off.change(save("other_loras_off"), inputs=others_off, outputs=status, show_progress="hidden")
        refresh.click(lambda: gr.update(choices=self._lora_choices(settings.get()["phase2_lora"])), outputs=lora)

    @classmethod
    def _summary(cls, cfg):
        from .runtime import lora_choice_label
        if not cfg["enabled"]:
            return "*Off: WanGP's default phase 2 (3 steps, FL2V Turbo v0.1) is used.*"
        pp_loaded = cls._pp_loaded()
        steps = "steps set by 2 Phases Plus" if pp_loaded else f"{int(cfg['phase2_steps'])} steps"
        choice = cfg["phase2_lora"]
        lora = lora_choice_label(choice)
        if choice not in (DEFAULT_LORA, NO_LORA):
            lora += " at 2 Phases Plus' Turbo multiplier" if cls._multiplier_tied(cfg) else f" x{float(cfg['phase2_lora_multiplier']):g}"
            if not os.path.isfile(str(choice)):
                lora += " - **file not found, the default will be used**"
        if choice != DEFAULT_LORA:
            if not cfg["replace_turbo"]:
                lora += (" (**not applied**: turn on \"Replace the built-in Turbo LoRA\")" if choice == NO_LORA
                         else ", alongside the built-in Turbo LoRA")
            else:
                lora += " (built-in Turbo LoRA removed)" if choice == NO_LORA else ", replacing the built-in Turbo LoRA"
        others = "other LoRAs off" if cfg["other_loras_off"] else "other LoRAs as set"
        upscaler = ""
        if pp_loaded:
            upscaler = " Also applied to H3 Upscaler tasks." if cfg["apply_to_upscaler"] else " H3 Upscaler tasks: not affected."
        return f"**Next two-phase generation:** phase 2 = {steps}; LoRA = {lora}; {others}.{upscaler}"
