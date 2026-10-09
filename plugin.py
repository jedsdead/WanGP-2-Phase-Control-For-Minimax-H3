import os

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import settings
from .settings import DEFAULT_LORA, NO_LORA

_LORA_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth")


class H3Phase2ControlPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "H3 Phase 2 Control"
        self.version = "0.1"
        self.description = "MiniMax H3 two-phase generation: choose the number of phase-2 steps and the LoRA used in phase 2."
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
    def create_ui(self):
        current = settings.get()
        gr.Markdown(
            "## H3 Phase 2 Control\n"
            "For MiniMax H3 **two-phase** generation (Guidance phases = 2). Phase 1 denoises at half resolution; "
            "phase 2 upscales and refines at full resolution. Wan2GP always runs phase 2 for **3 steps** with the "
            "**FL2V Turbo v0.1** LoRA at 1.0, and switches off any other LoRA with \"turbo\" in its name for phase 2. "
            "These settings override that and apply to the next generation."
        )
        enabled = gr.Checkbox(label="Enable", value=current["enabled"])
        steps = gr.Slider(1, 12, step=1, value=current["phase2_steps"], label="Phase 2 steps (3 = Wan2GP default)")
        with gr.Row():
            lora = gr.Dropdown(choices=self._lora_choices(current["phase2_lora"]), value=current["phase2_lora"],
                               label="Phase 2 LoRA", allow_custom_value=True, scale=4,
                               info="Any H3 LoRA file. It runs only in phase 2. You can also paste a full path.")
            refresh = gr.Button("Refresh list", scale=1)
        multiplier = gr.Slider(0.0, 2.0, step=0.05, value=current["phase2_lora_multiplier"],
                               label="Phase 2 LoRA multiplier (ignored for the Wan2GP default)")
        others_off = gr.Checkbox(value=current["other_loras_off"],
                                 label="Switch off all other LoRAs in phase 2 (otherwise they follow their own phase-2 multipliers, e.g. \"1;0.5\")")
        status = gr.Markdown(self._summary(current))

        def save(key):
            def handler(value):
                settings.set_value(key, value)
                return self._summary(settings.get())
            return handler

        enabled.change(save("enabled"), inputs=enabled, outputs=status, show_progress="hidden")
        steps.change(save("phase2_steps"), inputs=steps, outputs=status, show_progress="hidden")
        lora.change(save("phase2_lora"), inputs=lora, outputs=status, show_progress="hidden")
        multiplier.change(save("phase2_lora_multiplier"), inputs=multiplier, outputs=status, show_progress="hidden")
        others_off.change(save("other_loras_off"), inputs=others_off, outputs=status, show_progress="hidden")
        refresh.click(lambda: gr.update(choices=self._lora_choices(settings.get()["phase2_lora"])), outputs=lora)

    @staticmethod
    def _summary(cfg):
        from .runtime import lora_choice_label
        if not cfg["enabled"]:
            return "*Off: Wan2GP's default phase 2 (3 steps, FL2V Turbo v0.1) is used.*"
        lora = lora_choice_label(cfg["phase2_lora"])
        if cfg["phase2_lora"] not in (DEFAULT_LORA, NO_LORA):
            lora += f" x{float(cfg['phase2_lora_multiplier']):g}"
            if not os.path.isfile(str(cfg["phase2_lora"])):
                lora += " - **file not found, the default will be used**"
        others = "other LoRAs off" if cfg["other_loras_off"] else "other LoRAs as set"
        return f"**Next two-phase generation:** phase 2 = {int(cfg['phase2_steps'])} steps, {lora}, {others}."
