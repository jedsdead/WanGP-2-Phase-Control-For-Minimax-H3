"""Plugin settings, persisted to settings.json next to this file."""

import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_LORA = "default"   # Wan2GP's own choice: the FL2V Turbo v0.1 LoRA at 1.0
NO_LORA = "none"           # no phase-2 LoRA

DEFAULTS = {
    "enabled": True,
    "phase2_steps": 3,               # 3 = Wan2GP's own schedule, left untouched
    "phase2_lora": DEFAULT_LORA,     # DEFAULT_LORA, NO_LORA, or an absolute path to a LoRA file
    "phase2_lora_multiplier": 1.0,
    "other_loras_off": False,        # switch every other LoRA off in phase 2
}

_settings = dict(DEFAULTS)


def load():
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        _settings.update({key: stored[key] for key in DEFAULTS if key in stored})
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"[H3 Phase 2 Control] could not read settings.json, using defaults: {error}")


def save():
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(_settings, handle, indent=2)
    except Exception as error:
        print(f"[H3 Phase 2 Control] could not save settings.json: {error}")


def get():
    return dict(_settings)


def set_value(key, value):
    _settings[key] = value
    save()
