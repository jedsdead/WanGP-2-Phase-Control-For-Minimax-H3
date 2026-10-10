# H3 Phase 2 Control for Wan2GP

A Wan2GP plugin for MiniMax H3 **two-phase** generation (Guidance phases = 2): choose the LoRA and the number of steps
used in phase 2.

By default Wan2GP always runs phase 2 for 3 steps with the FL2V Turbo LoRA, and switches off any other turbo LoRA there.

## Features

- **Phase 2 steps**: 1 to 12 (3 is Wan2GP's default).
- **Phase 2 LoRA**: Wan2GP's default, no LoRA, or any LoRA from your H3 LoRA folder, at the multiplier you
  choose. It runs in phase 2 only.
- **Replace the built-in Turbo LoRA** (on by default). Turn it off to keep the Turbo LoRA and add your LoRA
  alongside it.
- **Switch off all other LoRAs in phase 2** (otherwise they follow their own `phase1;phase2` multipliers).

Single-phase generations are not affected. What was applied to each generation is written to the console
and to `phase2_control.log` in the plugin folder.

## 2 Phases Plus

Works alongside [2 Phases Plus](https://github.com/g3n3rativ3/wan2gp-2phases-plus). While it's enabled:

- The steps slider is disabled: its *Phase 2 Noise Level Start* sets the phase 2 steps.
- With *Replace the built-in Turbo LoRA* on, your LoRA replaces the Turbo LoRA at its *Phase 2 Turbo LoRA
  Multiplier*. Turn it off to keep the Turbo LoRA and add your LoRA alongside at your own multiplier.
- H3 Upscaler tasks are left to 2 Phases Plus unless you turn on *Also apply to H3 Upscaler tasks*.

## Install

In Wan2GP, open the **Plugins** tab, paste this repository's URL under *Install New Plugin*, install,
enable it and restart Wan2GP. The settings are in the **H3 Phase 2** tab.

## License

MIT
