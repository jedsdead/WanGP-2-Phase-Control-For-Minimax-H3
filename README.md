# H3 Phase 2 Control (Wan2GP plugin, 0.1.0)

Controls the second phase of MiniMax H3 **two-phase** generation (Guidance phases = 2).

Phase 1 denoises at half resolution; the latent is upscaled; phase 2 refines it at full
resolution. Wan2GP always runs phase 2 for **3 steps** with the **FL2V Turbo v0.1** LoRA at
1.0, and switches off any other LoRA with "turbo" in its name for phase 2.

## Settings (tab: H3 Phase 2)

- **Phase 2 steps**: 1 to 12. 3 is Wan2GP's own schedule and is left untouched.
  Other counts keep the same even spacing of noise levels after the "phase 2 noise level
  start" setting (5 steps: 0.9035 → 0.758 → 0.568 → 0.379 → 0.190 → 0).
- **Phase 2 LoRA**: Wan2GP's default, no LoRA, or any LoRA file in your H3 LoRA folder
  (or a pasted path). The chosen LoRA runs in phase 2 only, at the multiplier you set, and
  replaces the automatic FL2V Turbo LoRA. If it is already in your normal LoRA list, it
  keeps its phase-1 multiplier and only its phase-2 multiplier changes.
- **Switch off all other LoRAs in phase 2**: otherwise they follow their own phase-2
  multipliers, which Wan2GP already lets you set with the `phase1;phase2` syntax
  (for example `1;0.5`).

Settings apply to the next generation. Single-phase generations, PDD and VDN variants
are never changed.

## Checking what it did

Each two-phase generation writes what it applied to the console and to
`phase2_control.log` in this folder: the phase-2 noise levels, the phase-2 LoRA, and the
final phase-2 multiplier of every LoRA it changed.

## Notes

- The FL2V Turbo LoRA is distilled for few steps. More phase-2 steps with it is untested;
  more steps generally pairs better with a lighter or no distillation LoRA.
- A missing LoRA file falls back to Wan2GP's default, with a line in the log.
- Patches Wan2GP's H3 internals at run time (checked against commit 6479db3). Nothing in
  Wan2GP's files is edited; disabling the plugin restores the default behaviour.
