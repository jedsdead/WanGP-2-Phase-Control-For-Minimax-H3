# Changelog

## 0.3 (2026-10-20)

- Fixed "The following Loras files are missing or invalid" with a phase-2 LoRA on installs where the LoRA folder is
  a relative path (the default): the LoRA is now handed to Wan2GP by its full path.

## 0.2 (2026-10-12)

- Works with newer Wan2GP versions (the built-in Turbo LoRA is found by position; the steps slider is disabled where
  Wan2GP has its own phase 2 steps setting).
- Option to apply the phase-2 LoRA to H3 Upscaler tasks of 2 Phases Plus.

## 0.1

- First release.
