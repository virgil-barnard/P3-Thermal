# P3-Thermal

Local-first tooling for the Thermal Master P3 camera: a qualified native-frame
viewer/recorder, a synthetic calibration dataset generator, and an isolated GPU
upscaling environment.

## Projects

| Directory | Purpose |
| --- | --- |
| `p3thermal/` | P3 USB transport, protocol decoding, immutable native-frame recording, replay, and local browser viewer. |
| `thermal-calibration-lab/` | Reproducible synthetic physical-target sequences with native 256x192 observations and 2x/3x/4x simulator references. |
| `p3thermal-upscaling/` | Dedicated CUDA/PyTorch training, TorchScript export, and loopback-only learned-preview sidecar. |

Read `P3_Thermal_Software_Plan.md` for the architecture and acceptance gates,
and `PROGRESS.md` for verified evidence, known limits, and the current ordered
work. Each project has its own setup and usage instructions in its README.

## Boundaries

- Run P3 USB operations only from native Windows using the camera project's
  Windows virtual environment. Do not use USB/IP or WSL for camera access.
- Keep PyTorch and CUDA out of the qualified camera environment. Learned
  previews use a loopback-only sidecar from `p3thermal-upscaling`.
- Native recordings and `/api/frame` data remain raw immutable `uint16` camera
  frames. Learned images are derived previews only.
- The current synthetic contract contains 2x, 3x, and 4x references only.
  Synthetic data is an assumed-camera model, not physical P3 calibration or
  real-scene super-resolution evidence.

## Repository Hygiene

Virtual environments, generated datasets, recordings, training runs, exported
weights, logs, and caches are intentionally ignored. Recreate them from the
documented commands rather than committing them.
