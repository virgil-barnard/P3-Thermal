# P3 Thermal Progress

Last updated: 17 September 2026

## Current Milestone

M4 usable-application foundation. M0-M3 now have a tested capture, lossless
recording/replay, and local-viewer baseline. Sustained qualification, recovery,
temperature interpretation, and scientific processing remain open.

The active research direction is neural upsampling from the separate
`thermal-calibration-lab` synthetic generator, followed by real-time learned
preview in the local camera viewer. The verified generator contract currently
provides 2x, 3x, and 4x labels, so the initial one-checkpoint model serves those
three scales. 8x is deferred until generator support is implemented and
validated; it is not currently available training supervision.

## Completed Evidence

- Created `p3thermal` with immutable `DeviceProfile`, `ThermalFrame`, and
  `FrameSource` contracts.
- Native Windows P3 qualification uses PyUSB, `libusb-package`, and WinUSB on
  `USB\VID_3474&PID_45A2&MI_00`; Device Manager reports problem code 0.
- Device identity returned `P3`; device version is `0x0200`.
- Read-only inspection on 11 September 2026 recorded firmware `00.00.02.16`,
  hardware revision `P3-00.01`, part number `P30-1A00800007`, and serial
  `P3025043DF061400589` in
  `p3thermal/reports/p3-inspection-2026-09-11.json`.
- Interface 0 has bulk endpoints `0x84` and `0x05`. Interface 1 alternate
  setting 1 has bulk endpoints `0x81` and `0x02`; alternate setting 0 is idle.
- The stream start sequence returned `02/01` acknowledgements and produced a
  validated 197656-byte native frame over five and seven reads in separate runs.
- The refactored native Windows probe was run on 11 September 2026. It returned
  `model: P3`, two `02/01` acknowledgements, and a 197656-byte frame after five
  USB reads. It restored interface 1 to alternate setting 0 on exit.
- The public `Camera` API was run on native Windows on 11 September 2026. Its
  first frame had sequence `0`, stream epoch `1`, thermal and brightness shapes
  `(192, 256)`, quality `VALID`, and device counter `0`.
- A five-second accepted-frame native recording was captured at
  `p3thermal/reports/p3-session-clean-2026-09-11`. It has 67 structurally valid,
  `VALID` frames. Their host receive timestamps span 6.803 seconds, an observed
  9.70 delivered fps for this short run, not a sustained-rate claim.

## Implemented This Session

- Began the final comprehensive synthetic corpus at
  `thermal-calibration-lab/generated/comprehensive-2x-3x-4x`: expanded suite,
  three independent repetitions, seed `20260916`, 16 frames per clip, 24x
  supersampling, and eight boundary samples. The long native-Windows build is
  running resumably in process `6740`; do not train from the directory until it
  writes `manifest.json` and `validation.json` successfully. Added a guarded
  `thermal_calibration generate --resume` mode to reuse only complete,
  scenario-matching clips after an execution-time interruption.

- Reviewed the P3 nominal-intrinsics, generator, forward-model, and camera-viewer
  documentation. Recorded that the generator's verified current contract is
  2x/3x/4x, not 2x/4x/8x; 8x remains a deliberate future expansion.
- Updated the root architecture plan and workspace handoff instructions for a
  dedicated GPU training/export environment and bounded server-side learned
  preview. Browser WebGPU/WGSL is deferred pending latency and output-parity
  evidence.
- Created `p3thermal-upscaling` with its own Windows `.venv`, PyTorch
  `2.11.0+cu128`, pytest, and Ruff. CUDA is available on `NVIDIA GeForce RTX
  4090`.
- Added a compact shared-trunk PyTorch model with 2x, 3x, and 4x pixel-shuffle
  heads, synthetic acquisition-reference loader, group-split trainer, baseline
  metrics, and one TorchScript export artifact with runtime scale selection.
- Generated `thermal-calibration-lab/generated/smoke`: three two-frame clips
  (matched development stationary/dither pair plus held-out validation triangle;
  six frames total). Generator validation passed.
- Ran one CUDA epoch on the smoke dataset and exported
  `p3thermal-upscaling/artifacts/p3-upscaler-smoke.ts`. It loads and returns
  512x384, 768x576, and 1024x768 outputs for 2x, 3x, and 4x respectively. Its
  metrics are pipeline smoke evidence only, not model-quality evidence.
- Final offline verification: `p3thermal-upscaling/.venv/Scripts/python.exe -m
  pytest` reports 5 passed and Ruff clean; the calibration-lab smoke validation
  passed; `p3thermal/.venv/bin/python -m pytest` reports 17 passed and Ruff
  clean. No P3 hardware command was run in this session.
- Reconfirmed the complete training/export path on 17 September 2026 with a
  separate one-epoch CUDA run at `p3thermal-upscaling/runs/confirmation` and
  `artifacts/p3-upscaler-confirmation.ts`. Loading the artifact and invoking
  `forward(raw, scale)` on native-DN `(1, 1, 192, 256)` input returned
  `(1, 1, 384, 512)`, `(1, 1, 576, 768)`, and `(1, 1, 768, 1024)` at 2x, 3x,
  and 4x. These results confirm the pipeline and interface, not SR quality.
- Added learned-preview integration in `p3thermal`: a single latest-frame
  worker produces a separate derived preview slot; `/api/preview` exposes its
  little-endian uint16 output and source-sequence/scale headers; and the local
  viewer selects native, learned 2x, learned 3x, or learned 4x display. Native
  `/api/frame` and recordings remain unchanged.
- `p3thermal` offline tests now report 18 passed and Ruff clean. The real
  confirmation artifact ran on CUDA in the isolated GPU environment and
  produced 384x512, 576x768, and 768x1024 outputs. No camera command was
  issued during that automated verification.
- Replaced the pending in-process runtime deployment with a complete local
  sidecar boundary. `p3thermal-upscaling` now supplies
  `p3thermal_upscaling.serve`, which validates artifact metadata and serves
  TorchScript inference only on `127.0.0.1`; `p3thermal --inference-url` posts
  native frames to it. A real CUDA confirmation-artifact request through that
  HTTP boundary returned the expected 2x/3x/4x P3 shapes. The sidecar has no
  USB access and `p3thermal/.venv-windows` receives no PyTorch dependency.
- Hardened the viewer learned-preview handoff: it now accepts an image only
  when response dimensions are positive integers and the uint16 byte count
  exactly matches them, so pending/malformed responses retain the prior valid
  image instead of creating a zero-size canvas. The status panel now reserves a
  fixed two-line display area to prevent control-layout movement from wrapping
  status text. Offline verification remains 18 passed with Ruff clean.
- User-confirmed on 17 September 2026: the live Windows viewer successfully
  displayed the learned 2x preview through the loopback GPU sidecar. This is a
  functional integration observation, not a latency, recording-noninterference,
  or reconstruction-quality qualification.

- Added the root `AGENTS.md` with workspace boundaries, Windows-only USB rules,
  module ownership boundaries, and mandatory progress-record maintenance.
- Added `protocol.py`: 12-byte marker parsing, sync/counter validation,
  arbitrary-chunk reassembly with stale-byte resynchronization, and decoding of
  the 256x192 brightness/thermal planes from the native 256x386 payload.
- Added `transport.py`: P3 discovery, libusb backend selection, configuration,
  interface claiming, serialized control transactions, qualified stream start,
  bounded bulk reads, idle-alt-setting restoration, and resource release.
- Refactored `tools/capture_probe.py` to use the protocol and transport modules.
- Added five protocol tests covering fragmented input, stale bytes, malformed
  marker counters, native-plane decoding, frame size, and decoded-array
  immutability.
- Added a mocked transport test proving that a failed second interface claim
  releases interface 0 and disposes USB resources.
- Added `p3thermal inspect`, a read-only JSON diagnostic that reports supported
  identification registers, all USB interface alternate settings/endpoints, and
  runtime versions. The command completed successfully on the qualified Windows
  host and created `reports/p3-inspection-2026-09-11.json`.
- Added `Camera`, the sole streaming `FrameSource` owner. It assigns host
  sequence/timing, emits immutable thermal and brightness arrays, and preserves
  raw bytes and marker counter provenance without unsupported continuity claims.
- Added Camera tests for lifecycle/immutability, counter preservation, and
  bounded timeout faulting.
- Removed the unqualified cross-frame `counter_3` continuity heuristic after a
  real session showed it advances within each frame. Counters remain provenance,
  not missing-frame evidence.
- Added bounded read-timeout failure: three consecutive timeouts create events,
  transition Camera to `FAULTED`, and raise visibly instead of spinning.
- Added the versioned `P3F1` recording format: append-only checksummed records,
  manifest, index, valid-prefix replay recovery, and thermal NumPy export.
- Added loopback-only `serve` and `replay` workflows. The Canvas viewer receives
  only the latest immutable frame, so browser rendering cannot stall capture.
  It has raw display range, cursor readout, status, and recording controls.
- Added recording round-trip, truncation, checksum, service, and HTTP tests.
  Offline verification now reports 17 passed, Ruff clean, and `compileall` clean.
- Clarified that `record` saves a session without opening a viewer and now turns
  an existing output-directory collision into a concise CLI error.
- Expanded the local viewer with display-only digital zoom, palette selection,
  raw histogram/range controls, box/median preview filters, live preview rate,
  and drag ROI statistics. These never alter native frames or recordings.
- Separated cursor, ROI, and connection status fields so pointer updates cannot
  overwrite or race the status layout. Constrained all control-grid children to
  their pane and added adjustable low/high percentile sliders for auto range.
- Added white-hot, black-hot, rainbow, and ember palettes plus 0/90/180/270
  degree preview rotation. Cursor/ROI coordinates remain in native orientation.
- Offline verification after the Camera addition:
  `p3thermal/.venv/bin/python -m pytest` -> 11 passed;
  `p3thermal/.venv/bin/python -m ruff check .` -> all checks passed.
- Offline verification after the changes:
  `p3thermal/.venv/bin/python -m pytest` -> 8 passed;
  `p3thermal/.venv/bin/python -m ruff check .` -> all checks passed.

## Important Constraints

- P3 capture is native Windows only. WSL USB/IP attempts failed and were removed.
- A USB read is not a frame. Framing requires paired 12-byte markers around a
  197632-byte payload.
- Raw thermal values and shutter/gain behavior remain provisional. Do not claim
  temperature accuracy or issue unqualified controls.
- `p3-ir-camera` is protocol evidence only, currently referenced at commit
  `e3205dca5727682ff2d903585d1dce5a1d19f1f6`.
- The calibration lab uses nominal published intrinsics (`fx=fy=358.333333`,
  `cx=127.5`, `cy=95.5`) and assumed optics/noise; it is not physical P3
  calibration or validation of X3IR.
- Keep PyTorch/GPU packages in a dedicated training environment. Camera USB
  commands remain native Windows and always run through the installed
  `p3thermal/.venv-windows` interpreter.

## Next Tasks

1. Qualify repeated start/stop and a 30-minute native Windows recording; report
   delivery rate, timeouts, faults, memory, storage rate, and cleanup.
2. Create a five-or-more-point temperature dataset with a high-emissivity target,
   matched manufacturer-app screenshots/settings, and a traceable independent
   reference. Preserve geometry, ambient conditions, and emissivity with each
   lossless session.
3. Compare matching native ROIs with the app, fit an interpretation only against
   the independent reference, define its validated range/error, and then add a
   versioned temperature UI readout.
4. Build a controlled heated-point/aperture target and collect known sub-pixel
   translations plus fixed-target repeats. Estimate PSF, registration error,
   noise, and drift before implementing shift-and-add reconstruction.
5. Implement measured recovery after disconnect/timeout and preserve its events
   in recordings. Do not issue shutter or gain controls until qualified.
6. Generate a larger group-separated synthetic corpus with explicit seed,
   scenario distribution, storage budget, and held-out evaluation plan.
7. Train a meaningful 2x/3x/4x baseline, compare it with nearest/bicubic per
   scale, and retain metrics/model provenance with the exported artifact.
8. Run the loopback sidecar with learned preview against replay, then explicitly
   measure end-to-end latency, preview skips, output parity, and recording
   noninterference before a live-camera session or browser WebGPU/WGSL work.
