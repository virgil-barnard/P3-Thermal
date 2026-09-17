# Workspace Startup Instructions

## Working Areas

- `p3thermal/` is the custom Thermal Master P3 implementation. Make product
  changes there.
- `p3-ir-camera/` is an upstream reference checkout. Do not add product changes
  to it. Its WSL/USBIP guidance is not valid for this workspace's P3 setup.
- `thermal-calibration-lab/` produces synthetic, assumed-camera datasets. Its
  nominal P3 geometry and forward model are not measurements of the physical
  camera. Use its installed `.venv` for every generator or validation command.
- `p3thermal-upscaling/` is the separate GPU training and export environment.
  It consumes generated data and produces inference artifacts; it must not add
  PyTorch or training dependencies to the qualified camera environment.
- `P3_Thermal_Software_Plan.md` is the architecture and acceptance-gate source
  of truth. Read the relevant section before beginning a new milestone.
- `PROGRESS.md` is the durable session handoff. Update it when implementation,
  qualification evidence, tests, known limitations, or next work changes.

## P3 Hardware Rules

- Run P3 USB commands on native Windows, never WSL. Do not retry USB/IP setup.
- The P3 is `3474:45A2`. Only interface 0 (`MI_00`) is bound to `WinUSB` on the
  qualified host. Do not alter driver bindings unless the task specifically
  requires it.
- Close the manufacturer application and any other P3 process before a probe.
  One process must own the camera at a time.
- The verified hardware probe is
  `p3thermal/tools/capture_probe.py`, run with
  `p3thermal/.venv-windows/Scripts/python.exe` in PowerShell. It is allowed to
  start a stream and must restore interface 1 alternate setting 0 on cleanup.
- Never treat a USB bulk-read boundary as a frame boundary. Preserve raw bytes;
  only accept frames whose protocol markers and layout validate.
- Shutter and gain commands are not qualified. Do not send them as part of
  ordinary testing or implementation work.

## Development And Verification

- Never invoke a project with base `python` or a globally installed tool. Run
  commands through that project's virtual environment: `p3thermal/.venv/bin/python`
  for Linux-only camera tests, `p3thermal/.venv-windows/Scripts/python.exe` in
  native Windows PowerShell for camera commands, and
  `thermal-calibration-lab/.venv/Scripts/python.exe` for synthetic
  generation/tests on this installed Windows environment.
  Follow `p3thermal-upscaling/README.md` for its dedicated training venv.
- Use `p3thermal/.venv/bin/python -m pytest` and
  `p3thermal/.venv/bin/python -m ruff check .` for offline Linux verification.
- Hardware verification is explicit and reported separately from unit tests.
- Keep hardware-specific behavior in `transport.py`, byte-only framing and
  decoding in `protocol.py`, and camera ownership/lifecycle in `device.py`.
- Keep changes small, retain immutable `ThermalFrame` arrays, and add focused
  offline tests for protocol behavior before relying on hardware.
- The current synthetic file contract provides 2x, 3x, and 4x references only.
  Do not describe 8x supervision as available until the generator, validation,
  storage budget, and documentation have been expanded together.

## Progress Record

At the end of each meaningful implementation or hardware-validation step, update
`PROGRESS.md` with completed work, commands/results, open risks, and the next
ordered task. Do not replace measured facts with assumptions from the upstream
reference.
