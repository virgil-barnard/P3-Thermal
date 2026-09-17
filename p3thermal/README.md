# p3thermal

Custom, protocol-focused software for Thermal Master P3 cameras.

The camera is qualified for native Windows control through PyUSB, WinUSB, and
libusb. The current implementation captures validated native frames, records
lossless sessions, replays them offline, and provides a local viewer.

## Windows Camera Setup

Run camera commands in native Windows, not WSL. WSL has no direct access to
this USB device and USB/IP forwarding was not reliable during qualification.

Install Python 3.11 or newer, then create the Windows environment:

```powershell
cd C:\Users\athan\thermal\p3thermal
py -3.13 -m venv .venv-windows
.\.venv-windows\Scripts\python.exe -m pip install -e .
```

For tests and linting, install the development extra instead:

```powershell
.\.venv-windows\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-windows\Scripts\python.exe -m pytest
.\.venv-windows\Scripts\python.exe -m ruff check .
```

The P3 control interface needs the standard `WinUSB` driver. This is a one-time
Windows device setup:

1. Install and run [Zadig](https://zadig.akeo.ie/) as administrator.
2. Select `P3 (Interface 0)` with USB ID `3474:45A2`, interface `MI_00`.
3. Select `WinUSB` and choose **Install Driver**.
4. Close any manufacturer application before running this software.

The expected Device Manager state is `P3 (Interface 0)`, provider `libwdi`,
service `WinUSB`, and problem code `0`. The previous state on this machine was
problem code `28` (no driver), so there was no vendor driver to preserve.

Verify the camera and its native stream:

```powershell
.\.venv-windows\Scripts\python.exe tools\capture_probe.py
```

Successful output includes model response `P3`, two `02/01` command responses,
and a validated frame of `197656` bytes. The frame can arrive over several USB
reads; the probe deliberately validates start/end markers rather than assuming
one USB read is one image.

Create a read-only device profile report without starting the stream:

```powershell
.\.venv-windows\Scripts\python.exe -m p3thermal.cli inspect --out reports\p3-inspection.json
```

The report records supported identifiers, USB descriptors, endpoints, and the
runtime dependency versions. It is the required first diagnostic when qualifying
a new host or camera.

## Capture And View

Record five seconds of accepted frames; stream startup is not included in the
duration. This command saves data and exits; it does not open a viewer. Each
`--out` directory must be new so an existing session cannot be overwritten:

```powershell
.\.venv-windows\Scripts\python.exe -m p3thermal.cli record --duration 5 --out sessions\inspection
```

For a live feed, run this command and leave it running, then open
`http://127.0.0.1:8765` in a Windows browser:

```powershell
.\.venv-windows\Scripts\python.exe -m p3thermal.cli serve
```

The viewer shows native and optional learned 2x/3x/4x images side by side,
with independent 1-8x pixel-preserving digital zoom and histograms. It defaults
to Ember palette and 270-degree clockwise rotation, and provides six selectable palettes,
quarter-turn display rotation,
automatic/manual raw range, raw histogram, box/median preview filters, cursor
and drag-ROI min/mean/max readouts, quality status, preview rate, and recording
controls. Zoom sliders show their selected multiplier. Choose nearest-neighbor,
low-quality smoothed, or high-quality smoothed display interpolation; the same
selection is applied to both panes for visual comparison. All processing is
display-only: it never changes recorded native frames. It intentionally does not
show temperatures because native values have not yet been validated against the
manufacturer application and an independent reference.

For live inference, first run the `p3thermal-upscaling` loopback sidecar from
its dedicated environment. Then enable the learned-preview control without
adding Torch to this camera environment:

```powershell
.\.venv-windows\Scripts\python.exe -m p3thermal.cli serve --inference-url http://127.0.0.1:8766
```

Inference has one latest-frame worker and one output slot; it may skip preview
inputs under load while native capture and recording remain unchanged. The
learned image is synthetic-model output in native-DN units, not a
measured-resolution or temperature claim. A temporary sidecar request failure
is shown in learned-preview status and retried against the newest frame; it does
not stop capture, recording, or the preview worker.

The learned 2x viewer path was functionally confirmed on the qualified Windows
host on 17 September 2026. Latency, skipped-preview behavior, recording
noninterference, and real-scene reconstruction quality still require explicit
measurement.

Replay a recorded session without a camera:

```bash
.venv/bin/python -m p3thermal.cli replay reports/p3-session-clean-2026-09-11
```

Sessions contain a JSON manifest, append-only checksummed native-frame records,
and a rebuildable JSON-lines index. Replay accepts a truncated final record's
valid prefix and rejects records with a damaged checksum.

## Dependencies

The runtime uses NumPy, PyUSB, and, on Windows only, `libusb-package` to provide
the libusb DLL. The Windows USB driver is WinUSB. No USB/IP component is needed.

## Development

For Linux-only codec and unit-test development:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

## Current Scope

`Camera` is the sole streaming device owner; `tools/capture_probe.py` remains a
small hardware regression probe. Both restore interface 1 alternate setting 0
and release claimed interfaces on normal cleanup. Do not run another camera
application at the same time.

A short Windows run recorded 67 valid frames over 6.803 seconds of host receive
timestamps (9.70 delivered fps). It is neither a refresh-rate nor a
sustained-reliability claim. Gain/shutter controls, calibrated temperatures,
automatic recovery, and scientific super-resolution remain unqualified.

Protocol observations are based on the included community reference checkout at
`../p3-ir-camera`, commit `e3205dca5727682ff2d903585d1dce5a1d19f1f6`. This
project will retain its own implementation and tests.

## Calibration And Reconstruction Roadmap

Temperature labels require measured evidence, not an assumed raw-value formula.
Use a stable high-emissivity target that fills the central field of view. A
calibrated blackbody is preferred; a matte-black target with a traceable contact
probe is an interim reference with lower confidence. At five or more stable
temperatures across the intended range, preserve all of the following:

1. A new P3 lossless session, named with target and setpoint.
2. A manufacturer-application screenshot of its centre ROI/cursor reading and
   every visible setting.
3. The independent-reference reading, target emissivity, distance, ambient
   conditions, and a photograph/diagram of the setup.

Keep the camera geometry, target emissivity, and app settings fixed for each
series. First compare the same native-pixel ROI with the manufacturer app; that
only establishes compatibility. Fit and validate a versioned interpretation
against the independent reference before the viewer displays temperatures.

Super-resolution also begins with measurements. Use a small stable heated point
or aperture against a uniform cool background, and translate it with a
micrometer stage in known sub-pixel increments. Record lossless sessions at each
position, including fixed-target repeats for noise and drift. Those data provide
the point-spread function, registration error, detector temporal behavior, and
held-out geometry needed to evaluate a first registration-plus-shift-and-add
reconstruction against ordinary interpolation. Derived reconstructions must
remain separate from the original sessions and report uncertainty; visually
sharper output alone is not evidence of resolved thermal detail.
