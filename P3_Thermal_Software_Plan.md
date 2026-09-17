P3 Thermal Software: Implementation Plan

Version 0.1 • 11 September 2026 • Proposed design and initial qualification

Build a Python package that gives us direct control of the Thermal Master P3, reliable access to its numerical frames, a practical viewer, and recordings that can be replayed through custom processing. The first release should be useful for everyday thermal inspection and data collection. Moving-source estimation and super-resolution will use the same acquisition and replay interfaces as later research modules.

The current research goal is a GPU-trained neural upsampler. It will learn from
the synthetic calibration-lab observations, export one multi-scale checkpoint,
and run on each latest native live frame so the local viewer can compare native
interpolation with learned output in real time. This is a research display path:
it must never alter the immutable native frame, recording payload, temperature
interpretation, or camera ownership lifecycle.

This is a software plan. Initial native Windows qualification, decoding,
lossless recording/replay, and a local-viewer baseline are complete. Sustained
capture, calibrated temperature interpretation, recovery, and application
acceptance remain outstanding.

The central decision is to own the device API and application architecture. Implement the P3 protocol in a small user-space driver using PyUSB and libusb. Existing community protocol research provides an interoperability reference; the application will have its own capture, recovery, recording, and presentation behavior. At implementation time, record the reviewed upstream commit. Attribute any adapted code or documentation and retain its required license notices. Public protocol observations are evidence to verify against our camera.

PyUSB supplies Python access through a native USB backend, so the driver can share code across hosts while platform setup handles USB access. This design uses the operating system's existing USB facilities. PyUSB documentation

Decision

Initial choice

Reason

Project name

p3thermal, a working local package name

Short commands and a clear scope; public package-name availability remains unchecked.

Language

Python 3.11 or newer, with a tested version matrix

Fits the user's workflow and keeps deployment simple.

Core dependencies

NumPy and PyUSB, plus the host's libusb backend

Direct numerical access with a small acquisition environment.

Device scope

One P3 per capture process

Makes ownership and recovery easy to reason about; separate processes can later serve more cameras.

First hardware qualification

Native Windows on the existing desktop

The desktop is available before the Pi provisioning work is complete.

Second hardware qualification

Native Ubuntu Server on Raspberry Pi 5, ARM64

Provides an unattended sensor node for the proposed lab.

Interface

Local browser UI; optional FastAPI/Uvicorn extra

The same interface can view a desktop-connected or Pi-connected camera.

Frontend

Packaged HTML, JavaScript, and Canvas assets

Works locally without a frontend build tool or CDN.

Research execution

Desktop CPU/GPU through replay or a frame subscription

Heavy processing can evolve independently of acquisition.

The Windows desktop can retain its WSL2 analysis environment. Initial USB qualification runs natively on the host holding the camera. Data can move to WSL2 through recordings or the service API. A Pi connected to the camera would run its own capture process; the desktop would connect to that process.

The first hardware session will establish a device profile. The manufacturer identifies the P3 as a native 256×192 thermal camera with enhanced 512×384 output and an advertised 25 Hz refresh rate. Treat those as specifications, not measured performance of this unit. Manufacturer specifications

The community protocol reference reports USB ID 3474:45A2, separate control/streaming interfaces, bulk frame transfer, an IR brightness plane, a 16-bit thermal plane, and metadata. It describes temperature conversion as value / 64 - 273.15 and reports special framing around shutter correction. These are provisional profile inputs. P3 protocol reference

Capture the actual VID/PID, available serial, hardware/firmware versions, descriptors, OS binding, and library versions. A baseline can come from the manufacturer's app and a USB trace before our capture implementation is ready. Preserve sample transactions from normal capture and each supported control transition. Unknown metadata remains available as bytes. The device profile records which layouts, commands, and value interpretations have been verified and on which hardware.

Windows qualification includes checking both required USB interfaces and the current driver binding. libusb documents WinUSB support, restrictions on concurrent applications, and the lack of an actual USB-reset operation through WinUSB. If a binding change is needed, document the specific device/interface, its previous binding, and restoration procedure. Do not assume that a generic reset-based recovery strategy works on every host. libusb Windows documentation

The data flow keeps acquisition independent of display and analysis. One owner performs all USB operations. Consumers receive immutable frames through bounded queues with explicit overflow behavior.

flowchart TD
    C["P3 camera"] --> D["USB driver"]
    D --> F["Validated frames"]
    F --> R["Lossless recorder"]
    F --> P["Processing"]
    F --> V["Browser viewer"]
    R --> L["Replay"]
    L --> P
    P --> V
    V -. "Control requests" .-> D

Component

Responsibility

Boundary

transport.py

Enumeration, interface ownership, bounded reads, control transfers, backend capabilities, cleanup

Contains host-specific USB behavior.

protocol.py

Command encoding, frame assembly, layout decoding, profile selection

Accepts bytes and produces validated records; testable without hardware.

device.py

Connection state, command serialization, streaming, bounded recovery

The sole owner of a physical camera.

frames.py

Frame, device profile, settings, and event types

The shared contract for capture, replay, and processors.

recording.py

Versioned sessions, lossless storage, recovery, replay, export

Preserves numerical input and its provenance.

service.py and ui/

Status, live view, recording controls, playback, ROI tools

Uses the public device and frame APIs.

processing/

Background estimation, PSF fitting, tracking, reconstruction

Produces derived results linked to source frames.

cli.py

Diagnostics and repeatable acquisition commands

Calls the same library API used by the viewer.

Keep these as modules in one repository initially. Introduce additional packages or extension mechanisms only when a real second implementation needs them. Hardware and replay implement the same small FrameSource interface.

The USB driver should have a clear lifecycle. Its states are disconnected, opening, ready, streaming, recovering, and faulted. Commands specify their allowed states. Shutdown cancels pending work, completes bounded cleanup, releases claimed interfaces, and attempts to restore any kernel-driver attachment that the transport itself changed.

During streaming, a single I/O worker services a command queue at safe boundaries. Read timeouts, initialization delays, retry limits, and resynchronization rules belong to the tested device profile. A control call returns an explicit outcome; it cannot race an unrelated USB operation. Record requested settings and acknowledged settings separately. Associate frames with the effective setting only when that boundary is established; otherwise flag the transition interval as uncertain.

The frame assembler must tolerate split reads, short transfers, stale bytes, and interrupted frames. A USB read boundary is not an image boundary. Validate structure, sizes, markers, and any verified continuity information together before accepting a frame. On uncertainty, retain a diagnostic event and discard the unusable candidate. Recovery has a finite budget and ends in an actionable error if the backend cannot restore capture.

A community shutter-trigger error report makes this a concrete qualification case. Reproduce it only if applicable to our profile; build a regression fixture from our own captured behavior. Reported shutter failure

Every delivered frame should retain enough information to support later scientific processing. The proposed contract is:

Field group

Contents

Identity

Recording/session ID, stream epoch, host-assigned sequence, device-profile ID

Numerical payload

Immutable native thermal uint16 array; the other native plane and metadata retained losslessly

Received data

Preserved validated frame bytes; bounded transaction traces available in diagnostic mode

Timing

Host monotonic receive interval, wall-clock anchor, original device counters when available

State

Requested/acknowledged settings generation, gain interpretation, radiometry interpretation version

Quality

Structural validity, continuity uncertainty, control-transition interval, suspected correction event

Sequence values assigned by the host are not proof that the camera produced no missing frames. Device counters are not exposure timestamps until their meaning and clock relationship are established. Automatic shutter events may initially be inferred rather than confirmed; label that distinction explicitly.

Native numerical data is preserved before display normalization, denoising, or reconstruction. Temperature arrays are derived through a versioned interpretation function. Comparison with the manufacturer's app can establish consistency, while an independent reference is needed for an accuracy claim. Emissivity and environmental corrections must have a single documented owner so they are not applied twice. The UI shows the interpretation status alongside measurements.

### Calibration Dataset Protocol

Use a high-emissivity target that fills the central field of view. A calibrated
blackbody is the reference standard; a matte-black target with a traceable
contact probe is an explicitly lower-confidence interim reference. Do not change
unqualified gain or shutter settings. For at least five stable target
temperatures covering the intended use range, record a new lossless P3 session,
the matched manufacturer-app centre-ROI/cursor reading and settings, the
independent-reference reading, target emissivity, distance, ambient conditions,
and setup photograph/diagram. Keep geometry and application settings fixed
within a series.

The first analysis compares the same native-pixel ROI with the manufacturer app;
it establishes compatibility, not temperature accuracy. Fit a candidate raw to
temperature interpretation only against independent-reference data. Validate it
on held-out setpoints, state its range and residual error, version it, and retain
all source sessions. The UI may show that version only after these acceptance
criteria are met.

### Super-Resolution Dataset Protocol

Do not evaluate reconstruction on arbitrary scenes first. Build a stable small
heated point or aperture against a uniform cooler background. Translate either
the target or camera with a calibrated stage through known sub-native-pixel
offsets, collecting lossless sessions at each location and fixed-target repeats.
Measure the spatial point-spread function, registration residuals, noise, drift,
and temporal response from this dataset. A first algorithm is registration plus
shift-and-add; compare it with nearest-neighbor and bicubic interpolation using
held-out known target positions. Store every reconstruction in `derived/` with
its source session IDs, registration parameters, uncertainty, and evaluation
results. A sharper-looking image without a held-out measurement is not a
super-resolution acceptance result.

### Neural Upsampling Roadmap

The first model is a compact PyTorch network with shared feature extraction and
separate 2x, 3x, and 4x reconstruction heads. One exported artifact therefore
serves all currently generated label resolutions; the selected viewer scale
chooses its corresponding output head rather than loading another model. The
synthetic generator's actual file contract is 2x/3x/4x, not 2x/4x/8x. An 8x
head is deferred until the generator creates, validates, documents, and budgets
8x references. It must not be invented by repeatedly resizing a lower-scale
label.

Training uses observed `input.npz` frames as inputs, skips invalid frames, and
uses `frame_to_truth` to select the matching `acquisition_hr*_dn` target by
default. That target tests fine sampling under the declared optics; any
ideal-reference/deblurring experiment is separately named and evaluated. Split
by capture group, never adjacent frames or stationary/dither siblings. Report
per-scale held-out metrics against nearest and bicubic interpolation, plus model
latency, frame age, skipped-input count, and GPU/CPU execution device. Synthetic
scores are simulator evidence only, pending separate real P3 datasets.

Keep training dependencies in a dedicated `p3thermal-upscaling` virtual
environment. The first live integration runs the versioned TorchScript artifact
in a loopback-only GPU sidecar from that environment. The native-Windows camera
service owns a bounded latest-frame client worker and exposes its derived result
through the existing loopback viewer. This keeps PyTorch out of the qualified
camera environment. The preview may skip inputs but must never delay acquisition
or recording.
The learned 2x path has been functionally observed in the live Windows viewer;
this does not qualify its latency, recording noninterference, or reconstruction
quality. Those are separate acceptance measurements.
WebGPU/WGSL is a later optimization only after a measured end-to-end latency
baseline and output-parity test; it would otherwise duplicate model kernels,
preprocessing, artifact handling, and browser support.

Recording and replay form part of the first usable release. Use a session directory with a small, documented schema:

File

Purpose

manifest.json

Schema version, device/host profile, configuration, software revision, interpretation status, completion state

frames-000001.bin, subsequent chunks

Append-only, length-delimited validated frame records with sequence, timing, flags, and an integrity checksum

events.jsonl

Controls, reconnects, corrections, errors, quality changes, queue overflow, and recording stop reason

index.jsonl

Seek positions and timestamps; rebuildable by scanning valid chunk records

derived/

Exports and analysis outputs, each linked to input sessions and processor settings

Specify byte order, lengths, checksum coverage, and limits before implementing the writer. Cap chunk size and bound queued memory. A truncated final record should leave a readable valid prefix. Record the durability/flush policy; distinguish frames received, accepted, written, and committed to durable storage. Playback must work without a camera or vendor application. Numerical exports can produce NumPy files for experiments.

Proposed consumer policies are deliberately explicit:

Consumer

Behavior when it cannot keep up

Live viewer

Show the newest available frame and count skipped previews.

Recorder

On queue overflow or storage failure, terminate the affected recording visibly and preserve its recoverable prefix.

Research processor

Declare either latest-frame or ordered processing; report skipped inputs. Offline replay is the default for complete-sequence experiments.

Size storage from actual observed payloads during qualification. As a planning estimate, two 256×192 16-bit planes at 25 frames/second require approximately 4.92 MB/s, or 17.7 GB/hour before metadata and container overhead. A thermal-plane-only export is approximately 8.85 GB/hour. These are arithmetic estimates, not throughput measurements. Enforce a user-selected duration or byte limit and show remaining capacity.

The browser interface should make the camera immediately useful. Its first screen contains a live thermal image, temperature scale, cursor readout, ROI min/mean/max, connection state, capture rate, recording status, and relevant quality events. Users can lock the display range, change palette, rotate the view, and record. Display orientation and palette choices leave stored native coordinates unchanged.

Device controls include only verified gain and shutter operations at first. Playback uses the same view, with a timeline and visible events. Later, processing overlays can show tracks, fit uncertainty, source-relative reconstructions, and residuals. Keep routine inspection controls prominent and protocol diagnostics in a separate expandable panel.

The service starts on loopback. For access to a Pi-hosted service, the initial deployment uses SSH forwarding. One service owns the camera, with multiple read-only viewers if needed. Mutating operations pass through the command queue. The HTTP layer needs origin/host checks and a session credential for control requests. Direct LAN exposure can be designed once remote use becomes a requirement.

The proposed application surface is small:

Operation

Intended behavior

p3thermal doctor

Report host/backend readiness without changing driver bindings.

p3thermal devices

Enumerate candidates and available identifiers.

p3thermal inspect --serial SERIAL

Read the selected device's supported identification fields.

p3thermal record --serial SERIAL --duration 60 --out SESSION

Record a bounded numerical session.

p3thermal serve --serial SERIAL

Run the local viewer and acquisition service.

p3thermal replay SESSION

Open a recording through the same frame and viewer interfaces.

These commands and the following API are design targets, not currently installed software. If exactly one supported camera is present, selecting it can be automatic; ambiguous discovery requires a specific identifier.

from p3thermal import Camera, Recorder

with Camera(serial="SERIAL") as camera:
    with Recorder("session-directory") as recorder:
        with camera.stream() as frames:
            for frame in frames:
                recorder.write(frame)
                if recorder.elapsed_seconds >= 60:
                    break

A FrameSource supports deterministic close, frame iteration, and access to associated events. Hardware-specific controls live on Camera; replay offers seeking. Consumers receive documented read-only arrays whose storage remains valid for the lifetime of the frame. A processor returns derived results tagged with the input frame IDs and its configuration.

Installation should work from a normal virtual environment with python -m pip install -e .. Optional extras add the browser service and research dependencies. Linux instructions install the native backend and a narrowly scoped device-access group/rule. Windows instructions check the backend and binding independently of pip. TensorFlow or another GPU framework belongs in the research environment, so changing an experiment does not change the qualified capture environment. Record exact tested dependencies when packaging each release.

Implementation proceeds by acceptance gates. The first public-facing milestone is a working capture/record/replay application; research features follow it.

Milestone

Deliverable

Acceptance evidence

Hardware dependency

M0: device profile

Identification report and initial compatibility profile

Actual descriptors, host binding, firmware identifiers where available, and a short reference capture

Required

M1: package and codec

Library skeleton, frame contract, command codec, assembler, fixture corpus

Known fixtures decode; malformed, split, and interrupted inputs never become valid images

Offline initially; add real fixtures from M0

M2: acquisition

Device lifecycle, serialized controls, finite recovery, diagnostics

Sustained capture; repeated start/stop; shutter/gain transitions; disconnect/reconnect behavior; exact accepted/rejected counts

Required

M3: recording

Session format, writer, reader, numerical export

Bit-exact payload round trip; recovery of a valid prefix; explicit failure on full storage or exhausted recording queue

Offline fixtures plus real session

M4: usable application

Live viewer, measurements, recording controls, playback

Same input yields the same numerical readout live and on replay; a stalled viewer does not stall capture

Real camera and recording

M5: release qualification

Windows and Pi installation guides, package, diagnostic report

Repeat M2–M4 on the two selected hosts; cold-start operation and a 30-minute recording on each

Both hosts

M6: moving-source analysis

PSF profile, point-source fit, trajectories, residual view

Known-spacing experiment and held-out-frame evaluation with uncertainty and limitations

Controlled recordings

The 30-minute run is an initial engineering gate. Report observed frame rate, timeout/recovery events, unexplained discontinuities, memory growth, storage rate, and application frame age. It does not establish long-duration unattended reliability. Use a longer run only when a concrete deployment duration warrants it. Automated checks use pytest and Ruff; hardware tests remain explicitly marked and produce evidence reports rather than passing against a simulated camera.

The proposed issue backlog can be transferred to a repository when implementation begins. Each issue ends with a reviewable artifact and its acceptance evidence.

Issue

Work

Depends on

01

Package, configuration, public types, and FrameSource contract

—

02

Host diagnostics, device enumeration, profile report

01

03

P3 command codec, frame assembler, malformed-input fixtures

01

04

Camera lifecycle, command queue, cleanup, recovery

02, 03

05

Frame quality, timing semantics, versioned temperature interpretation

03, 04

06

Lossless recording schema, replay, recovery, export

01, 05

07

Local service, browser viewer, bounded subscriptions, controls

04, 05

08

Recording and playback workflow in the viewer

06, 07

09

Windows/Pi installation and release qualification

02–08

10

Measured PSF and moving-point estimation

05, 06, 08

The driver and numerical recording remain the foundation for the research direction discussed earlier. A stationary camera can observe moving heat sources; a processor can fit their subpixel trajectories and signal strengths, then investigate reconstruction of a stable moving object's thermal pattern. Use an independently characterized PSF first, model brightness changes, and account for detector temporal response. Report source localization and two-source resolution as separate outcomes. Keep fitted or reconstructed images in derived outputs so original recordings remain reusable.

The main unresolved questions are empirical. They determine qualification details, rather than preventing us from designing the software.

Unknown

How it will be resolved

Design consequence

Does this unit match the published protocol?

Identify hardware/firmware and compare recorded transactions with a profile.

Versioned profiles; unsupported layouts fail explicitly.

How does shutter correction interrupt this stream?

Capture normal, requested-correction, and recovery intervals.

Quality flags and verified resynchronization rules.

What do the numerical values and settings already include?

Compare the app and references across gain/settings transitions.

Versioned interpretation with documented correction ownership.

What timing information is trustworthy?

Measure host receive timing and investigate device counters/response lag.

Preserve timing provenance; avoid claiming exposure synchronization.

Which Windows binding works for both interfaces?

Inspect the current binding and qualify libusb access.

Targeted host setup and a backend-aware recovery path.

Can the Pi sustain the intended recording workload?

Run capture with the actual USB devices and storage.

Measured queue/storage limits and explicit recording failure behavior.

## Session Evidence: 11 September 2026

The custom package lives in `p3thermal`. Its initial frame contract and
`FrameSource` interface have offline tests. Native Windows qualification used
Python 3.13, PyUSB 1.3.1, NumPy 2.5.3, `libusb-package` 1.0.30.0, and a WinUSB
binding installed by Zadig for `USB\VID_3474&PID_45A2&MI_00`.

Verified device profile facts:

- VID:PID is `3474:45A2`; device version is `0x0200`.
- Interface 0 is vendor class and exposes bulk endpoints `0x84` and `0x05`.
- Interface 1 alternate setting 1 is vendor class and exposes bulk endpoints
  `0x81` and `0x02`; alternate setting 0 has no endpoints.
- A read-only model transaction returned `P3` with status responses `02`, then
  `03`.
- The stream lifecycle returned start-command acknowledgements `02/01` and
  delivered a structurally validated 197656-byte frame.
- One complete frame required multiple reads in practice (five and seven reads
  in two successful probes). USB read boundaries must never define frames.

`p3thermal/tools/capture_probe.py` is the current hardware regression probe. It
returns interface 1 to alternate setting 0 and releases both interfaces even if
capture fails. The reference checkout used for protocol comparison is
`p3-ir-camera` commit `e3205dca5727682ff2d903585d1dce5a1d19f1f6`.

## Concrete Next Steps

1. Repeat native Windows start/stop and run a 30-minute recording. Report frame
   delivery rate, timeout/fault events, memory growth, storage rate, and cleanup
   outcome; it is the first M2-M4 sustained qualification gate.
2. Capture matched P3/manufacturer-application samples and an independent
   temperature reference. Only then define a versioned temperature interpretation
   and show temperature readouts in the UI.
3. Measure disconnect and timeout behavior, then implement a finite recovery
   transition and preserve its event record in sessions. Keep shutter and gain
   disabled until their behavior is separately measured.
4. Measure a PSF and controlled motion dataset before building or claiming a
   super-resolution result. Use lossless recordings as the research input.

Do not return to WSL USB/IP for Windows capture. The custom camera workflow is
native Windows; WSL remains suitable for offline decoding, replay, and analysis.
