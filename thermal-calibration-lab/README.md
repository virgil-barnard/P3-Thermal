# Thermal Calibration Lab

Generate **physical-target synthetic thermal sequences** with known dimensions,
distances, camera intrinsics/extrinsics, temperature fields, optics, motion and
sensor noise. The default camera has native **256×192** samples. Each sequence
includes noisy observations and separate simulator-only references at **2×, 3×
and 4×**. An offline HTML inspector provides synchronized crops and playback.

This is a controlled **assumed camera**, not a calibrated P3, a reproduction of
X³IR, a thermal heat-transfer solver, or evidence of real-camera SR performance.
It is designed to expose those assumptions and make controlled tests possible.

## The included dataset

The separately supplied `thermal-calibration-dataset.zip` contains **42 sequences,
16 frames each: 672 native observations**. Open `preview.html` inside the extracted
dataset folder. Keep its `preview_data.js` and `clips` directories beside it.
The inspector needs no Python, server, cloud, or network connection.

| Role | Contents |
|---|---|
| Calibration | Two flat temperatures; two slanted-edge directions; finite pinhole and slit; four bars; three poses of a 7×5 aperture grid |
| Development / validation | Paired stationary and 16-phase dither sequences: 1 mm apertures, 2.8/5.6 mm center spacing, 0.5/1/2 m distances |
| Validation | Four triangle orientations at 1 and 2 m; equal-area single aperture; empty control |
| Challenge | Increased noise; wider/anisotropic blur; focus mismatch; independent target motion; foreground occlusion and camera roll; correction/drift/defects; lens distortion and board tilt; a cold target; linear and Gaussian temperature profiles |

These are illustrative controlled scenarios, not enough independent captures for
confidence intervals or trained-classifier claims. Challenge cases deliberately
reuse simple target families to isolate acquisition effects. Generate additional
captures, geometry and nuisance distributions before making generalization claims.

## Install and generate

In Windows Command Prompt, from this project directory:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m thermal_calibration generate --suite starter --output my-dataset
.venv\Scripts\python -m thermal_calibration validate my-dataset
start my-dataset\preview.html
```

On Linux/macOS, use `.venv/bin/python`. Only NumPy, SciPy and Pillow are runtime
dependencies. Neither TensorFlow nor PyTorch is needed for generation. The
project also installs the `thermal-calibration` command.

For eight representative cases use `--suite quick`. For several new capture
realizations use `--suite expanded --repetitions 3`. Set `--seed` explicitly.
The default 16 frames complete the 4×4 dither grid; a shorter sequence contains
only its initial phases. Generation refuses to overwrite a nonempty directory.
If an interrupted build has complete clips but no final manifest, repeat the
identical command with `--resume`. It verifies the original generation request,
reuses only complete scenario-matching clips, and regenerates incomplete
per-clip staging data.

For tighter numerical integration use `--supersample 24 --boundary-samples 8`.
This uses more memory and time. Supersample factors are multiples of 12 so the
2×/3×/4× reference grids share exactly the same field of view and integration.

## Assumed camera and coordinates

The default native intrinsic matrix is

```text
K = [[358.333333,   0.000000, 127.500000],
     [  0.000000, 358.333333,  95.500000],
     [  0.000000,   0.000000,   1.000000]]
```

The focal length in pixels is the initial assumption 4.3 mm / 0.012 mm. These
published nominal P3 dimensions motivate the example; they do not constitute
an empirical intrinsic calibration. In particular, focus-dependent intrinsics
and macro behavior are not modeled automatically.

- World/camera axes: x right, y down, z forward; physical coordinates in meters.
- Native pixel centers: `(u,v)=(0,0), (1,0), ...`; first pixel spans `[-0.5,0.5]`.
- `T_camera_from_world` maps a homogeneous world point into the camera frame.
- `T_world_from_board` maps the aperture plane's local x/y coordinates to world.
- Distance is the world z position of the board origin; it is not slant range to
  every point on a tilted plane.
- Intrinsics on a scale-s grid preserve pixel boundaries:
  `fx_s=s*fx`, `fy_s=s*fy`, `cx_s=s*(cx+0.5)-0.5`, and similarly for `cy`.
- Optional distortion uses `(k1,k2,p1,p2,k3)` in normalized camera coordinates.

An on-axis 2.8 mm separation projects to 2.0067, 1.0033 and 0.5017 native pixels
at 0.5, 1 and 2 m, respectively. Per-frame projected keypoints are saved; use
them instead of this small-angle example for tilted/distorted cases.

## Customize geometry, distances and the camera

Write a reproducible request, edit its explicit parameters, then generate:

```bat
python -m thermal_calibration plan --suite starter --output scenarios.json
python -m thermal_calibration generate --config scenarios.json --output custom-dataset
```

The config has a `scenarios` list. Camera fields are `width`, `height`, `fx`,
`fy`, `cx`, `cy`, and `distortion`. Target lengths have `_m` suffixes. Image
blur sigmas are in **native pixels**, so changing output scale does not change
the underlying optics. Target orientation is in degrees; an upright triangle
has its apex in negative board y before board/camera transforms.

Python construction works without the CLI:

```python
from thermal_calibration import Camera, Target, Scenario, Acquisition, generate_dataset

case = Scenario(
    id="my_pair",
    group_id="my_pair",
    split="validation",
    camera=Camera(fx=360, fy=358, cx=127.5, cy=95.5),
    distance_m=1.25,
    target=Target(kind="pair", radius_m=0.0005, separation_m=0.003),
    camera_motion="dither4",
    acquisition=Acquisition(contrast_c=1.0, frames=16),
)
generate_dataset([case], "my-pair-dataset")
```

## Two different high-resolution references

The distinction is intentional:

1. **`ideal_hr{2,3,4}_dn`:** instantaneous scene radiance projected through the
   camera, integrated over each fine pixel, **before optical and exposure blur**.
   Use this to assess combined deblurring and reconstruction.
2. **`acquisition_hr{2,3,4}_dn`:** the same assumed optics and finite exposure as
   the actual observations, sampled on a finer detector grid, without detector
   noise/nonuniformity. Use this to assess finer sampling without asking the
   method to undo the optics or motion blur.

`optical_native_dn` is the corresponding clean native acquisition. Integrating
each acquisition-HR reference back to native cells reproduces it within float
roundoff. The inspector's reference selector switches between the two meanings.

All reference arrays use the same arbitrary **linear digital-number (DN)** scale
as observed frames. They are not Celsius, P3 temperature codes, or an assertion
that actual P3 samples have linear radiance response. Metadata records the exact
conversion assumed in the simulator.

## File contract

Each `clips/CLIP_ID` directory contains:

| File / key | Meaning |
|---|---|
| `input.npz: frames` | uint16 `(T,H,W)` observed frames |
| `timestamps`, `sequence` | seconds and frame indices |
| `valid`, `reset`, `epoch` | capture validity and continuity events |
| `truth.npz: frame_to_truth` | index from each frame into shared reference arrays |
| `ideal_hr2_dn`, `ideal_hr3_dn`, `ideal_hr4_dn` | float32 `(N,H*s,W*s)` pre-optics reference |
| `acquisition_hr2_dn`, `acquisition_hr3_dn`, `acquisition_hr4_dn` | float32 `(N,H*s,W*s)` optical/exposure reference |
| `optical_native_dn` | float32 `(N,H,W)` clean native acquisition |
| `coverage_hr4` | float32 visible aperture coverage, including synthetic occlusion |
| `sensor_gain`, `sensor_offset_dn`, `defective_pixels` | simulator-only detector truth |
| `metadata.json` | full scenario, K at each scale, physical/keypoint labels, per-frame poses/events, file hashes |
| `preview/` | lossless PNG previews with a fixed, common per-sequence display mapping |

**N may be less than T.** Identical physical poses share a reference to avoid
storing repeated large arrays. Observations retain T independent noise samples.
Always use `frame_to_truth`, or the loader below, when associating references.
The truth NPZ uses lazy zip members: load an array once when iterating over many
frames instead of repeatedly decompressing its entire member.

```python
from thermal_calibration import iter_inputs, load_truth_frame

for sample in iter_inputs("my-dataset/clips/my_pair"):
    if not sample["valid"]:
        continue
    image = sample["image"]  # only this branch enters a self-supervised learner
    truth = load_truth_frame("my-dataset/clips/my_pair", sample["sequence"], scale=2)
    # truth["ideal_dn"] and truth["acquisition_hr_dn"] are evaluation-only here.
```

The arrays are directly usable with NumPy, TensorFlow/Keras or PyTorch. Keep
capture groups intact when dividing data. Do not randomly split adjacent frames
or the stationary/dither siblings. Calibration, development, validation and
challenge roles are in the manifest; they are not a guarantee of natural-scene
generalization.

## Connect to the earlier SR prototype

Ordinary clips can be opened by its `.npz` replay reader using `input.npz`.
For event cases, use the explicit event-aware example:

```text
python examples/compare_online_sr.py ../thermal-calibration-dataset/clips/pair_028_d100_dither4 --scale 2
```

That example requires the earlier `thermal-online-sr` package and PyTorch. It
passes only observed frames and continuity flags to the engine. The old generic
NPZ replay reader does not interpret the additional event arrays, so it must not
be used to assess correction handling without adaptation. The example reports
MSE including warmup/fallbacks, not detection or classification accuracy.

## Acquisition assumptions and limitations

See `docs/MODEL.md` for the full forward model and evaluation guidance.
The default PSF is an anisotropic Gaussian core (0.45×0.60 native pixels) with
a 2.5% broad halo; the nominal reconstruction assumption remains 0.35 pixels.
This deliberate mismatch prevents treating a perfectly known, matching blur
operator as evidence of robustness. Wider blur and defocus are separate cases.

The simulator includes independent read noise, gain/offset nonuniformity,
correlated row/column noise, quantization, optional defective pixels and drift,
finite exposure, different-depth occlusion, and an explicitly marked correction
event. It does not claim a measured P3 noise distribution, lens PSF or ISP.

Run the mathematical and storage checks with:

```text
python -m pip install -e ".[dev]"
python -m pytest -q
```

Validation evidence, including numerical-integration checks and inspector smoke
tests, is packaged under `evidence/`. There is no trained reconstruction model
or claimed SR improvement in this deliverable.
