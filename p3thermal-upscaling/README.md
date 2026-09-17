# P3 Thermal Upscaling

GPU-only research environment for training and exporting one compact neural
upsampler from `thermal-calibration-lab` synthetic datasets. It deliberately
does not share dependencies with the qualified P3 acquisition environment.

The current generator contract supplies 2x, 3x and 4x labels. The model has a
shared feature trunk and one reconstruction head per supported scale, so one
checkpoint/artifact supports all three. There is no 8x supervision or model head
yet; add it only with a validated 8x generator contract.

## Windows setup

Run all commands through this project's virtual environment, never base Python.
From native Windows PowerShell:

```powershell
cd C:\Users\athan\thermal\p3thermal-upscaling
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Select the CUDA wheel matching the installed NVIDIA driver from pytorch.org.
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not install the default PyPI CPU wheel before the CUDA wheel. If CUDA 12.8 is
not compatible with the driver, replace `cu128` with the supported index shown
by the PyTorch installer selector.

## Train and export

Generate a dataset with `thermal-calibration-lab/.venv/Scripts/python.exe`, then
train with groups intact. The `development` split is training data and
`validation` is held out. Calibration and challenge clips are excluded by
default.

```powershell
.\.venv\Scripts\python.exe -m p3thermal_upscaling.train `
  --dataset ..\thermal-calibration-lab\generated\smoke `
  --epochs 10 --batch-size 4 --output runs\baseline
.\.venv\Scripts\python.exe -m p3thermal_upscaling.export `
  --checkpoint runs\baseline\checkpoint.pt --output artifacts\p3-upscaler.ts
```

The default target is `acquisition_hr*_dn`, which retains the synthetic optics
and exposure blur. Invalid input frames are skipped and `frame_to_truth` maps
each observation to its target. The command writes per-scale nearest/bicubic and
model MSE metrics to `metrics.json`; these measure only the declared simulator.

The available `p3-upscaler-starter-baseline.ts` was trained on the validated
42-clip starter corpus. It is intentionally not named or treated as the final
model; a comprehensive independent-repetition corpus and real-P3 evaluation
remain required for that claim.

`p3-upscaler.ts` is a TorchScript artifact whose `forward(raw, scale)` accepts a
float tensor shaped `(N, 1, 192, 256)` in native DN units and returns native-DN
output at the selected 2x, 3x, or 4x resolution. It is intended for bounded
server-side inference in the native Windows camera service. Browser WebGPU is
not an initial target: it would require duplicate WGSL kernels and separate
artifact/preprocessing validation before it can be faster and equivalent.

## Live Inference Sidecar

Run the Torch-owning loopback service from this environment, then point the
qualified camera service at it. The sidecar accepts only local requests and
never accesses USB hardware:

```powershell
.\.venv\Scripts\python.exe -m p3thermal_upscaling.serve `
  --artifact artifacts\p3-upscaler-confirmation.ts --port 8766
```

In a separate PowerShell, run the camera viewer with
`--inference-url http://127.0.0.1:8766`. This isolates Torch from the qualified
camera environment while keeping learned preview latency local. Start with
replay, then explicitly measure latency, skipped previews, and recording
noninterference before a hardware session.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```
