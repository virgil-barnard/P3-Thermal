"""Export a trained checkpoint to the single live-inference TorchScript artifact."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

from .model import MultiScaleUpsampler, RawDnUpsampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a P3 upsampling checkpoint to TorchScript")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MultiScaleUpsampler(int(checkpoint["channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    # TorchScript remains the portable single-artifact format supporting runtime
    # scale selection. PyTorch 2.11 deprecates its API but still supports it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        artifact = torch.jit.script(RawDnUpsampler(model.eval()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact.save(str(args.output))
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps({"format": "p3thermal-upscaler", "version": 1, "scales": [2, 3, 4]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
