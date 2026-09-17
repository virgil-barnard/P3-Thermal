"""Train and evaluate the compact multi-scale synthetic-data baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SyntheticUpscalingDataset
from .model import MultiScaleUpsampler


def _loader(root: Path, split: str, scale: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SyntheticUpscalingDataset(root, split, scale),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _mse(
    model: MultiScaleUpsampler, loader: DataLoader, scale: int, device: torch.device
) -> dict[str, float]:
    totals = {"nearest": 0.0, "bicubic": 0.0, "model": 0.0}
    count = 0
    model.eval()
    with torch.inference_mode():
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            nearest = F.interpolate(source, scale_factor=scale, mode="nearest")
            bicubic = F.interpolate(source, scale_factor=scale, mode="bicubic", align_corners=False)
            prediction = model(source, scale)
            batch = source.shape[0]
            totals["nearest"] += F.mse_loss(nearest, target).item() * batch
            totals["bicubic"] += F.mse_loss(bicubic, target).item() * batch
            totals["model"] += F.mse_loss(prediction, target).item() * batch
            count += batch
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared-trunk P3 synthetic upsampler")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    model = MultiScaleUpsampler(args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loaders = {
        scale: _loader(args.dataset, "development", scale, args.batch_size, True)
        for scale in (2, 3, 4)
    }
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for scale, loader in loaders.items():
            for source, target in loader:
                source, target = source.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(source, scale), target)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * source.shape[0]
                count += source.shape[0]
        print(json.dumps({"epoch": epoch + 1, "train_mse": loss_sum / count}), flush=True)
    metrics = {
        "device": str(device),
        "target": "acquisition_hr*_dn",
        "validation": {
            str(scale): _mse(
                model,
                _loader(args.dataset, "validation", scale, args.batch_size, False),
                scale,
                device,
            )
            for scale in (2, 3, 4)
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = {"channels": args.channels, "state_dict": model.cpu().state_dict()}
    torch.save(checkpoint, args.output / "checkpoint.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
