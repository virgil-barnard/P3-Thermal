"""Explicit loader for synthetic observations and their matched acquisition targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticUpscalingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load one scale/split without mixing incompatible output sizes in a batch."""

    def __init__(self, root: str | Path, split: str, scale: int) -> None:
        if scale not in (2, 3, 4):
            raise ValueError("scale must be one of 2, 3, or 4")
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for clip in manifest["clips"]:
            if clip["split"] != split:
                continue
            folder = root / "clips" / clip["id"]
            with np.load(folder / "input.npz", allow_pickle=False) as inputs:
                frames = inputs["frames"]
                valid = inputs["valid"]
            with np.load(folder / "truth.npz", allow_pickle=False) as truth:
                target = truth[f"acquisition_hr{scale}_dn"]
                indices = truth["frame_to_truth"]
            for frame_index, image in enumerate(frames):
                if valid[frame_index]:
                    self.samples.append((image.copy(), target[int(indices[frame_index])].copy()))
        if not self.samples:
            raise ValueError(f"no valid {split!r} samples in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, target = self.samples[index]
        source = torch.from_numpy(image.astype(np.float32, copy=False)).unsqueeze(0)
        reference = torch.from_numpy(target.astype(np.float32, copy=False)).unsqueeze(0)
        center = source.mean()
        spread = source.std().clamp_min(16.0)
        return (source - center) / spread, (reference - center) / spread
