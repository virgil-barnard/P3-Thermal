"""Optional integration with the previously delivered thermal-online-sr package.

python examples/compare_online_sr.py DATASET/clips/CLIP_ID --scale 2
This runs ordered/offline and reports all valid frames, including
warmup and fallbacks. No truth or estimated camera parameters are passed to the SR engine.
"""

import argparse
import json

import numpy as np

from thermal_calibration import iter_inputs, load_truth_frame

parser = argparse.ArgumentParser()
parser.add_argument("clip")
parser.add_argument("--scale", type=int, choices=(2, 3, 4), default=2)
parser.add_argument("--sigma", type=float, default=0.35, help="Declared reconstruction assumption")
args = parser.parse_args()

import torch
from thermal_online_sr.engine import Config, OnlineSR

torch.set_num_threads(2)
engine = OnlineSR(Config(scale=args.scale, sigma=args.sigma))
rows = []
for sample in iter_inputs(args.clip):
    if not sample["valid"]:
        continue
    result = engine.process(
        sample["image"],
        sequence=sample["sequence"],
        timestamp=sample["timestamp"],
        epoch=sample["epoch"],
        reset=sample["reset"],
    )
    truth = load_truth_frame(args.clip, sample["sequence"], args.scale)["ideal_dn"]
    border = 4 * args.scale
    rows.append(
        {
            "frame": sample["sequence"],
            "status": result.metrics["status"],
            "mse_dn2": {
                k: float(
                    np.mean(
                        (
                            v[border:-border, border:-border].astype(np.float64)
                            - truth[border:-border, border:-border]
                        )
                        ** 2
                    )
                )
                for k, v in result.images.items()
                if k != "support"
            },
        }
    )
print(
    json.dumps(
        {"note": "Image MSE, not task accuracy; includes warmup/fallback frames", "rows": rows},
        indent=2,
    )
)
