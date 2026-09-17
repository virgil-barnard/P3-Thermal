"""python examples/read_sequence.py DATASET/clips/CLIP_ID"""

import sys

from thermal_calibration import iter_inputs, load_truth_frame

clip = sys.argv[1]
for observed in iter_inputs(clip):
    if not observed["valid"]:
        continue
    frame = observed["image"]  # uint16, H x W; feed only this branch to a learner.
    # Evaluation lives in a separate branch; never use truth as a model input.
    truth = load_truth_frame(clip, observed["sequence"], scale=3)
    print(observed["sequence"], frame.shape, truth["ideal_dn"].shape, "reset", observed["reset"])
