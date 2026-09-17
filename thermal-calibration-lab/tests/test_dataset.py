import json
from dataclasses import replace

import numpy as np
import pytest

from thermal_calibration.config import Acquisition, Camera, Scenario, Target
from thermal_calibration.dataset import (
    generate_dataset,
    iter_inputs,
    load_truth_frame,
    validate_dataset,
)
from thermal_calibration.scenarios import starter_suite


def test_dataset_roundtrip_input_truth_separation_and_preview(tmp_path):
    scenario = Scenario(
        id="tiny",
        group_id="tiny",
        camera=Camera(width=32, height=24, cx=15.5, cy=11.5),
        acquisition=Acquisition(frames=4),
        target=Target(kind="circle"),
    )
    root = tmp_path / "dataset"
    manifest = generate_dataset([scenario], root, progress=False)
    assert manifest["total_frames"] == 4
    assert validate_dataset(root)["all_passed"]
    folder = root / "clips" / "tiny"
    inputs = list(iter_inputs(folder))
    assert set(inputs[0]) == {"image", "sequence", "timestamp", "valid", "reset", "epoch"}
    assert inputs[0]["image"].dtype == np.uint16
    truth = load_truth_frame(folder, 3, scale=3)
    assert truth["ideal_dn"].shape == (72, 96)
    with np.load(folder / "truth.npz") as data:
        assert len(data["ideal_hr4_dn"]) == 1
        np.testing.assert_array_equal(data["frame_to_truth"], [0, 0, 0, 0])
    assert (root / "preview.html").is_file() and (root / "preview_data.js").is_file()
    assert (folder / "preview" / "truth4_000.png").is_file()
    restored = Scenario.from_dict(json.loads((folder / "metadata.json").read_text())["scenario"])
    np.testing.assert_allclose(restored.camera.matrix(), scenario.camera.matrix())
    with pytest.raises(FileExistsError):
        generate_dataset([scenario], root, progress=False)


def test_group_boundary_and_duplicate_ids_rejected(tmp_path):
    original = Scenario()
    with pytest.raises(ValueError, match="group"):
        generate_dataset([original, replace(original, id="second", split="validation")], tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        generate_dataset([original, original], tmp_path)


def test_suite_has_balanced_orientation_and_matched_motion_controls():
    cases = starter_suite()
    assert len(cases) == 42
    triangles = [s for s in cases if s.target.kind == "triangle" and s.split == "validation"]
    for distance in (1.0, 2.0):
        assert sorted(s.target.orientation_deg for s in triangles if s.distance_m == distance) == [
            0,
            90,
            180,
            270,
        ]
    paired = [s for s in cases if s.group_id == "pair_028_d100"]
    assert len(paired) == 2
    assert paired[0].target == paired[1].target
    assert paired[0].acquisition == paired[1].acquisition
