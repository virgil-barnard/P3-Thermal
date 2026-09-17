"""Dataset storage separates observed inputs from simulator-only truth."""

import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from PIL import Image

from .config import Camera, Scenario
from .render import Detector, Renderer


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _json_value(value):
    """Normalize tuples and arrays to their persisted JSON representation."""
    return json.loads(json.dumps(value, allow_nan=False))


def _write_png(path, data, limits):
    # Lossless preview only: numerical measurements always use NPZ float/uint arrays.
    image = np.rint(np.clip((data - limits[0]) / (limits[1] - limits[0]), 0, 1) * 255).astype(
        np.uint8
    )
    Image.fromarray(image).save(path, optimize=True)


def generate_clip(scenario, destination):
    """Render a whole sequence. Intermediate files are private until complete."""
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=False)
    preview = folder / "preview"
    preview.mkdir()
    renderer, detector = Renderer(scenario), Detector(scenario)
    a, c = scenario.acquisition, scenario.camera
    frames, valid, reset, epochs, frame_to_truth, labels = [], [], [], [], [], []
    unique, lookup = [], {}
    low_t = a.background_c + min(0, a.contrast_c * 1.7)
    high_t = a.background_c + max(0, a.contrast_c * 1.7)
    if high_t - low_t < 0.5:
        low_t, high_t = a.background_c - 0.25, a.background_c + 0.25
    limits = [
        a.dn_offset
        + a.dn_per_radiance
        * (renderer.tau * renderer.emitted(t) + (1 - renderer.tau) * renderer.atmosphere)
        for t in (low_t, high_t)
    ]
    for index in range(a.frames):
        timestamp = index / a.fps
        key = renderer.key(timestamp)
        if key not in lookup:
            lookup[key] = len(unique)
            sample = renderer.render(timestamp)
            unique.append(sample)
            uid = lookup[key]
            for scale in (2, 3, 4):
                _write_png(preview / f"truth{scale}_{uid:03d}.png", sample["ideal"][scale], limits)
                _write_png(
                    preview / f"acquisition{scale}_{uid:03d}.png",
                    sample["acquisition_hr"][scale],
                    limits,
                )
            _write_png(preview / f"optical_{uid:03d}.png", sample["optical_native_dn"], limits)
        uid = lookup[key]
        sample = unique[uid]
        observed, event = detector.sample(sample["optical_native_dn"], index)
        frames.append(observed)
        valid.append(event["valid"])
        reset.append(event["reset"])
        epochs.append(event["epoch"])
        frame_to_truth.append(uid)
        label = sample["label"] | {
            "frame": index,
            "timestamp_s": timestamp,
            "truth_index": uid,
            **event,
        }
        labels.append(label)
        _write_png(preview / f"observed_{index:03d}.png", observed, limits)

    np.savez_compressed(
        folder / "input.npz",
        frames=np.stack(frames),
        timestamps=np.arange(a.frames, dtype=np.float64) / a.fps,
        sequence=np.arange(a.frames, dtype=np.int64),
        valid=np.array(valid),
        reset=np.array(reset),
        epoch=np.array(epochs, dtype=np.int64),
    )
    # Repeated static/dither poses share exact ideal arrays. The index is explicit.
    arrays = {
        f"ideal_hr{scale}_dn": np.stack([sample["ideal"][scale] for sample in unique])
        for scale in (2, 3, 4)
    }
    arrays.update(
        {
            f"acquisition_hr{scale}_dn": np.stack(
                [sample["acquisition_hr"][scale] for sample in unique]
            )
            for scale in (2, 3, 4)
        }
    )
    arrays.update(
        optical_native_dn=np.stack([s["optical_native_dn"] for s in unique]),
        coverage_hr4=np.stack([s["coverage_hr4"] for s in unique]),
        frame_to_truth=np.array(frame_to_truth, dtype=np.int32),
        sensor_gain=detector.gain,
        sensor_offset_dn=detector.offset,
        defective_pixels=detector.bad,
    )
    np.savez_compressed(folder / "truth.npz", **arrays)
    metadata = {
        "format_version": 1,
        "scenario": scenario.to_dict(),
        "K_native": c.matrix().tolist(),
        "K_hr": {str(s): c.matrix(s).tolist() for s in (2, 3, 4)},
        "T_world_from_board": renderer.t_wb.tolist(),
        "coordinate_convention": "meters; x right, y down, z forward; integer native pixel centers",
        "true_effective_psf_sigma_yx_native_px": list(renderer.sigmas),
        "true_effective_halo_sigma_native_px": renderer.halo_sigma,
        "read_noise_std_dn_at_background": detector.read_std,
        "display_limits_dn": limits,
        "unique_truth_poses": len(unique),
        "frames": labels,
        "sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("input.npz", "truth.npz")
        },
    }
    _json(folder / "metadata.json", metadata)
    return {
        "id": scenario.id,
        "group_id": scenario.group_id,
        "split": scenario.split,
        "description": scenario.description,
        "target": scenario.target.kind,
        "profile": scenario.target.profile,
        "distance_m": scenario.distance_m,
        "contrast_c": a.contrast_c,
        "motion": scenario.camera_motion,
        "frames": a.frames,
        "unique_truth_poses": len(unique),
        "input": f"clips/{scenario.id}/input.npz",
        "truth": f"clips/{scenario.id}/truth.npz",
        "metadata": f"clips/{scenario.id}/metadata.json",
        "K_native": c.matrix().tolist(),
        "distortion": list(c.distortion),
        "display_limits_dn": limits,
        "preview_frames": labels,
        "nominal_projected_mm_per_pixel": 1000 * scenario.distance_m / c.fx,
    }


def _manifest_row(scenario, metadata):
    c, a = scenario.camera, scenario.acquisition
    return {
        "id": scenario.id,
        "group_id": scenario.group_id,
        "split": scenario.split,
        "description": scenario.description,
        "target": scenario.target.kind,
        "profile": scenario.target.profile,
        "distance_m": scenario.distance_m,
        "contrast_c": a.contrast_c,
        "motion": scenario.camera_motion,
        "frames": a.frames,
        "unique_truth_poses": metadata["unique_truth_poses"],
        "input": f"clips/{scenario.id}/input.npz",
        "truth": f"clips/{scenario.id}/truth.npz",
        "metadata": f"clips/{scenario.id}/metadata.json",
        "K_native": c.matrix().tolist(),
        "distortion": list(c.distortion),
        "display_limits_dn": metadata["display_limits_dn"],
        "preview_frames": metadata["frames"],
        "nominal_projected_mm_per_pixel": 1000 * scenario.distance_m / c.fx,
    }


def generate_dataset(scenarios, destination, progress=True, resume=False):
    scenarios = list(scenarios)
    ids, groups = set(), {}
    for s in scenarios:
        s.validate()
        if s.id in ids:
            raise ValueError(f"Duplicate scenario id {s.id}")
        ids.add(s.id)
        if s.group_id in groups and groups[s.group_id] != s.split:
            raise ValueError("A scenario group cannot cross dataset splits")
        groups[s.group_id] = s.split
    root = Path(destination)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(
            "Output directory must be empty; choose a new path to preserve prior data"
        )
    root.mkdir(parents=True, exist_ok=True)
    clips = root / "clips"
    clips.mkdir(exist_ok=True)
    request = _json_value({"scenarios": [s.to_dict() for s in scenarios]})
    request_path = root / "generation_request.json"
    if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise ValueError("Existing generation request does not match the requested scenarios")
    _json(request_path, request)
    rows, started = [], time.perf_counter()
    for index, scenario in enumerate(scenarios):
        one = time.perf_counter()
        folder = clips / scenario.id
        metadata_path = folder / "metadata.json"
        if resume and metadata_path.exists() and (folder / "input.npz").exists() and (folder / "truth.npz").exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["scenario"] != _json_value(scenario.to_dict()):
                raise ValueError(f"Existing clip does not match requested scenario: {scenario.id}")
            row = _manifest_row(scenario, metadata)
        elif folder.exists():
            if not resume:
                raise FileExistsError(f"Incomplete existing clip cannot be resumed: {folder}")
            # generate_clip writes metadata last, so this directory is private staging data.
            shutil.rmtree(folder)
            row = generate_clip(scenario, folder)
        else:
            row = generate_clip(scenario, folder)
        rows.append(row)
        if progress:
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(scenarios),
                        "id": scenario.id,
                        "seconds": round(time.perf_counter() - one, 2),
                        "unique_truth_poses": row["unique_truth_poses"],
                    }
                ),
                flush=True,
            )
    nominal = Camera() if not scenarios else scenarios[0].camera
    manifest = {
        "format": "thermal-calibration-lab",
        "format_version": 1,
        "kind": "Synthetic physical-target dataset; no measured P3 calibration",
        "nominal_camera": asdict(nominal),
        "nominal_K": nominal.matrix().tolist(),
        "nominal_reconstruction_sigma_px": 0.35,
        "truth_scales": [2, 3, 4],
        "clips": rows,
        "total_frames": sum(r["frames"] for r in rows),
        "generation_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": [
            "Assumed camera and spectral response; not actual P3 calibration",
            "Planar prescribed temperature fields, not heat-transfer simulation",
            "Finite quadrature and boundary subsampling, not analytic image truth",
            "Challenge cases are diagnostics, not independent natural-scene validation",
            "No vendor X3 implementation or reconstruction model is included",
        ],
    }
    _json(root / "manifest.json", manifest)
    browser = json.dumps(manifest, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (root / "preview_data.js").write_text(
        "window.THERMAL_DATA=" + browser + ";\n", encoding="utf-8"
    )
    template = Path(__file__).with_name("static") / "viewer.html"
    (root / "preview.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return manifest


def iter_inputs(clip_directory):
    """Only observed data/events; does not open simulator truth or label metadata."""
    with np.load(Path(clip_directory) / "input.npz", allow_pickle=False) as data:
        # NPZ members decompress on access; load each member once per sequence.
        arrays = {name: data[name] for name in data.files}
    for i in range(len(arrays["frames"])):
        yield {
            "image": arrays["frames"][i].copy(),
            "sequence": int(arrays["sequence"][i]),
            "timestamp": float(arrays["timestamps"][i]),
            "valid": bool(arrays["valid"][i]),
            "reset": bool(arrays["reset"][i]),
            "epoch": int(arrays["epoch"][i]),
        }


def load_truth_frame(clip_directory, frame, scale=2):
    """Explicit evaluation-only loader. Returned ideal target is instantaneous."""
    if scale not in (2, 3, 4):
        raise ValueError("Truth scales are 2, 3 and 4")
    with np.load(Path(clip_directory) / "truth.npz", allow_pickle=False) as truth:
        index = int(truth["frame_to_truth"][frame])
        return {
            "ideal_dn": truth[f"ideal_hr{scale}_dn"][index].copy(),
            "acquisition_hr_dn": truth[f"acquisition_hr{scale}_dn"][index].copy(),
            "optical_native_dn": truth["optical_native_dn"][index].copy(),
            "coverage_hr4": truth["coverage_hr4"][index].copy(),
        }


def validate_dataset(path, verify_hashes=True):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    rows, groups = [], {}
    for clip in manifest["clips"]:
        folder = root / "clips" / clip["id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        s = Scenario.from_dict(metadata["scenario"])
        groups.setdefault(s.group_id, set()).add(s.split)
        with (
            np.load(folder / "input.npz", allow_pickle=False) as inputs,
            np.load(folder / "truth.npz", allow_pickle=False) as truth,
        ):
            count, h, w = inputs["frames"].shape
            assert (count, h, w) == (s.acquisition.frames, s.camera.height, s.camera.width)
            assert inputs["frames"].dtype == np.uint16
            assert np.all(np.diff(inputs["timestamps"]) > 0)
            indices = truth["frame_to_truth"]
            assert len(indices) == count and np.all(indices >= 0)
            assert np.max(indices) < len(truth["optical_native_dn"])
            for scale in (2, 3, 4):
                array = truth[f"ideal_hr{scale}_dn"]
                assert array.shape == (len(truth["optical_native_dn"]), h * scale, w * scale)
                assert np.isfinite(array).all()
                optical_hr = truth[f"acquisition_hr{scale}_dn"]
                assert optical_hr.shape == array.shape and np.isfinite(optical_hr).all()
                integrated = optical_hr.reshape(len(optical_hr), h, scale, w, scale).mean(
                    axis=(2, 4)
                )
                np.testing.assert_allclose(
                    integrated, truth["optical_native_dn"], rtol=5e-7, atol=0.002
                )
            coverage = truth["coverage_hr4"]
            assert np.min(coverage) >= 0 and np.max(coverage) <= 1
            if verify_hashes:
                for name, digest in metadata["sha256"].items():
                    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            rows.append(
                {
                    "id": s.id,
                    "frames": count,
                    "valid_frames": int(inputs["valid"].sum()),
                    "unique_truth_poses": len(truth["optical_native_dn"]),
                    "status": "passed",
                }
            )
    assert all(len(roles) == 1 for roles in groups.values())
    result = {
        "clips": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "all_passed": True,
        "checks": rows,
    }
    _json(root / "validation.json", result)
    return result
