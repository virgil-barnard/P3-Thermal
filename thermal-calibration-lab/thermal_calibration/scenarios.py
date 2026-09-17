"""A declared scenario design, not randomized frames from one cartoon scene."""

from dataclasses import replace

import numpy as np

from .config import Acquisition, Camera, Scenario, Target
from .render import random_stream


def starter_suite(seed=20260916, frames=16, supersample=12, boundary_samples=4, camera=None):
    camera = camera or Camera()
    acquisition = Acquisition(
        frames=frames, supersample=supersample, boundary_samples=boundary_samples
    )
    suite = []

    def add(
        name,
        target,
        *,
        split="calibration",
        distance=1.0,
        group=None,
        contrast=2.0,
        motion="stationary",
        description="",
        overrides=None,
        **kwargs,
    ):
        rng = random_stream(seed, f"geometry:{group or name}")
        # Geometry is independently placed at a subpixel phase. Sibling motion
        # sequences share their exact target and noise realizations.
        center = (
            float(rng.uniform(-0.4, 0.4) * distance / camera.fx),
            float(rng.uniform(-0.4, 0.4) * distance / camera.fy),
        )
        target = replace(target, center_m=center)
        a = replace(acquisition, contrast_c=contrast, **(overrides or {}))
        scenario = Scenario(
            id=name,
            group_id=group or name,
            split=split,
            description=description or name.replace("_", " "),
            distance_m=distance,
            target=target,
            camera=camera,
            acquisition=a,
            camera_motion=motion,
            seed=seed,
            **kwargs,
        )
        scenario.validate()
        suite.append(scenario)

    add(
        "flat_20c",
        Target(kind="flat"),
        contrast=0,
        description="Uniform 20 C field: detector noise",
    )
    add(
        "flat_25c",
        Target(kind="flat"),
        contrast=5,
        description="Uniform 25 C field: response and noise",
    )
    add(
        "edge_horizontal_response",
        Target(kind="edge", orientation_deg=5),
        contrast=5,
        group="edges",
        description="Nearly vertical slanted edge; horizontal spatial response",
    )
    add(
        "edge_vertical_response",
        Target(kind="edge", orientation_deg=95),
        contrast=5,
        group="edges",
        description="Nearly horizontal slanted edge; vertical spatial response",
    )
    add(
        "finite_pinhole",
        Target(kind="circle", radius_m=0.0005),
        distance=0.5,
        contrast=5,
        description="1 mm diameter aperture at 0.5 m; never a delta source",
    )
    add(
        "finite_slit",
        Target(kind="slit", width_m=0.0008, height_m=0.025),
        contrast=5,
        description="0.8 mm by 25 mm slit at 1 m",
    )
    add(
        "four_bars",
        Target(kind="bars", width_m=0.002, height_m=0.03),
        contrast=2,
        description="Four finite bars; 2 mm width and equal gaps at 1 m",
    )
    for index, (distance, angles) in enumerate(
        [(0.5, (0, 0, 0)), (0.7, (15, -18, 7)), (1.0, (-12, 20, -8))]
    ):
        add(
            f"fiducial_grid_view{index}",
            Target(kind="grid", radius_m=0.0015),
            distance=distance,
            contrast=5,
            board_angles_deg=angles,
            group="fiducial_grid",
            description=f"7 by 5 aperture grid, 15 mm pitch, board pose {index}, {distance:g} m",
        )

    for separation in (0.0028, 0.0056):
        role = "development" if separation == 0.0028 else "validation"
        for distance in (0.5, 1.0, 2.0):
            group = f"pair_{round(separation * 10000):03d}_d{round(distance * 100):03d}"
            for motion in ("stationary", "dither4"):
                add(
                    f"{group}_{motion}",
                    Target(kind="pair", separation_m=separation),
                    split=role,
                    distance=distance,
                    group=group,
                    motion=motion,
                    description=f"1 mm apertures, {separation * 1000:g} mm center spacing, {distance:g} m; {motion}",
                )

    for distance in (1.0, 2.0):
        for orientation in (0, 90, 180, 270):
            group = f"triangle_d{round(distance * 100):03d}"
            add(
                f"{group}_rot{orientation:03d}",
                Target(kind="triangle", width_m=0.008, orientation_deg=orientation),
                split="validation",
                distance=distance,
                group=group,
                motion="dither4",
                description=f"8 mm equilateral triangle, {orientation} deg orientation, {distance:g} m",
            )

    add(
        "equal_area_single",
        Target(kind="circle", radius_m=np.sqrt(2) * 0.0005),
        split="validation",
        motion="dither4",
        description="One aperture with the area of two 1 mm apertures",
    )
    add(
        "blank_control",
        Target(kind="blank"),
        split="validation",
        motion="dither4",
        description="No target: measure false positives and false splitting",
    )

    common = {"split": "challenge", "motion": "dither4"}
    pair = Target(kind="pair", separation_m=0.0028)
    add("challenge_read_noise", pair, overrides={"read_noise_equivalent_k": 0.15}, **common)
    add(
        "challenge_broad_psf",
        pair,
        overrides={
            "sigma_x_px": 0.85,
            "sigma_y_px": 1.05,
            "halo_fraction": 0.10,
            "halo_sigma_px": 2.5,
        },
        **common,
    )
    add(
        "challenge_focus_mismatch",
        pair,
        distance=0.5,
        overrides={"focus_distance_m": 1.5, "defocus_px_per_diopter": 0.9},
        **common,
    )
    add(
        "challenge_independent_motion",
        Target(kind="triangle", width_m=0.018),
        split="challenge",
        target_velocity_m_s=(0.045, 0.006),
        overrides={"exposure_s": 0.09, "exposure_samples": 5},
        description="Moving 18 mm triangle; independent target motion and finite exposure",
    )
    add(
        "challenge_occlusion",
        Target(kind="triangle", width_m=0.018),
        split="challenge",
        motion="smooth",
        occluder=True,
        camera_roll_amplitude_deg=0.25,
        description="Moving foreground strip at a different depth, camera translation and roll",
    )
    add(
        "challenge_correction_drift",
        Target(kind="edge", orientation_deg=5),
        overrides={
            "offset_fpn_std_dn": 5.0,
            "gain_drift_per_s": 0.003,
            "offset_drift_dn_per_s": 2.0,
            "bad_pixel_fraction": 0.0005,
            "correction_frame": min(8, frames - 2),
        },
        **common,
    )
    distorted = replace(camera, distortion=(-0.20, 0.05, 0.001, -0.001, 0))
    # Separate off-axis case; use a fixed physical center to expose distortion.
    add(
        "challenge_tilt_distortion",
        Target(kind="bars", width_m=0.003, height_m=0.03),
        board_angles_deg=(12, -18, 7),
        **common,
    )
    suite[-1] = replace(
        suite[-1], camera=distorted, target=replace(suite[-1].target, center_m=(0.18, -0.10))
    )
    add("challenge_cold_target", Target(kind="triangle", width_m=0.012), contrast=-2, **common)
    add(
        "profile_linear_temperature",
        Target(kind="slit", width_m=0.035, height_m=0.025, profile="linear"),
        contrast=5,
        **common,
    )
    add(
        "profile_gaussian_temperature",
        Target(kind="circle", radius_m=0.018, width_m=0.030, profile="gaussian"),
        contrast=8,
        **common,
    )
    return suite


def quick_suite(**kwargs):
    all_cases = starter_suite(**kwargs)
    wanted = {
        "flat_20c",
        "edge_horizontal_response",
        "finite_pinhole",
        "pair_028_d100_stationary",
        "pair_028_d100_dither4",
        "triangle_d200_rot090",
        "challenge_independent_motion",
        "challenge_correction_drift",
    }
    return [s for s in all_cases if s.id in wanted]


def expanded_suite(repetitions=3, **kwargs):
    """New capture realizations/positions; keeps sibling groups in the same role."""
    seed = kwargs.pop("seed", 20260916)
    result = []
    for repetition in range(repetitions):
        for s in starter_suite(seed=seed + 10007 * repetition, **kwargs):
            result.append(
                replace(
                    s, id=f"r{repetition:03d}_{s.id}", group_id=f"r{repetition:03d}_{s.group_id}"
                )
            )
    return result
