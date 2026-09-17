from dataclasses import replace

import numpy as np
import pytest

from thermal_calibration.config import Acquisition, Camera, Scenario, Target
from thermal_calibration.physics import (
    band_radiance,
    board_transform,
    plane_coordinates,
    project_world,
    undistort,
    world_to_camera,
)
from thermal_calibration.render import Detector, Renderer


def small_scenario(**changes):
    return Scenario(
        camera=Camera(width=48, height=32, cx=23.5, cy=15.5),
        acquisition=Acquisition(
            frames=16,
            offset_fpn_std_dn=0,
            gain_fpn_std=0,
            row_noise_std_dn=0,
            column_noise_std_dn=0,
            halo_fraction=0,
        ),
        **changes,
    )


def test_metric_projection_and_distance_scaling():
    camera = Camera()
    points = np.array([[0, 0, 1], [0.0028, 0, 1], [0.0028, 0, 2]])
    projected = project_world(points, camera, np.eye(4))
    np.testing.assert_allclose(projected[0], [127.5, 95.5])
    np.testing.assert_allclose(projected[1, 0] - camera.cx, camera.fx * 0.0028)
    np.testing.assert_allclose(projected[2, 0] - camera.cx, (projected[1, 0] - camera.cx) / 2)


def test_scaled_pixel_center_convention():
    camera = Camera()
    for scale in (2, 3, 4):
        vector = np.array([0.02, -0.03, 1.0])
        native = camera.matrix() @ vector
        large = camera.matrix(scale) @ vector
        np.testing.assert_allclose(large[:2], (native[:2] + 0.5) * scale - 0.5)


def test_distorted_tilted_projection_ray_intersection_roundtrip():
    scenario = small_scenario(
        board_angles_deg=(15, -20, 7), camera_motion="smooth", camera_roll_amplitude_deg=0.3
    )
    scenario = replace(
        scenario, camera=replace(scenario.camera, distortion=(-0.2, 0.05, 0.001, -0.001, 0))
    )
    t_cw, t_wb = world_to_camera(scenario, 0.4), board_transform(scenario)
    plane = np.array([[-0.01, -0.02, 0], [0.02, 0.01, 0], [0.08, -0.03, 0]])
    world = plane @ t_wb[:3, :3].T + t_wb[:3, 3]
    pixels = project_world(world, scenario.camera, t_cw)
    x, y = undistort(
        (pixels[:, 0] - scenario.camera.cx) / scenario.camera.fx,
        (pixels[:, 1] - scenario.camera.cy) / scenario.camera.fy,
        scenario.camera.distortion,
    )
    xx, yy = plane_coordinates(x, y, t_cw, t_wb)
    np.testing.assert_allclose(np.c_[xx, yy], plane[:, :2], atol=1e-10)


def test_dither_is_known_positive_image_displacement_and_settled():
    scenario = small_scenario(camera_motion="dither4")
    renderer = Renderer(scenario)
    positions = np.array([renderer.annotation(t / 10)["center_native_px"] for t in range(16)])
    shifts = positions - positions[0]
    np.testing.assert_allclose(shifts, [[i % 4 / 4, i // 4 / 4] for i in range(16)], atol=1e-9)
    for t in range(16):
        np.testing.assert_allclose(
            world_to_camera(scenario, t / 10 - 0.018), world_to_camera(scenario, t / 10 + 0.018)
        )


def test_band_radiance_monotonic_and_lut_accurate():
    values = band_radiance(np.array([-10.0, 20, 30, 100]))
    assert np.all(np.diff(values) > 0)
    assert (
        20 < float(values[1]) < 100
    )  # Band-integrated radiance, not exitance or per-micron units.
    for t in (17.123, 24.456, 89.321):
        np.testing.assert_allclose(band_radiance(np.array([t]))[0], band_radiance(t), rtol=2e-7)


def test_flat_field_preserved_by_optics_and_integration():
    scenario = small_scenario(target=Target(kind="flat"))
    renderer = Renderer(scenario)
    sample = renderer.render(0)
    expected = (
        scenario.acquisition.dn_offset + scenario.acquisition.dn_per_radiance * renderer.emitted(22)
    )
    np.testing.assert_allclose(sample["optical_native_dn"], expected, rtol=2e-7)
    for image in sample["ideal"].values():
        np.testing.assert_allclose(image, expected, rtol=2e-7)


def test_compact_target_flux_matches_projected_area_and_is_stable_across_phase():
    scenario = small_scenario(
        target=Target(kind="circle", radius_m=0.0005), camera_motion="dither4"
    )
    renderer = Renderer(scenario)
    contrast = scenario.acquisition.dn_per_radiance * (renderer.emitted(22) - renderer.emitted(20))
    projected_radius = scenario.camera.fx * scenario.target.radius_m / scenario.distance_m
    expected = np.pi * projected_radius**2 * contrast
    fluxes = []
    for frame in (0, 1, 5, 10, 15):
        sample = renderer.render(frame / 10)
        image = sample["ideal"][4]
        fluxes.append(float((image.astype(np.float64) - image[0, 0]).sum() / 16))
        optical = sample["optical_native_dn"]
        flux = float((optical.astype(np.float64) - optical[0, 0]).sum())
        assert abs(flux / expected - 1) < 0.035
    np.testing.assert_allclose(fluxes, expected, rtol=0.03)


def test_render_quadrature_convergence():
    scenario = small_scenario(
        target=Target(kind="circle", radius_m=0.0005, center_m=(0.0007, 0.0003))
    )
    low = Renderer(scenario).render(0)
    high = Renderer(
        replace(scenario, acquisition=replace(scenario.acquisition, supersample=24))
    ).render(0)
    a, b = low["optical_native_dn"], high["optical_native_dn"]
    signal = max(float(b.max() - b.min()), 1e-6)
    assert float(np.max(np.abs(a - b))) / signal < 0.04


def test_finite_exposure_broadens_moving_target():
    scenario = small_scenario(
        target=Target(kind="circle", radius_m=0.001), target_velocity_m_s=(0.08, 0)
    )
    a = replace(scenario.acquisition, exposure_s=0.09, exposure_samples=5)
    instant = Renderer(replace(scenario, acquisition=replace(a, exposure_s=0))).render(0.75)
    integrated = Renderer(replace(scenario, acquisition=a)).render(0.75)

    def variance(image):
        weights = np.maximum(image.astype(float) - image[0, 0], 0).sum(axis=0)
        x = np.arange(len(weights))
        centroid = np.sum(weights * x) / weights.sum()
        return np.sum(weights * (x - centroid) ** 2) / weights.sum()

    assert variance(integrated["optical_native_dn"]) > variance(instant["optical_native_dn"]) + 0.3
    np.testing.assert_array_equal(instant["ideal"][4], integrated["ideal"][4])


def test_noise_scale_and_repeatability_independent_of_scene_renderer():
    scenario = small_scenario(target=Target(kind="flat"))
    renderer = Renderer(scenario)
    optical = renderer.render(0)["optical_native_dn"]
    first, second = Detector(scenario), Detector(scenario)
    samples = []
    for i in range(16):
        a, _ = first.sample(optical, i)
        b, _ = second.sample(optical, i)
        np.testing.assert_array_equal(a, b)
        samples.append(a.astype(float) - optical)
    assert abs(np.std(samples) / first.read_std - 1) < 0.03
    noisier = replace(
        scenario, acquisition=replace(scenario.acquisition, read_noise_equivalent_k=0.2)
    )
    np.testing.assert_array_equal(
        Renderer(noisier).render(0)["ideal"][4], renderer.render(0)["ideal"][4]
    )


def test_correction_event_has_explicit_validity_and_restart_boundary():
    scenario = small_scenario()
    scenario = replace(scenario, acquisition=replace(scenario.acquisition, correction_frame=3))
    detector = Detector(scenario)
    optical = np.full((32, 48), 10000, np.float32)
    flags = [detector.sample(optical, i)[1] for i in range(6)]
    assert flags[3]["valid"] is False
    assert flags[4]["valid"] and flags[4]["reset"] and flags[4]["epoch"] == 1
    assert not flags[5]["reset"]


@pytest.mark.parametrize("change", [{"distance_m": 0}, {"camera_motion": "unknown"}])
def test_invalid_scenario_rejected(change):
    with pytest.raises(ValueError):
        replace(small_scenario(), **change).validate()
