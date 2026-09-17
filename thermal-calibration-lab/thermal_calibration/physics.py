"""Perspective geometry and a declared idealized 8–14 um radiance model."""

from functools import lru_cache

import numpy as np


def rotation_xyz(degrees):
    x, y, z = np.deg2rad(degrees)
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def distort(x, y, coefficients):
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return (
        x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    )


def undistort(xd, yd, coefficients):
    if not any(coefficients):
        return xd, yd
    x, y = np.array(xd, copy=True), np.array(yd, copy=True)
    k1, k2, p1, p2, k3 = coefficients
    for _ in range(12):
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        tx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        ty = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - tx) / radial, (yd - ty) / radial
    ex, ey = distort(x, y, coefficients)
    if max(float(np.max(np.abs(ex - xd))), float(np.max(np.abs(ey - yd)))) > 1e-6:
        raise ValueError("Distortion inverse did not converge; use a weaker/bijective lens model")
    return x, y


def camera_pose(scenario, time_s):
    """Return camera center in world meters and camera-to-world rotation."""
    c, mode = scenario.camera, scenario.camera_motion
    if mode in {"dither2", "dither4"}:
        n = 2 if mode == "dither2" else 4
        index = int(np.floor(time_s * scenario.acquisition.fps + 0.5)) % (n * n)
        # Stage moves between exposures and settles; all exposure samples share pose.
        dx, dy = (index % n) / n, (index // n) / n
    elif mode == "smooth":
        amp = scenario.camera_motion_amplitude_px
        dx, dy = amp * np.sin(2.1 * time_s), amp * np.sin(1.7 * time_s + 0.3)
    else:
        dx, dy = 0.0, 0.0
    center = np.array([-dx * scenario.distance_m / c.fx, -dy * scenario.distance_m / c.fy, 0.0])
    roll = scenario.camera_roll_amplitude_deg * np.sin(1.4 * time_s)
    return center, rotation_xyz((0, 0, roll))


def world_to_camera(scenario, time_s):
    center, r_wc = camera_pose(scenario, time_s)
    t_cw = np.eye(4)
    t_cw[:3, :3] = r_wc.T
    t_cw[:3, 3] = -r_wc.T @ center
    return t_cw


def project_world(points, camera, t_cw):
    p = np.asarray(points, np.float64)
    q = p @ t_cw[:3, :3].T + t_cw[:3, 3]
    if np.any(q[..., 2] <= 0):
        raise ValueError("Projection contains points behind the camera")
    x, y = distort(q[..., 0] / q[..., 2], q[..., 1] / q[..., 2], camera.distortion)
    return np.stack([camera.fx * x + camera.cx, camera.fy * y + camera.cy], -1)


def board_transform(scenario):
    t_wb = np.eye(4)
    t_wb[:3, :3] = rotation_xyz(scenario.board_angles_deg)
    t_wb[:3, 3] = (0, 0, scenario.distance_m)
    return t_wb


def plane_coordinates(xn, yn, t_cw, t_wp):
    t_cp = t_cw @ t_wp
    inverse = np.linalg.inv(t_cp[:3, [0, 1, 3]])
    denominator = inverse[2, 0] * xn + inverse[2, 1] * yn + inverse[2, 2]
    if np.any(denominator <= 0):
        raise ValueError("Plane is not entirely in front of the viewed rays")
    x = (inverse[0, 0] * xn + inverse[0, 1] * yn + inverse[0, 2]) / denominator
    y = (inverse[1, 0] * xn + inverse[1, 1] * yn + inverse[1, 2]) / denominator
    return x, y


@lru_cache(maxsize=1)
def _spectral_quadrature():
    return np.polynomial.legendre.leggauss(128)


def band_radiance_exact(temperature_c):
    """Uniform spectral response, 8–14 um; 128-point Gauss-Legendre integration.

    Spectral radiance integrated over wavelength, W m^-2 sr^-1. This is an
    assumption, not a P3 spectral-response or temperature-conversion calibration.
    """
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    nodes, weights = _spectral_quadrature()
    wavelength = 11e-6 + nodes * 3e-6
    temperature = np.asarray(temperature_c, np.float64)[..., None] + 273.15
    if np.any(temperature <= 0):
        raise ValueError("Temperature below absolute zero")
    spectral = 2 * h * c**2 / wavelength**5 / np.expm1(h * c / (wavelength * k * temperature))
    return np.sum(spectral * weights, axis=-1) * 3e-6


@lru_cache(maxsize=1)
def _radiance_table():
    temperatures = np.linspace(-40.0, 150.0, 19001)
    return temperatures, band_radiance_exact(temperatures)


@lru_cache(maxsize=256)
def _scalar_radiance(temperature_c):
    return float(band_radiance_exact(temperature_c))


def band_radiance(temperature_c):
    if np.ndim(temperature_c) == 0:
        return _scalar_radiance(float(temperature_c))
    temperatures, values = _radiance_table()
    if np.min(temperature_c) < temperatures[0] or np.max(temperature_c) > temperatures[-1]:
        raise ValueError("Profile temperatures must lie in [-40,150] Celsius")
    return np.interp(temperature_c, temperatures, values).astype(np.float32)


def radiance_derivative(temperature_c):
    return (band_radiance(temperature_c + 0.01) - band_radiance(temperature_c - 0.01)) / 0.02
