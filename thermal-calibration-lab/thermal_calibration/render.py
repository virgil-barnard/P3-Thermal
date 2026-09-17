"""Independent forward rendering of finite thermal apertures through a camera.

No reconstruction, learned model, estimated flow, or benchmark scores enter this
module. Optics precede detector integration. Noise is added in detector space.
"""

import hashlib
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from .physics import (
    band_radiance,
    board_transform,
    plane_coordinates,
    project_world,
    radiance_derivative,
    undistort,
    world_to_camera,
)


def random_stream(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def block_average(image, factor):
    h, w = image.shape
    if h % factor or w % factor:
        raise ValueError("Image dimensions must divide the integration factor")
    return image.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def triangle_vertices(side):
    return np.array(
        [
            [0, -side / np.sqrt(3)],
            [side / 2, side / (2 * np.sqrt(3))],
            [-side / 2, side / (2 * np.sqrt(3))],
        ]
    )


def target_local(x, y, target, shift):
    x, y = x - target.center_m[0] - shift[0], y - target.center_m[1] - shift[1]
    theta = np.deg2rad(target.orientation_deg)
    return np.cos(theta) * x + np.sin(theta) * y, -np.sin(theta) * x + np.cos(theta) * y


def signed_distance(x, y, target, shift):
    """Positive inside; convex-edge distances are sufficient for boundary refinement."""
    x, y = target_local(x, y, target, shift)
    if target.kind == "blank":
        return np.full(np.broadcast_shapes(x.shape, y.shape), -np.inf, np.float32)
    if target.kind == "flat":
        return np.full(np.broadcast_shapes(x.shape, y.shape), np.inf, np.float32)
    if target.kind == "edge":
        return x
    if target.kind == "circle":
        return target.radius_m - np.hypot(x, y)
    if target.kind == "grid":
        column = np.clip(
            np.rint(x / target.grid_spacing_m + (target.grid_columns - 1) / 2),
            0,
            target.grid_columns - 1,
        )
        row = np.clip(
            np.rint(y / target.grid_spacing_m + (target.grid_rows - 1) / 2), 0, target.grid_rows - 1
        )
        xx = x - (column - (target.grid_columns - 1) / 2) * target.grid_spacing_m
        yy = y - (row - (target.grid_rows - 1) / 2) * target.grid_spacing_m
        return target.radius_m - np.hypot(xx, yy)
    if target.kind == "pair":
        half = target.separation_m / 2
        return np.maximum(
            target.radius_m - np.hypot(x - half, y), target.radius_m - np.hypot(x + half, y)
        )
    if target.kind == "slit":
        return np.minimum(target.width_m / 2 - np.abs(x), target.height_m / 2 - np.abs(y))
    if target.kind == "bars":
        out = np.full(np.broadcast_shapes(x.shape, y.shape), -np.inf, np.float32)
        for position in (-3, -1, 1, 3):
            bar = np.minimum(
                target.width_m / 2 - np.abs(x - position * target.width_m),
                target.height_m / 2 - np.abs(y),
            )
            out = np.maximum(out, bar)
        return out
    if target.kind == "triangle":
        vertices = triangle_vertices(target.width_m)
        out = np.full(np.broadcast_shapes(x.shape, y.shape), np.inf, np.float32)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            dx, dy = b - a
            edge = (dx * (y - a[1]) - dy * (x - a[0])) / np.hypot(dx, dy)
            out = np.minimum(out, edge)
        return out
    raise ValueError(f"Unknown target: {target.kind}")


@lru_cache(maxsize=2)
def _rays(key, q, pad):
    w, h, fx, fy, cx, cy, *distortion = key
    u = (np.arange((w + 2 * pad) * q, dtype=np.float32) + 0.5) / q - 0.5 - pad
    v = (np.arange((h + 2 * pad) * q, dtype=np.float32) + 0.5) / q - 0.5 - pad
    x, y = np.meshgrid((u - cx) / fx, (v - cy) / fy)
    return undistort(x, y, distortion)


class Renderer:
    def __init__(self, scenario):
        scenario.validate()
        self.s, self.a, self.c = scenario, scenario.acquisition, scenario.camera
        self.q = self.a.supersample
        extra = 0.0
        if self.a.focus_distance_m is not None:
            extra = self.a.defocus_px_per_diopter * abs(
                1 / self.s.distance_m - 1 / self.a.focus_distance_m
            )
        self.sigmas = (
            float(np.hypot(self.a.sigma_y_px, extra)),
            float(np.hypot(self.a.sigma_x_px, extra)),
        )
        self.halo_sigma = float(np.hypot(self.a.halo_sigma_px, extra))
        max_sigma = max(*self.sigmas, self.halo_sigma if self.a.halo_fraction else 0)
        self.pad = int(np.ceil(4 * max_sigma)) + 1
        key = (
            self.c.width,
            self.c.height,
            self.c.fx,
            self.c.fy,
            self.c.cx,
            self.c.cy,
            *self.c.distortion,
        )
        self.xn, self.yn = _rays(key, self.q, self.pad)
        self.t_wb = board_transform(scenario)
        self.reflected = band_radiance(self.a.reflected_c)
        self.atmosphere = band_radiance(self.a.atmosphere_c)
        self.background = self.emitted(self.a.background_c)
        self.tau = float(np.exp(-self.a.attenuation_per_m * self.s.distance_m))

    def emitted(self, temperature):
        return (
            self.a.emissivity * band_radiance(temperature)
            + (1 - self.a.emissivity) * self.reflected
        )

    def target_shift(self, time_s):
        center_time = (self.a.frames - 1) / (2 * self.a.fps)
        return np.asarray(self.s.target_velocity_m_s) * (time_s - center_time)

    def crop(self, image):
        margin = self.pad * self.q
        return image[
            margin : margin + self.c.height * self.q, margin : margin + self.c.width * self.q
        ]

    def key(self, time_s):
        pose = world_to_camera(self.s, time_s)
        values = [*pose.ravel(), *self.target_shift(time_s)]
        # Smooth motion, rolling target or occlusion also changes exposure averaging.
        if self.s.camera_motion == "smooth" or self.s.camera_roll_amplitude_deg or self.s.occluder:
            values.append(time_s)
        return tuple(np.round(values, 12))

    def coordinates_at(self, u, v, t_cw, t_wp=None):
        xn, yn = undistort(
            (u - self.c.cx) / self.c.fx, (v - self.c.cy) / self.c.fy, self.c.distortion
        )
        return plane_coordinates(xn, yn, t_cw, self.t_wb if t_wp is None else t_wp)

    def coverage(self, x, y, t_cw, time_s):
        target, shift = self.s.target, self.target_shift(time_s)
        distance = signed_distance(x, y, target, shift)
        coverage = (distance >= 0).astype(np.float32)
        n = self.a.boundary_samples
        if n > 1 and target.kind not in {"blank", "flat"}:
            # Conservative footprint bound for the supported modest tilts/distortions.
            margin_m = 3.0 * self.s.distance_m / (min(self.c.fx, self.c.fy) * self.q)
            iy, ix = np.nonzero(np.abs(distance) < margin_m)
            if len(ix):
                offsets = (np.arange(n) + 0.5) / n - 0.5
                ox, oy = np.meshgrid(offsets, offsets)
                u = (ix[:, None] + 0.5 + ox.ravel()) / self.q - 0.5 - self.pad
                v = (iy[:, None] + 0.5 + oy.ravel()) / self.q - 0.5 - self.pad
                sx, sy = self.coordinates_at(u, v, t_cw)
                coverage[iy, ix] = (signed_distance(sx, sy, target, shift) >= 0).mean(axis=1)
        return coverage

    def scene(self, time_s):
        t_cw = world_to_camera(self.s, time_s)
        x, y = plane_coordinates(self.xn, self.yn, t_cw, self.t_wb)
        alpha = self.coverage(x, y, t_cw, time_s)
        target = self.s.target
        if target.profile == "uniform":
            emitter = self.emitted(self.a.background_c + self.a.contrast_c)
        else:
            lx, ly = target_local(x, y, target, self.target_shift(time_s))
            if target.profile == "linear":
                level = np.clip(1 + target.profile_gradient * 2 * lx / target.width_m, 0, 2)
            else:
                sigma = target.width_m / 4
                level = np.exp(-(lx * lx + ly * ly) / (2 * sigma * sigma))
            emitter = self.emitted(self.a.background_c + self.a.contrast_c * level)
        image = self.background + alpha * (emitter - self.background)
        full_area = float(self.crop(alpha).sum())
        occluder_pose = None
        if self.s.occluder:
            # A different-depth opaque strip: parallax and independently moving occlusion.
            occluder_pose = np.eye(4)
            occluder_pose[2, 3] = 0.8 * self.s.distance_m
            duration = max((self.a.frames - 1) / self.a.fps, 1 / self.a.fps)
            occluder_pose[0, 3] = 0.025 * (2 * time_s / duration - 1)
            ox, _ = plane_coordinates(self.xn, self.yn, t_cw, occluder_pose)
            # Exact area boundary refinement is for target apertures; the occluder
            # uses the documented high-grid midpoint approximation.
            hidden = np.abs(ox) < 0.002
            alpha[hidden] = 0
            image[hidden] = self.background
        image = self.tau * image + (1 - self.tau) * self.atmosphere
        visible = float(self.crop(alpha).sum())
        return (
            image.astype(np.float32),
            self.crop(alpha),
            {
                "visible_fraction_of_in_frame_target": visible / full_area if full_area else 1.0,
                "occluder_T_world_from_plane": None
                if occluder_pose is None
                else occluder_pose.tolist(),
            },
        )

    def optics(self, image):
        core = gaussian_filter(
            image, tuple(v * self.q for v in self.sigmas), mode="nearest", truncate=4.0
        )
        weight = self.a.halo_fraction
        if weight:
            halo = gaussian_filter(image, self.halo_sigma * self.q, mode="nearest", truncate=4.0)
            core *= 1 - weight
            core += weight * halo
        return self.crop(core)

    def render(self, time_s):
        instantaneous, coverage, extra = self.scene(time_s)
        truth = {
            scale: (
                self.a.dn_offset
                + self.a.dn_per_radiance * block_average(self.crop(instantaneous), self.q // scale)
            ).astype(np.float32)
            for scale in (2, 3, 4)
        }
        moving = (
            self.s.camera_motion == "smooth"
            or self.s.camera_roll_amplitude_deg
            or any(self.s.target_velocity_m_s)
            or self.s.occluder
        )
        if moving and self.a.exposure_s and self.a.exposure_samples > 1:
            averaged = np.zeros_like(instantaneous)
            n = self.a.exposure_samples
            for phase in (np.arange(n) + 0.5) / n - 0.5:
                image = (
                    instantaneous
                    if abs(phase) < 1e-12
                    else self.scene(time_s + phase * self.a.exposure_s)[0]
                )
                averaged += image / n
        else:
            averaged = instantaneous
        optical_high = self.optics(averaged)
        optical = self.a.dn_offset + self.a.dn_per_radiance * block_average(optical_high, self.q)
        acquisition_hr = {
            scale: (
                self.a.dn_offset
                + self.a.dn_per_radiance * block_average(optical_high, self.q // scale)
            ).astype(np.float32)
            for scale in (2, 3, 4)
        }
        return {
            "ideal": truth,
            "optical_native_dn": optical.astype(np.float32),
            "acquisition_hr": acquisition_hr,
            "coverage_hr4": block_average(coverage, self.q // 4).astype(np.float32),
            "label": self.annotation(time_s) | extra,
        }

    def annotation(self, time_s):
        target = self.s.target
        theta = np.deg2rad(target.orientation_deg)
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        if target.kind == "triangle":
            points = triangle_vertices(target.width_m)
        elif target.kind == "grid":
            gx = (
                np.arange(target.grid_columns) - (target.grid_columns - 1) / 2
            ) * target.grid_spacing_m
            gy = (np.arange(target.grid_rows) - (target.grid_rows - 1) / 2) * target.grid_spacing_m
            points = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        elif target.kind == "pair":
            points = np.array([[-target.separation_m / 2, 0], [target.separation_m / 2, 0]])
        else:
            points = np.array([[0.0, 0.0]])
        local = points @ rotate.T + np.asarray(target.center_m) + self.target_shift(time_s)
        world = np.c_[local, np.zeros(len(local))] @ self.t_wb[:3, :3].T + self.t_wb[:3, 3]
        projected = project_world(world, self.c, world_to_camera(self.s, time_s))
        center_local = np.asarray(target.center_m) + self.target_shift(time_s)
        center_world = self.t_wb[:3, :3] @ np.r_[center_local, 0] + self.t_wb[:3, 3]
        center_px = project_world(center_world[None], self.c, world_to_camera(self.s, time_s))[0]
        label = {
            "target_kind": target.kind,
            "orientation_deg": target.orientation_deg,
            "keypoints_board_m": local.tolist(),
            "keypoints_native_px": projected.tolist(),
            "center_native_px": center_px.tolist(),
            "T_camera_from_world": world_to_camera(self.s, time_s).tolist(),
            "target_offset_board_m": self.target_shift(time_s).tolist(),
            "distance_plane_origin_m": self.s.distance_m,
        }
        if target.kind == "pair":
            label["projected_center_separation_px"] = float(
                np.linalg.norm(projected[1] - projected[0])
            )
        return label


class Detector:
    """Read noise, sensor-fixed response, AR(1) row/column noise, defects and events."""

    def __init__(self, scenario):
        self.s, self.a, self.c = scenario, scenario.acquisition, scenario.camera
        shape = (self.c.height, self.c.width)
        fixed = random_stream(self.s.seed, f"sensor:{shape}")
        self.offset = fixed.normal(0, 1, shape).astype(np.float32) * self.a.offset_fpn_std_dn
        self.gain = (1 + fixed.normal(0, 1, shape) * self.a.gain_fpn_std).astype(np.float32)
        self.bad = fixed.random(shape) < self.a.bad_pixel_fraction
        self.hot = fixed.random(shape) < 0.5
        # Paired motion conditions share the same detector-noise realizations.
        self.rng = random_stream(self.s.seed, f"read:{self.s.group_id}")
        self.row = self.rng.normal(0, self.a.row_noise_std_dn, (shape[0], 1))
        self.column = self.rng.normal(0, self.a.column_noise_std_dn, (1, shape[1]))
        self.read_std = (
            self.a.read_noise_equivalent_k
            * self.a.dn_per_radiance
            * self.a.emissivity
            * radiance_derivative(self.a.background_c)
        )

    def sample(self, optical_dn, frame):
        a, time_s = self.a, frame / self.a.fps
        after = a.correction_frame is not None and frame > a.correction_frame
        fixed_scale = 0.25 if after else 1.0
        signal = optical_dn - a.dn_offset
        mean = (
            a.dn_offset
            + signal * (1 + (self.gain - 1) * fixed_scale) * (1 + a.gain_drift_per_s * time_s)
            + self.offset * fixed_scale
            + a.offset_drift_dn_per_s * time_s
        )
        rho = a.correlated_noise_rho
        self.row = rho * self.row + np.sqrt(1 - rho * rho) * self.rng.normal(
            0, a.row_noise_std_dn, self.row.shape
        )
        self.column = rho * self.column + np.sqrt(1 - rho * rho) * self.rng.normal(
            0, a.column_noise_std_dn, self.column.shape
        )
        noisy = mean + self.row + self.column + self.rng.normal(0, self.read_std, mean.shape)
        noisy[self.bad & self.hot] = 65535
        noisy[self.bad & ~self.hot] = 0
        correction = frame == a.correction_frame
        if correction:
            noisy = np.full_like(mean, np.median(mean)) + self.rng.normal(
                0, self.read_std, mean.shape
            )
        saturated = float(np.mean((noisy < 0) | (noisy > 65535)))
        image = np.rint(np.clip(noisy, 0, 65535)).astype(np.uint16)
        return image, {
            "valid": not correction,
            "reset": bool(after and frame == a.correction_frame + 1),
            "epoch": int(after),
            "event": "simulated_correction" if correction else "none",
            "saturated_fraction": saturated,
        }
