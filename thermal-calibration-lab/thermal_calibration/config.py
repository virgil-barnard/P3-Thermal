"""All dimensions are SI except explicitly named pixels, degrees, and Celsius."""

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int = 256
    height: int = 192
    fx: float = 4.3 / 0.012
    fy: float = 4.3 / 0.012
    cx: float = 127.5
    cy: float = 95.5
    # Brown-Conrady order: k1, k2, p1, p2, k3; dimensionless normalized coordinates.
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def validate(self):
        if not (16 <= self.width <= 1024 and 16 <= self.height <= 1024):
            raise ValueError("Camera dimensions must be in 16..1024")
        if not all(
            math.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy, *self.distortion)
        ):
            raise ValueError("Camera parameters must be finite")
        if self.fx <= 0 or self.fy <= 0 or len(self.distortion) != 5:
            raise ValueError("Positive focal lengths and five distortion coefficients required")

    def matrix(self, scale=1):
        # Integer-valued pixel centers; resampling preserves pixel footprint boundaries.
        return np.array(
            [
                [self.fx * scale, 0, (self.cx + 0.5) * scale - 0.5],
                [0, self.fy * scale, (self.cy + 0.5) * scale - 0.5],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Target:
    kind: str = "pair"
    center_m: tuple = (0.0, 0.0)
    radius_m: float = 0.0005
    separation_m: float = 0.0028
    width_m: float = 0.008
    height_m: float = 0.012
    orientation_deg: float = 0.0
    # Applied inside the geometric aperture: uniform, linear, or gaussian.
    profile: str = "uniform"
    profile_gradient: float = 0.6
    grid_rows: int = 5
    grid_columns: int = 7
    grid_spacing_m: float = 0.015


@dataclass(frozen=True)
class Acquisition:
    frames: int = 16
    fps: float = 10.0
    exposure_s: float = 0.04
    exposure_samples: int = 3
    supersample: int = 12
    boundary_samples: int = 4
    # Independent assumed optics: anisotropic Gaussian core plus broad halo.
    sigma_x_px: float = 0.45
    sigma_y_px: float = 0.60
    halo_fraction: float = 0.025
    halo_sigma_px: float = 1.8
    # Optional extra *Gaussian approximation* to defocus, never an optical lens solver.
    focus_distance_m: float | None = None
    defocus_px_per_diopter: float = 0.0
    # Uniform 8–14 um response; units are linear W m^-2 sr^-1 before DN conversion.
    emissivity: float = 0.95
    background_c: float = 20.0
    contrast_c: float = 2.0
    reflected_c: float = 20.0
    atmosphere_c: float = 20.0
    attenuation_per_m: float = 0.0
    dn_offset: float = 1000.0
    dn_per_radiance: float = 180.0
    read_noise_equivalent_k: float = 0.035
    offset_fpn_std_dn: float = 1.0
    gain_fpn_std: float = 0.001
    row_noise_std_dn: float = 0.3
    column_noise_std_dn: float = 0.3
    correlated_noise_rho: float = 0.8
    gain_drift_per_s: float = 0.0
    offset_drift_dn_per_s: float = 0.0
    bad_pixel_fraction: float = 0.0
    correction_frame: int | None = None


@dataclass(frozen=True)
class Scenario:
    id: str = "pair_example"
    group_id: str = "pair_example"
    split: str = "development"
    description: str = "Two finite warm apertures"
    distance_m: float = 1.0
    board_angles_deg: tuple = (0.0, 0.0, 0.0)
    target: Target = field(default_factory=Target)
    camera: Camera = field(default_factory=Camera)
    acquisition: Acquisition = field(default_factory=Acquisition)
    camera_motion: str = "stationary"
    camera_motion_amplitude_px: float = 1.5
    camera_roll_amplitude_deg: float = 0.0
    target_velocity_m_s: tuple = (0.0, 0.0)
    occluder: bool = False
    nominal_sigma_px: float = 0.35
    seed: int = 20260916

    def validate(self):
        self.camera.validate()
        a, t = self.acquisition, self.target
        if not self.id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.id):
            raise ValueError("Scenario id must contain only lowercase letters, numbers, _ or -")
        if self.split not in {"calibration", "development", "validation", "challenge"}:
            raise ValueError("Invalid dataset split")
        if not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError("Positive finite target distance required")
        if t.kind not in {
            "blank",
            "flat",
            "edge",
            "circle",
            "pair",
            "slit",
            "bars",
            "triangle",
            "grid",
        }:
            raise ValueError("Unknown target geometry")
        if t.profile not in {"uniform", "linear", "gaussian"}:
            raise ValueError("Unknown temperature profile")
        if min(t.radius_m, t.width_m, t.height_m) <= 0:
            raise ValueError("Target dimensions must be positive")
        if t.kind == "pair" and t.separation_m < 2 * t.radius_m:
            raise ValueError("Pair apertures must not overlap")
        if t.kind == "grid" and (
            min(t.grid_rows, t.grid_columns) < 2 or t.grid_spacing_m <= 2 * t.radius_m
        ):
            raise ValueError("Grid needs at least two rows/columns and non-overlapping apertures")
        if self.camera_motion not in {"stationary", "dither2", "dither4", "smooth"}:
            raise ValueError("Unknown camera trajectory")
        if a.frames < 1 or a.fps <= 0 or not 0 <= a.exposure_s <= 1 / a.fps:
            raise ValueError("Invalid frame timing")
        if a.supersample < 12 or a.supersample % 12 or a.boundary_samples not in (1, 2, 4, 8):
            raise ValueError(
                "Supersample must be a positive multiple of 12; boundary samples 1,2,4,8"
            )
        if not 1 <= a.exposure_samples <= 15:
            raise ValueError("Exposure quadrature samples must be in 1..15")
        if min(a.sigma_x_px, a.sigma_y_px, a.halo_sigma_px) < 0:
            raise ValueError("PSF sigmas must be nonnegative")
        if not (0 <= a.halo_fraction <= 1 and 0 < a.emissivity <= 1):
            raise ValueError("Invalid mixture weight or emissivity")
        if not 0 <= a.correlated_noise_rho < 1 or not 0 <= a.bad_pixel_fraction <= 0.05:
            raise ValueError("Invalid correlated noise or defect rate")
        if a.correction_frame is not None and not 0 <= a.correction_frame < a.frames - 1:
            raise ValueError("Correction event must precede the last frame")
        if a.focus_distance_m is not None and a.focus_distance_m <= 0:
            raise ValueError("Focus distance must be positive")
        if a.dn_per_radiance <= 0 or a.attenuation_per_m < 0:
            raise ValueError("Invalid radiance conversion")
        if (
            min(
                a.read_noise_equivalent_k,
                a.offset_fpn_std_dn,
                a.gain_fpn_std,
                a.row_noise_std_dn,
                a.column_noise_std_dn,
                a.defocus_px_per_diopter,
            )
            < 0
        ):
            raise ValueError("Noise and defocus amplitudes must be nonnegative")
        for value in asdict(a).values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Acquisition values must be finite")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["target"] = Target(**value.get("target", {}))
        value["camera"] = Camera(**value.get("camera", {}))
        value["acquisition"] = Acquisition(**value.get("acquisition", {}))
        obj = cls(**value)
        obj.validate()
        return obj
