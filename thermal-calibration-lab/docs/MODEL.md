# Declared forward model and test interpretation

All parameters in this document are assumptions or exact synthetic design
choices. They do not describe a measured P3 unit. The simulator is deliberately
independent of the earlier PyTorch reconstruction code.

## Geometry and radiance

A mask is an analytic shape in a plane with known metric pose. The camera ray
for each subpixel sample is undistorted and intersected with that plane. Targets
are finite circles, pairs, slits, bars, triangles, uniform fields, edges or a
regular aperture grid. Geometry, not a low-resolution raster resized upward,
defines every image. Shape boundaries receive extra integration samples.

Uniform mask/background temperature defaults to 20 C. An aperture exposes an
emitter whose prescribed temperature is background plus contrast. Inside an
aperture the emitter is either uniform, linear in local x, or a Gaussian bump:

```text
uniform:  T = T_background + contrast
linear:   T = T_background + contrast * clip(1 + gradient * 2*x/width, 0, 2)
gaussian: T = T_background + contrast * exp(-(x*x+y*y)/(2*(width/4)^2))
```

The coordinates in these formulas follow the moving, oriented target. They are
prescribed temperature profiles, not a prediction from a heat equation. The
front/rear plate air gap, conduction, emissivity variation, thermal inertia,
real surface roughness and angular reflectance are not solved.

Band radiance is the Planck spectral radiance integrated over 8–14 micrometers
with an assumed uniform spectral response. Exact SI h, c and k are used with
128-point Gauss–Legendre quadrature. A 0.01 C interpolation table accelerates
spatially varying profiles. Surface radiance is

```text
L_surface = emissivity * L_band(T) + (1-emissivity) * L_band(T_reflected)
L_camera  = tau * L_surface + (1-tau) * L_band(T_atmosphere)
tau       = exp(-attenuation_per_meter * board_distance)
DN        = dn_offset + dn_per_radiance * L_camera
```

The optional atmosphere is a scalar gray approximation at board distance. It
does not include wavelength-dependent absorption or turbulence. Its default
attenuation is zero. The camera's actual spectral response and DN conversion
remain unknown. Automatic gain control, palette mapping, compression and any
proprietary temporal image processing are not modeled. A physical temperature contrast is therefore a simulator label,
not a real-camera NETD or temperature-accuracy claim.

## Sampling and optics

The default latent integration grid has 12 samples per native-pixel axis. Near
aperture boundaries, a 4×4 subgrid estimates fractional coverage in each latent
cell. The supported suite uses modest distortion/tilt; the boundary candidate
region conservatively bounds the projected cell footprint for these scenarios.
Extreme oblique planes or distortion require checking convergence and this bound.

Samples lie at `(index+0.5)/q - 0.5` in native coordinates. A padded field of view
extends beyond the largest truncated PSF support so filtering does not invent a
replicated image edge inside the saved field. Gaussian support is truncated at
four sigma. Latent field values are area approximations, not point-source deltas.

The optics are a normalized mixture of an axis-aligned anisotropic Gaussian core
and an isotropic Gaussian halo. No Airy pattern, measured transfer function or
field-dependent aberration is claimed. Focus mismatch adds an explicitly
phenomenological Gaussian width proportional to absolute diopter mismatch:

```text
extra_sigma = defocus_px_per_diopter * abs(1/distance - 1/focus_distance)
effective_sigma_axis = sqrt(base_sigma_axis^2 + extra_sigma^2)
```

This is a nuisance model for testing sensitivity, not a physical thin-lens defocus
calculation. To model a measured lens, replace it with the measured PSF/OTF family
and re-run the independent numerical-integration tests.

Finite exposure averages scene radiance at the declared quadrature times, then
applies the constant shift-invariant PSF. These operations commute under this
declared model. Native pixels integrate rectangular footprints with unit fill
factor. The finer acquisition references use the same optics/exposure but smaller
pixel footprints. Ideal references omit optical/exposure blur and refer to the
exposure midpoint. The finite different-depth occluder is sampled on the latent
grid; it does not receive the extra target-boundary refinement.

## Motion and events

Stationary capture repeats the same physical sampling positions. Dither2 uses a
2×2 half-pixel grid; dither4 uses a 4×4 quarter-pixel grid. Stage motion is assumed
to complete between exposures. Translation in meters is computed from distance
and intrinsics, and the renderer performs the actual perspective projection.
The image shifts are exactly prescribed for the untilted, undistorted base case;
use saved projections for more general cases.

Smooth camera translation, optional roll and independent target velocity have
continuous trajectories. A foreground strip in the occlusion case sits at 80%
of board distance, producing a different depth and a separate trajectory.

The correction event is an explicit test fixture: a frame is replaced by a noisy
uniform field, marked invalid, then fixed-pattern amplitudes are reduced. The
next valid frame has a new epoch and reset flag. This is not a reverse-engineered
P3 shutter/NUC sequence. Learning algorithms must not consume invalid frames.

## Detector noise

Read noise is Gaussian, with DN standard deviation derived from the radiance
slope at the background temperature and a declared equivalent-temperature sigma.
This is one noise component; total system NETD is not thereby calibrated.
Gain/offset fields are fixed to detector coordinates. Row and column noise follow
AR(1) processes with their declared stationary standard deviations. Drift and
defective pixels are optional and applied before uint16 clipping/quantization.

Seed streams are derived with SHA-256, not Python's randomized hash. Camera-fixed
fields depend on seed and sensor shape; per-sequence noise depends on group ID.
Stationary/dither siblings deliberately share the same noise draws and geometry.
Expanded captures change seeds and therefore are additional assumed sensor/noise
realizations. Array reproducibility is qualified by recorded package versions;
manifest wall time and archive byte metadata need not be identical across runs.

## Evaluation policy

- Estimate blur/noise from calibration-role clips. Do not pass simulator PSF,
  noise maps, target masks, true shifts or HR references to an algorithm that
  claims self-supervised operation. Oracle variants must be explicitly labeled.
- Choose thresholds/hyperparameters on development-role clips. Keep matched
  motion siblings and complete sequences together; never split adjacent frames.
- Use held-out validation geometries/conditions and repeated captures for final
  measurements. The small starter collection alone cannot establish generalization.
- Empty and equal-area single-aperture controls test false alarms/false splitting.
  A two-peak sharpness score alone is not a two-target detection metric.
- For triangle orientation, balance the four labels and randomize capture phase.
  Report accuracy against size/contrast, including chance level and uncertainty.
- Use clean acquisition-HR references for finer-sampling tests and ideal references
  for joint deblur/reconstruction tests. State which one each metric uses.
- MTF/sharpness measured after nonlinear enhancement is diagnostic, not proof of
  resolved detail. Judge target decisions and temporal failure modes as well.
- Verify numerical convergence when targets are extremely small, blur is narrow,
  motion is fast, or camera distortion/plane tilt are stronger than the starter cases.

The missing final step is a physical experiment. Replace parameter assumptions
with ranges estimated from real P3 measurements, then test on separate real
captures. This generator makes controlled assumptions testable; it does not
remove the need for those measurements.

## Sources for conventions and target design

- [OpenCV camera geometry and distortion conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
- [NIST exact physical constants](https://physics.nist.gov/cuu/Constants/).
- [P3 nominal detector pitch and focal length](https://thermalmaster.com/products/p3-thermal-camera-for-iphone-and-android).
- [Optikos thermal target calibration arrangement](https://www.optikos.com/wp-content/uploads/2022/05/Calibrating-The-Thermal-Camera_web.pdf).
- [Inframet infrared target types](https://www.inframet.com/Data_sheets/Targets_IR.pdf).
- [TNO triangle-orientation test methodology](https://publications.tno.nl/publication/103983/zNI2tc/bijl-2009-sensor.pdf).

