# Validation record

Generated and checked on 2026-09-16. These checks validate this simulator and
its outputs; they are not tests of a physical P3 or evidence of SR accuracy.

## Complete generated dataset

- 42 sequences, 672 observations, 671 valid frames.
- One deliberately invalid correction frame, followed by an explicit reset and new epoch.
- Scenario roles: {'calibration': 10, 'development': 6, 'validation': 16, 'challenge': 10}.
- Native arrays: uint16 256×192. Float32 references: 512×384, 768×576 and 1024×768.
- Every saved sequence passed shape, finite-value, coverage-range, timestamp,
  reference-index, acquisition-HR-to-native integration, SHA-256 integrity and
  group-split checks. See `dataset-validation.json` for per-clip results.
- All 5376 referenced preview paths exist and are nonempty (shared
  references appear more than once in this count).
- Generation elapsed 21.7 minutes in this environment.
- Recorded numerical environment: {'python': '3.12.14', 'numpy': '2.3.5', 'scipy': '1.17.0'}.

## Mathematical checks

`python -m pytest -q`: **16 passed**. Ruff check and format check both passed.
Tests cover metric projection, scaled pixel-center conventions, distorted/tilted
ray inversion, controlled dither phases, radiance quadrature, flat preservation,
finite aperture flux, rendering convergence, exposure blur, noise repeatability,
correction events and dataset round trips.

Additional numerical measurements are in `numerical-checks.json`:

- A 1 mm aperture at 1 m is 0.3583 native pixels across. Across all 16 quarter-pixel
  phases, the pre-optics integrated signal differs from analytic projected area
  by at most 0.152%; the optical signal differs by at most
  0.028%.
- Increasing the latent grid from 12× to 24× changes the tested native optical
  sample by at most 0.061523 DN, or
  0.685% of the fine-reference peak signal.
  This is one challenging finite-aperture convergence check, not a universal
  accuracy bound for arbitrary configurations.
- The generated flat-field frame-difference standard deviation divided by sqrt(2)
  is 4.713820 DN. The assumed read-noise sigma
  is 4.702884 DN; temporal row/column noise and quantization
  also contribute to the measured number.

## Inspector and integration

The offline HTML inspector passed a Chromium 153.0.8010.0 smoke test:
local file loading, 2×/3×/4× references, pre/post-optics selection, shared gamma,
synchronized crops, frame controls, playback, role filtering and mobile layout.
Desktop and mobile screenshots accompany `inspector-check.json`.

The optional example processed all 16 native frames from
`pair_056_d050_dither4` through the earlier thermal-online-sr engine at 2×.
`earlier-prototype-integration.json` records branch MSE values and statuses,
including warmup. This verifies the input/output interface; whole-image MSE is
not a detection/classification metric and is not presented as a quality result.

## Limits

No real camera was measured. PSF, noise, spectral response, DN conversion and
thermal profiles are declared assumptions. Target profiles are prescribed fields,
not heat-flow simulations. There is no vendor X³ implementation or trained
classifier in this deliverable. The starter set does not establish natural-scene
generalization or uncertainty intervals. See `docs/MODEL.md` for the full model.
