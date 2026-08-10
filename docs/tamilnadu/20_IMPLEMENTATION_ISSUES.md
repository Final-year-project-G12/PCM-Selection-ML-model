# 20 — Implementation Issues and Troubleshooting

The following issues are identified in the Tamil Nadu pipeline:

### 1. Active Deaccumulation Bug (Phase 2 / Step 2)
- **Problem**: `deaccumulate()` in `02_combine_tamilnadu.py` applies a `diff()` between consecutive downloaded hours. Since the downloaded NetCDF files contain hourly fluxes, this diffing step corrupts the radiation fields.
- **Consequence**: Noon ERA5 GHI is near-zero (mean ~50 W/m²), resulting in a Pearson correlation of only **r = 0.3963** against NASA POWER.
- **Fix**: Replace `deaccumulate()` with stateless clipping (`accum_to_flux(s) = s.clip(lower=0)`), matching the Rajasthan fix.

### 2. Missing Quantile-Mapping Bias Correction (Phase 2 / Step 7)
- **Problem**: `04_preprocess_tamilnadu.py` does not implement quantile mapping.
- **Consequence**: The corrupted, near-zero GHI is normalized and clustered directly, contaminating the climate signatures.
- **Fix**: Fit and apply empirical quantile mapping to daytime GHI as designed in `03b_agreement_analysis.py`.

### 3. 1000x Flow Rate Unit Error (Phase 3 / Target Derivation)
- **Problem**: In `04b_climate_signature.py`, water flow rate is set to `60.0 / 1000 / 60` (which is `0.001` kg/s), missing a density multiplication factor of 1000.
- **Consequence**: `L_required` is 51–54 kJ/kg instead of a realistic range. The latent heat floor `0.7 * L_required = 36` kJ/kg is too low, making the screening constraint a no-op.
- **Fix**: Correct the flow rate to `(60.0 / 60.0) = 1.0` kg/s (for 60 L/min), or redefine the water draw volume to 300 L total as done in Rajasthan.

### 4. GMM Covariance Overfitting (Phase 4 / Clustering)
- **Problem**: `05_cluster_tamilnadu.py` fits a GMM with `covariance_type="full"`.
- **Consequence**: Fitting 1,890 covariance parameters on only 133 points overdetermines the model, leading to membership probability saturation (probs ≈ 1.0) and poor generalization.
- **Fix**: Change covariance type to `diag` (diagonal), matching the Rajasthan fix.

### 5. Physics Tank Model Simplification (Phase 7 / Physics Validation)
- **Problem**: `10_physics_validation.py` does not include ambient heat losses from the water tank.
- **Consequence**: The tank stays hot overnight, leading to artificially high solar fractions (90–99%) and preventing the PCM from freezing (0-1 cycles/year).
- **Fix**: Add a convective/conductive heat loss term to ambient temperature (`- U * A * (Tw - Ta) * dt`) in the Euler solver.
