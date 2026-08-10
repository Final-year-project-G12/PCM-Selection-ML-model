# 05 — Phase 3 Audit: Climate Signature Construction

True script: `04b_climate_signature.py` (disk: `04c_interactive_postprocess_qc (3).py`).
**This file contains the single most important finding in the Tamil Nadu audit.**

## Status: code complete, never executed — contains an unfixed, known-category bug

## Purpose

Same as Rajasthan: reduce each point's record to one climate-signature vector, merging Tier-1
(sun-event proxies) with Tier-2 (true daily integrals), deriving `Tm_target_C`/`L_required_kJ_per_kg`,
5 interaction terms, PCA, and a standardized clustering-ready matrix.

## Tm_target_C — correct, matches Rajasthan and the framework doc exactly

```python
T_DELIVERY_C = 50.0
DT_APPROACH_C = 7.0
TM_TARGET_C = T_DELIVERY_C + DT_APPROACH_C   # 57°C, constant for every point
```
The code's own comment header marks this line **"CORRECTED (unchanged from before)"** — i.e., this
specific constant already reflects the framework doc's v3.0 sign correction (`T_delivery + ΔT_approach`,
not the erroneous v1.0/v2.0 subtraction). This part of the file is sound.

## L_required_kJ_per_kg — THE FINDING: uses the pre-correction, buggy formula

```python
DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60      # = 0.001 kg/s
CP_WATER = 4.186
ASSUMED_PCM_MASS_KG = 50.0
...
sig["T_mains_est_C"] = sig["Ta_mean"] - 2.0
q_night_kw = DRAW_RATE_KG_PER_S * CP_WATER * (T_DELIVERY_C - sig["T_mains_est_C"])
sig["L_required_kJ_per_kg"] = (q_night_kw * 3600 * 7) / ASSUMED_PCM_MASS_KG
```

**Cross-referenced directly against `era5-rajasthan/04_climate_signature_rajasthan.py`**, whose own
docstring explicitly names and fixes this exact formula (its "Correction 3," superseded by
"Correction 4," both quoted in full in `docs/era5_rajasthan/05_PHASE_3_AUDIT.md`):

> *"DRAW_RATE_KG_PER_S = 60.0/1000/60 (=0.001 kg/s, an erroneous extra L→m³ division with no matching
> ×1000 water-density step back to kg/s, off by 1000×)... §6.3's Q_night formula was being fed a
> 60 L/min rate sustained for 7 hours (25,200 L total) — no source for that figure anywhere in the
> plan doc. The plan doc's own §6.3 citation is Avargani et al. 2021: '300 L of hot water at
> 60±2°C for 7 h of operation' — a TOTAL volume over the window, not a per-minute rate."*

**Tamil Nadu's file contains this exact, byte-for-byte-identical unfixed constant** (`60.0/1000/60`),
with no correction, no `NIGHT_DRAW_TOTAL_L` constant, no Avargani citation, and no docstring caveat
about the result being an idealized ceiling. Only the `Tm_target_C` line is marked "CORRECTED" in
this file — `L_required` carries no such marker, consistent with it never having been touched.

### Concrete numeric consequence

For a representative point with `Ta_mean≈27°C` (so `T_mains_est_C≈25°C`):
- **TN's current formula**: `q_night_kw = 0.001 × 4.186 × 25 = 0.1047 kW`; `L_required =
  0.1047×3600×7/50 ≈ 52.7 kJ/kg` — a **small, easily-achievable-looking** requirement, comfortably
  within range of common organic PCMs (paraffins ~150–250 kJ/kg).
- **Rajasthan's corrected formula** (`NIGHT_DRAW_TOTAL_KG=300`): `Q_night = 300×4.186×25 = 31,395 kJ`;
  `L_required = 31,395/50 ≈ 628 kJ/kg` — an **order-of-magnitude larger ceiling**, deliberately framed
  in Rajasthan's docstring as an idealized upper bound "no single-material PCM is expected to fully
  reach," which is exactly why Rajasthan's Phase 5 feasibility filter currently returns 0 survivors at
  its nominal threshold (see `docs/era5_rajasthan/07_PHASE_5_AUDIT.md`).

### Why this matters more than it might first appear

Tamil Nadu's Phase 5 (`07_feasibility_filter.py`) uses the *same* `L ≥ 0.7×L_required` constraint
Rajasthan uses. Because TN's `L_required` is roughly an order of magnitude too low, **TN's feasibility
filter will pass this constraint far too easily** — the opposite failure mode from Rajasthan's
(which fails almost everything). Neither is scientifically correct: Rajasthan's ceiling is
defensibly strict-but-honest (self-documented as an idealized upper bound, with a documented
κ-relaxation policy to compensate); Tamil Nadu's is **silently too permissive**, which is more
dangerous methodologically because it produces a *plausible-looking, passing* result rather than an
obviously-broken one that forces a design conversation. **This should be fixed before Tamil Nadu's
pipeline is ever run for real** — the fix is a direct, already-available copy from
`04_climate_signature_rajasthan.py`'s Correction 4.

## Everything else in this file — verified sound

- **Tier-1 sun-event proxy formulas**: `Ta_mean/p95/p05` (percentiles of daily-mean sun-event temp),
  `DTR_proxy = T_noon − T_sunrise`, `GHI_daily_kWh_proxy` via a half-sine approximation
  (`(2/π)×(GHI_noon/1000)×daylength_hours`), `kt_mean/std_proxy`, `SAI_proxy` (ratio of sums),
  `cloudy_frac_proxy`/`CCI_proxy` (run-length-encoded), `HDD18/CDD24_proxy` (base 18/24°C),
  `HSI` (heat-stress index — `RH_mean × fraction of readings within 3°C dew-point depression`, **a
  different formula from Rajasthan's `HSI_sunrise` = Thom's THI** — these are not the same index
  despite the similar name; TN's version has no cited literature source found during this audit),
  `seasonality_proxy` (CV of monthly-mean noon GHI), `monsoon_index` (proxy-only, self-documented),
  `elev_proxy` (pressure-ratio pseudo-elevation, see `02_DATA_SOURCES_AND_VARIABLES.md`).
- **Tier-2 merge**: canonical-column mapping with an explicit, documented per-row fallback to the
  `_proxy` value wherever the true Tier-2 value is NaN — `sig[canon] = sig[true_col].where(notna,
  proxy_col)`. Hard-fails (`raise FileNotFoundError`) if the Tier-2 file is missing at all — no silent
  Tier-1-only fallback.
- **5 interaction terms**: `GHI×kt_std`, `DTR×cloudy_frac`, `RH×(Ta_mean−Tm_target)`,
  `wind×(Ta_mean−Tsoil_proxy)` (where `Tsoil_proxy_C = Ta_mean − 3.0`, a simplistic soil-lag proxy
  distinct from Rajasthan's ground-temperature approach), `CCI×(1−SAI)`.
- **PCA block**: `["Ta_mean","Ta_p95","Ta_p05","HDD18","CDD24","RH_mean","elev_proxy"]` — 7 columns
  (vs. Rajasthan's 8, and notably **includes `RH_mean`**, which Rajasthan's PCA block does not — a
  genuine, deliberate scope difference, not an error, since TN's coastal-vs-inland humidity contrast
  is a more prominent climate axis than Rajasthan's). `PCA(n_components=0.95, random_state=42)`.
- **Standardization**: excludes raw PCA-block columns, `lat`/`lon`, `population`, and every
  `_proxy`/`_true`/`_true_mean`-suffixed duplicate column — confirmed by direct code read, matching
  the "cluster only on the canonical merged index, never on geography" design principle shared with
  Rajasthan.

## Literature support

Same Tm_target/T_delivery citations as Rajasthan should apply here (framework doc §6.3, Avargani et
al. 2021) — **but currently do not**, since the `L_required` formula predates that correction in this
file. `HSI` here has no identified citation (distinct from Rajasthan's correctly-cited Thom 1959
THI) — worth sourcing or relabeling as an original index if kept.

## Validation

None possible yet (never run). The correlation-heatmap/distribution/map diagnostic outputs this
script would produce (`signature_correlation_heatmap.png`, etc.) do not exist on disk.

## Outputs (expected)

`climate_signature_tamilnadu.csv` — one row per point (~133 expected), `pca_loadings.csv`, 3
diagnostic PNGs.

## Dependencies

Requires Phase 2's `tamilnadu_cleaned_physical.csv` and `tier2_signature_tamilnadu.csv`. Feeds Phase
4 (clustering) and Phase 5 (feasibility targets) — **the `L_required` bug propagates directly into
Phase 5's feasibility filter results**, making this the single highest-priority fix in the entire
Tamil Nadu codebase.

## Problems / risks

- **The `L_required` bug (above) — highest priority fix in this entire pipeline.**
- `HSI`'s formula is unsourced/uncited, unlike Rajasthan's correctly-attributed equivalent.
- `Tsoil_proxy_C = Ta_mean − 3.0` is an unsourced simplification, similar in spirit to (but numerically
  distinct from) Rajasthan's equally-unsourced `T_mains_est_C = Ta_mean − 2.0` — both should be
  replaced with a real ground-temperature lag correlation before either pipeline's derived quantities
  are presented as fully literature-grounded.

## Status

**NOT READY — contains a known-category, high-impact, unfixed formula bug.** This is the one file in
the entire Tamil Nadu audit that should be corrected before the pipeline is run for the first time,
not after.
