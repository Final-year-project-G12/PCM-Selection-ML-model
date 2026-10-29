# 16 — Climate Signature: Feature-to-PCM-Property Mapping (Assam)

## Governing Design Principle (Framework Plan §6.1)

> "Every index must answer the question 'which PCM property does this constrain, and by what
> physical mechanism?'. If that sentence cannot be completed, the index is removed."

All 19 Assam signature indices across the **129 spatial coordinates** satisfy this criterion.
(`04b_climate_signature.py` computes all 19 from the event-sampled physical dataset alone — see
`02_DATA_SOURCES_AND_VARIABLES.md` for the verified formulas; there is no working Tier-1/Tier-2
split inside the current signature script, even though the table below is organized under that
older framing and `02b_build_daily_aggregates_assam.py`'s Tier-2 outputs exist on disk unused.)

---

## Feature → Thermal Behavior → PCM Property Map

### "Tier 1" — Sun-Event Statistics (ERA5, event-sampled: sunrise/noon/sunset, not hourly)

| Feature | Physical Mechanism | PCM Property Constrained |
|---|---|---|
| `Ta_mean` | Annual mean ambient temperature → baseline storage tank loss | $T_m^{\text{target}}$ sizing ($T_{\text{del}} - \Delta T_{\text{approach}}$) |
| `Ta_p95` | Extreme hot design-day temperature → maximum charging environment | Upper boundary of melting window; thermal safety |
| `Ta_p05` | Extreme cold design-day temperature → nocturnal discharge deficit | Storage loss coefficient; cold-side thermal cycling |
| `HDD18` | Heating degree days → seasonal domestic heating demand | Feeds PCA thermodynamic block; regime separation |
| `CDD24` | Cooling degree days → summer thermal demand baseline | Feeds PCA thermodynamic block; regime separation |
| `RH_mean` | Annual mean relative humidity → condensation and corrosion risk | Corrosion resistance requirement; container material |
| `GHI_daily_kWh` | Mean daily global horizontal irradiation | $L_{\text{required}}$ latent heat capacity sizing |
| `DTR` | Diurnal Temperature Range → daily thermal expansion/contraction | Thermal cycling durability ($\ge 300$ cycles) |
| `HSI` | **Correction:** actually `RH_mean × mean(fraction of events with (Ta − Td) < 3 K)` — a dew-point-proximity index, not $RH_{\text{mean}} \times GHI_{\text{daily}}$ | **Not** the corrosion-veto trigger in code. `07_feasibility_filter_final.py`'s corrosion check uses each cluster's `RH_mean_mean` (not `HSI`) as the humidity proxy, and — because no PCM in `pcm_database_final.csv` has `is_inorganic=True` — the veto can never actually fire regardless of HSI or RH (see `07_PHASE_5_AUDIT.md`, `08_PHASE_6_AUDIT.md`). Treat HSI as a reported climate-character index, not an active screening input. |

### "Tier 2" — Daily-Integral Indices (produced by `02b_build_daily_aggregates_assam.py`, but **not consumed** by `04b_climate_signature.py`)

The features below are described in earlier drafts as Tier-2 daily-integral indices feeding the
signature. In the current code they are **either computed differently (from event-sampled data,
in `04b` directly) or not computed at all**:

| Feature | Actual status |
|---|---|
| `kt_mean`, `kt_std` | Computed in `04b` from event-level `era5_CSI`, not from a Tier-2 daily clearness index |
| `cloudy_frac` | Computed in `04b` as a fraction of *events* (not days) with `GHI/GHI_clearsky < 0.35` |
| `monsoon_index` | Computed in `04b` from ERA5 event-sampled precipitation (POWER's `PRECTOTCORR` is never downloaded) |
| `CCI` | Computed in `04b` as `1 − std(daily cloudy-event fraction)`, not a "Cloud Cover Index" and not a longest-run count |
| `SAI` | Computed in `04b` from a noon-GHI-based daily proxy, not a true POWER daily integral ratio |
| `precipitation_annual` | **Not computed anywhere** — does not appear in `climate_signatures_raw.csv` |
| `Ta_min_true`, `Ta_max_true` | **Not computed in `04b`** — these exist per-day in `daily_aggregates_assam.csv` but are never carried into the signature |
| `elev_proxy` | Computed in `04b` from mean `era5_P_atm` ÷ 1013.25 — this one is accurately described |

---

## Four Climate Representations in the Pipeline

1. **Raw 19-Index Signature**: Dimensions preserved in physical units across all 129 points (`climate_signatures_raw.csv`).
2. **Standardized Matrix**: Normalized to zero mean and unit variance across the 129 coordinates (`climate_signatures_matrix.csv`).
3. **PCA Thermodynamic Block**: Principal component reduction applied strictly to the 7 correlated thermodynamic indices (`Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`). Solar and variability indices are held separate to preserve physical interpretability.
4. **Final Locked GMM Input Representation (5 Features)**:
   - To prevent full-covariance over-parameterization on $N=129$ points, the final locked Phase 3 GMM model clusters on **5 core physical features**:
     $$\{ GHI_{\text{mean}}, Ta_{\text{mean}}, DTR, RH_{\text{mean}}, wind_{\text{mean}} \}$$
   - This formulation captures solar energy, thermal baseline, diurnal cycling, monsoon humidity, and convective cooling, producing an unambiguous global BIC minimum at $K=3$ ($\text{BIC} = 1574.94$).

---

## Uniform Melting Target: $T_m^{\text{target}} = 44.0^\circ\text{C}$

Assam uses a uniform target across all 129 points:
- $T_{\text{delivery}} = 50.0^\circ\text{C}$ (Indian domestic SWH standard)
- $\Delta T_{\text{approach}} = 6.0\text{ K}$ (Heat exchanger approach)
- $T_m^{\text{target}} = 50.0 - 6.0 = \mathbf{44.0^\circ\text{C}}$

Because $T_{a,\text{mean}}$ varies moderately across Assam ($22.6^\circ\text{C}$ to $25.9^\circ\text{C}$ across regimes), the $44.0^\circ\text{C}$ target is applicable state-wide without requiring regional capping adjustments.
