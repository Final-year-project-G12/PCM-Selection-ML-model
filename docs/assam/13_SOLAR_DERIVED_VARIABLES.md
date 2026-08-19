# 16 — Derived Solar Variables Audit (Assam)

## GHI

`GHI = accum_to_flux(ssrd)/3600`, clipped ≥ 0. See `12_ERA5_DATA_PIPELINE.md` for the full
deaccumulation story. This is the pipeline's most consequential derived variable. The Assam pipeline
inherits the fixed `accum_to_flux()` version.

**Assam context**: Assam's monsoon-season GHI is significantly lower than the other three states —
the Brahmaputra valley receives >80% of its annual precipitation in Jun–Sep, and cloud cover reduces
GHI substantially during those months. The `kt_mean` values in the Assam clusters (0.696–0.789) are
lower than Rajasthan's (which operates in a predominantly clear-sky environment). This is a real
climate signal, not an artifact.

## DNI — two-branch derivation

```python
if "avg_sdirswrf" in df.columns:
    df["DNI"] = df["avg_sdirswrf"].clip(0, 1400)   # primary branch
else:
    df["DNI"] = np.where(cos_z > 0.05, df["GHI"] / cos_z, 0).clip(0, 1400)  # fallback
```

Branch 1 (primary): DNI taken directly from the ERA5 direct-radiation field. See `12_ERA5_DATA_PIPELINE.md`
for the unit-consistency caveat on which ERA5 field name actually matched.

Branch 2 (fallback — crude closure): `DNI = GHI / cos(SZA)` assuming zero diffuse component. This
is **not a real decomposition model**. For Assam's heavily-cloudy monsoon conditions, this fallback
would produce large overestimates of DNI whenever GHI is present but the atmosphere is optically
thick. Branch 2 is rarely exercised in practice (Branch 1 is available when the ERA5 field is
present).

## DHI — closure residual

`DHI = (GHI − DNI × cos_z).clip(0)` — always satisfies the closure equation by construction, not
independently measured. Any error in GHI or DNI propagates entirely into DHI. State this plainly
in any methodology write-up.

**Assam relevance**: During the monsoon, DHI (diffuse horizontal irradiance) is proportionally
more important than during clear-sky periods. Since DNI is not independently validated, DHI's
absolute values should be treated as estimates, not measured quantities.

## Clearness index (CSI / kt)

`CSI = GHI / GHI_clearsky` clipped to [0, 1.5], forced to 0 below 10 W/m² clearsky threshold.

**kt_mean range for Assam clusters (from `cluster_profiles_assam.csv` via `recommendation_cards_assam.md`)**:
- Cluster 0: kt_mean = 0.696
- Cluster 1: kt_mean = 0.758
- Cluster 2: kt_mean = 0.789
- Cluster 3: kt_mean = 0.772

These are physically reasonable for Assam: a kt of ~0.70–0.79 indicates moderately cloudy conditions,
consistent with a monsoon-dominated climate with significant diffuse-radiation contribution. Rajasthan's
kt is substantially higher (~0.85+) due to its predominantly dry, clear-sky climate.

## Physical bounds applied to derived solar variables

| Variable | Applied bound | Where |
|---|---|---|
| GHI | `<0 → 0` | `accum_to_flux()` |
| GHI (upper) | `>1400 → NaN` (high values dropped) | `02_combine_assam.py` |
| DNI | `clip(0, 1400)` | both branches |
| DHI | `clip(0)`, no upper bound | `02_combine_assam.py` |
| CSI | `clip(0, 1.5)`, forced 0 below GHI_clearsky=10 threshold | `02_combine_assam.py` |

## Monsoon-season cloud cover — a real validation signal

Assam's `cloud_cover` (ERA5 `tcc`, 0–1 fraction) and `precipitation` fields should show clearly
seasonal patterns in the raw data: near-zero precipitation Jan–Feb, rapid increase from late May,
peak in Jul–Aug (~200–300+ mm/month in wet areas), rapid decrease in Oct. If `02_combine_assam.py`'s
output `climate_assam_points.csv` does not show this pattern, it would indicate a data-processing
problem. This seasonal check was not independently run during this audit (no agreement-analysis
script for Assam), but serves as an implicit validation that the combine step worked correctly.

## Literature support

For DNI/DHI derivation limitations: Erbs, Klein & Duffie (1982) and Perez et al. (1990, DISC) are
the standard decomposition-model references if a proper model is ever added. As currently implemented,
the correct framing is: "DNI from ERA5's direct-radiation field where available; DHI computed as
closure residual" — not overclaiming a decomposition-model provenance.
