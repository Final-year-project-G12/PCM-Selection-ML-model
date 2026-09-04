# 05 — Phase 2.5 / 3 Audit: Quality Control & Climate Signature

## Phase 2.5 — Quality Control (`04_preprocess_assam.py`)

**Status**: COMPLETE (Authoritative Final)

### Processing & Bounds Checking
Reads `climate_assam_points.csv` and applies physical bounds verification and outlier flagging across all **129 spatial coordinates**, outputting individual parquet partitions to `data/processed/preprocessed/parquet/{point_id}.parquet`.

| Parameter | Lower Bound | Upper Bound | Enforcement Policy |
|---|---|---|---|
| `era5_GHI` | 0 W/m² | 1400 W/m² | Out-of-bounds flagged; physically valid extremes retained |
| `era5_T_amb` | -30 °C | 55 °C | Bounds verification |
| `era5_RHum` | 0 % | 100 % | Bounds verification |
| `era5_T_dew` | -30 °C | 40 °C | Bounds verification |
| `era5_W_spd` | 0 m/s | 50 m/s | Bounds verification |
| `era5_P_atm` | 850 hPa | 1060 hPa | Bounds verification |
| `era5_cloud_cover` | 0.0 | 1.0 | Fractional cloud cover |
| `era5_precipitation` | 0 mm | 200 mm | Precipitation bounds |

### Multivariate Outlier Detection (IsolationForest)
- **Algorithm**: scikit-learn `IsolationForest` applied to detect multivariate anomalies without assuming Gaussian normality, well-suited to Assam's heavy-tailed monsoon extremes.
- **Strict Retention Policy**: Outliers are **flagged but never deleted** (`is_outlier` boolean flag carried forward to preserve energy totals).
- **Output Files**: Exactly **129 parquet files** (`ASP_0001.parquet` through `ASP_0129.parquet`).

---

## Phase 3 — Climate Signature Construction (`04b_climate_signature.py`)

**Status**: COMPLETE (Authoritative Final)

### Four Climate Representations

To maintain scientific traceability, the pipeline distinguishes four distinct stages of climate representation:

1. **Raw Physical Climate Signature (`climate_signatures_raw.csv`)**:
   - Exactly **129 rows × 18 physical indices** in dimensional units (°C, kWh/m²/day, %, mm).
   - Structured into:
     - *Thermodynamic block (7 indices)*: `Ta_mean`, `Ta_p95`, `Ta_p05`, `HDD18`, `CDD24`, `RH_mean`, `elev_proxy`.
     - *Solar block (4 indices)*: `GHI_daily_kWh`, `kt_mean`, `SAI`, `CCI`.
     - *Variability / Climate character (5 indices)*: `DTR`, `cloudy_frac`, `monsoon_index`, `HSI`, `precipitation_annual`.
     - *Derived targets (2 indices, not clustered)*: `Tm_target` (44.0°C), `L_required`.

2. **Standardized Clustering Matrix (`climate_signatures_matrix.csv`)**:
   - Standardized (zero mean, unit variance) across all 129 points.
   - Normalization is applied strictly **after** temporal aggregation, avoiding aggregation bias.

3. **PCA Representation (`pca_loadings.csv`)**:
   - Principal Component Analysis applied strictly to the 7-feature thermodynamic block to diagnose latent covariance without collapsing the solar radiation signal.

4. **Final GMM Input Representation (5 Core Physical Features)**:
   - Auditing revealed that fitting a full-covariance Gaussian Mixture Model on 18–19 dimensions with $N=129$ points resulted in severe over-parameterization ($D(D+1)/2 = 190$ covariance parameters per cluster).
   - To ensure statistical power and cluster stability, the final locked GMM clustering in Phase 3 uses **5 core physical features**:
     - `GHI_mean` (Solar resource)
     - `Ta_mean` (Thermal baseline)
     - `DTR` (Diurnal thermal range)
     - `RH_mean` (Monsoon moisture proxy)
     - `wind_mean` (Convective boundary layer cooling)
   - This 5-feature representation produced an unambiguous global BIC minimum at **$K=3$** ($\text{BIC} = 1574.94$), with bootstrap ARI = $0.6289$.

---

## System Target Derivations

- **Uniform Melting Target**: $T_m^{\text{target}} = 44.0^\circ\text{C}$ across all 129 sites, based on $T_{\text{delivery}} = 50.0^\circ\text{C}$ (Indian domestic SWH standard) and heat exchanger approach $\Delta T = 6.0\text{ K}$.
- **Soil Temperature Fallback**: In the absence of recorded shallow ground temperatures, $T_{\text{soil,mean}} \approx T_{a,\text{mean}}$ is documented and applied.
- **Mains Temperature Approximation**: $T_{\text{mains,est}} = T_{a,\text{mean}} - 2.0\text{ K}$, determining the site-specific thermal charging deficit and $L_{\text{required}}$.
