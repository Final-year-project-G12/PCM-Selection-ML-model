# Cross-Region Pipeline Comparison: Tamil Nadu, Rajasthan, and Assam

This document summarizes the methodological differences, climate characteristics, and pipeline outcomes across the three implemented states.

## 1. Spatial Sampling & Climate Profile

| Dimension | Rajasthan | Tamil Nadu | Assam |
| :--- | :--- | :--- | :--- |
| **Grid Points (87.5% pop)** | 320 points | 133 points | 128 points |
| **Climate Archetype** | Arid/Desert, high diurnal range, clear-sky | Tropical/Coastal, hot, humid, moderate-high solar | Humid subtropical, heavy monsoon, low/diffuse solar |
| **Monsoon Season Def.** | Jun–Aug (minor internal mismatch in scripts) | Varies (Northeast & Southwest monsoons) | Jun–Sep (IMD standard, internally consistent) |
| **Elevation Handling** | Per-point from ERA5 geopotential | Point-specific interpolation | Fixed 100m default (underestimates hill districts) |

## 2. Preprocessing & Quality Control (Phase 2)

| Dimension | Rajasthan | Tamil Nadu | Assam |
| :--- | :--- | :--- | :--- |
| **Outlier Detection** | Hampel Filter (Univariate) | Hampel Filter (Univariate) | **IsolationForest** (Multivariate) |
| **GHI Outlier Handling** | Deliberately excluded (Hampel overcorrected clouds) | Flagged/winsorized | Included naturally (ensemble trees handle heavy tails) |
| **Imputation Method** | Linear → Point-seasonal mean | Linear → K-Means spatial median → MICE | Linear → Point-seasonal mean |
| **ERA5 vs NASA POWER** | Validated (Caught deaccumulation bug) | Validated | **Validated** (MBE=1.1%, strong agreement) |
| **Bias Correction** | Quantile Mapping (Computed but not applied upstream) | Quantile Mapping (Applied in Step 2b) | **None Needed** (`BACKBONE` decision) |

## 3. Climate Signatures & Feasibility (Phases 3–5)

| Dimension | Rajasthan | Tamil Nadu | Assam |
| :--- | :--- | :--- | :--- |
| **Clustering ($k$)** | $k=3$ (Thar, Semi-arid, Aravalli) | $k=5$ (Coastal, Inland, Hills, etc.) | $k=4$ (Brahmaputra, Barak, Hills, Plains) |
| **Target Melting Temp ($T_m$)** | Regime-specific (~44–48°C), worst-month capped | Regime-specific | **Uniform 44°C** (narrow $T_a$ range) |
| **Corrosion Veto (HSI)** | Not triggered (dry climate, HSI < p75) | Active in coastal clusters | **Actively triggered** in humid valley clusters |
| **Latent Heat Sizing ($L_{req}$)**| ~610–643 kJ/kg (Structurally unreachable by database) | Fixed 300 L/day draw (v3.1 update) | Same structurally unreachable floor constraint |

## 4. MCDM Ranking & Physics Validation (Phases 6–8)

| Dimension | Rajasthan | Tamil Nadu | Assam |
| :--- | :--- | :--- | :--- |
| **MCDM Consensus (Kendall's W)**| Weak in Cluster 0 (0.4375) | Measured across 4 methods | **Strong** (0.807–0.845) |
| **Top PCM Recommendation** | Fragmented (RT47, RT44HC, etc.) | Mixed | **Unanimous RT44HC** (driven by uniform $T_m$) |
| **Physics Sim (Phase 7)** | Negative result (Spearman $\rho \le -0.096$) | Completed (Tank UA=2.0 W/K active) | Negative result (Spearman $\rho \le 0.286$) |
| **Criterion Contributions** | Not implemented | Missed in Phase 8 implementation | **Implemented** (percentage breakdown per PCM) |
| **Reproducibility Arch.** | Excellent (`run_all.py`, `provenance_lib.py`) | Good (v3.1 bug fixes in place) | Partial (Missing orchestrator and hard-fail provenance) |

> [!TIP]
> **Key Methodological Evolution**
> The pipeline shows clear iterative improvement across the regions:
> - **Rajasthan** caught the critical ERA5 deaccumulation bug and fixed the GMM covariance type (`diag`).
> - **Tamil Nadu** refined the system draw volume to a realistic 300 L/day and applied the quantile mapping upstream.
> - **Assam** improved the outlier detection mechanism by adopting `IsolationForest` (better suited for monsoon data than the Hampel filter) and implemented the *Criterion Contributions* explainability requirement in the final recommendation cards. It also achieved structural parity in Phase 2 by formally computing cross-source validation, mathematically proving the raw ERA5 data is reliable enough to skip synthetic quantile mapping.
