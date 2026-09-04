# 21 — Reproducibility & Verification Audit (Assam)

## Reproducibility Checklist & Governance Status

| Verification Item | Status | Authoritative Implementation Notes |
|---|---|---|
| **Random Seeds** | **PASS** | `random_state=42` set on GMM, K-Means, and bootstrap sampling; `10_physics_validation.py` is fully deterministic. |
| **GMM & Scaler Persistence** | **PASS** | `scaler_assam.joblib` and `gmm_model_assam.joblib` saved in `data/processed/clustering/`; reproduces exact $K=3$ cluster assignments without re-fitting. |
| **scikit-learn Version Pinning** | **PASS** | Recorded in output cluster files (`sklearn_version: 1.9.0`). |
| **Master Manifest & Inventory** | **PASS** | `final_output_manifest.csv` catalogs all **31 artifacts** with schema, row counts, provenance, and active/historical status flags. |
| **Automated Verification Suite** | **PASS** | `final_project_verification.py` executes 10 comprehensive verification modules covering all 11 phases with a **100% pass rate**. |
| **Regression Test Coverage** | **PASS** | Dedicated verification test suites (`verify_phase5_phase6.py`, `verify_phase7.py`, `verify_phase8.py`, `verify_phase9.py`, `verify_phase10.py`) pass with zero errors. |
| **Geographic Coordinates** | **PASS** | Deterministic WorldPop + GADM intersection on ERA5 0.25° grid yields exact same **129 population-weighted points** (`ASP_0001`–`ASP_0129`). |
| **Temporal Data Coverage** | **PASS** | Strict chronological interval: 2016-01-01 to 2025-12-31 (10 full years). Exactly **467,367 valid daily records**. |
| **Phase 9 Numerical Convergence** | **PASS** | First-Law conservation error = $0.0000\%$; SSRD reconstruction error = $0.000000\%$; 100% spin-up convergence satisfied. |
| **Phase 10 Comparative Diagnostics** | **PASS** | Dual-level validation script confirms exact mathematical rank inversions and negative correlations. |
| **Output Naming & Formatting** | **PASS** | Standardized naming `{artifact}_assam.csv` and unified thesis output directories (`final_outputs/tables/`, `final_outputs/visuals/`). |

---

## Master Verification Suite (`final_project_verification.py`)

The pipeline includes a standalone, automated verification suite that enforces repository-wide integrity across all 11 phases.

### Key Verification Modules
1. **Grid Completeness**: Validates that all 129 points (`ASP_0001` to `ASP_0129`) are present and valid.
2. **Climate Regimes ($K=3$)**: Confirms the $K=3$ GMM full-covariance solution, cluster sizes (33, 61, 35), and medoids (`ASP_0012`, `ASP_0092`, `ASP_0028`).
3. **PCM Database Provenance**: Asserts 58 deduplicated records across 41 columns with complete provenance (`source_type`, `value_status`) and strict dual-phase $C_p$ averaging.
4. **Feasibility Governance**: Confirms $n_{\text{confirmed}}=[0,0,0]$ and preserves `n-Tetracosane C24` as a Conditional candidate.
5. **MCDM Governance**: Confirms formal K=3 MCDM status is `NOT PERFORMED` and preserves historical K=4 rankings.
6. **Monte Carlo Governance**: Asserts $n_{\text{draws}}=0$ and status `SKIPPED` under K=3.
7. **Physics Simulation Integrity**: Confirms 24 simulation runs across 10 years, First-Law error $\le 0.05\%$, and 100% convergence.
8. **Phase 10 Comparison**: Asserts negative Spearman rank correlation ($\rho < -0.40$) and confirms verdict `NOT PHYSICALLY SUPPORTED`.
9. **Publication Deliverables**: Verifies the presence and validity of all 10 thesis tables and 10 thesis figures.
10. **Full Regression**: Runs all 5 subordinate verification scripts with zero exit errors.
