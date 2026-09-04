# 18 — Research Gap Mapping & Novelty Positions (Assam)

## Novelty Positioning vs. Broader Research Gaps

Two distinct taxonomies operate within this project:
- **N1–N6** (Objective 1 Framework Plan §3, Table 3): The specific scientific novelties claimed for this data-driven climate-signature, clustering, screening, and validation pipeline.
- **RG1–RG5**: Literature research gaps for the **broader multi-objective project** (hardware prototypes, real-time DRL control, grid-demand matching).

---

## Phase → N (Novelty Claim) Mapping Across All 11 Phases

| Phase | Primary Novelty Claim | Implementation Reality in Assam |
|---|---|---|
| **1 — Spatial Grid** | **N6: Population-Weighted Sampling** | Exactly **129 population-weighted grid points** (`ASP_0001`–`ASP_0129`), achieving **87.8% population coverage** aligned to ERA5's native 0.25° grid. |
| **2 — Preprocessing & Cross-Source Validation** | **Cross-Source Reanalysis Rigor** | Full 10-year ERA5 and NASA POWER cross-validation (`03b_agreement_analysis_assam.py`); **1.1% GHI MBE** confirms the **`BACKBONE`** decision. Exactly **467,367 daily rows** processed. |
| **2.5 — Quality Control** | **Non-Destructive QC** | IsolationForest multivariate outlier detection across 129 parquet files; outliers are flagged but never deleted. |
| **3 — Climate Regime Clustering** | **N1: Discovered Climate Regimes** | Final locked **$K=3$ GMM (full covariance)** on 5 core physical features (`GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, `wind_mean`). Unambiguous global BIC minimum at $K=3$ ($\text{BIC} = 1574.94$), bootstrap ARI = $0.6289$. |
| **4 — SWH Design Specification** | **Engineering Target Derivation** | Standard 100 L/day demand, 50 kg PCM, 100 kg water, $T_m^{\text{target}} = 44.0^\circ\text{C}$ ($T_{\text{del}} = 50.0^\circ\text{C}, \Delta T = 6.0\text{ K}$). |
| **5 — Curated PCM Database** | **N3: Audited 42–70°C PCM Band** | Curated **58-row production database** (`pcm_database_final.csv`) with strict provenance (`source_type`, `value_status`) and strict dual-phase $C_p$ averaging (zero silent single-phase fallback). |
| **6 — Feasibility Filtering** | **N3: Multi-Constraint Screening** | Strict 7-constraint filtering without automatic relaxation. Final $K=3$ governance: $n_{\text{confirmed}} = [0, 0, 0]$, 1 conditional candidate (`n-Tetracosane C24`, $T_m=52.0^\circ\text{C}$ in Cluster 0). Historical $K=4$ survivor set: 8 PCMs. |
| **7 — MCDM Ranking Engine** | **N4: Multi-Method Consensus** | Final $K=3$ governance: **`NOT PERFORMED`** ($n_{\text{confirmed}}=0$). Historical pre-audit $K=4$ ranking evaluated TOPSIS, GRA, PROMETHEE II, VIKOR, Borda, and Copeland. |
| **8 — Monte Carlo Analysis** | **Stochastic Uncertainty Modeling** | Final $K=3$ governance: **`SKIPPED`** ($n_{\text{draws}}=0$). Historical pre-audit $K=4$ benchmark executed 5,000 Dirichlet-perturbed draws. |
| **9 — Sub-Hourly Physics Validation** | **N5: Multi-Year Physics Testing** | Full **10-year dynamic simulation** at $\Delta t = 300\text{ s} / 150\text{ s}$ across 8 historical PCMs and 3 final medoids (24 runs). 4-state path-dependent enthalpy model with supercooling hysteresis; First-Law cumulative error $= 0.0000\%$. |
| **10 — Validation Comparison** | **N5: Decision Theory vs. Physics** | Dual-level comparison: Delivery-rank Spearman $\rho = -0.52$ to $-0.64$, Top-1 agreement $= 0.0\%$, Top-3 overlap $= 0.0\%$. Scientific verdict: **`NOT PHYSICALLY SUPPORTED`**, demonstrating that Gaussian target proximity fails to predict transient solar fraction under strict delivery cutoffs ($50^\circ\text{C}$). |
| **11 — Final Outputs Consolidation** | **Reproducibility & Traceability** | Master manifest (`final_output_manifest.csv`, 31 entries), 10 thesis tables, 10 thesis figures, and automated master test suite (`final_project_verification.py`, 100% pass rate). |

---

## Broader Research Gap Mapping (RG1–RG5)

- **RG1 (Real-Time Control)**: Explicitly out of scope for Objective 1.
- **RG2 (Hardware Prototype)**: Out of scope for Objective 1; provides the validated thermal modeling basis for future experimental work.
- **RG3 (Demand Alignment)**: Addressed through 100 L/day morning (50 L) and evening (50 L) domestic tapping schedules.
- **RG4 (Experimental Validation)**: Sub-hourly 10-year physics simulation bridges empirical gaps prior to hardware builds.
- **RG5 (Predictive Optimization Under Uncertainty)**: Directly advanced via the 10-year multi-regime reanalysis pipeline, provenance-aware property database, and multi-criteria comparison.
