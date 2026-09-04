# 20 — Implementation Issues: Consolidated Audit & Resolution Log

Ranked by severity. Fully audited and resolved items demonstrate the project's rigorous self-correction methodology.

---

## 1. Resolved & Audited High-Impact Items

### 1.1 PCM Database Curation & Expansion (Phase 5) — RESOLVED
- **Prior Defect**: The preliminary prototype contained only 25 rows (`pcm_database_assam.csv`) with unvetted literature entries and missing thermophysical properties.
- **Audited Resolution**: Expanded to **58 deduplicated PCM records** across 41 columns (`pcm_database_final.csv`). Established cell-level provenance tracking (`source_type`, `value_status`: Reported, Imputed, Missing). Enforced a strict dual-phase specific heat capacity policy ($C_{p,\text{avg}} = 0.5(C_{p,\text{solid}} + C_{p,\text{liquid}})$), eliminating the silent single-phase fallback bug.

### 1.2 Cross-Source Reanalysis Validation (Phase 2) — RESOLVED
- **Prior Defect**: Assam lacked an explicit cross-source agreement analysis, leaving reanalysis solar data unverified against independent satellite observations.
- **Audited Resolution**: Implemented and executed `03b_agreement_analysis_assam.py`. Daytime GHI comparison against NASA POWER yielded a Mean Bias Error (MBE) of **1.1%**, well within the $\le 10\%$ threshold. Generated the authoritative **`BACKBONE`** decision (`bias_decision_assam.txt`), mathematically justifying the unmanipulated flow of ERA5 data.

### 1.3 Sub-Hourly Dynamic Physics Validation (Phase 9) — RESOLVED
- **Prior Defect**: Early validation relied on a simplified lumped daily tank model.
- **Audited Resolution**: Built a full **10-year chronological dynamic simulation** (`10_physics_validation.py`) running at $\Delta t = 300\text{ s}$ (with $\Delta t = 150\text{ s}$ numerical sensitivity check) across 24 evaluations (8 PCMs × 3 medoids). Implemented a 4-state path-dependent enthalpy formulation with supercooling hysteresis, duration-overlap SSRD reconstruction with energy conservation verified before nighttime clamping; nighttime-clamp losses were separately quantified. Verified First-Law cumulative energy conservation error at **$0.0000\%$** and SSRD reconstruction conservation error at **$0.000000\%$** before nighttime clamping.

### 1.4 Decision-Theoretic vs. Dynamic Physics Divergence (Phase 10) — RESOLVED
- **Prior Ambiguity**: Unclear whether historical MCDM rankings were physically supported.
- **Audited Resolution**: Executed a dual-level assessment (`10_validation_comparison.py`). Level 1 confirmed that under $K=3$ governance, formal MCDM was **`NOT PERFORMED`** ($n_{\text{confirmed}}=[0,0,0]$). Level 2 retrospective evaluation revealed negative rank correlation ($\rho = -0.52$ to $-0.64$, Top-1 agreement $0\%$). Scientific verdict: **`NOT PHYSICALLY SUPPORTED`**, conclusively diagnosed as a thermodynamic threshold mismatch between the $50^\circ\text{C}$ delivery requirement and the $44^\circ\text{C}$ Gaussian MCDM target.

### 1.5 Pipeline-Version Inconsistency ($K=4$ vs. $K=3$) — RESOLVED
- **Prior Defect**: Silently mixing preliminary 4-cluster results with the finalized 3-cluster model.
- **Audited Resolution**: Explicitly demarcated historical $K=4$ artifacts from final locked $K=3$ outputs across all documentation and tables.

---

## 2. Governed Pipeline Invariants & Documented Assumptions

### 2.1 Feasibility Screening Governance (Phase 6)
- Under final locked $K=3$ forcing and strict 7-constraint filtering without arbitrary relaxation, $n_{\text{confirmed}} = [0, 0, 0]$. Exactly one candidate (`n-Tetracosane C24`, $T_m=52.0^\circ\text{C}$) qualified under conditional status in Cluster 0 ($L = 255.0\text{ kJ/kg} \ge L_{\text{required}} = 252.0\text{ kJ/kg}$).
- The 8 historical survivors evaluated in Phase 9 (`RT44HC`, `savE OM42`, `C22H46`, `savE OM46`, `RT45HC`, `savE OM50`, `Myristic-Palmitic eutectic`, `savE OM48`) are designated strictly as *historical pre-audit candidates*, not final confirmed survivors.

### 2.2 Monte Carlo Uncertainty Governance (Phase 8)
- Under current $K=3$ governance, Monte Carlo uncertainty analysis is officially **`SKIPPED`** ($n_{\text{draws}} = 0$) due to insufficient confirmed candidates ($n < 2$).
- The 5,000-draw Monte Carlo analysis is preserved strictly as a historical pre-audit $K=4$ benchmark.

### 2.3 Topographic Elevation Proxy (Phase 1)
- Fixed default baseline elevation of 100 m above sea level represents the Brahmaputra alluvial plains where $>85\%$ of the sampled population resides. While hill borders have higher true elevations, the 100 m baseline serves as an approved, consistent proxy across the state.

### 2.4 Soil & Mains Water Temperature Approximations
- $T_{\text{soil,mean}} \approx T_{a,\text{mean}}$ is adopted in the absence of measured ground data.
- Mains water temperature is estimated as $T_{\text{mains}} = T_{a,\text{mean}} - 2.0\text{ K}$, directly parameterizing $L_{\text{required}}$.
