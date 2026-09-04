# 20 — Literature Mapping (Assam)

## Method

Sources checked: (1) `PCM-Selection-ML-model/Sources/` — 21 full paper summaries, (2) the framework
doc's own §15 IEEE reference list, (3) `references.bib` (37 entries). Every citation below was
checked against one of these, not asserted from general training knowledge alone. Citations marked
"not confirmed in project bib" should be added before formal submission.

Classification of supporting evidence:
- **Strong**: Directly supports the implemented methodology, formulation, or governing equations.
- **Moderate**: Supports the general methodology, algorithm class, or qualitative behavior.
- **Project-Specific**: Implementation choice, heuristic, or assumption without direct external literature support.
- **Gap**: Necessary methodological citation absent from `references.bib` that should be added.

---

## Methodology-Component → Implementation → Literature Matrix

| Component | Implementation | Supporting Source | Evidence Strength |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | Phases 1–2 | Hersbach et al. (2020), *QJRMS* 146(730) — per framework doc §15 | Strong |
| NASA POWER as cross-check | Phases 1–2 | NASA POWER project documentation & validation reports — per framework doc §15 | Strong |
| Cross-source solar validation (MBE = 1.1%) | `03b_agreement_analysis_assam.py` | Empirical benchmarking against satellite data; justifies `BACKBONE` decision | Strong (empirical consistency; not ground-truth proof) |
| Solar position algorithm (SPA) | pvlib, `00b_build_suntimes.py`, `02_combine_assam.py` | Reda & Andreas (2004), *Solar Energy* 76(5) | Strong — Gap: not confirmed in `references.bib`; add |
| Clear-sky model (Ineichen) | `02_combine_assam.py` | Ineichen & Perez (2002), *Solar Energy* 73(3) | Strong — Gap: not confirmed in project bib; add |
| pvlib software library | Throughout solar preprocessing | Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) | Strong, per framework doc §15 |
| Humidity-stress index (HSI) | `04b_climate_signature.py` | Thom (1959), *Weatherwise* 12(2) — Discomfort/THI formulation | Strong — load-bearing for climatic humidity characterization |
| Night-discharge design basis ($L_{\text{required}}$) | `04b_climate_signature.py`, `05b_swh_design_specification.py` | Avargani et al. (2021), *J. Energy Storage* 42 | Strong |
| SWH system design constants ($M_w=100\,\text{kg}$, $M_{\text{pcm}}=50\,\text{kg}$, $T_{\text{del}}=50^\circ\text{C}$, $T_{m,\text{target}}=44^\circ\text{C}$) | Phase 4 (`05b_swh_design_specification.py`) | Standard domestic SWH sizing (100 L/day demand, morning/evening 50 L draws) | Project-Specific — engineering design specification |
| Mains water temperature estimate: $T_{\text{mains,est}} = \max(5.0, T_{a,\text{mean}} - 6.0)$ | Phase 4 (`05b_swh_design_specification.py`) | None — engineering approximation for sub-surface thermal damping in Assam | Project-Specific assumption / unsourced model parameter — Gap: needs published local empirical correlation |
| GMM clustering & optimal $K$-selection | Phase 3 (`05_cluster_assam.py`) | Minimum BIC ($1574.94$ at $K=3$) across 5 thermodynamic features; silhouette validation | Moderate — cluster count statistically supported; external Assam meteorological classification absent |
| Bootstrap clustering stability | Phase 3 (`05_cluster_assam.py`) | Resampling stability methodology; mean bootstrap ARI $\approx 0.6289$ (median 0.6542) for $K=3$ | Moderate — objectively verified across 500 resamples |
| PCM candidate database & provenance | Phase 5 (`06_build_pcm_database_final.py`) | 58 deduplicated PCMs with strict provenance; RT-series validated by Martínez et al. (2025), *Heliyon* 11; Singh et al. (2025), *Sol. Energy Mater. Sol. Cells* 293 | Strong |
| PCM database imputation (MICE+RF+PMM) | Offline preprocessing (`PCM_data/`) | van Buuren & Groothuis-Oudshoorn (2011), *J. Stat. Softw.* 45(3) | Strong — Gap: add citation to project bib |
| Strict feasibility screening ($n_{\text{confirmed}}=[0,0,0]$) | Phase 6 (`07_feasibility_filter_final.py`) | Criterion-by-criterion screening audit; zero tolerance for unverified data | Project-Specific / Quality Governance |
| MCDM method family (TOPSIS, GRA, PROMETHEE, VIKOR, Borda) | Historical Phase 7 (`08_mcdm_ranking.py`); Final governance: **NOT PERFORMED** (`08_mcdm_ranking_final.py`) | Originating literature: Hwang & Yoon (1981) for TOPSIS, Brans & Vincke (1985) for PROMETHEE, Opricovic (1998) for VIKOR, Deng (1982) for GRA | Gap — originating papers missing from `references.bib`. Formal $K=3$ MCDM ranking was NOT PERFORMED due to $n_{\text{confirmed}}=0$ |
| Gaussian $T_m$-fitness heuristic ($\sigma=4\,\text{K}$ around $44^\circ\text{C}$) | Historical Phase 7 (`08_mcdm_ranking.py`) | Framework doc §9.2 only; no empirical or physical basis | Project-Specific heuristic — discredited by Phase 10 physics validation (higher $T_m \ge 50^\circ\text{C}$ delivers superior performance) |
| Monte Carlo stability analysis | Historical Phase 8: 5,000 draws executed on historical $K=4$; Current Phase 8: **SKIPPED** ($n_{\text{draws}}=0$) | Framework doc §9.6 | Moderate for historical $K=4$ execution; Final $K=3$ correctly skipped per governance |
| Coupled Water–PCM Grey-Box Thermal Model | Phase 9 (`10_physics_validation.py`) | Model-class foundation: Bony & Citherlet (2007), *Energy and Buildings* 39(9); Energy conservation & first-law formulation: Barqawi (2025), *Muthanna J. Eng. Technol.* 13(3) | Strong (for model class & energy balance equations). Sub-hourly discretization, 4-state hysteresis loop, and boundary clipping are project-specific numerical implementations |
| Domestic draw profile (07:00 & 19:00 IST) | Phase 9 (`10_physics_validation.py`) | ASHRAE Standard 90.2 §8.9.4 (bimodal domestic draw profile) | Moderate — qualitative bimodal schedule supported; exact 50 L / 50 L volume split is project-specific |
| Historical solar fraction benchmark (54–84%) | Pre-audit calibration reference | Framework doc Table 16 (historical indicative target; Phase 9 10-year simulation yielded 47.9–75.3% across candidates) | Historical reference — calibration guide only, not final Phase 9 output |
| Dual-Level MCDM vs Physics Comparison | Phase 10 (`10_validation_comparison.py`) | Retrospective validation framework; Spearman rank correlation ($\rho = -0.52$ to $-0.64$) | Strong — rigorous empirical test of whether MCDM rankings are physically supported |

---

## Assam-Specific Climate Interpretation

None of the 21 papers in `Sources/` provide a dedicated microclimate or thermal-storage zoning specific to Assam or Northeast India.

The final model establishes **strictly $K=3$ climate regimes** based on objective statistical optimization (minimum BIC = $1574.94$, optimal silhouette score, and bootstrap stability ARI $\approx 0.6289$):
- **Cluster 0 (33 grid points, 25.6%)**: Medoid `ASP_0012` — characterized by higher ambient temperature ($T_{a,\text{mean}} = 25.89^\circ\text{C}$), moderate solar radiation ($348.4\,\text{W/m}^2$), lower relative humidity ($75.8\%$), and higher wind speed ($1.66\,\text{m/s}$).
- **Cluster 1 (61 grid points, 47.3%)**: Medoid `ASP_0092` — characterized by maximum solar resource ($GHI_{\text{mean}} = 373.0\,\text{W/m}^2$, $4.08\,\text{kWh/m}^2/\text{day}$), high ambient temperature ($25.10^\circ\text{C}$), and high humidity ($79.0\%$).
- **Cluster 2 (35 grid points, 27.1%)**: Medoid `ASP_0028` — cooler, lower-solar regime ($T_{a,\text{mean}} = 22.59^\circ\text{C}$, $GHI_{\text{mean}} = 330.3\,\text{W/m}^2$) associated with elevated terrain and peripheral foothill areas.

> [!IMPORTANT]
> **Scientific Framing of Regional Regimes**:
> The 3 Assam clusters are **macro-climatic regimes derived from thermodynamic feature similarity**, not administrative boundaries or contiguous geographic districts. Geographic contiguity is not mathematically enforced.
> 
> *Historical / Superseded Context*: An early preliminary $K=4$ diagnostic run informally hypothesized a 4-region division (Brahmaputra valley, hill districts, Barak valley, western plains). That exploratory interpretation was completely superseded by the rigorous BIC minimum at $K=3$ on the 5 core thermodynamic features.

---

## Recommendation: "Methods & Tools" Reference Block Needed

Before formal submission, the following originating citations should be added to the shared project `references.bib` to resolve remaining citation gaps:

1. **Reda & Andreas (2004)**: *Solar Energy* 76(5), 577–589 — Solar Position Algorithm (SPA) for solar zenith/azimuth angles.
2. **Ineichen & Perez (2002)**: *Solar Energy* 73(3), 151–157 — Ineichen clear-sky global and beam irradiance model.
3. **Holmgren, Hansen & Mikofski (2018)**: *Journal of Open Source Software* 3(29), 884 — `pvlib python` library.
4. **Hwang & Yoon (1981)**: *Multiple Attribute Decision Making: Methods and Applications*, Springer-Verlag — TOPSIS method.
5. **Brans & Vincke (1985)**: *Management Science* 31(6), 647–656 — PROMETHEE outranking method.
6. **Opricovic (1998)**: *Multicriteria Optimization in Civil Engineering*, Faculty of Civil Engineering, Belgrade — VIKOR compromise ranking.
7. **Deng (1982)**: *Control and Decision* 1, 288–294 — Grey Relational Analysis (GRA).
8. **van Buuren & Groothuis-Oudshoorn (2011)**: *Journal of Statistical Software* 45(3), 1–67 — MICE framework for multivariate imputation.
9. **Bony & Citherlet (2007)**: *Energy and Buildings* 39(9), 1065–1072 — Numerical model and experimental validation of a solar storage tank with PCM.
10. **Barqawi (2025)**: *Muthanna Journal of Engineering and Technology* 13(3) — Dynamic energy-balance equations for PCM solar water heating.
11. **Empirical Indian Ground/Mains Temperature Correlation**: An empirical reference for mains water temperature in sub-Himalayan / Northeast India to replace the project-specific assumption ($T_{\text{mains,est}} = \max(5.0, T_{a,\text{mean}} - 6.0)$).

