# 20 — Literature Mapping (Assam)

## Method

Sources checked: (1) `PCM-Selection-ML-model/Sources/` — 21 full paper summaries, (2) the framework
doc's own §15 IEEE reference list, (3) `references.bib` (37 entries). Every citation below was
checked against one of these, not asserted from general training knowledge alone. Citations marked
"not confirmed in project bib" should be added before formal submission.

## Methodology-component → implementation → literature matrix

| Component | Implementation | Supporting source | Strength |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | Phases 1–2 | Hersbach et al. (2020), *QJRMS* — per framework doc §15 | Strong |
| NASA POWER as cross-check | Phases 1–2 | NASA POWER project documentation — per framework doc §15 | Strong |
| Solar position (SPA) | pvlib, `00b`/`02_combine_assam.py` | Reda & Andreas (2004), *Solar Energy* 76(5) | Strong — not confirmed in `references.bib`; add |
| Clear-sky model (Ineichen) | `02_combine_assam.py` | Ineichen & Perez (2002), *Solar Energy* 73(3) | Strong — not confirmed in project bib; add |
| pvlib software | throughout | Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) | Strong, per framework doc §15 |
| Humidity-stress index (HSI) | `04b_climate_signature.py` | Thom (1959), *Weatherwise* 12(2) — THI formula | Strong — **load-bearing for Assam's corrosion veto**; correctly attributable |
| Night-discharge design basis (L_required) | `04b_climate_signature.py` | Avargani et al. (2021), *J. Energy Storage* | Strong |
| PCM candidate band (42–70°C) | Phase 5 | Framework doc Table 5; Singh et al. (2025), *Solar Energy Materials and Solar Cells* 293 | Strong |
| GMM clustering, k-selection | `05_cluster_assam.py` | Framework doc §7.2; BIC + silhouette + ARI reported | Moderate — cluster count methodology supported; external classification not yet wired |
| Bootstrap stability (ARI) | `05_cluster_assam.py` | Framework doc §7.3 | Moderate — correctly done, ARI_mean=0.716 reported honestly |
| PCM property database (RT-series) | `06_build_pcm_database.py` | Martínez et al. (2025), *Heliyon* 11 — validates RT-family; Singh et al. (2025) for literature additions | Strong |
| MCDM method family (TOPSIS/PROMETHEE/VIKOR/GRA) | `08_mcdm_ranking.py` | No dedicated originating-paper citation confirmed in `references.bib` | **Gap** — add Hwang & Yoon 1981 (TOPSIS), Brans & Vincke 1985 (PROMETHEE), Opricovic 1998 (VIKOR), Deng 1982 (GRA) |
| Gaussian Tm-fitness σ=4K | `08_mcdm_ranking.py` | Framework doc §9.2 only | Weak/self-sourced — state plainly |
| Monte Carlo uncertainty propagation (5,000 draws) | `08_mcdm_ranking.py` | Framework doc §9.6 — Assam correctly matches spec | Moderate — correctly cited |
| Criterion Contributions (explainability) | `09_recommendation_cards.py` | Framework doc Table 18 — Assam adds this (TN missed it) | Moderate — implementation-defined; no external citation needed |
| Phase 7 lumped-enthalpy ODE | `10_physics_validation.py` | Barqawi, F. A. (2025), *Muthanna J. Eng. Technol.* 13(3) | Strong — in `Sources/`, equations used directly |
| Phase 7 model-class justification | `10_physics_validation.py` | Bony & Citherlet (2007), *Energy and Buildings* 39(9) | Strong — model class confirmed |
| Draw-profile shape | `10_physics_validation.py` | ASHRAE Standard 90.2 §8.9.4 (two-peak draw profile) | Partial — qualitative shape correct; exact fractions not verbatim reproduced |
| Phase 7 solar fraction benchmark (54–84%) | `10_physics_validation.py` | Framework doc Table 16 | Strong — correctly used as calibration check |
| PCM database imputation (MICE+RF+PMM) | `PCM_data/01_preprocess.py` | No dedicated citation confirmed | **Gap** — cite van Buuren & Groothuis-Oudshoorn (2011) for MICE framework |
| T_mains_est_C = Ta_mean − 2.0 | `04b_climate_signature.py` | **None — explicitly unsourced in-code** | **Weak / open gap** — needs a published correlation |

## Assam-specific literature note

None of the 21 papers in `Sources/` are specific to Northeast India or Assam's climate. The GMM
clustering result (k=4, interpretable as Brahmaputra valley / hill districts / Barak valley /
western plains) is supported by geographic knowledge but not by a Assam-specific climate-classification
study. For the thesis, a one-sentence acknowledgment that the k=4 physical interpretation is based
on geographic domain knowledge and internal BIC/silhouette statistics rather than an external
Assam-specific climate classification would be the correct framing.

## Recommendation: "Methods & Tools" reference block needed

Add before formal submission:
- Reda & Andreas (2004) — SPA
- Ineichen & Perez (2002) — clear-sky model
- Holmgren et al. (2018) — pvlib
- Hwang & Yoon (1981) — TOPSIS
- Brans & Vincke (1985) — PROMETHEE
- Opricovic (1998) — VIKOR
- Deng (1982) — GRA
- van Buuren & Groothuis-Oudshoorn (2011) — MICE
- Bony & Citherlet (2007) — PCM tank model class
- A published correlation for mains-water temperature vs. ambient air temperature in India

These gaps are **shared with Rajasthan and Tamil Nadu** — adding them once to the project's
`references.bib` covers all four states simultaneously.
