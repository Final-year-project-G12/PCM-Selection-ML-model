# 17 — Literature Mapping

**Documentation note (2026-09-02):** Standalone concept files `10_TEMPORAL_PROCESSING.md` and
`11_SPATIAL_PROCESSING.md` have been consolidated into `03_PHASE_1_AUDIT.md` and
`04_PHASE_2_AUDIT.md` respectively, with full justification for each method. The research gap
mapping has been moved into `00_MASTER_OVERVIEW.md` under the new "Research gaps addressed
(N1–N6 novelty mapping)" section. This file (`17_LITERATURE_MAPPING.md`) remains the authoritative
reference for all methodology-component-to-source mappings.

## Method

Sources checked, in priority order: (1) `PCM-Selection-ML-model/Sources/` — 21 full paper summaries
(the project's own curated, previously-read literature), (2) the framework doc's own §15 IEEE
reference list, (3) `references.bib` (37 entries) and `.claude/references.md` (24 unique
ResearchRabbit entries + a duplicate of `references.bib`). Every citation below was checked against
one of these three, not asserted from general training knowledge alone, except where explicitly
marked "not independently verified in this project's bibliography" — those are standard, correct
citations for well-known methods (e.g. Reda & Andreas SPA, Ineichen clear-sky) that were not found in
this specific project's reference files during this audit and should be added before formal
submission.

## Methodology-component → implementation → literature matrix

| Component | Implementation | Supporting source | Strength |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | Phase 1–2 | Hersbach et al. (2020), *QJRMS* — per framework doc §15 | Strong (product-defining citation) |
| NASA POWER as cross-check | Phase 1–2 | NASA POWER project documentation — per framework doc §15 | Strong |
| Solar position (SPA) | `pvlib`, `00b`/`02` | Reda & Andreas (2004), *Solar Energy* 76(5) | Strong, but not confirmed present in `references.bib`/`.claude/references.md` — add before submission |
| Clear-sky model (Ineichen) | `02_combine_rajasthan.py` | Ineichen & Perez (2002), *Solar Energy* 73(3) | Strong, not confirmed in project bib — add |
| pvlib software | throughout | Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) | Strong, per framework doc §15 |
| Humidity-stress index (HSI_sunrise) | `signature_lib.py` | Thom (1959), *Weatherwise* 12(2) — THI, correctly cited in-code | Strong, directly attributable |
| Night-discharge design basis (L_required) | `04_climate_signature_rajasthan.py` | Avargani et al. (2021), *J. Energy Storage* | Strong, direct citation with a corrected units interpretation (see `05_PHASE_3_AUDIT.md`) |
| Worst-month sizing cap (Tm_target_capped_C) | same | Durin et al. (2018), "Worst Month and Critical Period Methods..." | Strong, direct and appropriately applied |
| Field-evidence sanity check for the cap | same | Nahar (2003), tested at Jodhpur | Direct, present as a bare citation in `.claude/references.md` — needs a complete BibTeX entry |
| T_mains lag estimate | same | **none** — explicitly documented in-code as not derived from any published correlation | **Weak / open gap** — see recommendation below |
| GMM clustering, k-selection heuristics | `05_cluster_rajasthan.py` | *Building and Environment* (2024) India climate-classification study (silhouette 0.21 vs −0.2 NBC); a 2026 thermal-comfort clustering study (mean silhouette 0.235) | Moderate — cited with enough specificity to be traceable but full BibTeX entries not located in this pass |
| External classification validation | `05_cluster_rajasthan.py` | Beck et al. (2018), *Scientific Data* 5, DOI:10.1038/sdata.2018.214 (Köppen-Geiger) | Strong citation, **now wired in for real (2026-08-11)** — ARI=0.19/NMI=0.32 vs. GMM. NBC/ECBC remains unwired. |
| PCM candidate band (42–70°C) | Phase 5 | Framework doc Table 5, cross-referenced against Singh et al. (2025), *Solar Energy Materials and Solar Cells* 293 (states 40–70°C as the optimal SWH PCM band) | Strong, closely matching independent literature |
| PCM property values (RT-series validation) | PCM database | Martínez et al. (2025), *Heliyon* 11 — directly measures/validates RT54HC/RT55/RT64HC, the same product family in this project's database, and finds large literature-vs-measured discrepancies for some | Strong and directly relevant — should be cited as a caveat on manufacturer-datasheet trust, not just a property source |
| Gaussian Tm-fitness σ=4K | `08_mcdm_ranking_rajasthan.py` | Framework doc §9.2 only — "not independently literature-calibrated," per the code's own docstring | Weak/self-sourced — state plainly, do not overclaim external validation |
| PROMETHEE II q/p thresholds | same | Framework doc §9.4 | Implementation-defined, documented as such |
| TOPSIS unit-test fixture | same | Oluah (2020) — 72.12% thermal-conductivity domination cited as a cautionary comparator | Direct, used correctly as both a regression-test anchor and an interpretive comparator |
| MCDM method family (TOPSIS/PROMETHEE/VIKOR/GRA) | same | No dedicated MCDM-methodology paper found cross-referenced in `references.bib`/`.claude/references.md` | **Gap** — these are standard, well-established methods, but a formal write-up should cite each method's originating paper (Hwang & Yoon 1981 for TOPSIS; Brans & Vincke 1985 for PROMETHEE; Opricovic 1998 for VIKOR; Deng 1982 for GRA) |
| PCM database imputation (MICE-style + RF + custom PMM-like donor blend) | `PCM_data/01_preprocess.py` | No dedicated imputation-methodology paper found in this project's bibliography | **Gap** — cite the general MICE framework (van Buuren & Groothuis-Oudshoorn 2011) and note explicitly that the donor-blend step is a project-original variant, not textbook PMM (see `07_PHASE_5_AUDIT.md`) |
| Quantile mapping (bias correction) | `03b_agreement_analysis.py` | No dedicated citation found in this project's bibliography | **Gap** — cite Cannon et al. (2015) or an equivalent standard reference |
| Phase 7 lumped-enthalpy ODE structure (3-phase pre-melt/melt/post-melt) | `physics_lib.py` | Barqawi, F. A. (2025), *Muthanna J. Eng. Technol.* 13(3):1-14, doi:10.52113/3/eng/mjet/2025-13-03/-1-14 | Strong — already in `Sources/` (read in full pre-Phase-7), DOI independently re-verified this session, equations used directly (not paraphrased from memory) |
| Phase 7 model-class justification (lumped PCM-in-tank, the basis for TRNSYS Type 860) | same | Bony, J. & Citherlet, S. (2007), *Energy and Buildings* 39(9):1065-1072 | Strong — independently confirmed via web search this session (not previously in `Sources/`), cited for model-CLASS justification only, not claimed as a literal Type 860 replication |
| Phase 7 draw-profile SHAPE (two-peak, morning+evening) | same | ASHRAE Standard 90.2 §8.9.4 Table 8-4, built on Perlman & Mills (1985), *ASHRAE Transactions* | **Partial/honest gap** — the qualitative two-peak shape is real and cited, but the exact 24 published hourly fractions were not independently retrievable this session; `physics_lib.py`'s own docstring flags this explicitly as a parametric reconstruction of the documented SHAPE, not a verbatim reproduction of the standard's table — do not cite specific hourly percentages from this pipeline as if reproducing that table |
| Phase 7 draw-total volume (300 kg/day) | same | Avargani et al. (2021) — same citation Phase 3 already uses for `L_required_kJ_per_kg`'s 300 L/7h basis, reused as the FULL DAY total rather than a night-only ceiling (a different, explicitly stated use of the same cited figure) | Strong, cross-phase-consistent citation reuse |
| Phase 7 collector parameters (A_c, h_c, efficiency, PCM bed surface-to-volume ratio) | same | Barqawi (2025), same paper as above | Strong for the ORIGINAL values; **recalibrated** during Phase 7's own calibration pass (collector area, implicit loss coefficient) — recalibration reasoning documented in `physics_lib.py`'s CALIBRATION section, not silently changed |

## Sources/ folder papers — relevance summary (21 papers read in full)

The 21 papers in `Sources/` are overwhelmingly **PCM-material / PCM-SWH-system / AI-for-thermal-systems**
domain literature (Abdellatif 2025, Al-Mamun 2023, Assareh 2023, Barghi 2026, Barqawi 2025, Chen 2025,
Chopra 2023, Duraivel 2025, Eldokaishi 2022, Emami 2026, Ghodusinejad 2026, Hamzat 2025, Kou 2025, Liu
2025, Mansouri 2025, Martínez 2025, Mohammed 2025, Odoi-Yorke 2025, Singh 2025, Terfai 2025, Yan 2025)
— they substantiate this project's PCM-selection rationale, MCDM-in-PCM-context precedents (Assareh
2023's TOPSIS/LINMAP/AHP; Chen 2025's GRA), and ML-for-thermal-systems framing well. **None of them are
methodology-support papers for ERA5/reanalysis handling, pvlib solar geometry, quantile mapping, or
MCDM statistical foundations specifically** — this is a real, confirmed gap (searched by title
keyword against both `references.bib` and `.claude/references.md`; only two incidental matches, Chen
2025 for "grey relational" and Chopra 2023 for "Monte Carlo," both already counted above). Köppen
classification is now covered (Beck et al. 2018, above). **Barqawi (2025), already in this list as a
PCM-SWH domain paper, now ALSO serves as a direct methodology-support citation for Phase 7's
lumped-enthalpy simulation equations** — its equations are used directly, not just cited for
framing.

## Recommendation

Before formal submission, add a dedicated "Methods & Tools" reference block covering: Reda & Andreas
(2004), Ineichen & Perez (2002), Holmgren et al. (2018), Hwang & Yoon (1981), Opricovic (1998), Deng
(1982), Brans & Vincke (1985), van Buuren & Groothuis-Oudshoorn (2011), and a quantile-mapping
reference (e.g. Cannon et al. 2015) — none of these are currently in `references.bib` or
`.claude/references.md`, and all are directly load-bearing for claims this pipeline actually makes.
Also complete the bare Nahar (2003) citation note into a full BibTeX entry, and add Durin et al.
(2018) and a formal Thom (1959) entry, since both are directly quoted/used in code but not present in
either bibliography file. **New, added 2026-08-11**: add Bony & Citherlet (2007) — the Phase 7
model-class justification, independently confirmed via web search this session but not yet a formal
BibTeX entry in either bibliography file (Barqawi 2025 is already present via `Sources/`).
