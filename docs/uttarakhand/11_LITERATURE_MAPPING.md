# 11 — Literature Mapping

> **Consolidation note.** Temporal- and spatial-processing justifications now live in
> `03_PHASE_1_AUDIT.md`; ERA5 de-accumulation, solar geometry, derived solar variables,
> cross-source validation and quality control now live in `04_PHASE_2_AUDIT.md`; the climate
> signature's feature-to-PCM-property mapping now lives in `05_PHASE_3_AUDIT.md`. This file records
> only what the Uttarakhand source files actually cite, and what must be added.

## Method

Every entry below was checked against the **contents of `era5-uttarakhand/`** — the Python scripts,
their docstrings and comments, and the four markdown files (`README.md`, `README_PREPROCESSING.md`,
`PREPROCESSING_STEPS.md`, `NEXT_STEPS.md`, `VERIFICATION_METHODOLOGY.md`). Nothing was asserted from
general knowledge and nothing was imported from another state's documentation.

The governing plan document (`Objective1_PCM_Climate_Framework_Plan_v3`, cited in-script as "plan
v3.0") is **not present inside `era5-uttarakhand/`**. Every plan reference is therefore recorded as
*"cited by the script"*, not verified against the document.

---

## The complete citation footprint of `era5-uttarakhand/`

This is the headline finding of this file, and it is short.

**Exactly two author-year literature citations appear anywhere in the Uttarakhand pipeline:**

| Citation | Where | Context |
|---|---|---|
| **Al-Mamun 2023** | `07b_charging_feasibility.py`, line 58 | "roughly consistent with Al-Mamun2023's cited FPC 25-100C operating band" — justifying the 70 °C `REFERENCE_GOOD_DAY_TEMP_C` ceiling |
| **Singh 2025** | `06_build_pcm_database.py`, line 36 | Historical only: "The old script appended 7 hardcoded Singh2025 literature rows; those PCMs are now fully included in the 55-row CSV" |

Neither is a full reference — both are bare author-year strings in a comment. The Singh 2025
reference describes a **superseded** code path, not the current one.

Everything else the pipeline cites is an internal plan-document section or table.

## Plan-document references cited in-script

| Reference | Cited by |
|---|---|
| plan v3.0 §4.3 "Repair 1" | `02b_build_daily_aggregates.py`, `04b_climate_signature.py`, `README_PREPROCESSING.md` |
| plan v3.0 §4.5 / D2 | `06_build_pcm_database.py` |
| plan §5 (13-step methodology table) | `04_preprocess_uttarakhand.py` |
| plan §5.2 (scaling trap) | `04_preprocess_uttarakhand.py` |
| plan Table 9 (physical bounds; checks #2, #4, #7) | `04_preprocess_uttarakhand.py`, `03_plots_raw.py` |
| plan v3.0 §6 | `04b_climate_signature.py` |
| plan v3.0 §6.2 (never cluster on geography) | `04b_climate_signature.py`, `05_cluster_uttarakhand.py` |
| plan v3.0 §6.3 (constant `Tm_target` design rule) | `08_mcdm_ranking.py` |
| plan Table 8 (equivalent) | `04b_climate_signature.py` docstring |
| plan v2.0 §7 (GMM over K-Means; 0.15–0.35 silhouette band) | `05_cluster_regions.py` — **note the v2.0 version lag** |
| plan v3.0 §8 + Table 12 (feasibility filters) | `07_feasibility_filter.py`, `07b_charging_feasibility.py`, `06_build_pcm_database.py` |
| plan v3.0 §9 | `08_mcdm_ranking.py` |
| plan v3.0 §9.2 (Gaussian Tm fitness; σ = 4 K "justified from HX approach temperature") | `08_mcdm_ranking.py` |
| plan v3.0 §9.5 (low Kendall's W is a reportable finding) | `08_mcdm_ranking.py` |
| plan v3.0 Table 13 (AHP priors) | `08_mcdm_ranking.py` |
| plan v3.0 §11 + Table 18 (recommendation cards) | `09_recommendation_cards.py` |
| plan Table 16 (annual solar fraction 54–84 %) | `README.md`, `NEXT_STEPS.md` |
| plan v3.0 "Repair 2" (per-point elevation) | `NEXT_STEPS.md`, `README_PREPROCESSING.md` |

## Data products named (with URLs, no citations)

| Product | Named in | Citation present? |
|---|---|---|
| ERA5 / Copernicus CDS `reanalysis-era5-single-levels` | `01`, `config.py` | **No** |
| NASA POWER hourly point API | `01b` | **No** |
| GADM v4.1 India admin-1 | `00a` (URL given) | **No** |
| WorldPop India 2020 UN-adjusted 100 m | `00a` (URL given) | **No** |
| `pvlib` | `00b`, `02` | **No** |
| Ineichen clear-sky model | `02` (as the string `"ineichen"`) | **No** |
| SPA solar-position algorithm | `00b` (`method="spa"`, described as "pvlib's SPA algorithm — no manual equation-of-time code") | **No** |

---

## Methodology-component → implementation → literature matrix

| Component | Implementation | Supporting source in `era5-uttarakhand/` | Strength |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | `01`, `02` | Product named, URL implied via `cdsapi` | **Gap** — no product citation |
| NASA POWER as independent cross-check | `01b`, `02`, `02b` | Product + endpoint named | **Gap** — no citation |
| Population-weighted sampling (GADM + WorldPop, 87.5 %) | `00a` | URLs given | **Gap** — no methodology citation |
| Solar position (SPA) | `00b` (`method="spa"`), `02` (**unpinned**) | Algorithm named | **Gap** — needs Reda & Andreas (2004) |
| Clear-sky model (Ineichen) | `02` | Model named | **Gap** — needs Ineichen & Perez (2002) |
| `pvlib` software | `00b`, `02` | Library named | **Gap** — needs Holmgren et al. (2018) |
| ERA5 accumulated-field de-accumulation | `deaccumulate()` in `02` | Docstring reasoning only | **Weak / self-sourced** — and empirically contradicted, see `04_PHASE_2_AUDIT.md` Part A.3 |
| DNI from ERA5 direct field; DHI as closure residual | `compute_solar()` in `02` | Docstring states the fallback is not a decomposition model | **Gap** — needs a decomposition reference if a real model is ever added |
| Two-tier signature (sun-event + daily integral) | `04b` + `02b` | plan v3.0 §4.3 "Repair 1" | Plan-sourced only |
| `Tm_target = T_delivery + ΔT_approach = 57 °C` | `04b` | plan v2.0/v3.0 corrected rule; `PREPROCESSING_STEPS.md` explains the sign | Plan-sourced only |
| `T_mains_est_C = Ta_mean − 2.0` | `04b` | **None — unsourced in-code** | **Weak / open gap** |
| `L_required` night-discharge sizing (0.001 kg/s × 7 h × 50 kg) | `04b` | **None** | **Weak / open gap** — no `SHARE_PCM` factor either |
| `HSI = RH_mean × fraction(T_amb − T_dew < 3 K)` | `04b` | **None** | **Gap** — needs a humidity-stress-index reference |
| PCA on the thermodynamic block only | `04b` | plan v3.0 §6 | Plan-sourced |
| GMM over K-Means; soft membership | `05_cluster_uttarakhand.py` | plan v2.0 §7 rationale quoted in `05_cluster_regions.py` | Plan-sourced; internally argued |
| Silhouette band 0.15–0.40 | `05_cluster_uttarakhand.py` | plan §7 (0.15–0.35) widened in-script with a stated reason | Plan-sourced + local judgment |
| PCM band 42–70 °C | `06`, `07` | plan v3.0 Table 12 | Plan-sourced |
| PCM property imputation (MICE + RF + PMM) | `PCM_data/PCM_data/01_preprocess.py` | Method described at length; **no citation** | **Gap** — needs a MICE reference |
| Feasibility filters (melting window, latent-heat floor κ = 0.7, cycling ≥ 300, supercooling ≤ 8 K) | `07` | plan v3.0 §8 / Table 12 | Plan-sourced |
| Gaussian Tm-fitness transform, σ = 4 K | `08` | plan v3.0 §9.2 "justified from HX approach temperature" | **Weak / self-sourced** — state plainly |
| Shannon-entropy criterion weighting | `08` | **None** | **Gap** |
| TOPSIS | `08` | **None** | **Gap** |
| Grey Relational Analysis (ζ = 0.5) | `08` | **None** | **Gap** |
| Borda-count consensus | `08` | **None** | **Gap** |
| Kendall's W | `08` | plan v3.0 §9.5 for the *interpretation* only | **Gap** for the statistic itself |
| AHP prior weights | `08` | plan v3.0 Table 13; **explicitly labelled "an honest placeholder, not a claimed AHP result"** | Correctly self-limited |
| Flat-plate collector 25–100 °C operating band | `07b` | **Al-Mamun 2023** | The pipeline's only substantive citation |
| Annual solar fraction 54–84 % benchmark | `README.md`, `NEXT_STEPS.md` | plan Table 16 | Plan-sourced; never used (Phase 7 absent) |
| Grey-box lumped-enthalpy tank model | — | Named in `README.md` / `NEXT_STEPS.md`; **not implemented** | See `09_PHASE_7_AUDIT.md` |

---

## Uttarakhand-specific literature note

**No Uttarakhand-specific or Himalayan-specific climate reference appears anywhere in
`era5-uttarakhand/`.** The state-specific reasoning that does exist is stated as geographic domain
knowledge in prose:

- `05_cluster_uttarakhand.py`: "the high-altitude Himalayan belt around Chamoli/Pithoragarh vs. the
  Doon Valley around Dehradun vs. the Terai plains around Udham Singh Nagar/Haridwar are very
  plausibly different regimes — elevation alone spans roughly 200-2000m of populated terrain here."
- `README_PREPROCESSING.md`: "Uttarakhand's populated terrain genuinely spans roughly 200m (Terai
  plains near Udham Singh Nagar/Haridwar) to 2000m (hill towns)."
- `07b_charging_feasibility.py`: "this state's higher-altitude clusters see LESS reliable clear-sky
  access than the plains (more cloud/fog persistence, not less)."
- `PREPROCESSING_STEPS.md`: "hot foothill/Terai summer Apr–Jun, southwest monsoon Jun–Sep, cold
  high-altitude winter Dec–Feb."

None of these is cited. For a thesis, the correct framing is that **the K = 5 partition's physical
interpretation rests on geographic domain knowledge and internal BIC/silhouette statistics, not on
an external Uttarakhand-specific climate classification.** No external classification
(Köppen-Geiger, NBC/ECBC Indian climate zones) is wired into the pipeline — see
`06_PHASE_4_AUDIT.md`.

A related documentation error worth flagging: `03_plots_raw.py`'s docstring describes its seasonal
check as a sanity check against "hot dry Apr-Jun, **NE monsoon Oct-Dec**" — the northeast monsoon
is not Uttarakhand's regime. `PREPROCESSING_STEPS.md` has the correct climatology for the same
plot. Only the docstring's interpretation guidance is wrong; the plot itself is unaffected.

---

## Recommended "Methods & Tools" reference block

Every item below is a **gap** in `era5-uttarakhand/` — a method the pipeline uses without a
citation. These must be added before formal submission.

**Data products**
- ERA5 product citation (Copernicus Climate Data Store / ECMWF)
- NASA POWER project documentation
- GADM v4.1 and WorldPop (UN-adjusted 2020 India raster)

**Solar geometry and radiation**
- Solar Position Algorithm — for `method="spa"` in `00b` (and, once pinned, in `02`)
- Ineichen clear-sky model — for `get_clearsky(model="ineichen")`
- `pvlib` software citation
- A beam/diffuse decomposition reference — **only if** the `GHI/cos(SZA)` fallback in
  `compute_solar()` is ever replaced by a real model. As currently written the honest framing is
  "DNI from ERA5's mean direct short-wave radiation flux where available; DHI computed as a closure
  residual."

**Statistics and imputation**
- MICE / chained-equations imputation — for `IterativeImputer` in `04` and for
  `PCM_data/PCM_data/01_preprocess.py`'s MICE + RF + PMM chain
- Predictive Mean Matching — for the donor-based fill in the PCM cleaner
- Hampel filter / MAD-based outlier detection — for `04` step 3
- Gaussian Mixture model selection by BIC — for `05`
- Silhouette, Davies-Bouldin, Calinski-Harabasz — for `05`'s selection table

**MCDM**
- TOPSIS
- Grey Relational Analysis
- Shannon-entropy criterion weighting
- Borda count
- Kendall's coefficient of concordance (W)
- AHP — **only** as the source of the *form* of the prior. `08` explicitly labels its fixed weights
  "an honest placeholder, not a claimed AHP result", and no pairwise elicitation was performed. Any
  write-up must say so.

**PCM and thermal**
- A source for the 42–70 °C SWH-specific PCM band (currently plan-Table-12-sourced only)
- A published correlation for mains-water temperature vs ambient air temperature in India — to
  replace the unsourced `T_mains_est_C = Ta_mean − 2.0`
- A source for the night-discharge `L_required` sizing basis, and an explicit statement about the
  absence of a PCM fractional-contribution (`SHARE_PCM`) factor
- A humidity-stress-index reference for `HSI`
- Al-Mamun (2023) — **already cited in `07b`**; needs a full bibliographic entry
- The 54–84 % annual-solar-fraction benchmark's underlying references (currently plan Table 16
  only) — needed **only if** Phase 7 is implemented

**Explicitly self-sourced — state as such rather than citing**
- The Gaussian Tm-fitness transform with σ = 4 K (plan v3.0 §9.2 only)
- The `deaccumulate()` reset-hour treatment (docstring reasoning only, and empirically contradicted
  — see `04_PHASE_2_AUDIT.md` Part A.3)
- `07b`'s `REFERENCE_GOOD_DAY_TEMP_C = 70`, `MIN_ACHIEVABLE_TEMP_C = 42` and `POOR_DAY_Z = 1.28`,
  which the script itself calls "stated assumptions, not measured values"

---

## What this mapping does not claim

- That any of the above citations exist in `era5-uttarakhand/` today. **Two do** (Al-Mamun 2023,
  Singh 2025); everything else in the recommended block is absent.
- That the plan document's sections and tables say what the scripts say they say. The plan document
  is not in this folder and was not consulted.
- That the citation gaps are unique to Uttarakhand. This file makes no comparison to any other
  state's pipeline.
