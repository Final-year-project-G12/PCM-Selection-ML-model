# ASSAM — FULL PIPELINE (Consolidated, Code-Grounded)

**Objective 1 — Climate-Region-Aware PCM Recommendation Framework**
Group 12 · B.Tech CSE Final Year · Amrita School of Engineering · Guide: Dr. T. Deepika
Governing document: `Objective1_PCM_Climate_Framework_Plan_v3.docx` (v3.0)

---

## 0. What this document is, and how it was produced

This is the single consolidated reference for the **Assam** state pipeline — the third
of four states, implemented after Rajasthan and Tamil Nadu. It replaces the need to read
the 22 separate files in `docs/assam/` one at a time.

It differs from `RAJASTHAN_FULL_PIPELINE.md` and `UTTARAKHAND_FULL_PIPELINE.md` in one
important way. Those documents are **concatenations** of their per-state audit files. This
document is a **reconciliation**: every claim in `docs/assam/*.md` was checked line-by-line
against the actual source in `era5-assam/`, and where the two disagree, **the code wins and
the disagreement is recorded explicitly** (§3).

That reconciliation was worth doing. The Assam audit set was written partly from
intent and partly from the Tamil Nadu/Rajasthan scripts it was adapted from, and it has
drifted from the code in several places that matter for the thesis — including two claims
that are currently load-bearing in the "what makes Assam different" narrative and do not
survive contact with the source.

### 0.1 Reading record

Every file below was read in full for this consolidation.

**Documentation — `docs/assam/` (22 files)**

| File | Covers |
|---|---|
| `00_MASTER_OVERVIEW.md` | Pipeline map, phase status, design choices vs Rajasthan |
| `01_PROJECT_CONTEXT.md` | Identity, D1–D8, N1–N6, phase numbering |
| `02_DATA_SOURCES_AND_VARIABLES.md` | ERA5/POWER/WorldPop/GADM, 18-index signature, PCM DB |
| `03_PHASE_1_AUDIT.md` | Data collection |
| `04_PHASE_2_AUDIT.md` | Combine + daily aggregates |
| `05_PHASE_3_AUDIT.md` | QC (2.5) + climate signature (3) |
| `06_PHASE_4_AUDIT.md` | Clustering, BIC table, bootstrap ARI |
| `07_PHASE_5_AUDIT.md` | PCM database, feasibility filter |
| `08_PHASE_6_AUDIT.md` | MCDM stack, per-cluster results |
| `09_PHASE_7_8_AUDIT.md` | Physics validation, recommendation cards |
| `09_ERA5_DATA_PIPELINE.md` | Deaccumulation story, unit table, DNI caveat |
| `10_TEMPORAL_PROCESSING.md` | UTC handling, SPA, seasons |
| `11_SPATIAL_PROCESSING.md` | Grid alignment, boundary, elevation |
| `12_SOLAR_GEOMETRY.md` | pvlib usage, Ineichen, night handling |
| `13_SOLAR_DERIVED_VARIABLES.md` | GHI/DNI/DHI/CSI derivation |
| `14_ERA5_POWER_VALIDATION.md` | Agreement analysis, BACKBONE decision |
| `15_QUALITY_CONTROL.md` | Bounds, outliers, imputation |
| `16_CLIMATE_SIGNATURE.md` | Feature → PCM-property mapping |
| `17_LITERATURE_MAPPING.md` | Citation matrix and gaps |
| `18_RESEARCH_GAP_MAPPING.md` | Phase → N and Phase → RG mapping |
| `20_IMPLEMENTATION_ISSUES.md` | Ranked issue list |
| `21_REPRODUCIBILITY.md` | Reproducibility checklist |
| `22_FINAL_READINESS_REPORT.md` | Readiness verdict |

Note the internal numbering of these files is inconsistent — the `NN_` filename prefix and
the `# NN —` heading inside disagree in eight of them (e.g. `15_QUALITY_CONTROL.md` opens
with `# 18 — Quality Control Audit`, `22_FINAL_READINESS_REPORT.md` opens with
`# 11 — Final Readiness Report`). Cross-references inside the docs point at the *heading*
numbers, so several of them resolve to nothing. This document uses filenames throughout.

**Source — `era5-assam/` (32 files)**

| File | Phase | Role |
|---|---|---|
| `config.py` | — | Path anchoring, CDS credential loading |
| `.gitignore` | — | Excludes `data/` from version control |
| `00a_build_population_grid.py` | 1 | WorldPop × GADM → 0.25° population-weighted points |
| `00b_build_suntimes.py` | 1 | pvlib SPA sunrise/noon/sunset table |
| `01_download_era5_assam.py` | 1 | CDS ERA5 download, instant + accum |
| `01b_download_nasapower.py` | 1 | NASA POWER hourly point download |
| `00_unzip_accum.py` | 1 | Repairs CDS zip-disguised-as-`.nc` files |
| `02_combine_assam.py` | 2 | ERA5 + POWER merge, units, solar geometry |
| `02b_build_daily_aggregates_assam.py` | 2 | True daily integrals → Tier-2 indices |
| `03_plots_raw.py` | 2 | Pre-cleaning raw QA plots |
| `03b_agreement_analysis_assam.py` | 2 | ERA5 vs POWER agreement, BACKBONE/QUANTILE_MAP |
| `04_preprocess_assam.py` | 2.5 | Bounds, IST, imputation, outliers, bias correction, parquet |
| `04b_climate_signature.py` | 3 | Per-site signature, interactions, PCA, scaling |
| `05_cluster_assam.py` | 4A | GMM full-covariance clustering, BIC, bootstrap ARI |
| `05b_level_b_seasonal_assam.py` | 4B | Seasonal PCM sensitivity (**does not run** — §3.4) |
| `05_plot_assam.py` | — | 17-figure visualisation suite (maps/timeseries/stats/features/solar) |
| `06_build_pcm_database.py` | 5 | Manufacturer + literature PCM database (**crashes** — §3.1) |
| `07_feasibility_filter.py` | 5 | 7-constraint hard filter, auto-relaxation |
| `07b_charging_feasibility.py` | 5 | Optional heuristic Tm_target regime cap |
| `08_mcdm_ranking.py` | 6 | TOPSIS/GRA/PROMETHEE II/VIKOR + Borda/Copeland + 5,000-draw MC |
| `09_recommendation_cards.py` | 8 | Per-cluster markdown cards + criterion contributions |
| `10_physics_validation.py` | 7 | Grey-box lumped-enthalpy tank simulation |
| `comparison_plots_assam.py` | — | 8 cross-step comparison figures |
| `generate_assam_plots.py` | — | 13-plot Objective-1 figure set (Plotly/Folium) |
| `fast_generate_raw_signatures.py` | 3 | **Rival** signature implementation (§3.3) |
| `check_points_in_assam.py` | 1 | Point-in-polygon boundary QA |
| `verify_grid_points.py` | 1 | Grid point listing (**hardcoded `m:/` path**) |
| `verify_01_preprocessing_assam.py` | QA | Preprocessing verification figures |
| `verify_02_clustering_assam.py` | QA | Clustering verification figures |
| `verify_03_feasibility_assam.py` | QA | Feasibility verification figures |
| `verify_04_ranking_assam.py` | QA | MCDM verification figures |
| `PLOTS_GUIDE.md` | — | Plot inventory and execution commands |

**Supporting files also inspected**

- `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` (55 rows) — the Phase 5 input
- `PCM_data/data/PCM_Properties_55records_42_70C_dense.csv`
- `PCM_data/data/PCM_Properties_cleaned_mice_pmm.csv`
- `PCM_data/data/05_imputation_provenance.csv`

### 0.2 Verification status — read this before quoting any number

**`era5-assam/data/` does not exist in this working tree.** `era5-assam/.gitignore`
excludes `data/raw/`, `data/processed/`, `data/preprocessed/` and `data/plots/` in their
entirety, so no pipeline output — no signature CSV, no cluster assignment, no MCDM table,
no physics result — is present or version-controlled.

Consequences, stated plainly:

- Every **algorithm, constant, formula, threshold and file path** in this document was read
  directly from source and is verified.
- Every **numeric result** (cluster sizes, BIC table, ARI, Kendall's W, Spearman rho, solar
  fractions, top-3 rankings) is **quoted from `docs/assam/*.md`**, which in turn quotes a
  pipeline run that is not reproducible from this checkout. These are marked
  *(reported, unverified)* throughout.
- Two independent artifacts corroborate the cluster-level numbers — the header comment block
  of `05_plot_assam.py` and the cluster colour map in the same file both record the same
  4-cluster structure, medoid IDs and kt values as `docs/assam/06_PHASE_4_AUDIT.md`. That is
  weak corroboration (same run, transcribed twice), not verification.
- Because of §3.1 and §3.3, **it is not established that the reported Phase 5–8 numbers can be
  regenerated by the current code at all.** Treat them as a record of a historical run.

---

## 1. Project context

### 1.1 Identity and scope

Objective 1 turns 10 years of reanalysis climate data into population-weighted climate
regimes, derives PCM performance targets per regime, and ranks candidate phase-change
materials against those targets with an auditable, multi-method, uncertainty-aware pipeline.

Sub-goals SG1–SG4 (signature construction, regime discovery, feasibility + ranking, physics
validation) are bounded. **Out of scope for Objective 1:** hardware prototyping, DRL control,
real-time operation.

Assam is the third of four target states (§1.3, Table 1 of the framework doc), chosen to span
distinct climate archetypes: arid/semi-arid (Rajasthan), humid subtropical/monsoon-heavy
(Assam), coastal tropical (Tamil Nadu), high-relief montane (Uttarakhand). Assam is the
monsoon-dominated archetype — the highest annual rainfall of the four, high year-round
humidity, and an intra-state gradient from the Brahmaputra floodplain to the hill districts
(Karbi Anglong, Dima Hasao).

### 1.2 Phase numbering (authoritative)

| Phase | Name | Assam script(s) |
|---|---|---|
| 1 | Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` |
| 2 | Preprocessing and Cross-Source Validation | `02`, `02b`, `03b` |
| 2.5 | Quality Control | `04_preprocess_assam.py` |
| 3 | Climate Signature Construction | `04b_climate_signature.py` |
| 4 | Climate Regime Clustering | `05_cluster_assam.py` (Level A), `05b` (Level B) |
| 5 | Feasibility Filtering (+ PCM database build) | `06`, `07b`, `07` |
| 6 | Multi-Criteria Ranking Engine | `08_mcdm_ranking.py` |
| 7 | Physics-Based Validation | `10_physics_validation.py` |
| 8 | Explanation and Final Output | `09_recommendation_cards.py` |

### 1.3 Deliverables (§1.4, Table 2)

| ID | Deliverable | Status |
|---|---|---|
| D1 | Validated climate dataset | Code complete — `climate_assam_points.csv` |
| D2 | Climate signature + PCA | Code complete — but see §3.3 (two rival implementations) |
| D3 | Regime clusters + external validation | Partial — k=4 produced; Köppen-Geiger **not** wired in |
| D4 | PCM feasibility-survivor set | **Blocked** — see §3.1 |
| D5 | MCDM ranking + MC confidence | Code complete; depends on D4 |
| D6 | Physics-validated ranking | Code complete; depends on D4/D5 |
| D7 | Recommendation cards | Code complete; three field-name bugs (§3.5) |
| D8 | Methodology write-up | This document + `docs/assam/` |

### 1.4 Novelty positions (§3, Table 3)

| ID | Claim | Assam implementation, as coded |
|---|---|---|
| N1 | Discovered regimes, not hand-picked zones | GMM full covariance, k=4, BIC + silhouette + 500-bootstrap ARI. **Internal statistics only** — no external classification. |
| N2 | Two-tier climate signature | Tier 1 sun-event (ERA5) + Tier 2 daily-integral (NASA POWER hourly → daily). Implemented; index list differs from the docs (§7). |
| N3 | Corrected 42–70 °C SWH-specific PCM band | Enforced in `06` (`in_absolute_band`) and `07` (constraint 2). |
| N4 | Top-3 + method-agreement reporting | Borda + Copeland + Kendall's W per cluster; 5,000-draw MC. Implemented as documented. |
| N5 | Physics-validated ranking | `10_physics_validation.py` implemented and run; genuine negative result reported. |
| N6 | Population-weighted sampling | `00a`, `COVERAGE_TARGET = 0.875`. Implemented as documented. |

**Do not conflate N1–N6 with RG1–RG5.** N1–N6 are the framework doc's own novelty
positioning for Objective 1. RG1–RG5 (no real-time control, no integrated prototype, poor
demand alignment, limited experimental validation, no predictive optimisation) come from a
separate literature-scoring artifact and belong to the broader multi-objective project.
Objective 1 directly addresses **RG5** only; Phase 7 touches RG4 indirectly; Phase 8 output
*feeds* RG2/RG3 rather than addressing them; RG1 is explicitly out of scope.

---

## 2. Pipeline map, as actually implemented

```
PHASE 1 — DATA COLLECTION
  00a_build_population_grid.py     → processed/population_grid_points.csv
                                     (WorldPop 100 m × GADM Assam → 0.25° ERA5-aligned
                                      cells, ranked by population, minimal prefix ≥ 87.5%)
  00b_build_suntimes.py            → processed/suntimes.csv
                                     (pvlib SPA sunrise/transit/sunset, UTC, 2016-01-01…2025-12-31)
  01_download_era5_assam.py        → raw/era5/points/era5_AS_points_{YYYY}_{MM}_{instant,accum}.nc
                                     (240 CDS calls: 10 yr × 12 mo × 2 var-types, ONE bbox)
  01b_download_nasapower.py        → raw/nasapower/power_{point_id}_{YYYY}.json
                                     (HOURLY point API, 128 pts × 10 yr = 1,280 calls)
  00_unzip_accum.py                → repairs .nc files that are actually ZIP archives
        ↓
PHASE 2 — PREPROCESSING & CROSS-SOURCE VALIDATION
  02_combine_assam.py              → processed/climate_assam_points.csv
                                     (unit conversion, Magnus RH, pvlib solar geometry,
                                      ssrdc-preferred clear-sky, ERA5+POWER nearest-hour
                                      match ≤3 h, 4-season labels)
  02b_build_daily_aggregates_assam.py
                                   → processed/daily_aggregates_assam.csv
                                     processed/tier2_signature_assam.csv
                                     (re-reads the FULL hourly POWER cache; IST day buckets;
                                      ≥20 h/day required)
  03_plots_raw.py                  → plots/raw/*.png            (read-only QA)
  03b_agreement_analysis_assam.py  → processed/era5_power_agreement_assam.csv
                                     processed/bias_decision_assam.txt
                                     processed/quantile_maps_assam.joblib (only if QUANTILE_MAP)
        ↓
PHASE 2.5 — QUALITY CONTROL
  04_preprocess_assam.py           → preprocessed/parquet/{point_id}.parquet
                                     preprocessed/qc_report.txt
                                     (bounds→NaN, UTC→IST, night masking, interpolate+
                                      climatological fill, drop site-years >5% missing,
                                      3σ + IsolationForest flagging, UNCONDITIONAL
                                      quantile-map bias correction, kt recompute)
        ↓
PHASE 3 — CLIMATE SIGNATURE CONSTRUCTION
  04b_climate_signature.py         → processed/climate_signatures_raw.csv    (19 indices/site)
                                     processed/climate_signatures_matrix.csv (17 + PCs, scaled)
                                     processed/pca_loadings.csv
                                     preprocessed/climate_signature_report.txt
  [fast_generate_raw_signatures.py → OVERWRITES climate_signatures_raw.csv with
                                     incompatible definitions — see §3.3]
        ↓
PHASE 4 — CLIMATE REGIME CLUSTERING
  05_cluster_assam.py              → clustering/bic_selection_assam.csv
                                     clustering/kmeans_comparison_assam.csv
                                     clustering/bootstrap_stability_assam.csv
                                     clustering/cluster_assignments_assam.csv
                                     clustering/cluster_profiles_assam.csv
                                     clustering/scaler_assam.joblib
                                     clustering/gmm_model_assam.joblib
                                     plots/cluster_map_assam.png
  05b_level_b_seasonal_assam.py    → [never executes — inputs do not exist, §3.4]
        ↓
PHASE 5 — PCM DATABASE + FEASIBILITY FILTERING
  06_build_pcm_database.py         → pcm/pcm_database_assam.csv   [KeyError, §3.1]
  07b_charging_feasibility.py      → adds Tm_target_C_regime_capped to cluster_profiles (optional)
  07_feasibility_filter.py         → pcm/feasibility_survivors_assam.csv
        ↓
PHASE 6 — MULTI-CRITERIA RANKING
  08_mcdm_ranking.py               → pcm/mcdm_topk_assam.csv
                                     pcm/mcdm_full_scores_assam.csv
                                     pcm/monte_carlo_stability_assam.csv
        ↓
PHASE 7 — PHYSICS-BASED VALIDATION
  10_physics_validation.py         → pcm/physics_validation_results_assam.csv
                                     pcm/physics_validation_spearman_assam.csv
        ↓
PHASE 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py       → pcm/recommendation_cards_assam.md
```

There is **no `run_all_assam.py`**. Scripts must be invoked manually in the order above.
Rajasthan has an orchestration script; Assam does not.

---

## 3. Discrepancy register — where `docs/assam/` and `era5-assam/` disagree

Ranked by consequence for the thesis. Every item below was confirmed by reading the source.

### 3.1 CRITICAL — The corrosion veto cannot fire, and `06_build_pcm_database.py` crashes

This is the most consequential finding in this audit, because it removes the single claim
that `docs/assam/` uses to differentiate Assam from Rajasthan.

**The claim.** Six separate documents state that the humidity-driven corrosion veto is
"load-bearing for Assam": `00_MASTER_OVERVIEW.md` lists it in the design-choices table,
`07_PHASE_5_AUDIT.md` gives it a dedicated subsection, `16_CLIMATE_SIGNATURE.md` calls it
"a real climate-discriminating result", `18_RESEARCH_GAP_MAPPING.md` calls it "N3's
strongest Assam-specific contribution", and `22_FINAL_READINESS_REPORT.md` lists it third
among the five strongest components.

**The mechanism, in code.** `06_build_pcm_database.py` sets the flag:

```python
db["corrosion_class"] = np.where(
    db["pcm_type"].astype(str).str.contains("Inorganic", na=False),
    "check_manually", "low_organic")
```

and `07_feasibility_filter.py` vetoes on it:

```python
cluster_is_high_hsi = cluster_hsi > hsi_p75_global
df["pass_corrosion"] = ~((df["corrosion_class"] == "check_manually") & cluster_is_high_hsi)
```

**The finding.** No PCM in the project has `pcm_type` containing "Inorganic".
`PCM_Properties_cleaned_mice_pmm_detailed.csv` holds **55 rows, every one of them
"Organic…"** (`Organic (RT-line)`, `Organic n-alkane`, `Organic fatty acid`,
`Organic PCM`, `Organic blend`, `Organic/composite blend`, `Organic bio-based PCM`,
`Organic/eutectic composite`, `Organic/polymer blend`). A grep for the literal string
`Inorganic` across all four PCM CSVs in `PCM_data/` returns nothing. All 7 literature rows
added by `literature_rows()` are hardcoded `"pcm_type": "Organic"`.

Therefore `corrosion_class` is `"low_organic"` for every row, `pass_corrosion` is `True`
for every PCM in every cluster, and **the veto is inert regardless of HSI**. The
`hsi_p75_global` threshold is computed and printed but changes nothing. Any survivor-count
difference between clusters comes from the melting window and latent-heat floor, not from
corrosion.

**A second, independent failure in the same file.** `load_manufacturer_rows()` reads:

```python
out["family"] = np.where(df["is_rt_line"] == 1, "Rubitherm RT", "PLUSS savE")
```

**No PCM CSV in the repository has an `is_rt_line` column.** Against every available source
file this raises `KeyError: 'is_rt_line'` on the first call, so `06_build_pcm_database.py`
cannot produce `pcm_database_assam.csv`, and Phases 5–8 cannot run end-to-end from a clean
checkout.

**Third: the database size claim is stale in the opposite direction.** Every doc says the
database is "25 rows vs the 40–60-row target" and treats undersizing as the single blocking
item. The actual source is 55 manufacturer rows spanning Tm 40.5–70.0 °C, plus 7 literature
rows = **62 rows**. The sibling file is even named
`PCM_Properties_55records_42_70C_dense.csv`. The 40–60-row target has been met and exceeded.
The 25-row table printed in `07_PHASE_5_AUDIT.md` lists products (`savE® HS36`, `savE® OM35`,
`RT35`, `savE® OM37`, `RT38`, `savE® OM39`) that **do not appear in the current CSV at all** —
it describes a superseded database.

**What this means for the write-up.** Three things follow, and none of them are optional:

1. Delete the corrosion-veto differentiator from the Assam narrative, or re-establish it by
   adding genuinely inorganic candidates (salt hydrates — the `savE® HS` line — with
   `pcm_type` containing "Inorganic"). As it stands the claim is unsupported.
2. Fix `06` before any Phase 5–8 result is regenerated. Either derive `family` from the
   `manufacturer` / `pcm_type` columns that do exist, or add `is_rt_line` upstream in
   `PCM_data/`.
3. Retire "the PCM database is undersized" as the headline blocker. The pool is 62 rows.
   The Phase 7 negative result needs a different explanation (§3.6 offers one).

### 3.2 CRITICAL — `04_preprocess_assam.py` ignores the BACKBONE decision

`docs/assam/14_ERA5_POWER_VALIDATION.md` states:

> "When `04_preprocess_assam.py` runs, it dynamically reads the `bias_decision_assam.txt`
> file. Because the decision is `BACKBONE`, the script correctly **bypasses** the empirical
> quantile mapping step, allowing the raw, structurally correct ERA5 data to flow into the
> downstream clustering phases unmodified."

`04_preprocess_assam.py` never opens `bias_decision_assam.txt`. It has no conditional at
all. Step [7] runs unconditionally:

```python
if "era5_GHI" in df.columns and "power_ALLSKY_SFC_SW_DWN" in df.columns:
    df["era5_GHI_corrected"] = df["era5_GHI"].copy()
    for name, grp in df.groupby(["point_id", "season_code"]):
        ...
        corrected = np.interp(era_vals, np.sort(era_vals), np.sort(ref_vals))
        df.loc[grp.index, "era5_GHI_corrected"] = corrected
    df["era5_GHI"] = df["era5_GHI_corrected"]      # ← overwritten in place
```

So per-(point, season) empirical quantile mapping onto NASA POWER **is always applied**, and
the original `era5_GHI` is overwritten. Every downstream phase consumes bias-corrected GHI.

Three separate problems come out of this:

- The documented claim is the exact inverse of the behaviour. `22_FINAL_READINESS_REPORT.md`
  goes further and cites the BACKBONE decision as proof that "no synthetic bias correction
  was necessary" — while the code applies one to every row.
- `03b_agreement_analysis_assam.py`'s decision output is **dead**. Nothing reads
  `bias_decision_assam.txt`, and `quantile_maps_assam.joblib` (written only in the
  QUANTILE_MAP branch) is never loaded by anything. `04` builds its own maps inline.
- The correction is applied at a finer grain than the one that was validated. `03b` decides
  globally on season-stratified GHI; `04` corrects per (point, season) — 128 × 4 = 512
  separate empirical maps, many fitted on relatively few points.

Note also that `docs/assam/04_PHASE_2_AUDIT.md`, `15_QUALITY_CONTROL.md` and
`20_IMPLEMENTATION_ISSUES.md` (item 5) all still assert that no agreement analysis exists
for Assam and that `bias_decision_assam.txt` was never produced. `14_ERA5_POWER_VALIDATION.md`
and `22_FINAL_READINESS_REPORT.md` assert the opposite. The script exists; the docs simply
were not updated together.

**Decide and document one of two positions:** either the quantile correction is intended
(then say so, and delete the BACKBONE narrative), or BACKBONE is intended (then add
`if decision == "QUANTILE_MAP":` around step [7]). The current state is indefensible either
way because the paper and the code disagree.

### 3.3 CRITICAL — Two scripts write `climate_signatures_raw.csv` with different formulas

`04b_climate_signature.py` and `fast_generate_raw_signatures.py` both write
`data/processed/climate_signatures_raw.csv`. They compute different quantities under the
same column names. Whichever ran last defines the signature that Phase 4 clusters on.

| Index | `04b_climate_signature.py` (Phase 3 proper) | `fast_generate_raw_signatures.py` |
|---|---|---|
| `HSI` | `RH_mean × mean(fraction of events with Ta − Td < 3 K)` | `RH_mean × mean(T_dew)` |
| `CCI` | `CCI_true` — **longest consecutive cloudy-day run, in days** | `mean(era5_cloud_cover)` (0–1 fraction) |
| `SAI` | `Σ GHI_daily / Σ GHIcs_daily` over all usable days | `fraction of events with CSI > 0.6` |
| `cloudy_frac` | fraction of days with `kt_daily < 0.35` | fraction of events with `cloud_cover > 0.7` |
| `kt_mean`, `kt_std` | POWER daily `kt` (all-sky ÷ clear-sky daily integrals) | ERA5 event-level `CSI` |
| `GHI_daily_kWh` | true daily integral, Σ hourly POWER ÷ 1000 | `mean(era5_GHI) × 24 / 1000` |
| `DTR` | daily `Ta_max − Ta_min` from hourly POWER, averaged | monthly `max − min` of event temps, averaged |
| `HDD18` / `CDD24` | daily-mean degree-days from POWER | `Σ over events / 24` |
| `Ta_mean/p95/p05` | **noon-event** ERA5 `T_amb` mean/quantiles | all-event ERA5 `T_amb` mean/quantiles |
| `Tm_target`, `L_required`, 5 interaction terms, PCA | produced | **absent** |

These are not refinements of each other; `CCI` alone differs by three orders of magnitude
(days vs a 0–1 fraction), which would dominate any standardised clustering matrix.

Additional problems with `fast_generate_raw_signatures.py`:

- It reads `data/preprocessed/assam_cleaned_physical.csv`, which **no script in the pipeline
  produces**. `04_preprocess_assam.py` writes per-point parquet, not that CSV.
- It requires an `is_daytime` column that `02` and `04` never create.
- Its docstring claims "all 18 physical climate signature indices"; it emits 19 plus 4
  metadata columns.

**Recommendation:** delete `fast_generate_raw_signatures.py`, or rename its output to
`climate_signatures_raw_fast.csv` so it can never silently win a race with Phase 3. Until
then, no signature-derived result has determinate provenance.

### 3.4 HIGH — Phase 4 Level B is dead code

`05b_level_b_seasonal_assam.py` is described in its own docstring as the source of "the most
interesting result" and as direct empirical motivation for the Objective 3 DRL controller. It
cannot run. Its dependency gate:

```python
missing = [f for f in (PHYSICAL_FILE, ASSIGN_FILE, PROFILE_FILE, PCM_FILE, SCORES_FILE)
           if not f.exists()]
if missing: ... return
```

fails on two of the five, always:

- `PHYSICAL_FILE = preprocessed/assam_cleaned_physical.csv` — never produced by anything.
- `SCORES_FILE = pcm/mcdm_full_scores_by_cluster.csv` — `08_mcdm_ranking.py` writes
  `pcm/mcdm_full_scores_assam.csv`. The filename is inherited from another state's pipeline.

Its constants have also drifted from the scripts it claims to mirror:

| Constant | `05b` | Authoritative source |
|---|---|---|
| `WINDOW_LOWER_OFFSET` | 5.0 | 6.0 (`07_feasibility_filter.py`) |
| Tm fitness | symmetric σ = 4.0 | asymmetric σ = 4.0 / 1.5 (`08_mcdm_ranking.py`) |
| Cycling criterion | `cycles_confidence` | `cycles_confidence_imputed` (`09`) |
| `L_required` basis | third variant — see §8 | — |
| `DRAW_RATE_KG_PER_S` | `60/1000/60` = 0.001 kg/s, commented "60 L/min" | off by 1000× |

No Assam documentation file mentions Level B at all. Either fix the two paths and reconcile
the constants, or state in the thesis that Level B seasonal analysis was implemented for
Assam but not executed.

### 3.5 HIGH — Three field-name bugs silently blank the recommendation cards

`05_cluster_assam.py` writes cluster profiles with **`_mean`/`_std` suffixes**:

```python
row[f"{col}_mean"] = np.average(vals, weights=w.loc[vals.index])
row[f"{col}_std"]  = vals.std()
```

`09_recommendation_cards.py` then looks the columns up **without** the suffix:

```python
SIGNATURE_DISPLAY = ["GHI_daily_kWh", "Ta_mean", "DTR", "kt_mean", "cloudy_frac",
                     "CCI", "HDD18", "CDD24", "RH_mean", "HSI", "monsoon_index"]
for col in SIGNATURE_DISPLAY:
    if col in prof and prof[col] == prof[col]:
        lines.append(f"| {col} | {prof[col]:.3f} |")
```

None of those names exist in the profile row (they are `GHI_daily_kWh_mean`, `Ta_mean_mean`,
…), so **the "Climate signature (population-weighted mean)" table in every card is written
with a header and zero rows.**

Two more of the same kind in the same function:

- `prof.get('Tm_target_C', float('nan'))` — the column is `Tm_target_mean`, so every card
  prints `Tm_target = nan C`.
- `"total_population_covered" in prof` — the column is `total_population`, so the
  "Population covered" line is never emitted.

`l_req_kj_kg` is computed from `L_required_kWh_mean`, which **does** exist, so the
L_required figure is the one derived value that prints correctly.

**A fourth bug, different in kind.** The criterion-contribution decomposition — the feature
`docs/assam/` presents as Assam's own methodological contribution over Tamil Nadu — reads a
boolean:

```python
CRITERIA = ["f_Tm", "latent_heat_margin_ratio", "rho_H_MJ_m3", "TC_W_mK",
            "cycles_confidence_imputed"]
```

`cycles_confidence_imputed` is created in `08_mcdm_ranking.py` as
`df["cycles_confidence"].isna()` — an *imputation flag*, not the criterion value. The
"Cycling_Stability" contribution percentage is therefore computed from True/False, so it
reports 0% for every PCM with a known cycle count and a positive share only for PCMs whose
cycling data was missing — the exact inverse of the intended meaning. The correct column is
`cycles_confidence`. `weight_cols` already maps back to `weight_cycles_confidence`, so only
the value column is wrong; this is a one-word fix.

### 3.6 HIGH — The physics model cannot charge PCMs above ≈ Ta + 16 °C

`10_physics_validation.py` drives the tank from a collector coil temperature:

```python
COLLECTOR_EFF = 0.40
tc = tamb + COLLECTOR_EFF * isolar / 20.0
```

At a peak irradiance around 800 W/m², `tc ≈ tamb + 0.40 × 800 / 20 = tamb + 16 °C`. With
Assam cluster mean temperatures of 26–28 °C, the collector node tops out near **44 °C**. A
PCM with Tm above roughly 45 °C can therefore never complete a melt cycle in this model,
regardless of its properties.

This is not incidental — it is the mechanism behind two headline results:

- The reported solar fractions in `09_PHASE_7_8_AUDIT.md` show RT44HC (Tm 43 °C) and
  C22H46 (Tm 44.5 °C) at 82–85%, while RT45HC (Tm 47 °C) sits at 51–52%. That is the
  Tm > 44 °C cliff, not a material property difference.
- `08_mcdm_ranking.py` was subsequently patched with an **asymmetric** Tm fitness whose
  comment names this exact diagnosis: *"Asymmetric fitness: diagnosed from Phase 7 physics
  validation. SF crashes for Tm > target because the collector can't reach it."*

So a modelling artefact in Phase 7 has been fed back into the Phase 6 scoring function that
Phase 7 is supposed to independently validate. That circularity should be stated plainly, or
broken.

**Compounding this, the file's own docstring contradicts its constants:**

| Assumption | Docstring (copied verbatim into `docs/assam/09_PHASE_7_8_AUDIT.md`) | Code |
|---|---|---|
| Collector–tank coil area | 2.5 m² | `A_C_M2 = 2.0` |
| Collector efficiency | 0.70 | `COLLECTOR_EFF = 0.40` |
| Draw mass | 75 kg × 2/day | `DRAW_MASS_KG = 100.0` |

The docs reproduce the docstring, so the published assumption table is wrong on three of
nine rows. A 100 kg draw from a 150 kg tank, twice daily, is also a much more aggressive
duty cycle than 75 kg — worth stating explicitly since it drives the solar fraction.

`MAX_PCMS_PER_CLUSTER = 20` carries the comment "not expected to bind given ~25-row
database". With 62 rows it may well bind; check it after §3.1 is fixed.

### 3.7 MEDIUM — Canonical cluster relabeling does not exist in Assam

`06_PHASE_4_AUDIT.md`, `21_REPRODUCIBILITY.md` and `22_FINAL_READINESS_REPORT.md` all state
that clusters are relabeled by ascending mean latitude immediately after the GMM fit, and
`22_FINAL_READINESS_REPORT.md` lists this as the first of Assam's "strongest components"
("Assam benefits from the fix already being in place").

`05_cluster_assam.py` contains no such step. `hard_labels = gmm_final.fit_predict(X)` is
written straight to `cluster_assignments_assam.csv` with no reordering. Cluster IDs are
whatever the GMM's component ordering produces. They are reproducible only because
`random_state=42` is fixed — not because they are canonical. Any change to the feature set,
the sklearn version, or `n_init` can permute the labels, and nothing downstream would
notice.

### 3.8 MEDIUM — NASA POWER: hourly, not daily; and no precipitation

`02_DATA_SOURCES_AND_VARIABLES.md` and `03_PHASE_1_AUDIT.md` describe POWER as a **daily**
product with parameters `ALLSKY_SFC_SW_DWN, T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR`.

`01b_download_nasapower.py` uses the **hourly** endpoint:

```python
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
POWER_PARAMETERS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M"
```

Differences that matter:

- **`CLRSKY_SFC_SW_DWN` is downloaded** and the docs omit it — it is what makes the Tier-2
  daily clearness index `kt_daily` a true clear-sky ratio rather than a model estimate.
- **`PRECTOTCORR` is not downloaded.** `02b`'s own docstring says so explicitly and warns
  that `monsoon_index` therefore comes from the ERA5 3×/day precipitation proxy, *not* from a
  true Tier-2 daily integral. `02_DATA_SOURCES_AND_VARIABLES.md` nonetheless lists
  `monsoon_index` and `precipitation_annual` as Tier-2 POWER indices. `precipitation_annual`
  is not computed anywhere.
- `T2M_MAX`/`T2M_MIN` are not requested; `Ta_max_true`/`Ta_min_true` are derived in `02b` as
  the max/min of the 24 hourly `T2M` values.
- Wind is `WS10M` (10 m), not `WS2M`.

### 3.9 MEDIUM — ERA5 variable list and the `accum_to_flux` name

`02_DATA_SOURCES_AND_VARIABLES.md` lists `msl` (mean sea level pressure) and omits
`ssrdc`. The download script requests:

```python
INSTANT_VARS = ["2m_temperature", "2m_dewpoint_temperature",
                "10m_u_component_of_wind", "10m_v_component_of_wind",
                "total_cloud_cover", "surface_pressure"]
ACCUM_VARS  = ["surface_solar_radiation_downwards",
               "surface_solar_radiation_downward_clear_sky",
               "mean_surface_direct_short_wave_radiation_flux",
               "surface_thermal_radiation_downwards",
               "total_precipitation"]
```

- Pressure is **`surface_pressure` (`sp`)**, not `msl`. `02` converts `sp / 100 → P_atm`.
- **`ssrdc` (clear-sky GHI) is downloaded** and the docs never mention it, despite it being
  the more defensible clear-sky source (§6.3).
- The deaccumulation function is named **`deaccumulate()`** in `02_combine_assam.py`, not
  `accum_to_flux()` as `09_ERA5_DATA_PIPELINE.md` and `13_SOLAR_DERIVED_VARIABLES.md` state,
  and the non-negative clip is applied at the call site rather than inside it. Functionally
  identical — stateless, no differencing — but the name and snippet in the docs do not exist
  in the codebase and should not be quoted verbatim into a paper.
- ERA5 download is **one bounding box per year-month**, 240 calls total, with per-point
  nearest-neighbour extraction happening later in `02`. `03_PHASE_1_AUDIT.md` describes it as
  "per-point download", which would be 128 × 120 calls.

### 3.10 MEDIUM — The 18-index signature is 19 indices, and the list is wrong

See §7 for the authoritative list. Summary of the difference:

- **In the docs but not computed:** `precipitation_annual`, `Ta_min_true`, `Ta_max_true`
  (these exist per-day in `daily_aggregates_assam.csv` but are never carried into the
  signature).
- **Computed but absent from the docs:** `kt_std`, `wind_mean`, `seasonality`, `GHI_mean`,
  and the **five interaction terms** (`ix_GHI_x_kt_std`, `ix_DTR_x_cloudy`,
  `ix_RH_x_dT_store`, `ix_wind_x_dT_soil`, `ix_CCI_x_1mSAI`). The Uttarakhand documentation
  set records its five interaction terms; the Assam set does not mention them at all.
- The **HSI formula** in the docs (`RH_mean × GHI_daily`) is not the coded formula
  (`RH_mean × fraction of events within 3 K of dew point`). Since HSI is presented as
  Assam's key climate discriminator, this matters.
- **`CCI` is not a "Cloud Cover Index."** It is `CCI_true`, the longest consecutive run of
  cloudy days, in days.

### 3.11 LOW — smaller drift

| Item | Docs | Code |
|---|---|---|
| CSI clip | `clip(0, 1.5)` | `clip(0, 1.2)` in both `02` and `04` |
| Cloudy-day threshold | `kt < 0.4` | `KT_CLOUDY_THRESHOLD = 0.35` |
| Imputation, short gaps | "linear interpolation ≤ 3 consecutive events" | `interpolate(limit=1)` — one step |
| Imputation, long gaps | "point-seasonal mean, then point-event fallback" | `(point, month, event)` mean; no third stage |
| Out-of-bounds values | "flagged but not deleted" | set to `np.nan` (`df.loc[out_of_bounds, col] = np.nan`) |
| Site-year deletion | not mentioned | site-years with >5% missing are **dropped** |
| Outlier detection | IsolationForest only | 3σ per (point, month, event) on `T_amb`/`GHI`/`W_spd` **first**, then IsolationForest (`contamination=0.01`) |
| IST conversion | "No IST conversion exists in the pipeline" (`10_TEMPORAL_PROCESSING.md`) | `04` creates `time_ist` and validates noon lands in 10:00–13:00 IST; `02b` buckets days in `Asia/Kolkata` |
| Tm fitness | symmetric σ = 4 K | asymmetric: σ = 4.0 below target, σ = 1.5 above |
| AHP | "`AHP_PAIRWISE_MATRIX = None`" | no such variable; `AHP_PRIOR_BASE` dict is used directly |
| `T_mains` offset | `Ta_mean − 2.0` in `04b` | `Ta_mean − 6.0` in `04b`; `−2.0` appears in `10` and `05b` (§8) |
| Per-cluster Tm cap | "no per-cluster capping in Assam" | `07b_charging_feasibility.py` adds `Tm_target_C_regime_capped`; `07` prefers it when present |

---

## 4. Phase 1 — Data collection

**Scripts:** `00a_build_population_grid.py`, `00b_build_suntimes.py`,
`01_download_era5_assam.py`, `01b_download_nasapower.py`, `00_unzip_accum.py`

### 4.1 Population-weighted sampling grid (`00a`)

- **Boundary:** GADM v4.1 India admin-1 GeoJSON, filtered `NAME_1 == "Assam"`, first
  geometry.
- **Population:** WorldPop unconstrained global mosaic, India, UN-adjusted, 100 m, 2020
  (`ind_ppp_2020_UNadj.tif`, ~1.5 GB — note this file is what triggered the GitHub size
  limits; it is gitignored).
- **Method:** clip raster to boundary (`rio_mask`, `nodata=0`), zero negative sentinels,
  bin pixels into 0.25° cells **anchored to ERA5's own grid origin** (`ERA5_ORIGIN_LAT=90.0`,
  `ERA5_ORIGIN_LON=-180.0`) via per-raster-row `np.bincount`, rank cells by population
  descending, keep the minimal prefix reaching `COVERAGE_TARGET = 0.875`.
- **Output:** `population_grid_points.csv` — `point_id, lat, lon, population, weight`, with
  `weight` renormalised over the selected subset only.
- **IDs:** `ASP_0001` upward, assigned sequentially *after* selection
  (`f"ASP_{i+1:04d}"`), so IDs are contiguous by construction. The docs' claim of "gaps due
  to boundary rejection" is not supported by the code — nothing rejects an ID mid-sequence.
  The 128-vs-129 confusion in the docs is more likely an artifact of
  `check_points_in_assam.py` and `verify_grid_points.py` both hardcoding "129" in printed
  strings.
- **Elevation:** none attached. Rajasthan has `00c_attach_elevation.py`; Assam does not.
  `02` uses `DEFAULT_ALT_M = 100` for all points.

Grid alignment is the quiet strength here: because sampling cells coincide with ERA5 grid
nodes, the later nearest-neighbour lookup introduces no population-to-cell misalignment. Two
sampling points falling in the same ERA5 cell would receive identical readings — expected
and harmless under this design.

**Honest framing for the thesis:** the 128 points are *population-representative*, not a
uniform state-wide survey. Sparsely populated hill terrain (parts of Karbi Anglong, Dima
Hasao) is underrepresented relative to its land area. Say this beside any spatial map.

### 4.2 Sun-event time table (`00b`)

`pvlib.location.Location.get_sun_rise_set_transit(dates, method="spa")` — Reda & Andreas
(2004), method **explicitly pinned**. Altitude is passed as `0` here (not 100 m), which is
immaterial for sun times.

Date range `2016-01-01` … `2025-12-31` inclusive = 3,653 days (10 × 365 + leap days 2016,
2020, 2024). Expected rows = 128 × 3,653 × 3 = **1,402,752**. The script is idempotent: it
skips if `suntimes.csv` already covers all current `point_id`s unless `--force` is passed.

### 4.3 ERA5 download (`01`)

- **Product:** `reanalysis-era5-single-levels`, hourly, `data_format: netcdf`,
  `download_format: unarchived`.
- **Area:** the envelope of all population points padded 0.5°, as a **single bounding box**.
- **Split:** `INSTANT_VARS` and `ACCUM_VARS` are requested separately, so 10 years × 12
  months × 2 types = **240 calls**.
- **Hour selection:** the clever part. `compute_hour_windows()` reads `suntimes.csv`, takes
  the set of UTC hours at which each event actually occurs across all points and dates, and
  passes it to `circular_hour_window()`, which finds the largest *unobserved* circular gap,
  keeps the complementary arc, pads by `HOUR_MARGIN = 1`, and wraps modulo 24. This handles
  events near the UTC midnight boundary correctly. Accum hours additionally include each
  instant hour's predecessor (`(h-1) % 24`) — a hedge kept from when the pipeline still
  intended to difference accumulations.
- **Idempotency:** `StatusTracker` writes `download_status_points.csv` keyed on
  `(year, month, var_type)`; existing files > 50 kB are skipped; 3 retries with 30 s waits.

### 4.4 NASA POWER download (`01b`)

Hourly point API, `community=RE`, `time-standard=UTC`, one JSON per (point, year) →
128 × 10 = **1,280 files**. Sentinel `-999` is mapped to `NaN` at read time. Status tracking
via `download_status_power.csv`; 1 s courtesy sleep between requests. No API key required.

This full hourly cache is what makes Phase 2's Tier-2 layer free: `02` reads only 3 of ~8,760
hours per point-year, and `02b` later re-reads the same files in full at zero additional
download cost. That is a genuinely good design decision and worth a sentence in the paper.

### 4.5 CDS zip repair (`00_unzip_accum.py`)

CDS API v2 sometimes returns a ZIP even when `unarchived` is requested. The script sniffs
magic bytes (`PK` = zip, `CDF`/`\x89HDF` = NetCDF), extracts the inner `.nc`, and replaces
the file in place. Safe to re-run. Run it after `01` and before `02`.

### 4.6 Standalone boundary QA

- `check_points_in_assam.py` — downloads a **different** boundary source
  (`udit-001/india-maps-data`, not the GADM used by `00a`) and runs point-in-polygon on all
  grid points, writing `assam_grid_points.csv` / `outside_assam_grid_points.csv`. Useful as
  an independent check precisely *because* it uses a different polygon.
- `verify_grid_points.py` — prints the grid with an Assam bounding-box check
  (24.1–28.2 N, 89.6–96.1 E). **Contains a hardcoded absolute path**
  (`m:/Final_year_pro/PCM-Selection-ML-model/...`) and cannot run in this checkout without
  editing.

---

## 5. Phase 2 — Preprocessing and cross-source validation

### 5.1 `02_combine_assam.py`

Reads all ERA5 NetCDFs into memory, extracts per-point series by **independent 1-D `argmin`
on the lat and lon axes** (correct nearest-neighbour for a regular rectilinear grid; no
interpolation), applies unit conversions and solar geometry, matches both sources to each sun
event, and streams the result to CSV point-by-point.

**Unit conversions (exact, from `apply_unit_conversions`)**

| ERA5 field | Operation | Output column | Unit |
|---|---|---|---|
| `t2m` | `− 273.15` | `T_amb` | °C |
| `d2m` | `− 273.15` | `T_dew` | °C |
| `T_amb`, `T_dew` | Magnus (a=17.625, b=243.04), clipped 0–100 | `RHum` | % |
| `u10`, `v10` | `√(u²+v²)`, `(deg(atan2(u,v))+360) % 360` | `W_spd`, `W_dir` | m/s, ° |
| `sp` | `/ 100` | `P_atm` | hPa |
| `tcc` | pass-through | `cloud_cover` | 0–1 |
| `ssrd` | `deaccumulate() / 3600`, `clip(0)` | `GHI` | W/m² |
| `ssrdc` | `deaccumulate() / 3600`, `clip(0)` | `GHI_clearsky_era5` | W/m² |
| `msdwswrf`\|`fdir`\|`msdrswrf` | `clip(0)` — **no `/3600`** | `avg_sdirswrf` → `DNI` | W/m² |
| `strd` | `deaccumulate() / 3600`, `clip(0)` | `LW_down` | W/m² |
| `tp` | `deaccumulate() × 1000`, `clip(0)` | `precipitation` | mm |

Post-conversion range guards: `GHI < 0 → 0`, `GHI > 1400 → NaN`, `T_amb` outside
[−10, 60] °C → `NaN`, `RHum` clipped [0, 100].

**The deaccumulation decision.** `deaccumulate()` is a pass-through:

```python
def deaccumulate(s):
    return pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
```

No differencing. This is the *fixed* behaviour inherited from the Rajasthan audit, where a
naive `diff()` against the previous hour produced near-zero GHI (noon Pearson r ≈ 0.01
against NASA POWER — physically implausible). The empirical finding was that this specific
CDS request configuration returns each hour as its own ~1-hour accumulation, not a running
total. Assam was built after that fix and never carried the bug.

Frame this in the paper as **a pipeline-specific empirical finding**, not as a general claim
about the CDS API. Cite Hersbach et al. (2020) for ERA5 itself, not for this.

**The `avg_sdirswrf` unit caveat.** The column matcher accepts three ERA5 fields with two
different unit conventions — `fdir` is accumulated (would need `/3600`), `msdwswrf` and
`msdrswrf` are mean-rate (already W/m²) — and applies `clip(0)` uniformly with no division.
`01` requests `mean_surface_direct_short_wave_radiation_flux`, which maps to `msdwswrf`, so
the no-conversion branch is almost certainly right in practice. It has not been verified by
opening a `.nc` file and inspecting `ds.data_vars`; do that before describing DNI as
unit-validated. If the matched field were ever `fdir`, DNI would be overestimated 3600×.

**Cross-source matching.** `nearest_row()` uses
`index.get_indexer([target], method="nearest")` and rejects any match more than
`MAX_MATCH_HOURS = 3` from the true sun-event instant, applied independently to ERA5 and
POWER. The actual matched timestamp is **not persisted** — only the requested `time_utc` is
written — so match quality cannot be audited after the fact. Inherited from the Rajasthan
design; worth fixing.

**Season mapping (`SEASON_MAP`)**

| Months | Season | Code |
|---|---|---|
| Dec, Jan, Feb | Winter | 1 |
| Mar, Apr, May | Pre-Monsoon | 2 |
| Jun, Jul, Aug, Sep | Monsoon | 3 |
| Oct, Nov | Post-Monsoon | 4 |

A **4-month monsoon** (Jun–Sep), which is the IMD convention for Northeast India and the
correct choice for Assam. It is used consistently in `02`, `02b`, `04b` (`monsoon_index`
sums months 6–9), `03_plots_raw.py`, `05b` and `05_plot_assam.py` — Assam does **not** have
the Jun–Aug / Jun–Sep inconsistency that Rajasthan had between `02` and `02b`. This is a
genuine, defensible improvement and is safe to claim.

**Output columns.** `point_id, lat, lon, population, weight, date, event, time_utc,
grid_lat, grid_lon`, then `era5_*` for each of `ERA5_OUTPUT_VARS` and `power_*` for each of
`POWER_VARS`, then `month, DOY, year, season, season_code`. Row count ≈ points × 3,653 × 3.

`ETR` (extraterrestrial radiation) is computed in `compute_solar` but is not in
`ERA5_OUTPUT_VARS`, so it is discarded.

### 5.2 `02b_build_daily_aggregates_assam.py`

Re-reads the **full** hourly POWER cache and builds the daily integrals the sun-event merge
discarded. Key decisions:

- Hourly index is converted to **`Asia/Kolkata`** before day bucketing, so "day" means local
  civil day, not UTC day. (This directly contradicts
  `docs/assam/10_TEMPORAL_PROCESSING.md`'s claim that no IST conversion exists.)
- `MIN_HOURS_PER_DAY = 20` — days with more than 4 missing hours are **dropped**, not
  averaged over fewer hours. Good practice, and worth stating.
- `KT_CLOUDY_THRESHOLD = 0.35`.

**Daily outputs** (`daily_aggregates_assam.csv`, one row per point-day):
`GHI_daily_kWh` (Σ hourly all-sky ÷ 1000), `GHIcs_daily_kWh` (Σ hourly clear-sky ÷ 1000),
`kt_daily` (ratio, clipped [0, 1.2], NaN when clear-sky ≤ 0.05), `Ta_mean_true`,
`Ta_max_true`, `Ta_min_true`, `DTR_true`, `RH_mean_true`, `wind_mean_true`.

**Per-point Tier-2 outputs** (`tier2_signature_assam.csv`):

| Column | Definition |
|---|---|
| `n_days_used` | count of days meeting the 20-hour rule |
| `GHI_daily_kWh_mean` | mean daily GHI integral |
| `kt_daily_mean`, `kt_daily_std` | mean / std of daily clearness index |
| `SAI_true` | `Σ GHI_daily / Σ GHIcs_daily` — a true solar-availability ratio |
| `cloudy_frac_true` | fraction of days with `kt_daily < 0.35` |
| `CCI_true` | **longest consecutive run of cloudy days, in days** (integer) |
| `DTR_true_mean` | mean daily `Ta_max − Ta_min` |
| `Ta_mean_true`, `Ta_p95_true`, `Ta_p05_true` | mean and quantiles of daily-mean temperature |
| `HDD18_true`, `CDD24_true` | Σ max(0, 18 − Ta_daily_mean), Σ max(0, Ta_daily_mean − 24) |
| `RH_mean_true`, `wind_mean_true` | means of daily means |
| `seasonality_true` | std ÷ mean of the 12 monthly-mean daily GHI values |

`CCI_true` deserves emphasis: it is a **run-length**, not a cloud fraction. In the write-up
call it "longest consecutive cloudy-day run" and give its units in days. Describing it as a
"Cloud Cover Index" (as `docs/assam/02_DATA_SOURCES_AND_VARIABLES.md` does) is wrong and will
confuse a reviewer looking at a value of, say, 14 on a 0–1 axis.

### 5.3 `03b_agreement_analysis_assam.py`

Compares four ERA5/POWER pairs — GHI (daytime only, `era5_SZA < 90`), `T_amb`, `RHum`,
`W_spd` — stratified by season, computing MBE, RMSE and Pearson r for each
(season, variable). Writes `era5_power_agreement_assam.csv`.

Decision rule:

```python
bias_percentage = abs(ghi_mbe_mean / ghi_true_mean) * 100
decision = "QUANTILE_MAP" if bias_percentage > 10 else "BACKBONE"
```

written to `bias_decision_assam.txt`; empirical quantile maps are dumped to
`quantile_maps_assam.joblib` only in the QUANTILE_MAP branch.

`docs/assam/14_ERA5_POWER_VALIDATION.md` reports the outcome as **MBE = 1.1% → BACKBONE**
*(reported, unverified — the output CSV is not in this checkout)*. A 1.1% GHI bias against
an independent satellite product is a strong result and is worth reporting **on its own
terms**, as evidence that the deaccumulation handling is correct.

But see §3.2: nothing downstream reads this decision, and `04` bias-corrects regardless.

### 5.4 `03_plots_raw.py` — pre-cleaning QA

Six read-only diagnostics on `climate_assam_points.csv`, with explicit stop criteria:

| Plot | Checks | Stop criterion |
|---|---|---|
| A — point map | sampling design, population weighting | — |
| B — event profile | **timezone sanity**: GHI/T_amb must peak at the `noon` event | if noon is not the peak, timezone bug |
| C — ERA5 vs POWER scatter | 5 variable pairs incl. `GHI_clearsky` vs `CLRSKY_SFC_SW_DWN`; writes `C_era5_vs_power_stats.csv` | GHI MBE < 20 W/m² |
| D — missing-data heatmap | per point × variable null rates | — |
| E — seasonal boxplots | monsoon GHI suppression, winter temperature minimum | — |
| F — multi-year trend | year-by-year noon means | no step change in any single year |

This is the best-designed QA artifact in the Assam pipeline — the stop criteria are concrete
and the checks are the right ones. **Caveat:** its docstring and the D-plot title hardcode
"117 points", a stale number inherited during adaptation. Everything else in the repository
says 128.

---

## 6. Phase 2.5 — Quality control (`04_preprocess_assam.py`)

Nine numbered steps. This script does considerably more than the documentation describes.

**[1] Physical bounds.** `BOUNDS` (Table 9 of the plan doc):

| Variable | Lower | Upper |
|---|---|---|
| `era5_GHI` | 0 | 1400 W/m² |
| `era5_T_amb` | −30 | 55 °C |
| `era5_RHum` | 0 | 100 % |
| `era5_T_dew` | −30 | 40 °C |
| `era5_W_spd` | 0 | 50 m/s |
| `era5_P_atm` | 850 | 1060 hPa |
| `era5_cloud_cover` | 0 | 1 |
| `era5_precipitation` | 0 | 200 mm |

Out-of-bounds values are **set to `NaN`**, not merely flagged. The docs describe this as
"flagged but not deleted"; that phrasing belongs to the *outlier* step, not this one.

**[2] Timezone.** `time_ist = time_utc.tz_convert("Asia/Kolkata")`, followed by a real
assertion: the mean IST hour of the `noon` event must land in [10, 13], else `[FAIL]` is
logged. A good check, and direct evidence that the "no IST conversion" claim in the docs is
wrong.

**[3] Humidity.** RH > 100 clipped to 100.

**[4] Night masking.** `era5_GHI` and `era5_GHI_clearsky` forced to 0 wherever
`era5_SZA ≥ 90°`.

**[5] Missing values.** Three actions, in order:
1. `interpolate(method="linear", limit=1)` within each point — bridges single-step gaps only.
2. Fill remaining with the `(point_id, month, event)` group mean — a climatological
   same-month-same-event value.
3. **Drop entire site-years** where any imputed column still exceeds 5% missing.

Step 3 is a deletion the documentation does not mention at all. It can silently reduce the
per-point record length, which propagates into every downstream aggregate. Log and report
how many site-years it removes.

**[6] Outlier detection — two stages, both flag-only.**
1. Per `(point_id, month, event)` **3σ** test on `era5_T_amb`, `era5_GHI`, `era5_W_spd`.
2. **IsolationForest** (`contamination=0.01`, `random_state=42`) on
   `[T_amb, GHI, RHum, W_spd]` with `fillna(0)`.

Both set `is_outlier = 1`. Nothing is deleted. Note the `fillna(0)` before fitting: a missing
value becomes a literal zero in the feature space, which for GHI is indistinguishable from
night and may bias what the forest considers anomalous.

The multivariate approach is a genuine improvement over Rajasthan's univariate Hampel filter,
which had to be patched to exclude GHI/CSI after it winsorised real cloud-driven variability.
That comparison is fair and worth making — but describe Assam's method as **"3σ screening
followed by IsolationForest"**, not IsolationForest alone.

**[7] Solar bias correction.** Unconditional per-(point, season) empirical quantile mapping
of `era5_GHI` onto `power_ALLSKY_SFC_SW_DWN`, overwriting `era5_GHI` in place and keeping
`era5_GHI_corrected` alongside. Pre- and post-correction MBE are logged. **See §3.2** — this
is the step the documentation claims is bypassed.

**[8] Clear-sky index.** `era5_CSI = GHI / GHI_clearsky` where `GHI_clearsky > 10`, clipped
[0, 1.2], else 0. Sanity gate: median kt over positive values should fall in [0.55, 0.75].

**[9] Storage.** One `{point_id}.parquet` per site under `preprocessed/parquet/`, plus a
round-trip row-count test on the first point. Report at `preprocessed/qc_report.txt`.

`CSI = 0` is ambiguous by construction — it means either "genuinely no solar radiation" or
"ratio suppressed below the 10 W/m² clear-sky threshold". Not distinguishable from the output
alone; state it.

---

## 7. Phase 3 — Climate signature (`04b_climate_signature.py`)

### 7.1 Inputs and outputs

**In:** `preprocessed/parquet/*.parquet`, `daily_aggregates_assam.csv`,
`tier2_signature_assam.csv`, `population_grid_points.csv`.
**Out:** `climate_signatures_raw.csv` (physical units), `climate_signatures_matrix.csv`
(standardised, PCA-reduced), `pca_loadings.csv`, `preprocessed/climate_signature_report.txt`.

### 7.2 The signature as actually computed — 19 indices

| # | Index | Source | Exact definition in code |
|---|---|---|---|
| 1 | `Ta_mean` | parquet | mean of `era5_T_amb` over **noon events only** |
| 2 | `Ta_p95` | parquet | 95th percentile, noon events only |
| 3 | `Ta_p05` | parquet | 5th percentile, noon events only |
| 4 | `GHI_mean` | parquet | mean of `era5_GHI` over noon events with GHI > 0 |
| 5 | `HSI` | parquet | `RH_mean × mean((Ta − Td) < 3 K)` across all events |
| 6 | `monsoon_index` | parquet | Σ ERA5 precip in months 6–9 ÷ Σ annual ERA5 precip |
| 7 | `elev_proxy` | parquet | mean `era5_P_atm` ÷ 1013.25 |
| 8 | `DTR` | Tier 2 | `DTR_true_mean` |
| 9 | `GHI_daily_kWh` | Tier 2 | `GHI_daily_kWh_mean` |
| 10 | `kt_mean` | Tier 2 | `kt_daily_mean` |
| 11 | `kt_std` | Tier 2 | `kt_daily_std` |
| 12 | `SAI` | Tier 2 | `SAI_true` |
| 13 | `CCI` | Tier 2 | `CCI_true` (longest cloudy run, days) |
| 14 | `cloudy_frac` | Tier 2 | `cloudy_frac_true` |
| 15 | `HDD18` | Tier 2 | `HDD18_true` |
| 16 | `CDD24` | Tier 2 | `CDD24_true` |
| 17 | `RH_mean` | Tier 2 | `RH_mean_true` |
| 18 | `wind_mean` | Tier 2 | `wind_mean_true` |
| 19 | `seasonality` | Tier 2 | `seasonality_true` |

Three of these carry caveats worth a sentence each in the methodology:

- **`Ta_mean`, `Ta_p95`, `Ta_p05` are noon-event statistics**, not annual means. They are
  systematically warmer than a true annual mean. Note that `Ta_mean_true` — a genuine
  daily-mean-derived value — *is* available in `tier2_signature_assam.csv` and is simply not
  used for these three fields. Whether that is intentional should be decided and documented.
- **`monsoon_index` uses ERA5 3×/day precipitation**, not a true daily integral, because
  `PRECTOTCORR` was not downloaded from POWER. `02b`'s docstring says so and offers the fix
  (add the parameter, re-run `01b` only — no CDS queue time).
- **`HSI` is a dew-point-proximity index**, not the RH × irradiance product the docs
  describe.

### 7.3 Derived PCM targets

```python
T_DELIVERY  = 50.0      # °C, Indian domestic hot-water target
DT_APPROACH = 6.0       # K, heat-exchanger approach
TM_TARGET   = 44.0      # °C — uniform across all Assam sites
M_DRAW_KG   = 100.0
CP_WATER    = 4186.0

sig["T_mains_est"]    = (sig["Ta_mean"] - 6.0).clip(lower=5.0)
sig["L_required_kWh"] = M_DRAW_KG * CP_WATER * (T_DELIVERY - sig["T_mains_est"]) / 3_600_000
```

`Tm_target = 44 °C` is uniform for all 128 sites. Assam's cluster mean temperatures span only
26.3–28.2 °C, so a worst-month capping rule (as used for Rajasthan) would not produce
materially different values between regimes. Document that reasoning explicitly — otherwise
it reads as if the target was simply fixed by fiat.

The uniform target has a direct methodological consequence that Phase 6 handles correctly:
raw latent heat carries **zero** cluster-specific information, so `08_mcdm_ranking.py` ranks
on `latent_heat_margin_ratio = L / L_required` instead (§9.2).

Note `T_MAINS_DEFAULT = 18.0` is defined and used for a one-off `Q_NIGHT_KWH` print, then
immediately superseded by the per-site `Ta_mean − 6.0` formula.

### 7.4 The five interaction terms (§6.4)

Absent from all Assam documentation; present in the code.

| Term | Formula | Intent |
|---|---|---|
| `ix_GHI_x_kt_std` | `GHI_mean × kt_std` | charging energy weighted by unreliability |
| `ix_DTR_x_cloudy` | `DTR × cloudy_frac` | cycling stress under intermittent charging |
| `ix_RH_x_dT_store` | `RH_mean × (Ta_mean − Tm_target)` | condensation risk at the store surface |
| `ix_wind_x_dT_soil` | **`wind_mean × Ta_mean`** | wind-cooling load proxy |
| `ix_CCI_x_1mSAI` | `CCI × (1 − SAI)` | combined autonomy requirement |

`ix_wind_x_dT_soil` needs care. It is assigned twice:

```python
sig["ix_wind_x_dT_soil"] = sig["wind_mean"] * (sig["Ta_mean"] - sig["Ta_mean"])  # = 0
sig["ix_wind_x_dT_soil"] = sig["wind_mean"] * sig["Ta_mean"]                     # overwrites
```

The first line is the literal `wind × (Ta − Tsoil)` design, which collapses to exactly zero
under the `Tsoil ≈ Ta_mean` approximation. The second replaces it with `wind × Ta_mean`. The
effective term is the second, and **its name no longer describes it**. Rename it
(`ix_wind_x_Ta`) before publication, and drop the dead first line.

### 7.5 Tsoil approximation

Soil temperature was not downloaded. `Tsoil_mean ≈ Ta_mean` (standard shallow-soil fallback),
stated in the script docstring and user-approved. Its only consumer was
`ix_wind_x_dT_soil`, which is now `wind × Ta_mean` anyway — so in the shipped code the
approximation has **no numerical effect on anything**. That is the accurate statement; the
docs' framing that it "carries an inherited caveat" downstream overstates its reach.

### 7.6 PCA and the clustering matrix

PCA block: `Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy` — standardised, then
`PCA(n_components=0.95, svd_solver="full")` (retain 95% variance; component count is
data-determined, not fixed). Loadings written to `pca_loadings.csv`.

Solar and variability indices are deliberately kept **out** of PCA to preserve the signal
that actually discriminates regimes for PCM selection. That is the right call and is worth
defending explicitly in the paper as a design decision, not an oversight.

**Final clustering matrix = 17 named features + n PCs**, all standardised (zero mean, unit
variance) *after* aggregation — avoiding Plan §5.2 Trap 1 (normalising before aggregation):

`DTR, GHI_daily_kWh, kt_mean, kt_std, SAI, CCI, cloudy_frac, wind_mean, seasonality,
GHI_mean, HSI, monsoon_index` (12) + the 5 interaction terms + `PC1…PCn`.

Excluded: the 7 PCA-block columns, `point_id`, `Tm_target`, `L_required_kWh`, `T_mains_est`.

---

## 8. The `L_required` unit chain — three incompatible definitions

This is the single most confusing thread in the Assam codebase and it needs one authoritative
statement in the thesis. Three scripts compute a quantity called `L_required` on three
different bases.

**Definition A — `04b_climate_signature.py` (kWh/day, whole-system).**

```
L_required_kWh = M_DRAW_KG(100) × CP_WATER(4186) × (50 − T_mains_est) / 3.6e6
T_mains_est    = clip(Ta_mean − 6.0, min=5.0)
```

With Ta_mean ≈ 26.8 °C → T_mains ≈ 20.8 °C → **≈ 3.40 kWh/day**. This is a nightly energy
demand, not a material property. Despite the name, it is not in kJ/kg.

**Definition B — `07_feasibility_filter.py` (kJ/kg, per-PCM-mass).**

```python
PCM_MASS_KG = 50.0
profiles["L_required_kJ_per_kg"] = profiles["L_required_kWh_mean"] * 3600.0 / PCM_MASS_KG
```

3.40 kWh/day × 3600 ÷ 50 kg ≈ **245 kJ/kg**, consistent with the 232–249 kJ/kg range reported
in `docs/assam/07_PHASE_5_AUDIT.md`. The `κ = 0.7` floor therefore sits at ~162–174 kJ/kg,
which a large fraction of the database clears.

**Definition C — `05b_level_b_seasonal_assam.py` (kJ/kg, different basis again).**

```python
DRAW_RATE_KG_PER_S = 60/1000/60          # = 0.001 kg/s  (comment says "60 L/min" — wrong by 1000×)
q_night_kw = DRAW_RATE_KG_PER_S * CP_WATER(4.186 kJ/kgK) * (50 − t_mains)
l_required = q_night_kw * 3600 * 7 / ASSUMED_PCM_MASS_KG(50)
t_mains    = ta_mean − 2.0
```

≈ 0.122 kW × 3600 × 7 ÷ 50 ≈ **62 kJ/kg** — about a quarter of Definition B. A fourth
`T_mains` offset (`−2.0`) also appears in `10_physics_validation.py`.

**Why this matters, and how it relates to `CLAUDE.md` §3.1.** The project-level correction
recorded in `CLAUDE.md` §3.1 (the combined sensible + latent framework, `SHARE_PCM = 0.5`,
implemented in the Rajasthan and Tamil Nadu signature scripts) is **not present in the Assam
code**. Assam reaches a reachable `L_required` by a different route entirely: dividing the
whole-tank nightly demand by an assumed 50 kg PCM mass. The two approaches are not
equivalent, and neither is derivable from the other.

Practically this is fine — 245 kJ/kg is a physically sensible per-kg latent-heat floor and it
does not zero out the candidate pool the way Rajasthan's uncorrected 610–643 kJ/kg did. But
the *justification* differs by state, and a reviewer comparing the four states will notice.

**Recommended actions:**
1. Rename `L_required_kWh` → `Q_night_kWh` in `04b`. It is a demand, not a latent-heat
   requirement, and the current name is what makes the chain confusing.
2. State `PCM_MASS_KG = 50` as an explicit design assumption in the methodology, with a
   sentence on where 50 kg comes from. It is currently a bare literal inside `07`.
3. Reconcile Assam's basis with the `SHARE_PCM` framework used for Rajasthan/Tamil Nadu, or
   document deliberately why Assam differs.
4. Fix `05b`'s formula and its 1000× comment, or delete the script (§3.4).
5. Settle on one `T_mains` correlation across `04b` (−6.0), `10` (−2.0) and `05b` (−2.0).
   None of the offsets has a cited source in any of the four states; a published correlation
   (ASHRAE, CIBSE, or an India-specific mains-temperature study) would close a real gap.

---

## 9. Phases 4–8 — algorithms as coded

### 9.1 Phase 4 Level A — clustering (`05_cluster_assam.py`)

| Parameter | Value |
|---|---|
| Algorithm | `GaussianMixture(covariance_type="full")` |
| `K_CANDIDATES` | 2 … 10 |
| `K_FINAL` | 4 |
| Silhouette accept band | `SILHOUETTE_LO = 0.15`, `SILHOUETTE_HI = 0.45` |
| `N_BOOTSTRAP` | 500 |
| `RANDOM_STATE` | 42 |
| `n_init` | 5 (sweep), 10 (reference/final), 3 (bootstrap) |
| Comparison | `KMeans(n_init=10)` across the same k range |
| Metrics | BIC, silhouette, Davies-Bouldin, Calinski-Harabasz |

**Full covariance is the right choice and the docstring justifies it well:** Assam's indices
are genuinely correlated (`monsoon_index` with `RH_mean`, GHI with CCI), and diagonal
covariance would split elongated clusters. Soft membership also matters at the Brahmaputra
valley fringe where two regimes genuinely overlap.

**Bootstrap ARI** is computed by resampling with replacement, refitting a GMM on the
resample, and comparing against the labels the *full-data reference model* assigns to the
same indices — `adjusted_rand_score(labels_ref[idx], labs_b_boot)`. `stable = ARI_mean ≥ 0.75`.

**Reproducibility artifacts.** `gmm_model_assam.joblib` is a genuine fitted model.
`scaler_assam.joblib` is **not** — the matrix arriving from `04b` is already standardised, and
the code fits a `StandardScaler` on already-standardised data purely so downstream scripts can
call `.transform()` uniformly. The code comments say so; the docs present it as a fitted
preprocessing scaler. Describe it as an identity placeholder.

**No canonical relabeling** (§3.7). **No external validation** — `05` prints Köppen-Geiger /
NBC-ECBC comparison as an explicit *manual* step in its closing banner.

**Cluster profiles** are population-weighted means with `_mean`/`_std` suffixes, computed on
`climate_signatures_raw.csv` (physical units) rather than the standardised matrix — the right
choice for reporting. Population weighting is applied at sample selection and at profile
reporting, but deliberately **not** a third time inside the GMM fit.

### 9.2 Phase 5 — PCM database and feasibility filter

`06_build_pcm_database.py` — see §3.1 for the two blocking defects. When it works, it
produces:

| Derived column | Formula |
|---|---|
| `TC_W_mK` | `(TC_liquid + TC_solid) / 2` |
| `supercooling_K` | `Tm_melting − Tm_freezing` |
| `rho_H_MJ_m3` | `density_solid` (fallback `density_liquid`) `× latent_heat / 1000` |
| `Cp_avg_kJ_kgK` | mean of liquid/solid Cp with mutual fallback |
| `cycles_confidence` | `log1p(cycles_tested) / log1p(max cycles)`, NaN if untested |
| `in_absolute_band` | `Tm ∈ [42, 70]` |
| `corrosion_class` | `"check_manually"` if `pcm_type` contains "Inorganic", else `"low_organic"` |

`07b_charging_feasibility.py` (optional, run before `07`) derives a poor-day solar
reliability proxy and caps the target:

```python
REFERENCE_GOOD_DAY_TEMP_C = 65.0   # vs Tamil Nadu's 70 — reflects Assam's higher cloud fraction
POOR_DAY_Z = 1.28                  # ≈ 5th percentile under a normal approximation
poor_day_kt  = clip(kt_mean − 1.28 × kt_std, min=0.05)
reliability  = clip(poor_day_kt / kt_mean, 0, 1)
achievable_T = 42 + reliability × (65 − 42)
Tm_target_C_regime_capped = min(Tm_target_C, achievable_T)
```

`07_feasibility_filter.py` — seven constraints:

| # | Constraint | Parameter |
|---|---|---|
| 1 | Melting window | `Tm ∈ [Tm_target − 6, Tm_target + 8]` = [38, 52] °C at 44 °C |
| 2 | Absolute band | `Tm ∈ [42, 70]` °C |
| 3 | Latent-heat floor | `L ≥ 0.7 × L_required` |
| 4 | Cycling | `≥ 300` cycles if known; retained with `"not_reported"` flag if unknown |
| 5 | Supercooling | `|supercooling_K| ≤ 8 K` if known; retained if unknown |
| 6 | Corrosion | exclude `check_manually` when `cluster_HSI > global p75` — **inert, §3.1** |
| 7 | Safety | keyword veto: "highly flammable", "extremely flammable", "toxic" |

Auto-relaxation: if fewer than `MIN_SURVIVORS = 5` pass, the melting window widens by
`RELAX_STEP_K = 2.0` per step, up to `MAX_RELAX_STEPS = 4` (±8 K total). The relaxation
applied is recorded per row in `window_relax_applied`. Status is printed as `LOW` (<5),
`OK` (5–25), or `HIGH` (>25).

Note constraints 1 and 2 interact: at `Tm_target = 44` the window is [38, 52] but the
absolute band floor is 42, so the effective lower bound is **42 °C** and the first two
relaxation steps do nothing on the low side.

The κ-relaxation is an ad-hoc pass, not a settled policy. Decide it permanently and document
it — this is the same open item flagged in `CLAUDE.md` §3.1's Option-B guidance, which
recommends a light prescreen gate plus latent heat as one MCDM criterion among many, rather
than a hard binary gate.

### 9.3 Phase 6 — MCDM (`08_mcdm_ranking.py`)

**Criteria** (all benefit-direction after transformation):

| Criterion | Definition | AHP prior |
|---|---|---|
| `f_Tm` | asymmetric Gaussian fitness (below) | 0.24 / 0.80 |
| `latent_heat_margin_ratio` | `L / L_required` | 0.20 / 0.80 |
| `rho_H_MJ_m3` | volumetric latent heat | 0.12 / 0.80 |
| `TC_W_mK` | thermal conductivity | 0.13 / 0.80 |
| `cycles_confidence` | log-scaled cycling stability | 0.11 / 0.80 |

**Asymmetric Tm fitness** — the docs describe a symmetric σ = 4 K Gaussian; the code is:

```python
def gaussian_tm_fitness(tm, tm_target, sigma_lower=4.0, sigma_upper=1.5):
    sigma = np.where(tm <= tm_target, sigma_lower, sigma_upper)
    return np.exp(-((tm - tm_target) ** 2) / (2 * sigma ** 2))
```

Penalising Tm above target ~2.7× more steeply than below. See §3.6 on the circularity this
introduces.

**Weights:** `w_final = 0.5 × entropy + 0.5 × AHP`, renormalised
(`ENTROPY_AHP_LAMBDA = 0.5`). Entropy weights are Shannon entropy of the min-max normalised
decision matrix. No pairwise AHP elicitation was performed — the framework doc's Table 13
priors are used unmodified. State this plainly in the methodology.

**Methods:** TOPSIS (closeness coefficient), GRA (`ζ = 0.5`), PROMETHEE II (V-shape with
`q = 0.10`, `p = 0.30`), VIKOR (`v = 0.5`, lower Q better, with an acceptable-advantage
check against `dq = 1/(n−1)`).

**Consensus:** Borda (primary, drives `consensus_rank`), Copeland pairwise (cross-check,
with a `borda_copeland_agree` flag on the #1), and Kendall's W per cluster.

**Monte Carlo:** `N_MONTE_CARLO_DRAWS = 5000` — matching plan §9.6 exactly, where Rajasthan
used 1,000 as a documented deviation. Per draw: Dirichlet weight perturbation
(`concentration = 30`) plus Gaussian property noise (Tm ±1 K absolute; L ±5%, TC ±10%,
ρH ±8% relative). Outputs top-3 inclusion probability, top-1 retention rate, and mean
Spearman rho against the baseline ranking. `MC_RANDOM_SEED = 42`.

One design note worth disclosing: **the Monte Carlo re-ranks with TOPSIS only**, not with the
full four-method Borda consensus. So the reported inclusion probabilities measure the
stability of the TOPSIS ranking under perturbation, not of the consensus ranking that the
cards actually report. Defensible on cost grounds; say so rather than letting a reviewer find
it.

### 9.4 Phase 7 — physics validation (`10_physics_validation.py`)

Grey-box lumped-enthalpy tank, two coupled nodes (tank water `Tw`, PCM `Tp`/melt fraction),
three phases (pre-melt sensible → isothermal plateau at Tm → post-melt sensible), **backward
Euler** at `dt = 3600 s`, driven for one full real year of the cluster representative point's
daily climate. ODE structure from Barqawi (2025); model class supported by Bony & Citherlet
(2007).

**Constants as coded** (not as documented — §3.6):

| Parameter | Code value |
|---|---|
| Tank water mass `M_W_KG` | 150 kg |
| Coil area `A_C_M2` | **2.0 m²** |
| Water-coil HTC `H_C_WM2K` | 1500 W/m²K |
| Collector efficiency `COLLECTOR_EFF` | **0.40** |
| PCM volume `V_PCM_M3` | 0.035 m³ |
| PCM-water HTC `H_P_WM2K` | 800 W/m²K |
| PCM surface area `A_P_M2` | 3.5 m² |
| Draws | 2/day at 07:00 and 19:00 local, **100 kg each** |
| Delivery target | 50 °C |
| Default PCM density / Cp | 800 kg/m³ / 2000 J/kgK when unreported |
| Benchmark band | 54–84% annual solar fraction (plan Table 16) |
| `MAX_PCMS_PER_CLUSTER` | 20 |

**Representative point selection.** `pick_medoid_point()` returns the point with the highest
`max_membership_prob` — the most *typical* member by GMM posterior, which is not the same as
a geometric medoid. `09_recommendation_cards.py` labels it correctly as
"Medoid point (highest membership confidence)"; the phase docs just say "medoid". Use the
longer phrase.

**Representative year.** `pick_representative_year()` takes the year with the most usable
days; clusters whose representative point has fewer than 300 usable days are skipped.

**Drivers.** Sunrise/sunset are read from `suntimes.csv` and shifted UTC → IST by a hardcoded
`+5:30`, falling back to (05:45, 18:00) if absent. Irradiance is a half-sine over the
daylight window scaled so its integral matches that day's real `GHI_daily_kWh`. Ambient
temperature is a daily sinusoid between the day's real `Ta_min_true`/`Ta_max_true`, peaking
at 14:00 and troughing at 05:00 local. Mains temperature is `Ta_mean − 2.0` (a fourth offset,
§8).

**Solar fraction** = Σ over draws of `DRAW_MASS × Cp × max(0, min(Tw, 50) − T_mains)` ÷
Σ `DRAW_MASS × Cp × max(0, 50 − T_mains)`. There is no auxiliary heater in the model; SF is
the fraction of demand the tank can meet unaided, capped at 1 per draw.

**Agreement metric.** `spearmanr(consensus_rank, annual_solar_fraction)` with the sign flipped
(`rho_agreement = -rho`), because rank 1 is best while a higher solar fraction is better.
Interpretation bands per plan Table 17: > 0.8 strong, 0.4–0.8 partial, < 0.4 weak.
Requires ≥ 3 candidates.

### 9.5 Phase 8 — recommendation cards (`09_recommendation_cards.py`)

One markdown card per cluster containing: points in regime, medoid, climate signature table,
derived targets, Phase 5 screening summary, top-3 table with all four method scores plus MC
inclusion probability, Kendall's W with an interpretation band (≥0.8 strong, ≥0.6 moderate,
else "genuinely ambiguous"), **analytical criterion contributions**, the Phase 7 simulation
table, and the cluster's Spearman rho.

Criterion contributions are computed by min-max normalising the criteria across the cluster's
candidates, multiplying by the stored weights, and expressing each as a percentage of the
row sum — an explainability decomposition without SHAP. Contributions below 0.5% are
suppressed from the printed breakdown.

This satisfies the plan's Table 18 explainability mandate that the Tamil Nadu Phase 8
implementation omitted, and is a genuine Assam contribution worth backporting. **But see
§3.5** — three field-name bugs blank most of the card header, and the cycling contribution
reads a boolean flag instead of the criterion.

---

## 10. Results as recorded

Everything in this section is quoted from `docs/assam/*.md`. **None of it is reproducible
from this checkout** (§0.2), and §3.1/§3.2/§3.3 all bear on whether it can be regenerated.

### 10.1 Cluster structure *(reported, unverified)*

| Cluster | n | Population | Ta_mean | kt_mean | Representative point | Character |
|---|---|---|---|---|---|---|
| 0 | 24 | ~1.70 M | 26.3 °C | 0.696 | ASP_0013 (27.375, 94.875) | Northeast hill/transition |
| 1 | 52 | ~3.25 M | 26.8 °C | 0.758 | ASP_0017 (26.875, 94.125) | Upper Brahmaputra valley |
| 2 | 11 | ~0.93 M | 28.2 °C | 0.789 | ASP_0008 (24.875, 92.875) | Barak valley / southern Assam |
| 3 | 41 | ~5.55 M | 28.2 °C | 0.772 | ASP_0001 (26.125, 91.625) | Western plains / Guwahati belt |

Total 128 points, ≈11.4 M population at the 87.5% coverage target. The medoid coordinates and
kt values come from the header of `05_plot_assam.py`, which transcribes them from
`recommendation_cards_assam.md`.

One inconsistency to resolve before publication: `docs/assam/11_SPATIAL_PROCESSING.md`
assigns Guwahati to **Cluster 1**, while `05_plot_assam.py` assigns the Guwahati belt to
**Cluster 3** — and Cluster 3's representative point (26.125, 91.625) is in fact in Kamrup
district, near Guwahati. The plotting script appears correct. Cluster 1's representative
point (26.875, 94.125) is upper Assam (Sivasagar/Dibrugarh). Fix the geographic labels in
the docs against the actual coordinates.

kt of 0.70–0.79 is physically reasonable for a monsoon-dominated climate with a large diffuse
fraction, and is meaningfully lower than Rajasthan's predominantly clear-sky regime. That is
a real climate signal, not an artifact.

### 10.2 k selection *(reported, unverified)*

| k | BIC | Silhouette | DB | CH | In band |
|---|---|---|---|---|---|
| 2 | −1910.4 | 0.457 | 0.915 | 82.2 | False |
| 3 | −3024.8 | 0.309 | 1.203 | 71.7 | True |
| **4** | **−3322.3** | **0.321** | **1.152** | **62.1** | **True** |
| 5 | −3982.7 | 0.271 | 1.343 | 51.7 | True |
| 6 | −4555.8 | 0.292 | 1.165 | 48.5 | True |
| 7 | −4762.3 | 0.273 | 1.280 | 44.4 | True |
| 8 | −4851.7 | 0.277 | 1.250 | 49.4 | True |
| 9 | **−5138.4** | 0.309 | 1.180 | 49.7 | True |
| 10 | −4578.1 | 0.300 | 1.231 | 46.5 | True |

BIC keeps falling to k = 9. **k = 4 was chosen for interpretability, not by BIC minimum** —
say that directly rather than implying BIC selected it. The supporting argument: k = 2's high
silhouette (0.457) is achieved by a split that collapses all Brahmaputra diversity and lies
outside the accept band; k = 4 improves on k = 3 in silhouette (0.321 vs 0.309) and maps onto
interpretable Assam geography.

### 10.3 Stability *(reported, unverified)*

`k_final = 4`, `n_bootstrap = 500`, **ARI_mean = 0.716**, ARI_std = 0.139,
**`stable = False`** (threshold 0.75).

Report this honestly as written: the k = 4 partition is reasonably stable but **does not**
meet the strong-stability criterion, which is consistent with Assam's genuinely gradual
climate transitions — particularly at the Cluster 0 hill/valley boundary. It sets the
uncertainty correctly; it does not invalidate the result.

K-Means silhouette is ~0.31 flat across k = 2…10 with no elbow; GMM's k = 2 silhouette
(0.457) exceeds the K-Means best, supporting the full-covariance choice.

### 10.4 Feasibility and MCDM *(reported, unverified — and see §3.1)*

Survivors per cluster: **6, 6, 8, 8** (clusters 0–3). `L_required` 232–249 kJ/kg.

Top-3 identical in all four clusters: **RT44HC** (Tm 43 °C, L 250) → **RT45HC** (47 °C, 230)
→ **C22H46 docosane-class paraffin** (44.5 °C, 249).

Kendall's W = 0.807 (clusters 0, 1), 0.845 (clusters 2, 3) — strong four-method concordance.
MC top-3 inclusion: RT44HC 95–96%, RT45HC 62–68%, C22H46 27–39%.

The unanimous #1 is the mathematically correct outcome of a uniform `Tm_target = 44 °C`:
RT44HC sits 1 K from target with the highest latent heat in the pool. `08_mcdm_ranking.py`
detects this case and prints the correct framing itself — the cluster differentiation from
Phase 4 matters for **system sizing and seasonal analysis**, not for PCM identity.

Two caveats now attach to these numbers. First, the survivor counts were attributed in the
docs to corrosion-veto differences that cannot occur (§3.1). Second, the products named do
not all match the current PCM CSV — `RT45HC` appears there with L = 240, not 230, and
`C22H46` appears as `n-Docosane (C22)`. These results came from a different database
snapshot.

### 10.5 Physics validation *(reported, unverified)*

| Cluster | n | Spearman rho | p | Interpretation |
|---|---|---|---|---|
| 0 | 6 | 0.257 | 0.623 | weak |
| 1 | 6 | 0.257 | 0.623 | weak |
| 2 | 8 | 0.286 | 0.493 | weak |
| 3 | 8 | 0.167 | 0.693 | weak |

Mean rho = 0.242. All four clusters weak (< 0.4); no p-value approaches significance.

Solar fractions: RT44HC 82.1% (C1) / 84.8% (C2), C22H46 82.9% / 85.3%, savE OM42 82.6% /
85.1%, RT45HC 51.7% / 52.1%.

**This is a genuine negative result, correctly computed and honestly reported.** It is
publishable as such — plan Table 17 explicitly treats all three outcome bands as reportable
if diagnosed. But the diagnosis in `docs/assam/` should be revised. The docs attribute it to
an undersized PCM database; §3.1 shows the pool is 62 rows, and §3.6 offers a more specific
and better-supported explanation:

1. **Collector-temperature ceiling.** `COLLECTOR_EFF = 0.40` with the `/20.0` divisor caps
   the collector node near Ta + 16 °C ≈ 44 °C. PCMs above ~45 °C cannot complete a melt
   cycle, so RT45HC's 51–52% is a model boundary effect, not a material result. This alone
   accounts for most of the rank scatter.
2. **Ceiling effect at the top.** RT44HC, C22H46 and savE OM42 all land within 3 percentage
   points of each other (82–85%), above the 84% benchmark ceiling — so their relative order
   is decided by simulation noise.
3. **Small n.** Six to eight candidates gives Spearman rho very little power; the p-values
   (0.49–0.69) confirm it.
4. **Degenerate between-cluster differences.** With a uniform `Tm_target` and similar
   `L_required`, the four clusters' MCDM rank vectors are nearly identical, so there is
   little signal for the correlation to find.

Note also that the benchmark band (54–84%) was derived from dry-climate SWH literature and
may not transfer to a humid-monsoon regime. That is an inherited model assumption, not a code
error — but it should be stated when reporting out-of-band results.

---

## 11. Cross-cutting notes

### 11.1 Temporal processing

- **Storage and matching are UTC throughout.** `suntimes.csv`, ERA5 native times, and the
  POWER request (`time-standard=UTC`) are all UTC, and `02` matches on UTC.
- **IST appears in three places**, contrary to `10_TEMPORAL_PROCESSING.md`: `04` creates
  `time_ist` and validates the noon event against it; `02b` buckets days in `Asia/Kolkata`;
  `10` shifts sunrise/sunset by a hardcoded +5:30.
- Assam spans ~89.7–96.0 °E, so eastern points have earlier UTC sunrise. The
  `circular_hour_window()` algorithm in `01` handles the midnight wrap correctly.
- Any figure shown to a general audience needs an explicit UTC → IST conversion at
  presentation time. `05_plot_assam.py`'s diurnal-profile plot labels its axis "Hour of Day
  (UTC)" correctly — but note that a "diurnal profile" built from three sun-event samples per
  day is not a true diurnal curve, and should not be described as one.

### 11.2 Spatial processing

ERA5 single-levels at 0.25° (~28 km), sampling grid aligned to the same origin, strict
nearest-neighbour lookup (two independent 1-D `argmin`s), no interpolation. Assam's GADM
boundary is a single polygon, so no multi-part geometry handling is needed.

**Elevation is fixed at 100 m for all points** (`DEFAULT_ALT_M`). Correct for the Brahmaputra
plains where most of the population lives; it underestimates Karbi Anglong and Dima Hasao
(300–900 m+). This propagates into `P_atm`, the altitude-dependent Ineichen turbidity lookup,
and `elev_proxy`. A documented, accepted approximation — considerably less consequential here
than it would be for Uttarakhand.

### 11.3 Solar geometry

`compute_solar()` builds a `pvlib.location.Location(lat, lon, altitude=100, tz="UTC")` per
point and calls `get_solarposition(times)` and `get_clearsky(times, model="ineichen")`.

- **`get_solarposition` is called without an explicit `method=`**, relying on the installed
  pvlib default (NREL SPA in current releases). `00b` *does* pin `method="spa"`. Pin it in
  `02` too — a one-line change that closes a reproducibility gap.
- **Clear-sky source selection** is the notable upgrade the docs miss entirely. `02` prefers
  ERA5's own `ssrdc` and only falls back to pvlib Ineichen when it is absent, recording which
  was used in a `clearsky_source` audit column:

  ```python
  if "GHI_clearsky_era5" in df.columns and df["GHI_clearsky_era5"].notna().any():
      df["GHI_clearsky"], df["clearsky_source"] = df["GHI_clearsky_era5"], "era5_ssrdc"
  else:
      df["GHI_clearsky"], df["clearsky_source"] = cs["ghi"].values, "pvlib_ineichen"
  ```

  This is the more defensible choice for Assam and it sidesteps the Linke-turbidity concern
  the docs raise (pvlib's default turbidity climatology does not capture Assam's pre-monsoon
  biomass-burning aerosol load or monsoon humidity interaction). Report the `clearsky_source`
  distribution — if it is `era5_ssrdc` throughout, the Ineichen caveat does not apply to the
  shipped results at all.

Assam spans ~24.1–27.8 °N; solar-noon zenith ranges from ~43° at the winter solstice to near
0° at the summer solstice.

### 11.4 Derived solar variables

- **GHI:** `ssrd / 3600`, clipped ≥ 0, > 1400 → NaN. Then night-masked and quantile-corrected
  in `04`.
- **DNI:** two branches. Primary — `avg_sdirswrf.clip(0, 1400)` straight from the ERA5
  direct-radiation field. Fallback — `GHI / cos(SZA)` where `cos_z > 0.05`, which assumes
  zero diffuse and is **not a decomposition model**. Under Assam's optically thick monsoon
  skies the fallback would badly overestimate DNI; it is rarely exercised because the ERA5
  field is present.
- **DHI:** `(GHI − DNI × cos_z).clip(0)` — a closure residual, satisfied by construction, not
  independently measured. Any error in GHI or DNI propagates entirely into DHI. During the
  monsoon, when the diffuse component matters most, DHI should be described as an estimate.
- **CSI:** `GHI / GHI_clearsky` where clear-sky > 10 W/m², clipped **[0, 1.2]** in both `02`
  and `04`.

The honest framing for a paper is: "DNI from ERA5's direct-radiation field where available;
DHI computed as a closure residual" — do not claim decomposition-model provenance. Erbs et
al. (1982) and Perez et al. (1990, DISC) are the references if a real model is ever added.

---

## 12. Plotting and verification suite — audit

Five scripts produce figures. They are useful for eyeballing results, but **`verify_*` is a
misnomer**: several of these compute nothing and one fabricates its data.

| Script | Outputs | Assessment |
|---|---|---|
| `03_plots_raw.py` | `plots/raw/` × 6 | **Best of the set.** Real checks with explicit stop criteria. Stale "117 points" in labels. |
| `05_plot_assam.py` | `plots/{maps,timeseries,statistics,features,solar_resource}/` × 17 | Comprehensive; 5 Folium maps + 12 figures. D-plots partly vestigial (below). |
| `generate_assam_plots.py` | `plots/assam_objective1/` × 9 (of "13") | Plotly + Folium interactive cards. |
| `comparison_plots_assam.py` | `plots/comparison/` × 8 | Cross-step sanity figures. |
| `verify_01..04_*.py` | `plots/verify_*/` × 19 | See defects below. |

**Defects found, by script:**

`verify_01_preprocessing_assam.py`
- Reads `preprocessed/assam_cleaned_physical.csv` — never produced by any script.
- Plot [1/4] looks for `GHI_max`, `Ta_mean_proxy`, `Ta_max_proxy`, `RH_mean_proxy`,
  `Ws_mean_proxy` — **none exist** in `climate_signatures_raw.csv` (the real names are
  `GHI_mean`, `Ta_mean`, `RH_mean`, `wind_mean`). The figure silently renders empty.
- Docstring says "Hampel filter summary" — that is Rajasthan's method, not Assam's.
- Summary card hardcodes `Status: PASS`.

`verify_02_clustering_assam.py`
- Reads `bic["bic"]` lowercase; `05_cluster_assam.py` writes **`BIC`** uppercase. The BIC
  curve is never plotted — only the silhouette twin axis renders.
- The "cluster profiles" bar chart takes `cols[:5]` of the profile frame, which begins with
  `n_points` and `total_population`, then labels the y-axis "Normalized Feature Value".

`verify_03_feasibility_assam.py`
- **The "Filter Stage Candidate Funnel" figure is fabricated.** The intermediate stages are
  literal placeholders, not measured counts:
  ```python
  counts = [len(db), int(len(db)*0.7), int(len(db)*0.5), len(feas)]
  ```
  A figure that presents invented numbers under the heading "Verify" should be fixed or
  deleted before anyone sees it.
- Survivor counts use `feas.groupby("cluster_id").size()`, which counts **every evaluated
  PCM**, not `passes_all` survivors. The chart titled "Feasible PCM Candidates per Cluster"
  therefore reports the full database size per cluster.

`verify_04_ranking_assam.py`
- Summary card hardcodes **"Monte Carlo Iterations : 1,000 runs"**. The pipeline uses 5,000 —
  and "5,000 draws, matching the plan spec, unlike Rajasthan's 1,000" is one of Assam's
  headline claims. This figure contradicts it.
- Hardcodes `Status: PASS`.

`comparison_plots_assam.py`
- Comparison 5 labels its histogram "Feasible survivors (n=len(feas))" using all rows rather
  than `passes_all`.
- Comparison 2 looks for `Ta_mean_proxy` (nonexistent) and falls back to the first column
  containing `"T_"`; its ±25 °C / ±35 °C guide lines are inherited Tamil Nadu heuristics with
  no Assam derivation.

`generate_assam_plots.py`
- `p04_05()` counts all `feas` rows as survivors — same defect as above.
- Advertises 13 plots; `main()` calls 9 (numbers 6, 9 and 12 are absent).

`05_plot_assam.py`
- D2 (rolling-mean) and D1 (lag correlations) look for `GHI_roll*` / `GHI_lag*` columns that
  the Assam parquet never contains — D1 falls back to plain feature correlations, D2 skips.
- D3 plots a 70/15/15 **train/val/test temporal split**. Objective 1 has no supervised model;
  this is vestigial from the forecasting pipeline it was adapted from. Remove it or exclude
  it from the thesis figure set.
- Its `CLUSTER_COLORS` comments disagree with `docs/assam/11_SPATIAL_PROCESSING.md` on which
  cluster is the Guwahati belt (§10.1).

`PLOTS_GUIDE.md` marks all four verification suites **"✅ PASS"**, including "Verify 04 |
high inter-method Spearman correlation (>0.85)". Given the above, those PASS marks are
asserted, not computed. Remove them or replace them with the actual measured criteria.

---

## 13. Reproducibility audit

| Item | Status | Note |
|---|---|---|
| Random seeds | **PASS** | `random_state=42` on GMM, K-Means, bootstrap RNG, IsolationForest, MC |
| GMM persistence | **PASS** | `gmm_model_assam.joblib` |
| Scaler persistence | **PARTIAL** | `scaler_assam.joblib` is an identity placeholder fitted on already-standardised data |
| sklearn version recorded | **PASS** | `sklearn_version` column in every clustering output |
| Monte Carlo draws | **PASS** | 5,000, matching plan §9.6 |
| Time ranges | **PASS** | 2016-01-01…2025-12-31 hardcoded consistently |
| API parameters | **PASS** | CDS variable lists and POWER parameter strings are in version control |
| Geographic determinism | **PASS** | GADM + WorldPop + fixed ERA5-aligned 0.25° grid |
| Output naming | **PASS** | consistent `{artifact}_assam.csv`, except `05b`'s expected `mcdm_full_scores_by_cluster.csv` |
| Logging | **PASS** | download status CSVs, `qc_report.txt`, `climate_signature_report.txt` |
| Canonical cluster relabeling | **FAIL** | claimed in three docs; **not in the code** (§3.7) |
| Signature provenance | **FAIL** | two scripts write the same file with different formulas (§3.3) |
| Phase 5 executability | **FAIL** | `06` raises `KeyError` against every available PCM CSV (§3.1) |
| Level B executability | **FAIL** | two input paths do not exist (§3.4) |
| Pinned dependencies | **FAIL** | no `requirements.txt` / `environment.yml` in `era5-assam/` |
| Solar-position method pin | **FAIL** | `02` relies on the pvlib default; `00b` pins `method="spa"` |
| Full-chain orchestration | **ABSENT** | no `run_all_assam.py` |
| Cross-phase provenance | **PARTIAL** | no `provenance_lib.py`; consistency checked inline in `09` only |
| Dataset version pinning | **PARTIAL** | ERA5 is periodically reprocessed by ECMWF; no per-file download-date manifest |
| Output artifacts in VCS | **ABSENT** | `.gitignore` excludes all of `data/` — nothing is auditable after the fact |

**Fixes, ordered by effort ÷ impact:**

1. `pip freeze > era5-assam/requirements.txt` — zero code change, closes the largest gap.
2. Fix `06_build_pcm_database.py` (§3.1) — Phases 5–8 do not run without it.
3. Delete or rename `fast_generate_raw_signatures.py` (§3.3) — restores signature provenance.
4. Add canonical relabeling by ascending mean latitude to `05_cluster_assam.py`, so the docs
   become true and cluster IDs survive refits.
5. Decide the BACKBONE-vs-quantile-map question in `04` (§3.2) and make code and text agree.
6. Pin `get_solarposition(method="spa")` in `02_combine_assam.py`.
7. Create `run_all_assam.py` in the order of §2.
8. Commit the small result CSVs (cluster profiles, MCDM top-k, Spearman) — they are kilobytes,
   and without them no reported number in `docs/assam/` can be checked.
9. Fix the four `09_recommendation_cards.py` field names (§3.5).
10. Add provenance fingerprinting (mtime + size + row count) at the entry of `07`, `08`, `10`.

---

## 14. Literature mapping

| Component | Implementation | Source | Strength |
|---|---|---|---|
| ERA5 reanalysis backbone | Phases 1–2 | Hersbach et al. (2020), *QJRMS* 146(730) | Strong |
| NASA POWER cross-check | Phases 1–2 | NASA POWER project documentation | Strong |
| Solar position (SPA) | `00b`, `02` | Reda & Andreas (2004), *Solar Energy* 76(5) | Strong — **not in `references.bib`; add** |
| Clear-sky model (Ineichen) | `02` fallback branch | Ineichen & Perez (2002), *Solar Energy* 73(3) | Strong — **not in bib; add.** Note `ssrdc` is now preferred over Ineichen |
| pvlib | throughout | Holmgren, Hansen & Mikofski (2018), *JOSS* 3(29) | Strong |
| Humidity-stress index | `04b` | Thom (1959), *Weatherwise* 12(2) — THI | **Weak as implemented.** The coded HSI is a dew-point-proximity product, not Thom's THI. Either re-derive or drop the citation. |
| Night-discharge sizing | `04b` | Avargani et al. (2021), *J. Energy Storage* | Moderate — but see §8; Assam's basis differs from `CLAUDE.md` §3.1 |
| PCM band 42–70 °C | Phase 5 | Framework doc Table 5; Singh et al. (2025), *Sol. Energy Mater. Sol. Cells* 293 | Strong |
| GMM clustering + k-selection | `05` | Framework doc §7.2 | Moderate — internal statistics only; no external classification |
| Bootstrap ARI stability | `05` | Framework doc §7.3 | Moderate — correctly implemented, honestly reported |
| PCM property database (RT series) | `06` | Martínez et al. (2025), *Heliyon* 11; Singh et al. (2025) | Strong |
| MCDM method family | `08` | **No originating citations in `references.bib`** | **Gap** — add Hwang & Yoon (1981) TOPSIS, Brans & Vincke (1985) PROMETHEE, Opricovic (1998) VIKOR, Deng (1982) GRA |
| Gaussian Tm fitness | `08` | Framework doc §9.2 only | Weak/self-sourced — and the shipped version is **asymmetric**, which §9.2 does not describe |
| Monte Carlo propagation | `08` | Framework doc §9.6 | Moderate — 5,000 draws matches spec |
| Criterion contributions | `09` | Framework doc Table 18 | Implementation-defined; no external citation needed |
| Lumped-enthalpy ODE | `10` | Barqawi (2025), *Muthanna J. Eng. Technol.* 13(3) | Strong — in `Sources/`, equations used directly |
| Model-class justification | `10` | Bony & Citherlet (2007), *Energy and Buildings* 39(9) | Strong |
| Draw profile | `10` | ASHRAE 90.2 §8.9.4 two-peak profile | Partial — shape correct; the coded 2 × 100 kg is not a verbatim reproduction |
| Solar-fraction benchmark 54–84% | `10` | Framework doc Table 16 | Strong as a calibration check; **dry-climate provenance** should be disclosed |
| MICE + RF + PMM imputation | `PCM_data/01_preprocess.py` | none confirmed | **Gap** — cite van Buuren & Groothuis-Oudshoorn (2011) |
| `T_mains` offsets (−2.0 / −6.0) | `04b`, `10`, `05b` | **none — unsourced in code** | **Gap** — needs a published mains-temperature correlation |

**Assam-specific note.** None of the 21 papers in `Sources/` is specific to Northeast India.
The k = 4 physical interpretation rests on geographic domain knowledge plus internal
BIC/silhouette statistics, not on an external Assam climate classification. Say so in one
sentence rather than letting the interpretation read as validated.

The bibliography gaps above are **shared with Rajasthan and Tamil Nadu** — adding them once
to the project `references.bib` covers all four states.

---

## 15. Consolidated implementation issues

### Blocking — Phases 5–8 cannot run from a clean checkout

1. **`06_build_pcm_database.py` raises `KeyError: 'is_rt_line'`** against every PCM CSV in the
   repository. (§3.1)
2. **The corrosion veto is inert** — no PCM has `pcm_type` containing "Inorganic", so the
   central Assam differentiator does not occur. (§3.1)
3. **Two scripts write `climate_signatures_raw.csv`** with mutually incompatible formulas, so
   the clustering input has no determinate provenance. (§3.3)

### High — results are computed, but the write-up misdescribes them

4. **`04_preprocess_assam.py` always quantile-corrects GHI**; the docs claim it bypasses
   correction under BACKBONE. (§3.2)
5. **Phase 4 Level B never executes** — two input paths do not exist. (§3.4)
6. **Recommendation cards render a blank signature table** and `Tm_target = nan`, and the
   cycling contribution reads a boolean flag. (§3.5)
7. **The physics model caps collector temperature near Ta + 16 °C**, which structurally
   penalises Tm > ~45 °C and was then fed back into Phase 6's fitness function. (§3.6)
8. **The Phase 7 docstring contradicts its own constants** on coil area, collector
   efficiency and draw mass — and the docs copy the docstring. (§3.6)
9. **No canonical cluster relabeling**, despite three documents claiming it. (§3.7)
10. **`verify_03`'s filter funnel is fabricated**; `verify_04` reports 1,000 MC draws
    against the pipeline's 5,000; `verify_02` never plots BIC. (§12)

### Medium — documentation accuracy

11. NASA POWER is hourly, includes `CLRSKY_SFC_SW_DWN`, and **excludes precipitation**. (§3.8)
12. ERA5 uses `surface_pressure`, downloads `ssrdc`, and the function is `deaccumulate()`. (§3.9)
13. The signature is **19 indices**, HSI and CCI are misdefined in the docs, and the five
    interaction terms are undocumented. (§3.10, §7)
14. `L_required` has three incompatible definitions across `04b`, `07` and `05b`, and Assam
    does not use the `SHARE_PCM` framework recorded in `CLAUDE.md` §3.1. (§8)
15. `07b_charging_feasibility.py` exists and can cap `Tm_target` per cluster; the docs say
    Assam has no per-cluster capping.
16. The Guwahati belt is assigned to different clusters by the docs and the plotting code.
    (§10.1)
17. Stale hardcoded counts: "117 points" in `03_plots_raw.py`, "129" in
    `check_points_in_assam.py` and `verify_grid_points.py`.

### Low — hygiene

18. `verify_grid_points.py` hardcodes an `m:/` absolute path.
19. No `requirements.txt`; `get_solarposition` unpinned in `02`.
20. No `run_all_assam.py`; no cross-phase provenance checking.
21. `ix_wind_x_dT_soil` is assigned twice and its surviving formula does not match its name.
22. The `docs/assam/` filename prefixes and internal heading numbers disagree in eight files,
    breaking cross-references.
23. `05_plot_assam.py` D3 plots a train/val/test split for a pipeline that has no supervised
    model.

### Resolved / not defects

- **ERA5 deaccumulation** — Assam inherited the fixed stateless-clip version from the start.
  A real benefit of building Assam after the Rajasthan audit.
- **Uniform `Tm_target` with a climate-relative latent-heat criterion** — using
  `L / L_required` rather than raw `L` is the correct handling of a uniform target, and is
  documented in `08_mcdm_ranking.py`.
- **Monsoon-month consistency** — Jun–Sep is used in every Assam script. Rajasthan had a
  `02`-vs-`02b` mismatch; Assam does not.
- **The Phase 7 negative result itself** — correctly computed and honestly reported. The
  diagnosis needs revising (§10.5); the result does not.

---

## 16. What can and cannot be claimed

### Safe to write up now

- The Phase 1–4 methodology end to end: 128 population-weighted points at 87.5% coverage,
  ERA5-grid-aligned sampling, sun-event-aligned ERA5 sampling with the circular-hour-window
  algorithm, dual-source ERA5 + NASA POWER acquisition, and the two-tier signature design.
- The **ERA5 `ssrdc`-preferred clear-sky index** with a `clearsky_source` audit trail — a
  methodological upgrade over pvlib-only clear-sky modelling that the docs never claimed.
- The **free Tier-2 layer**: `02b` re-reads the full hourly POWER cache that `02` sampled
  three hours from, at zero additional download cost.
- The **k = 4 GMM result with full covariance**, presented as *interpretability-driven within
  the accept band*, not as a BIC minimum, with BIC/silhouette/DB/CH all tabulated.
- **ARI = 0.716 ± 0.139, `stable = False`** — reported as borderline, consistent with Assam's
  gradual climate transitions.
- The **1.1% ERA5-vs-POWER GHI bias** as evidence the deaccumulation handling is right —
  reported on its own terms, decoupled from the BACKBONE claim until §3.2 is settled.
- **Monsoon-month consistency** (Jun–Sep everywhere) as an improvement over Rajasthan.
- **IsolationForest-based multivariate outlier flagging** (described accurately as 3σ + IF)
  as an improvement over Rajasthan's univariate Hampel filter.
- The **four-method MCDM stack with Borda + Copeland + Kendall's W** and **5,000-draw Monte
  Carlo** matching plan §9.6 — noting the MC re-ranks on TOPSIS only.
- **Criterion-contribution explainability** as an Assam addition (after fixing §3.5).
- The **Phase 7 negative result**, stated plainly, with the revised diagnosis in §10.5.

### Not claimable as things stand

- **That the corrosion veto differentiates Assam's clusters.** It cannot fire. (§3.1)
- **That the PCM database is the binding constraint at 25 rows.** It is 62. (§3.1)
- **That ERA5 data flows to clustering unmodified under a BACKBONE decision.** It is
  quantile-corrected unconditionally. (§3.2)
- **That cluster IDs are canonical across re-runs.** No relabeling exists. (§3.7)
- **That Phases 5–8 are reproducible.** `06` does not run. (§3.1)
- **That the clustering result is externally validated.** Köppen-Geiger is not wired in;
  `05` prints it as a manual step.
- **That AHP pairwise elicitation informed the weights.** It did not.
- **That the current Top-3 is final.** It came from a database snapshot that no longer exists
  in the repository.
- **That the verification suites "PASS".** Several compute nothing; one fabricates its data.

### Prerequisites for a final, non-provisional Assam result

1. Fix `06_build_pcm_database.py` and re-run Phases 5–8 against the real 62-row database.
2. Resolve the signature-provenance race (§3.3) and confirm which definition produced the
   published clusters.
3. Settle the bias-correction question (§3.2) and make code and text agree.
4. Add canonical cluster relabeling before any re-run.
5. Either add genuinely inorganic candidates so the corrosion veto becomes real, or remove it
   from the narrative.
6. Recalibrate `COLLECTOR_EFF` / `A_C_M2` in Phase 7 so the collector can reach the top of the
   42–70 °C band, then re-run — and break the Phase 7 → Phase 6 fitness feedback loop (§3.6).
7. Wire in Köppen-Geiger for Phase 4 external validation.
8. Commit the small result CSVs so numbers become auditable.

### Final verdict by phase

| Phases | Verdict |
|---|---|
| 1–2 (collection, combine, agreement) | **Ready.** Methodology sound; fix the §3.8/§3.9 descriptions and the "117 points" label. |
| 2.5 (QC) | **Ready with a decision required** — the unconditional bias correction must be either justified or gated. |
| 3 (signature) | **Not ready** until the two rival implementations are resolved (§3.3) and the index list is corrected. |
| 4 Level A (clustering) | **Ready with minor fixes** — add relabeling; present k = 4 honestly as interpretability-driven. |
| 4 Level B (seasonal) | **Not implemented in practice** — code exists, never executes. |
| 5–6 (feasibility, MCDM) | **Not ready** — `06` does not run; the corrosion narrative must go; re-run required. |
| 7 (physics) | **Ready as a negative result**, with the revised diagnosis and the corrected assumption table. Not ready as a *final* validation until the collector ceiling is fixed. |
| 8 (cards) | **Ready after four one-line field-name fixes.** |

---

## 17. Document change log

**2026-09-05 — initial consolidation.** Built by reading all 22 files in `docs/assam/` and
all 32 files in `era5-assam/` in full, plus the four PCM property CSVs in `PCM_data/`. The
file was empty (0 bytes) before this revision.

This is a **reconciliation**, not a concatenation: `docs/assam/` was treated as a claim set
and each claim checked against source. Twenty-three discrepancies were found and are recorded
in §3 and §15, three of them blocking. No numeric result was recomputed — `era5-assam/data/`
is gitignored and absent from this checkout (§0.2) — so all reported values are attributed to
their documentary source and marked unverified.

Where this document and `docs/assam/*.md` disagree, **this document reflects the code** and
the individual audit files should be corrected to match.
