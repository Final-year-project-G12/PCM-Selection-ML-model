# Documentation Consolidation Summary — Uttarakhand

## What this folder is

A complete audit trail of `PCM-Selection-ML-model/era5-uttarakhand/`, structured to match the
Rajasthan documentation set: **phase audits with all conceptual and methodological justification
embedded where the method is actually used**, rather than scattered across standalone concept
files.

Every state-specific fact in this set is sourced exclusively from `era5-uttarakhand/`. Nothing was
imported from the Rajasthan, Tamil Nadu or Assam documentation — those were consulted only for
format and level of detail.

---

## What was consolidated

The set was first drafted as 22 files following the Assam/Tamil Nadu layout, then consolidated into
the Rajasthan phase-audit layout. Seven standalone concept files were merged into the phase audits
where their content is used, and their originals deleted.

### Temporal Processing (formerly `10_TEMPORAL_PROCESSING.md`)

**Merged into:** `03_PHASE_1_AUDIT.md` -> "Temporal Processing Justification (Dates, Times,
Sunrise/Sunset)", with the merge-step portion in `04_PHASE_2_AUDIT.md` -> A.5.

Content merged:
- Study period (2016–2025, 3,653 days) and the 493,155-row arithmetic
- UTC as the sole time reference, and the single IST conversion in `04` step 6
- Sunrise/noon/sunset via pvlib SPA with `method="spa"` pinned — and the altitude-0 m vs
  altitude-1200 m inconsistency between `00b` and `02`
- Cross-midnight UTC handling and the `circular_hour_window()` algorithm that solves it
- De-accumulation predecessor logic and the documented 2016-01-01 edge case
- Nearest-in-time matching (3-hour rejection window) and the unrecorded matched-timestamp gap
- Sun-event-aligned vs fixed-clock-hour sampling, and why every lag/rolling/delta feature is
  therefore defined over *occurrences*
- The `monsoon_index` JJAS vs `SEASON_MAP` JJA inconsistency (flagged, unreconciled)
- The `03_plots_raw.py` docstring's wrong regional climatology

### Spatial Processing (formerly `11_SPATIAL_PROCESSING.md`)

**Merged into:** `03_PHASE_1_AUDIT.md` -> "Spatial Processing Justification".

Content merged:
- ERA5 grid alignment (0.25° anchored to ERA5's own origin) — verified against the 45 observed
  point coordinates
- GADM boundary handling and WorldPop aggregation; the 87.5 % coverage rule
- The full observed point set: 45 points, `UKP_0001–UKP_0045`, 10,475,711 population,
  28.875–30.625 °N / 77.875–80.125 °E
- Nearest-neighbour extraction and why no interpolation is used
- **Elevation handling** — the three inconsistent altitude assumptions, and why this is
  Uttarakhand's central spatial limitation
- Population weighting — where it is and is not applied
- Why the approach is appropriate for regime-level recommendation, and the
  population-representative-not-area-representative caveat

### ERA5 Data Pipeline (formerly `09_ERA5_DATA_PIPELINE.md`)

**Merged into:** `04_PHASE_2_AUDIT.md` -> A.3 "ERA5 Accumulated Fields & De-accumulation".

Content merged:
- The `deaccumulate()` implementation and its stated MARS-convention model
- The four independent lines of evidence for the GHI magnitude anomaly (raw event profile,
  cross-source statistics with two clean controls, downstream magnitudes, and the `era5_LW_down`
  fingerprint)
- What can and cannot be concluded from `era5-uttarakhand/` alone
- The exact one-file verification procedure that would settle it

### Solar Geometry (formerly `12_SOLAR_GEOMETRY.md`)

**Merged into:** `04_PHASE_2_AUDIT.md` -> A.6 "Solar Geometry (why it's computed this way)".

Content merged:
- `compute_solar()` and the **unpinned** `get_solarposition()` method (vs `00b`, which pins it)
- Ineichen clear-sky with default Linke turbidity — and the r = 0.9923 / MBE +5.3 W/m² validation
  that makes this the pipeline's strongest positive result
- The 1200 m flat-altitude assumption and the fact it is never written to output
- Night-time handling, division-by-zero protection, and the three-way ambiguity of `CSI = 0`
- `ETR` computed and discarded

### Solar-Derived Variables (formerly `13_SOLAR_DERIVED_VARIABLES.md`)

**Merged into:** `04_PHASE_2_AUDIT.md` -> A.7 "Solar-Derived Variables (construction &
assumptions)".

Content merged:
- GHI construction and observed magnitudes
- DNI's two-branch derivation, the three-name `avg_sdirswrf` matcher and its latent 3600× unit
  hazard, and the explicit statement that the fallback is **not** a decomposition model
- DHI as a closure residual with no independent basis
- CSI construction, and the key finding that the Tier-2 repair insulated the canonical solar block
  while `GHI_mean` remains exposed
- Cloud cover, precipitation and the heavily-flagged `era5_LW_down`
- The full physical-bounds table for solar variables with observed flag counts

### ERA5 vs NASA POWER Validation (formerly `14_ERA5_POWER_VALIDATION.md`)

**Merged into:** `04_PHASE_2_AUDIT.md` -> A.8 "Cross-Source Validation Decision — there isn't one".

Content merged:
- The full `C_era5_vs_power_stats.csv` table (n = 493,155)
- The three places the source files say a large MBE "gets addressed in 04", and the confirmation
  that nothing in `04` addresses it
- Variable pairs compared, and the absence of any stratification
- Detailed reading of the RHum (+11.4 %) and wind (−1.14 m/s) disagreements
- The which-side-does-each-clustering-column-come-from table, and the two-entry `CANON_MAP` fix

### Quality Control (formerly `15_QUALITY_CONTROL.md`)

**Merged into:** `04_PHASE_2_AUDIT.md` -> **PART B**, the full Phase 2 preprocessing and QC audit.

Content merged:
- The 13-step sequence in full, with the occurrence-not-hours schema note the script leads with
- Physical-bounds table with observed flag counts (`era5_LW_down` 363,525; `era5_P_atm` 182,899)
- The Hampel filter and why 10.0 % of `era5_cloud_cover` and 7.2 % of `era5_GHI` were flagged —
  clouds are weather, not errors
- The four-tier hierarchical imputation chain and the `impute_zone` clarification
- Feature engineering, lag/rolling/delta counts (18 + 24 + 3 = 45, matching the verified figure)
- The 4,050-row lag warm-up drop = 45 × 3 × 30 exactly
- Leakage-safe scaling and the step-13 hard gate
- The verified outcome: 489,105 rows, 89 columns, 100 % complete cases
- The cleaned-file distribution table and its three key observations
- `04c`'s six post-cleaning checks

### Climate Signature feature mapping (drafted as `16_CLIMATE_SIGNATURE.md`, never a separate file)

**Written directly into:** `05_PHASE_3_AUDIT.md` -> "Climate Signature Feature-to-PCM-Property
Mapping".

Content:
- Every Tier-1 and Tier-2 index mapped to its physical mechanism and the PCM property it constrains
- Why the two-tier design is necessary — including the demonstrable finding that Tier 2 insulated
  the clustering matrix from the pipeline's largest data defect
- PCA scope and why the solar block is deliberately kept out
- The table of indices that carry a known problem into the clustering matrix

### Research gap / novelty mapping (drafted as `18_RESEARCH_GAP_MAPPING.md`, never a separate file)

**Written directly into:** `00_MASTER_OVERVIEW.md` -> "Research gaps and novelty mapping".

Content:
- N1–N6 vs RG1–RG5 disambiguation
- Phase -> novelty-claim mapping with a delivered/partial/not-delivered verdict per phase
- **The central finding against the novelty claim**: this run does not demonstrate
  regime-differentiated PCM recommendation, and why
- Phase -> broader-project mapping
- What the mapping explicitly does not claim

### Implementation issues and reproducibility (drafted as `20_` and `21_`, never separate files)

**Written directly into:** `12_FINAL_READINESS_REPORT.md`.

Content: the ranked implementation-issues list (1 fixed, 22 open across three priority tiers), the
full reproducibility checklist with PASS/PARTIAL/FAIL per item, the three gaps that matter most,
and 12 recommended fixes ordered by effort-to-impact.

### Plots and verification suite (drafted as `23_` and `24_`, merged)

**Merged into one file:** `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`, mirroring
Rajasthan's `11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md` slot.

Content: the committed plot inventory, what each of the 13 Objective 1 plots actually shows, the
`passes_all` defect, the orphaned `data/plots/objective1/` directory, the never-run
`comparison_plots_uttarakhand.py`, the four verification scripts, the two preserved generations of
results, an internal-consistency cross-check, and 13 tabulated defects.

---

## Files deleted

- `09_ERA5_DATA_PIPELINE.md` -> `04_PHASE_2_AUDIT.md` A.3
- `10_TEMPORAL_PROCESSING.md` -> `03_PHASE_1_AUDIT.md` + `04_PHASE_2_AUDIT.md` A.5
- `11_SPATIAL_PROCESSING.md` -> `03_PHASE_1_AUDIT.md`
- `12_SOLAR_GEOMETRY.md` -> `04_PHASE_2_AUDIT.md` A.6
- `13_SOLAR_DERIVED_VARIABLES.md` -> `04_PHASE_2_AUDIT.md` A.7
- `14_ERA5_POWER_VALIDATION.md` -> `04_PHASE_2_AUDIT.md` A.8
- `15_QUALITY_CONTROL.md` -> `04_PHASE_2_AUDIT.md` PART B

Four further files planned under the Assam layout (`16_CLIMATE_SIGNATURE`,
`18_RESEARCH_GAP_MAPPING`, `20_IMPLEMENTATION_ISSUES`, `21_REPRODUCIBILITY`) were never created as
standalone files — their content was written directly into the consolidated targets.

---

## Resulting structure — 15 files

```
00_MASTER_OVERVIEW.md      [Pipeline status + architecture + novelty & research-gap mapping]
|
+- Phase audits (with full embedded justifications):
|  +- 03_PHASE_1_AUDIT.md  [+ Spatial & Temporal Processing Justification]
|  +- 04_PHASE_2_AUDIT.md  [Part A: combine, Tier-2 repair, ERA5 de-accumulation, solar geometry,
|  |                         derived solar variables, cross-source validation
|  |                        Part B: the full 13-step quality control + post-clean QA
|  |                        Part C: combined problems  |  Part D: combined status]
|  +- 05_PHASE_3_AUDIT.md  [+ Climate Signature Feature-to-PCM-Property Mapping]
|  +- 06_PHASE_4_AUDIT.md  [Regime clustering]
|  +- 07_PHASE_5_AUDIT.md  [PCM database + feasibility filtering]
|  +- 08_PHASE_6_AUDIT.md  [MCDM ranking engine]
|  +- 09_PHASE_7_AUDIT.md  [Physics validation — NOT IMPLEMENTED]
|  +- 10_PHASE_8_AUDIT.md  [Recommendation cards]
|
+- Context & reference:
|  +- 01_PROJECT_CONTEXT.md
|  +- 02_DATA_SOURCES_AND_VARIABLES.md
|  +- 11_LITERATURE_MAPPING.md
|  +- 11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md
|
+- Post-pipeline:
   +- 12_FINAL_READINESS_REPORT.md   [Implementation issues + reproducibility + final verdict]
   +- CONSOLIDATION_SUMMARY.md       [This file]
```

Down from 22 planned files to 15, matching Rajasthan's structure one-for-one.

---

## Why this consolidation matters

1. **Single source of truth per phase.** Each phase audit contains its own complete methodology
   justification, not scattered across five concept files.
2. **Reduced cognitive load.** A reader does not need to cross-reference multiple files to
   understand why a method was chosen.
3. **Preserved context.** Justifications are framed where the method is applied — the
   de-accumulation analysis sits next to the merge that performs it, and the quality-control audit
   sits next to the preprocessing it evaluates.
4. **Easier maintenance.** A methodology change updates one place.

---

## Evidence basis — a note specific to Uttarakhand

`era5-uttarakhand/.gitignore` excludes `data/raw/`, `data/processed/` and `data/preprocessed/`.
**None of the pipeline's result CSVs are committed** — not `qc_report.txt`, not `pca_loadings.csv`,
not `bic_selection_uttarakhand.csv`, not `cluster_profiles_uttarakhand.csv`, not
`mcdm_topk_by_cluster.csv`, not `recommendation_cards.md`.

Every observed number in this documentation set was therefore recovered from the **committed plot
tree**, by one of four methods:

| Method | Example |
|---|---|
| Parsing a committed CSV | `C_era5_vs_power_stats.csv` (cross-source statistics); `C_qc_flag_counts.csv` (QC counts) |
| Decoding embedded Plotly base64 payloads | Top-3 PCM ranks and properties from `13_recommended_pcm_summary_interactive.html` |
| Parsing Folium popup HTML | 45 point IDs, coordinates, populations and cluster assignments |
| Reading a rendered summary panel | 493,155 -> 489,105 rows; silhouette 0.279; Spearman −0.930 |
| Reproducing a computation against a committed source CSV | The 29-survivor feasibility count, from the committed PCM database |

Values recovered from a rendered chart rather than parsed from a file are marked **approximate** at
the point of use. Values that could not be recovered are marked **"not available in the source
files"** — principally Kendall's W per cluster, the BIC/silhouette selection table, `L_required`
per cluster, the PCA component count, and the VIF report.

**The single highest-value fix for this documentation set** is to commit the roughly ten small
result CSVs, or add a `.gitignore` exception for them. See `12_FINAL_READINESS_REPORT.md`.

---

## Consolidation status

**COMPLETE.** All seven standalone concept files have been merged into their phase audits and
deleted; the four never-created files were written directly into their targets. All cross-references
have been updated to point at the consolidated locations.
