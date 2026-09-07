# Documentation Consolidation Summary (2026-09-02, updated 2026-09-07)

> **How to read this file:** it is a *changelog*. The Wave 1–3 sections below describe intermediate
> states — some name files (e.g. `04b_PHASE_2_5_AUDIT.md`, `09_ERA5_DATA_PIPELINE.md`) that were
> created and then folded away again in a later wave. Only the **"Remaining documentation structure"**
> tree and the **Wave 4 / Wave 5** notes describe the *current* folder. Every file named as a
> consolidation *target* (`03_PHASE_1_AUDIT.md`, `04_PHASE_2_AUDIT.md`, `05_PHASE_3_AUDIT.md`,
> `00_MASTER_OVERVIEW.md`, `13_LITERATURE_MAPPING.md`) exists today; every file named with "formerly"
> or "✗" has been deleted.

## What was consolidated

This consolidation reduces documentation fragmentation by embedding conceptual and methodological content directly into the phase audits where it's actually used, with explicit justification for each method/technique. **Two waves of consolidation** were applied (initial + comprehensive).

### Temporal Processing (formerly `10_TEMPORAL_PROCESSING.md`)

**Moved to:** `03_PHASE_1_AUDIT.md` → "Temporal Processing Justification (Dates, Times, Sunrise/Sunset)" section

**Content merged:**
- UTC as sole time reference — justifies choice for downstream pipeline consistency
- Sunrise/noon/sunset via pvlib SPA — explains why this method is used (Reda & Andreas 2004)
- Cross-midnight UTC handling (circular-window algorithm) — documents the edge case it solves
- Leap years and date range — ground-truthed verification
- Nearest-in-time matching (3-hour rejection window) — explains the gap in Phase 2
- Sun-event-aligned vs. fixed-clock-hour sampling — key justification for why the pipeline's sampling approach preserves physical meaning
- Seasonal definitions — inconsistency flag (Jun–Aug vs Jun–Sep) persists for reconciliation

**Additional temporal content added to:** `04_PHASE_2_AUDIT.md` → "Temporal Processing in the Merge" section
- Nearest-in-time matching in the merge step
- Unrecorded matched timestamp gap (low-cost fix flagged)
- Missing/duplicated timestamp handling

### Spatial Processing (formerly `11_SPATIAL_PROCESSING.md`)

**Moved to:** `03_PHASE_1_AUDIT.md` → "Spatial Processing Justification" section

**Content merged:**
- ERA5 grid alignment (0.25° to ERA5's own grid origin) — load-bearing design choice for 1:1 population-to-ERA5 mapping
- Rajasthan boundary & population aggregation — justifies 87.5% coverage rule
- Nearest-neighbor grid extraction — explains why no interpolation is used
- Elevation handling — documents grid-cell-mean caveat and why it's acceptable for Rajasthan
- Why this spatial approach is appropriate — frames the design as correct for regime-level recommendations, not microclimate modeling

### Research Gap Mapping (formerly `18_RESEARCH_GAP_MAPPING.md`)

**Moved to:** `00_MASTER_OVERVIEW.md` → "Research gaps addressed (N1–N6 novelty mapping)" section

**Content merged:**
- Disambiguation of N1–N6 (framework doc novelty) vs RG1–RG5 (broader project gaps) — prevents misattribution
- Full Phase → N mapping table — shows how each phase contributes to Objective 1's novelty claims
- Full Phase → RG mapping table — shows how Objective 1 feeds into broader project objectives
- Important note section — clarifies that RG1–RG4 are addressed across multiple objectives, not by Objective 1 alone

### Second Wave: Methodology & Processing Details (12–16)

**Solar Geometry (formerly `12_SOLAR_GEOMETRY.md`)**
**Moved to:** `04_PHASE_2_AUDIT.md` → "Solar Geometry (why it's computed this way)" section
- Explains choice of pvlib SPA (need to pin method explicitly)
- Ineichen clear-sky model with default Linke turbidity
- Altitude/elevation handling for atmospheric-pressure assumptions
- Nighttime handling and division-by-zero protection for CSI

**Solar-Derived Variables (formerly `13_SOLAR_DERIVED_VARIABLES.md`)**
**Moved to:** `04_PHASE_2_AUDIT.md` → "Solar-Derived Variables (construction & assumptions)" section
- GHI computation from deaccumulated ERA5
- DNI two-branch derivation (direct field vs closure fallback) — explicitly NOT a decomposition model
- DHI as closure residual (not independent, all errors propagate)
- Clearness Index (CSI) construction with nighttime suppression
- Unit-consistency caveat on `avg_sdirswrf` field-matching logic (open issue)
- Physical bounds applied to all solar variables

**ERA5 vs NASA POWER Validation Decision (formerly `14_ERA5_POWER_VALIDATION.md`)**
**Moved to:** `04_PHASE_2_AUDIT.md` → "Cross-Source Validation Decision (why QUANTILE_MAP was chosen)" section
- Variable pairs compared and matching procedure
- Decision rule thresholds (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW)
- Actual Rajasthan numbers: r=0.8102, MBE=10.95 W/m² at solar noon
- Why QUANTILE_MAP was chosen (r ≥ 0.70 but failed stricter gates)
- Critical caveat: quantile-mapped GHI never persisted; Phase 3 reads uncorrected ERA5

**Quality Control (formerly `15_QUALITY_CONTROL.md`)**
**Moved to:** briefly `04b_PHASE_2_5_AUDIT.md`, then folded into `04_PHASE_2_AUDIT.md` Part B in Wave 4 (its current home)
- Part 1: Read-only sanity checks via `03_verify_climate_csv.py` (6 checks, schema/coverage/nulls/ranges/cross-source)
- Part 1b: Visual QC via `03_qc_plots.py` (8 interactive HTML visualizations)
- Part 2: Actual data cleaning via `03b_quality_check_rajasthan.py` (Hampel filter on T_amb/RHum/W_spd only, **deliberately excluding GHI/CSI** to preserve weather variability)
- Part 2b: Validation of cleaning via `03b_validate_quality_fix_rajasthan.py`
- Part 2c: Visual before/after via `03c_plots_raw_rajasthan.py` + `03b_quality_check_plots_rajasthan.py`
- Justification for excluding GHI/CSI: clouds are weather, not errors
- First correction (2026-08-11): initially applied Hampel to GHI/CSI, over-corrected cloud variability → fixed

**Climate Signature Feature Mapping (formerly `16_CLIMATE_SIGNATURE.md`)**
**Moved to:** `05_PHASE_3_AUDIT.md` → "Climate Signature Feature-to-PCM-Property Mapping" section
- Governing design principle: "Every index must answer which PCM property it constrains & by what mechanism"
- Full feature → thermal behavior → PCM requirement → PCM property mapping table
- Why two-tier (Tier 1 sun-event + Tier 2 daily-integral) design is necessary (neither alone is sufficient)
- HSI_sunrise properly attributed as Thom's (1959) THI, not original derivation
- Interaction terms: 5 compound-risk terms, each named and physically justified
- PCA scope: only temperature/elevation block, deliberately excluding discriminating solar/humidity/cycling indices
- Level-B ablation candidate: daylength (climatically tautological, not weather-driven)

---

## Files now eligible for deletion

These files can be safely deleted; their content is now embedded in the indicated audit files with full context and justification:

**Wave 1 (initial consolidation):**
- **`10_TEMPORAL_PROCESSING.md`** → consolidated into `03_PHASE_1_AUDIT.md` and `04_PHASE_2_AUDIT.md`
- **`11_SPATIAL_PROCESSING.md`** → consolidated into `03_PHASE_1_AUDIT.md`
- **`18_RESEARCH_GAP_MAPPING.md`** → consolidated into `00_MASTER_OVERVIEW.md`

**Wave 2 (methodology consolidation):**
- 12_SOLAR_GEOMETRY.md → consolidated into `04_PHASE_2_AUDIT.md`
- 13_SOLAR_DERIVED_VARIABLES.md → consolidated into `04_PHASE_2_AUDIT.md`
- 14_ERA5_POWER_VALIDATION.md → consolidated into `04_PHASE_2_AUDIT.md`
- 15_QUALITY_CONTROL.md → consolidated into `04b_PHASE_2_5_AUDIT.md` (new dedicated Phase 2.5 audit)
- 16_CLIMATE_SIGNATURE.md → consolidated into `05_PHASE_3_AUDIT.md`

**Wave 2 (methodology consolidation) — COMPLETE:**
- 12_SOLAR_GEOMETRY.md → consolidated into `04_PHASE_2_AUDIT.md`
- 13_SOLAR_DERIVED_VARIABLES.md → consolidated into `04_PHASE_2_AUDIT.md`
- 14_ERA5_POWER_VALIDATION.md → consolidated into `04_PHASE_2_AUDIT.md`
- 15_QUALITY_CONTROL.md → consolidated into `04_PHASE_2_AUDIT.md`
- 16_CLIMATE_SIGNATURE.md → consolidated into `05_PHASE_3_AUDIT.md`

**Wave 3 (Phase 7-8 completion reports) — COMPLETE:**

---

## Files still requiring cross-reference updates

**`13_LITERATURE_MAPPING.md`** (formerly numbered `17_`) has been updated with a header note explaining the consolidation and pointing readers to the new locations of temporal/spatial processing justifications and research gap mapping content.

---

## Remaining documentation structure

The Rajasthan `docs` folder now follows this hierarchy (15 files, down from 26 at the start). Phase 2 and Phase 2.5 share a single audit (`04_PHASE_2_AUDIT.md`); the ERA5 deaccumulation deep-dive, quality-control detail, and literature-mapping notes all live inside existing audits now, not in standalone files:

```
00_MASTER_OVERVIEW.md          [Overall pipeline status + N1-N6 novelty & RG1-RG5 research gap mapping]
│
├─ Phase audits (with full embedded justifications & completion reports):
│  ├─ 03_PHASE_1_AUDIT.md      [+ Spatial & Temporal Processing Justification]
│  ├─ 04_PHASE_2_AUDIT.md      [Phase 2 + Phase 2.5: Temporal Processing, Solar Geometry, Derived Variables,
│  │                            ERA5 deaccumulation, Cross-Source Validation, Quality Control / Hampel / imputation]
│  ├─ 05_PHASE_3_AUDIT.md      [+ Climate Signature Feature-to-PCM-Property Mapping]
│  ├─ 06_PHASE_4_AUDIT.md
│  ├─ 07_PHASE_5_AUDIT.md
│  ├─ 08_PHASE_6_AUDIT.md
│  ├─ 09_PHASE_7_AUDIT.md      [+ Completion report: bugs caught & fixed, inherited caveats, honest negative result]
│  └─ 10_PHASE_8_AUDIT.md      [+ Supercooling diagnostic, sensitivity analysis, Phase 9 epilogue]
│
├─ Context & reference:
│  ├─ 01_PROJECT_CONTEXT.md
│  ├─ 02_DATA_SOURCES_AND_VARIABLES.md
│  ├─ 11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md [Plotting audit + PLOTSV2 coverage]
│  └─ 13_LITERATURE_MAPPING.md [Updated with consolidation notes; formerly numbered 17_]
│
└─ Post-pipeline documentation:
   ├─ 12_FINAL_READINESS_REPORT.md
   └─ CONSOLIDATION_SUMMARY.md [This file]
```

Implementation-issue and reproducibility detail that earlier drafts kept in standalone `20_`/`21_`
files now lives in `00_MASTER_OVERVIEW.md` ("Current known issues") and `12_FINAL_READINESS_REPORT.md`.

---

## Why this consolidation matters

1. **Single source of truth per phase:** Each phase audit now contains its own complete methodology justification, not scattered across separate files
2. **Reduced cognitive load:** Readers don't need to cross-reference five files to understand why a particular method was chosen
3. **Easier maintenance:** Updates to methodology justification stay in one place (the phase audit) rather than requiring changes in multiple concept files
4. **Preserved context:** Justifications are now framed in the specific context where the method is actually applied (Phase 1/2), not as abstract standalone concepts

---

## Consolidation status

✅ **COMPLETE** — All ten standalone concept/methodology/gap/completion files have been consolidated into their respective phase audits and master overview.

**Wave 1 (early):** 3 files consolidated (temporal, spatial, research gaps)
**Wave 2 (methodology):** 5 files consolidated (solar geometry, solar variables, validation decision, quality control, climate signature)
**Wave 3 (Phase 7-8):** 2 files consolidated (completion report, summary guide)
**New Phase 2.5 audit created:** Dedicated documentation for the undocumented-until-now quality-control stage

**Wave 4 (Final consolidation — Phase 2 as single comprehensive file):**
- 09_ERA5_DATA_PIPELINE.md → consolidated into `04_PHASE_2_AUDIT.md`
- 04b_PHASE_2_5_AUDIT.md → consolidated into `04_PHASE_2_AUDIT.md`

**Cleanup action:** The twelve now-redundant files have been deleted from the repository:
- ✗ `10_TEMPORAL_PROCESSING.md`
- ✗ `11_SPATIAL_PROCESSING.md`
- ✗ `12_SOLAR_GEOMETRY.md`
- ✗ `13_SOLAR_DERIVED_VARIABLES.md`
- ✗ `14_ERA5_POWER_VALIDATION.md`
- ✗ `15_QUALITY_CONTROL.md`
- ✗ `16_CLIMATE_SIGNATURE.md`
- ✗ `18_RESEARCH_GAP_MAPPING.md`
- ✗ `19_PHASE_7_ONWARD.md` (completion report, bugs fixed, caveats inherited)
- ✗ `PHASE_7_8_SUMMARY.md` (high-level overview/navigation guide)
- ✗ `09_ERA5_DATA_PIPELINE.md` (deaccumulation deep-dive, critical bug fix details)
- ✗ `04b_PHASE_2_5_AUDIT.md` (quality control, Hampel filter, MICE imputation)

**Wave 5 (2026-09-07 — reconciliation with current `era5-rajasthan/` code):**
- `11_LITERATURE_MAPPING.md` renamed to `13_LITERATURE_MAPPING.md` to clear a duplicate `11_` prefix
  (the plotting audit keeps `11_`); H1 titles realigned to filenames in `12_` and `13_`.
- Doc-wide script-name refresh: `04_climate_signature_rajasthan.py` → `04b_climate_signature.py`,
  `07_feasibility_filter_rajasthan.py` → `07_feasibility_filter.py`,
  `08_mcdm_ranking_rajasthan.py` → `08_mcdm_ranking.py` (Rajasthan scripts renamed toward the shared
  cross-state convention).
- New audit coverage added in-place: `04_preprocess_rajasthan.py` (Phase 2), the per-phase
  visualization/interactive scripts (`00d`, `03b_coverage_viz`, `03b_qmap_before_after_viz`, `03d`,
  `04d`, `04e`, `04f`, `05e`, `05f`, `05g`, `check_supercooling_*`), and the `PLOTSV2/` plotting layer
  (documented in `11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`).
- Three byte-identical duplicate scripts removed from `era5-rajasthan/`: `10_physics_validation.py`
  (== `09_physics_validation_rajasthan.py`), `09_recommendation_cards.py`
  (== `10_recommendation_cards_rajasthan.py`), `03_plots_raw.py` (== `03c_plots_raw_rajasthan.py`).

**Result:** 15 focused, highly integrated documentation files (down from 26 at start). Single Phase 2 audit now contains:
- Complete preprocessing workflow (02_combine + 02b_daily_aggregates)
- Cross-source validation (ERA5 vs POWER agreement analysis)
- ERA5 deaccumulation details (the critical bug fix that enabled downstream analysis)
- Solar geometry & derived variables (why computed this way)
- Quality control & data cleaning (Hampel filter, imputation, validation)
- All temporal processing & visualization

Each phase audit now contains complete methodological justification for every technique, full completion reports with bug-fix histories, and all supporting details. No concept files to juggle — readers get everything they need for understanding *why* each choice was made in the single audit where it's used.
