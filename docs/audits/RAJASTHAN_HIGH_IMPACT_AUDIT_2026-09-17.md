# Rajasthan — High-Impact Audit (2026-09-17)

Scope: `era5-rajasthan/*.py`, `docs/rajasthan/*.md`, `era5-rajasthan/outputs/*`,
`pcm_shared_config.py`. Read-only audit; nothing in this pass changed any file.

**Baseline check — already-fixed items confirmed still correct in current code/outputs:**
SHARE_PCM=0.5, T_DELIVERY_C 50→60 cascaded correctly into `physics_lib.py`'s local
duplicate, M_W_KG 300→200 + COLLECTOR_UL_WM2K 2.0 + NIGHT_ISOLATION_FRACTION 0.03,
and the `11_seasonal_pcm_sensitivity.py` per-cluster-kappa fix are all genuinely
present and internally consistent with `outputs/recommendation_cards_rajasthan.md`
and `outputs/seasonal_pcm_sensitivity_rajasthan.md`. Not re-flagged below.

---

## 1. `docs/rajasthan/12_FINAL_READINESS_REPORT.md` is internally contradictory — stale numbers survive below the "CRITICAL UPDATE" banners meant to supersede them

**File:** `docs/rajasthan/12_FINAL_READINESS_REPORT.md`

- The top of the file (CRITICAL UPDATE #2/#3, ~lines 22–48) correctly states the
  current Phase 7 Spearman ρ = **0.105 / -0.190 / -0.091** for clusters 0/1/2,
  matching `outputs/recommendation_cards_rajasthan.md` exactly.
- Further down, "Completed phases" (~line 95) and "Strongest components" item 3
  (~line 116) still quote **ρ = -0.900 / -0.096 / -0.198** — numbers from a
  pre-correction run that were never overwritten when the banners were added.
- Also stale: "Weakest components" item 1 and "Prerequisites for a FINAL result"
  both say `PCM_Properties_cleaned_mice_pmm_detailed.csv` is "currently missing
  from disk" and frame the candidate pool as "55 rows." The file exists on disk
  now (`PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv`, 34,909 bytes),
  and `07_PHASE_5_AUDIT.md` plus the current recommendation cards both confirm the
  actual evaluated pool is **62** rows (55 manufacturer + 7 literature), not 55.

**Why high-impact:** this file is literally named "Final Readiness Report" — the
document most likely to be skimmed for numbers when writing the methodology/results
section. A reader who scrolls past the dated top banners (easy to do, since they're
appended notes, not a rewrite of the body) will pick up the wrong Spearman ρ values
or the wrong "blocking item" status.

**Fix direction:** rewrite the stale body sections (Completed phases, Strongest
components, Weakest components, Prerequisites) to match the banner numbers, or add
an explicit "superseded — see banner above" marker next to each stale figure, the
way `09_PHASE_7_AUDIT.md` and `10_PHASE_8_AUDIT.md` already do.

---

## 2. "Open scientific risk" about GHI quantile-mapping is stale — it has actually been resolved, but is still listed as an open decision

**Files:** `docs/rajasthan/12_FINAL_READINESS_REPORT.md` ("Scientific risks"
section), `docs/rajasthan/CONSOLIDATION_SUMMARY.md` (same claim: "Critical caveat:
quantile-mapped GHI never persisted; Phase 3 reads uncorrected ERA5").

- Code check: `04_preprocess_rajasthan.py` (Phase 2.5, ~lines 170–224) fits a
  per-season quantile mapper (ERA5 GHI → NASA POWER GHI) and applies it **in
  place** to `era5_GHI` (propagating into `era5_CSI`), writing the corrected
  values into `rajasthan_cleaned_physical.csv`.
- `04b_climate_signature.py` (~lines 226/324) reads
  `PHYSICAL_FILE = PREPROCESSED_DIR / "rajasthan_cleaned_physical.csv"` directly —
  Phase 3 **does** consume the quantile-mapped GHI, not raw ERA5.

**Why high-impact:** `12_FINAL_READINESS_REPORT.md` lists this as one of "the two
most consequential open scientific decisions" still needing to be made before
Phase 6's output is final — but it appears to already be implemented (most likely
once `04_preprocess_rajasthan.py` superseded `03b_quality_check_rajasthan.py` as
Phase 2.5). Leaving it listed as unresolved risks either (a) someone redoing work
that's already done, or (b) the methodology write-up describing an uncorrected-GHI
pipeline that doesn't match what actually ran.

**Fix direction:** confirm against `ghi_quantile_mapping_report.csv` (written by
the same step) and update both docs to mark this risk resolved, with a pointer to
the correction step and its before/after MBE numbers.

---

## 3. Cluster 0's n=4 candidate pool is a de facto committed design choice with no stated fallback

`12_FINAL_READINESS_REPORT.md`'s own "Scientific risks" section names "what the
permanent policy should be for the latent-heat feasibility constraint (accept
calibrated-κ, or switch to rank-by-proximity)" as unresolved, and
`07_PHASE_5_AUDIT.md` confirms Cluster 0 bottoms out at κ=0.0 with only 4
rescuable survivors out of 62 — driven by melting-window + charging-feasibility
overlap, not latent heat. This is correctly diagnosed elsewhere as a genuine
climate-driven finding, not a bug — but no doc states what happens if a reviewer
judges n=4 too small to support a defensible Top-3 ranking claim for that cluster.

**Why high-impact:** this is a real open methodological gap that could force a late
rewrite of roughly one-third of Objective 1's headline results table if raised in
review.

**Fix direction:** decide explicitly (accept n=4 with a stated justification, or
define a documented fallback such as rank-by-proximity) before the paper is
drafted — not a code fix, but should not remain implicit.

---

Everything else checked — `physics_lib.py`'s calibration math, the MCDM
entropy-weight cap, `run_all_rajasthan.py`'s orchestration order, and the seasonal-
sensitivity kappa fix — matches what the docs claim and did not surface a new
high-impact issue.
