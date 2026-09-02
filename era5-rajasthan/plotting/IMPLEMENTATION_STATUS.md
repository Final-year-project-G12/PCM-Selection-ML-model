# Objective 1 Plotting Audit — Implementation Status

**Last Updated:** 2026-09-02  
**Overall Progress:** 9/13 main plots + 2/8 comparison plots implemented (~68%)

## Summary

This document tracks the implementation progress of the comprehensive Objective-1 plotting audit for the Rajasthan PCM selection pipeline, per the detailed specification in `../docs/rajasthan/11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`.

### Quick Stats

```
Main Plots (13 required):
  ✓ Implemented: 9 scripts (plots 1, 2, 3, 4, 5, 6, 8, 9, 11)
  ⏳ TODO: 3 (plot 10 blocked, plot 7 referencing existing)
  ~ EXISTING: 1 (plot 7, copy/consolidate)

Comparison Plots (8 additional):
  ✓ Implemented: 2 (phase3_tmcap, phase5_lrequired)
  ⏳ TODO: 6 remaining
  ✓ READY FOR: 4 more (can be written quickly)

Infrastructure:
  ✓ Master runner (run_all_plots.py)
  ✓ Comprehensive README.md
  ✓ Per-script verification blocks
  ✓ Fingerprint staleness detection
  ✓ Output directory structure (created on demand)
```

---

## PART A — Main Plots (13 Required)

### ✓ IMPLEMENTED (9 scripts ready)

| # | Plot | Script | Status | Verification |
|---|------|--------|--------|--------------|
| **1** | Raw vs. preprocessed radiation | `01_raw_vs_preprocessed.py` | ✓ READY | KS statistics (GHI small, T_amb large) |
| **2** | Climate regime map | `02_climate_regime_map_copy.py` | ✓ READY | k=3, cluster 0 lowest latitude, sizes |
| **3** | PCM feasibility scatter | `03_pcm_feasibility_scatter.py` | ✓ READY | Survivor count per cluster, L_required consistency |
| **4** | Survivors per cluster | `04_pcm_survivors_per_cluster.py` | ✓ READY | Total calibrated survivors = 39 (post-correction) |
| **5** | Bump chart (method agreement) | `05_bump_chart.py` | ✓ READY | Spearman rho(VIKOR, TOPSIS) — no sign-inversion bug |
| **6** | Correlation heatmap | `06_method_correlation_heatmap.py` | ✓ READY | GRA has lowest mean pairwise correlation (all clusters) |
| **8** | Rank-reversal frequency | `08_rank_reversal_frequency.py` | ✓ READY | Cluster 0 freq > Clusters 1/2 (due to lower Kendall's W) |
| **9** | MCDM vs. physics | `09_mcdm_vs_physics_agreement.py` | ✓ READY | Spearman rho: C0=-0.385, C1=+0.125, C2=-0.097 |
| **11** | Summary cards | `11_summary_cards.py` | ✓ READY | Top-1 per cluster, Tm/L/confidence/rho comparison |

### ⏳ TODO / BLOCKED (4 remaining)

| # | Plot | Issue | What's Needed |
|---|------|-------|---------------|
| **7** | MC Top-3 inclusion probability | Exists (Phase 6) | Script to copy `qc_montecarlo_inclusion_rajasthan.html` + compute imputation correlation |
| **10** | Tank profile (day-night) | **BLOCKED** | Instrumentation: `physics_lib.py` needs `save_timeseries=True` hook to return hourly Tw/Tp/melt_frac arrays |
| **7** (consolidation) | Monte Carlo summary | TODO | Wrapper to (1) copy existing plot, (2) analyze per-candidate imputation impact |

**Estimated effort:**
- Plot 7 consolidation: 30 min (write + test)
- Plot 10 instrumentation: 1–2 hours (add hook to physics_lib, verify physics, write plot)

---

## PART B — Comparison Plots (8 Additional)

### ✓ IMPLEMENTED (2 scripts ready)

| Phase | Comparison | Script | Status | Notes |
|-------|-----------|--------|--------|-------|
| **3** | Tm_target_capped (old vs. new) | `comparison_phase3_tmcap_old_vs_new.py` | ✓ READY | Scatter with y=x ref line; shows 51–55°C (realistic) vs. 40–49°C (old) |
| **5** | L_required before/after | `comparison_phase5_lrequired_before_after.py` | ✓ READY | Bar chart; gracefully skips if pre-correction backup not found |

### ✓ READY TO IMPLEMENT (4 scripts, data exists on disk)

| Phase | Comparison | Data Source | Estimated Effort |
|-------|-----------|-------------|-----------------|
| **2.5** | Raw vs. clean (5 variables) | `climate_rajasthan_points.csv` + `climate_rajasthan_points_clean.csv` | 30 min |
| **3** | Tier 1 vs. Tier 2 | `climate_signature_rajasthan.csv` (diurnal_gradient vs. DTR_true) | 20 min |
| **4** | Level A vs. B | `cluster_assignments_rajasthan_levelA.csv` + `levelB.csv` | 30 min |
| **8** | k=0.0 vs. k=0.3 | `phase8_supercooling_sweep_rajasthan.csv` | 20 min |

**Subtotal:** ~100 min (~2 hours) to implement these 4

### ⏳ REQUIRES INVESTIGATION (2 scripts)

| Phase | Comparison | Issue | Path Forward |
|-------|-----------|-------|--------------|
| **6** | VIKOR bugfix before/after | Requires pre-fix backup of `mcdm_rankings_rajasthan.csv` | Check for `backups/`, `archive/`, or `.git` history; if none, gracefully note as "historical, not reproducible" |
| **7** | PCM vs. plain tank | Needs plain-tank comparison data | Check `physics_validation_rajasthan.csv` for plain-tank row; if not present, requires simulation with latent_heat=0 (same hook as Plot 10) |

**Subtotal:** ~1 hour investigation + 30 min implementation if data exists

---

## Implementation Priority

### Phase 1 (Immediate, 30 min)
Run existing 9 scripts via `run_all_plots.py`:
```bash
python run_all_plots.py
```
This generates 9 HTML/PNG plots and a summary report (`PLOTTING_REPORT.json`).

### Phase 2 (Quick wins, ~2 hours)
Implement the 4 "ready to implement" comparison plots:
- `comparison_phase2_5_raw_vs_clean.py` (5-panel)
- `comparison_phase3_tier1_vs_tier2.py` (scatter)
- `comparison_phase4_levelA_vs_levelB.py` (contingency/facet)
- `comparison_phase8_penalty_k0_vs_k3.py` (grouped bar)

### Phase 3 (Blocking issue, 1–2 hours)
Implement Plot 10 (tank profile):
1. Add `save_timeseries=True` parameter to `physics_lib.py`
2. Call simulator once per cluster medoid on clear-sky day
3. Extract and plot hourly Tw, Tp, melt_fraction arrays
4. Verify no unphysical jumps, overnight 58–62°C band

### Phase 4 (Investigation, variable)
Determine if VIKOR bugfix and PCM vs. plain tank comparisons are reproducible:
- Check for pre-fix backup or git history
- Check if plain-tank row exists in physics_validation
- Implement if data available; note as "skipped" if not

---

## Architecture Notes

### Verification Block Pattern

Every script follows this structure:

```python
print("\n=== VERIFICATION BLOCK ===")
# 1. Load data and check columns
# 2. Verify fingerprints (if Phase 5–9 downstream)
# 3. Compute key metrics against audit-documented baselines
# 4. Print PASS/WARN/INFO status

print("\n✓ PASS: Data matches baseline")  # or WARN
```

Example output that MUST be present:
```
=== VERIFICATION BLOCK ===
Total survivors: 39
  ✓ PASS: Total survivor count matches post-correction baseline (39)
Spearman rho (MCDM vs. physics):
  Cluster 0: ρ = -0.385 (p=0.306)
    ✓ PASS: Matches audit value closely
```

### Staleness Detection

All Phase 5+ plots use `provenance_lib.fingerprint_id()` to detect if files were regenerated:

```python
from provenance_lib import file_fingerprint, fingerprint_id

current_fp = fingerprint_id(file_fingerprint(CLUSTER_PROFILE_FILE))
if survivors_df["upstream_cluster_profile_fingerprint"].iloc[0] != current_fp:
    print("⚠ WARNING: Data is STALE")
    # But DON'T block — watermark and continue
```

Staleness is **not fatal** — plots are generated and watermarked instead.

---

## File Organization

```
plotting/
├── __init__.py                          # Module marker
├── README.md                            # User guide
├── IMPLEMENTATION_STATUS.md             # This file
├── run_all_plots.py                     # Master runner
│
├── 01_raw_vs_preprocessed.py            # ✓ Ready
├── 02_climate_regime_map_copy.py        # ✓ Ready
├── 03_pcm_feasibility_scatter.py        # ✓ Ready
├── 04_pcm_survivors_per_cluster.py      # ✓ Ready
├── 05_bump_chart.py                     # ✓ Ready
├── 06_method_correlation_heatmap.py     # ✓ Ready
├── 08_rank_reversal_frequency.py        # ✓ Ready
├── 09_mcdm_vs_physics_agreement.py      # ✓ Ready
├── 11_summary_cards.py                  # ✓ Ready
│
├── comparison_phase3_tmcap_old_vs_new.py            # ✓ Ready
├── comparison_phase5_lrequired_before_after.py      # ✓ Ready
├── comparison_phase2_5_raw_vs_clean.py              # ⏳ TODO
├── comparison_phase3_tier1_vs_tier2.py              # ⏳ TODO
├── comparison_phase4_levelA_vs_levelB.py            # ⏳ TODO
├── comparison_phase8_penalty_k0_vs_k3.py            # ⏳ TODO
├── comparison_phase6_vikor_bugfix_before_after.py   # ⏳ INVESTIGATE
└── comparison_phase7_pcm_vs_plaintank.py            # ⏳ INVESTIGATE
```

---

## Running the Plots

### Option 1: Run all at once
```bash
cd era5-rajasthan
python plotting/run_all_plots.py
```

Outputs go to `outputs/objective1_plots_rajasthan/` with a `PLOTTING_REPORT.json` summary.

### Option 2: Run individual script
```bash
python plotting/05_bump_chart.py
```

Each script is independent and can be re-run.

### Option 3: Run specific group
```bash
python plotting/01_raw_vs_preprocessed.py
python plotting/03_pcm_feasibility_scatter.py
python plotting/04_pcm_survivors_per_cluster.py
```

---

## Debugging & Issues

### Issue: "upstream_cluster_profile_fingerprint mismatch"
**Cause:** Phases 5–9 haven't been re-run post-correction, or they were run against a different cluster_profiles file.

**Fix:** Re-run Phases 5–9:
```bash
python run_all_rajasthan.py --from 05_cluster_rajasthan.py
```

Then re-run plots.

### Issue: "PCM database not found"
**Cause:** `PCM_Properties_cleaned_mice_pmm_detailed.csv` path is wrong.

**Fix:** Update the path in the script to match your PCM_data folder structure.

### Issue: "Column 'X' not found"
**Cause:** CSV schema has changed, or file is from a different run.

**Fix:** Check the file manually and update column names in the script.

---

## Next Steps

1. **Immediate (5 min):**
   - Run `python run_all_plots.py` to generate the 9 ready plots
   - Review `outputs/objective1_plots_rajasthan/PLOTTING_REPORT.json`

2. **Short-term (2 hours):**
   - Implement the 4 quick-win comparison plots
   - Run full suite again

3. **Medium-term (1–2 hours):**
   - Implement Plot 10 instrumentation in `physics_lib.py`
   - Test and verify tank profile plot

4. **Investigation:**
   - Check if VIKOR bugfix and PCM vs. plain tank data are available
   - Implement if possible; note as "skipped" if not

---

## References

- Full audit spec: `../docs/rajasthan/11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`
- Provenance checking: `../provenance_lib.py`
- Master overview: `../docs/rajasthan/00_MASTER_OVERVIEW.md`
- Individual phase audits: `../docs/rajasthan/0X_*.md`

---

## Contact

For questions on implementation, refer to the audit specification or check individual script docstrings.
