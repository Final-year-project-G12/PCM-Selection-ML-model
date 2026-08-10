# 09 — Phase 8 Audit: Recommendation Cards (and the Absence of Phase 7)

True script: `09_recommendation_cards.py` (disk: `config (3).py`).

## Phase 7 (Physics-Based Validation) does not exist in this pipeline

No grey-box thermal model, no calibration-benchmark code, no Spearman-rho-vs-MCDM-rank comparison —
confirmed absent by the full read of every script in this folder. This is **not a hidden gap**: the
project's own status document (`NEXT_STEPS.md`, true content of the disk file `01_preprocess
(1).py`) names it explicitly as accepted future work, and `FIXES.md` independently states, quoting
the framework doc: *"Do not skip this... it is the difference between an undergraduate exercise and
a publishable result."* Recommendation for this pipeline: mirror `docs/era5_rajasthan/19_PHASE_7_ONWARD.md`'s
already-detailed Phase 7 specification (same grey-box lumped-enthalpy model, same Avargani et al.
2021 calibration benchmarks — directly reusable since both pipelines share the same design basis)
once Tamil Nadu's Phase 5/6 outputs are real and corrected.

## Phase 8 — recommendation card generator

Confirmed a pure aggregation script: its own docstring states plainly, *"computes nothing new."*

### Inputs
`cluster_profiles_tamilnadu.csv`, `cluster_assignments_tamilnadu.csv`, `mcdm_topk_by_cluster.csv`,
`feasibility_survivors_by_cluster.csv` — all four required, existence-guarded (prints error and
returns if any missing, no silent partial-card generation).

### Card schema (per cluster, exact)
Cluster ID heading; point count; population covered (if present); approximate medoid point (nearest
member to the cluster's mean lat/lon, only if lat/lon columns exist); a climate-signature table over
11 named indices (`GHI_daily_kWh, Ta_mean, DTR, kt_mean, cloudy_frac, CCI, HDD18, CDD24, RH_mean,
HSI, monsoon_index`); derived targets line (`Tm_target`, `L_required`); candidates-screened count;
a Top-3 PCM table (Rank/PCM/Family/Tm/Latent heat/TOPSIS/GRA) if any survivors were ranked, else an
explicit fallback message recommending database expansion or window relaxation; a three-tier
Kendall's W agreement note (≥0.8 strong / ≥0.6 moderate-discuss / else weak-ambiguous); and a fixed
caveats paragraph noting missing thermal-conductivity/density/Cp for literature-added candidates and
partial constraint coverage, with a pointer to `07_feasibility_filter.py`'s docstring.

### Assessment
Well-designed, directly comparable in spirit to Rajasthan's specified (but also not-yet-built) Phase
8 card schema — the explicit fallback for under-survived clusters and the honest three-tier Kendall's
W labeling are both good, defensible design choices. No methodological content of its own to
critique (by design) — its output quality is entirely a function of Phases 3–6's correctness, which
means **this script should not be run to produce a "final" card set until the Phase 3 `L_required`
fix is applied and the pipeline is executed for real.**

## Literature support

None needed for this phase (pure aggregation).

## Validation

None possible yet — no `recommendation_cards.md` exists.

## Dependencies

Requires all of Phases 4–6's outputs, none of which currently exist.

## Status

**Phase 7: NOT IMPLEMENTED, explicitly deferred as future work (self-documented, not a hidden gap).
Phase 8: CODE COMPLETE, NEVER RUN, no methodological concerns of its own — entirely gated on
upstream phases being fixed and executed first.**
