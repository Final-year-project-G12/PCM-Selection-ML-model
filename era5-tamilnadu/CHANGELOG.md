# CHANGELOG — Objective 1, Tamil Nadu Pipeline

Everything changed in response to the two review documents
(`Objective1_TamilNadu_STATUS_AND_TODO (2).md` and `FIXES.md`). Grouped by
file. "Was" = what the earlier version did; "Now" = what changed.

---

## v3.1 Bug Fixes (August 2026 — critical correctness)

### `02_combine_tamilnadu.py`
- **Deaccumulation bug fixed.** Was: `deaccumulate()` with `pd.Series.diff()` corrupted GHI (noon r ≈ 0.40). Now: `accum_to_flux(s) = s.clip(lower=0)` — matches Rajasthan fix.

### `04_preprocess_tamilnadu.py`
- **Quantile mapping added (Step 2b).** Per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution. Saves `ghi_quantile_mapping_report.csv`.

### `03b_agreement_analysis.py` (NEW)
- Cross-source validation decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW). Outputs `era5_power_agreement_tamilnadu.csv`, scatter HTML, `bias_decision_tamilnadu.txt`.

### `04b_climate_signature.py` (already fixed in prior round)
- **1000× flow rate bug fixed.** Now uses `DRAW_VOLUME_L = 300` (Avargani et al. 2021).

### `11_level_b_seasonal_analysis.py`
- **Draw volume aligned with 04b.** Seasonal `L_required` now uses 300 L/day formula (was still using buggy `DRAW_RATE_KG_PER_S`).

### `05_cluster_tamilnadu.py` (already fixed in prior round)
- **GMM covariance fixed.** `covariance_type="diag"` (was `"full"`).

### `10_physics_validation.py` (already fixed in prior round)
- **Tank ambient heat loss added.** `UA_TANK_W_K = 2.0 W/K`.

### `config.py`
- Added `OUTPUTS_DIR` for agreement analysis outputs.

### All `docs/era5_tamilnadu/*.md` files
- Updated from "known issues" to "corrected (v3.1)" status.
- Added Literature Support sections referencing `sources/` summaries from `sources.zip`.

**Re-run required**: `02_combine` → full downstream chain for scientifically valid outputs.

---

## Bug fixes (correctness, not new features)

### `02b_build_daily_aggregates.py`
- **HDD18/CDD24 annualization.** Was: summed over the full 10-year record
  (so "HDD18" was ~10x a real annual figure). Now: divided by the number
  of distinct years actually present in each point's usable-day set.
- **CCI (consecutive-cloudy-day run) gap bridging.** Was: a dropped day
  (< 20/24 hours of NASA POWER coverage) could silently let two separate
  cloudy runs be counted as one continuous run. Now: any calendar-date
  gap > 1 day forces a run break before the max-run calculation. Also now
  reports `n_date_gaps_gt1day` per point and flags points with >20 gaps.

### `04b_climate_signature.py`
- **Same HDD18/CDD24 annualization bug**, same fix, applied to the Tier-1
  sun-event proxy version (`HDD18_proxy`/`CDD24_proxy`).

---

## Feature completions (things that were honestly flagged as not-yet-done, now done)

### `07_feasibility_filter.py`
- **Corrosion veto** (Table 12, filter 6) — now implemented: excludes any
  `corrosion_class == "check_manually"` PCM in a cluster whose HSI sits
  above the 75th percentile across all clusters. Currently a near-no-op
  given your mostly-organic 25-row database (only one inorganic
  candidate is flagged `"check_manually"`) — becomes load-bearing once
  you add real salt hydrates or extend to a more humid state.
- **Safety exclusion** (Table 12, filter 7) — now implemented: keyword
  veto against the flammability field (`"highly/extremely flammable"`,
  `"toxic"`). Also currently a no-op given your data (paraffins/fatty
  acids are "combustible," not "highly flammable" in standard hazard
  classification) — the mechanism is real, just unused by current rows.
- Docstring updated to reflect 7/8 Table 12 filters now implemented (only
  the true 5th-percentile-insolation charging-feasibility filter remains
  unimplemented in its literal form — `07b_charging_feasibility.py`'s
  heuristic is the closest available substitute, and Phase 7's simulated
  performance now supersedes the need for it in practice).

### `08_mcdm_ranking.py` (v2 — substantial rewrite)
- **Added PROMETHEE II** — net outranking flow with V-shape preference
  function, q=0.10/p=0.30 (fraction of the normalized [0,1] criterion
  range — documented simplification, see script docstring).
- **Added VIKOR** — compromise ranking Q_i (v=0.5), plus the standard
  acceptable-advantage/acceptable-stability check (flags when a single
  "winner" isn't statistically distinct and a compromise set should be
  reported instead).
- **Added 5,000-draw Monte Carlo stability analysis** (plan v3.0 Section
  9.6) — Dirichlet-perturbed weights + Gaussian-perturbed PCM properties
  (Tm ±1K, latent heat ±5%, conductivity ±10%), reporting per-PCM Top-3
  inclusion probability, Top-1 retention rate, and mean Spearman rho vs.
  the unperturbed baseline ranking.
- **Consensus upgraded** from 2-method Borda to 4-method Borda, with
  Copeland pairwise-majority computed as an explicit cross-check —
  disagreement between the two is now flagged in output rather than
  silently resolved.
- Kendall's W now computed across all 4 methods (was 2).

### `09_recommendation_cards.py` (v2)
- Now includes a **Phase 7 physics validation** table per cluster
  (simulated annual solar fraction, benchmark-band flag, complete
  cycles/year) and the per-cluster Spearman rho, when
  `10_physics_validation.py` has been run. Falls back gracefully (with a
  clear note) if it hasn't.
- Top-3 table now shows all 4 methods' scores + Monte Carlo Top-3%, not
  just TOPSIS/GRA.

---

## New scripts

### `10_physics_validation.py` — Phase 7, NOT deferred to future work
Grey-box lumped-enthalpy PCM tank model (3-phase: pre-melt sensible,
isothermal melting, post-melt sensible — adapted from Barqawi2025's ODE
structure, already in your literature summaries), solved with backward
Euler (implicit, unconditionally stable — needed because the tank's
thermal time constant here is short relative to an hourly step). Driven
by each cluster's medoid point's **real** 10-year daily GHI/temperature
data (`daily_aggregates_tamilnadu.csv` from `02b`) for one representative
year, not synthetic weather. Simulates every feasibility survivor per
cluster, computes annual solar fraction, checks it against the plan's
54-84% published benchmark band (Table 16), and computes Spearman rho
between the MCDM consensus rank and simulated performance per cluster
(Table 17's three-outcome interpretation — all three are publishable if
diagnosed, which the script's output does for you).

All tank/collector parameters (mass, coil area, HTC, collector
efficiency, draw schedule) are stated assumptions with literature
citations in the docstring — exactly the same honesty standard as every
other assumption already in this pipeline. This is a genuine grey-box
model, not a toy — but it is still a simplified lumped model, per the
plan's own explicit permission ("a crude model honestly described beats
an elaborate one that is wrong").

### `11_level_b_seasonal_analysis.py` — Phase 4, Level B (seasonal sensitivity)
The "nearly free" addition the plan calls out specifically for Tamil
Nadu's out-of-phase north-east monsoon. For each existing Level-A
cluster, recomputes L_required per season (Ta_mean varies seasonally;
Tm_target stays constant per the plan's rule) and re-ranks with a
single-method TOPSIS (using the SAME weights as the annual ranking, for a
fair comparison) per (cluster, season). Reports whether the #1 PCM flips
between seasons — a flip is direct empirical motivation for Objective 3's
adaptive controller, generated from your own data; no flip is also a
valid, reportable finding (the Tm_target rule is robust to seasonal
swings). This is the "nearly free" version explicitly permitted by the
plan, not the full independent-per-season-GMM-clustering version — say
so in your methodology.

---

## Still open (unchanged from the review — not addressed this round, by design)

1. **PCM database at ~25/40-60 rows.** Real gap, self-flagged since the
   first version of `06_build_pcm_database.py`. Add RT58/RT60/RT62HC,
   PLUSS OM55/OM65, and a properly-cited salt hydrate if you have time —
   nothing fabricated in this pipeline, so this stays a coverage gap
   until you source real values.
2. **External cluster validation** (ARI vs. Köppen-Geiger / NBC-ECBC
   zones, plan v3.0 Section 7.5) — not implemented. Needs an external
   Köppen/NBC shapefile or lookup table joined to your 133 points, which
   this pipeline doesn't currently have. Lower priority for a TN-only
   scope per both reviews, but the step the plan calls out as "what earns
   credibility" for the clustering — add if you extend to more states.
3. **Elevation** — still the flat 150m proxy. Both reviews agree this is
   fine for Tamil Nadu's gentle terrain; only becomes non-optional for
   Uttarakhand's 200m-7000m range.
4. **`monsoon_index` stays proxy-only** — NASA POWER precipitation was
   never downloaded (see `02b`'s docstring). Unchanged, documented.

---

## What to actually run, in order, for a complete Objective 1

```
python 06_build_pcm_database.py        # (only if you haven't — confirm INPUT_CSV)
python 07_feasibility_filter.py         # now with corrosion + safety filters
python 08_mcdm_ranking.py               # now full 4-method + Monte Carlo (~5000 draws — allow a minute or two)
python 09_recommendation_cards.py       # now includes physics validation section
python 10_physics_validation.py         # Phase 7, run BEFORE the 09 above if you want it in the cards
python 11_level_b_seasonal_analysis.py  # optional but recommended — TN's monsoon story
```

Note the ordering nuance: run `10` before the final `09` if you want
physics results baked into `recommendation_cards.md` — `09` checks for
`10`'s output files and includes them automatically if present, so you
can also just run `09` again after `10` to regenerate the cards with the
physics section added.
