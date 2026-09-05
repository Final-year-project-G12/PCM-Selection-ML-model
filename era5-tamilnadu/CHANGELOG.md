# CHANGELOG — Objective 1, Tamil Nadu Pipeline

Everything changed in response to the two review documents
(`Objective1_TamilNadu_STATUS_AND_TODO (2).md` and `FIXES.md`). Grouped by
file. "Was" = what the earlier version did; "Now" = what changed.

---

## v3.2 Bug Fixes (Phase 7 physics solver — critical correctness)

Found during a cross-check of the Tamil Nadu pipeline against the
Rajasthan pipeline's documented, already-fixed bug history. Both bugs
below are the *same bug classes* Rajasthan's `physics_lib.py` audit
already names and fixes — this pipeline had independently reintroduced
them. v3.1 fixed the "no ambient loss" symptom, but the tank still
never actually cooled overnight because of these two solver bugs, so
Phase 7's output was still stuck at the pre-v3.1 failure signature
(85–100% solar fraction, 0–1 cycles/year, 0% in the 54–84% benchmark
band) even after v3.1 shipped.

### `10_physics_validation.py` — backward-Euler closed-form solve bug
- **Was**: `Tw_new` in the pre-melt (phase 1) and post-melt (phase 3)
  sensible branches was solved with numerator
  `(Tw + dt*a*tc + loss_coeff*tamb)*(1+dt*c) + dt*b*(Tp + dt*c*Tw)` —
  the trailing `dt*c*Tw` inside the PCM-coupling term is spurious; it
  does not appear when the 2×2 implicit system is solved algebraically
  (the correct numerator uses the *old* `Tp` alone: `dt*b*Tp`). This is
  the identical bug class the Rajasthan audit documents as "a wrong
  closed-form backward-Euler solve... caused unbounded temperature
  blow-up." Verified numerically with the script's own default
  parameters: the buggy formula pushed `Tw_new` to 69.2°C in a single
  step from a 45°C collector with no other heat source — thermodynamically
  impossible for this passive linear coupling. The corrected formula
  gives 44.5°C for the same inputs.
- **Fix applied**: numerator corrected to use `dt*b*Tp` (old `Tp` only)
  in both the phase-1 and phase-3 branches.
- **Observed effect**: every simulated PCM before this fix landed at
  85–100% annual solar fraction (0% within the 54–84% benchmark band,
  0–1 complete cycles/year) — the same "tank never actually discharges"
  signature the v3.1 ambient-loss fix was supposed to prevent. After
  this fix alone (before the night-isolation fix below), 10% of runs
  fell in-band.

### `10_physics_validation.py` — missing night/idle collector-coupling isolation
- **Was**: the collector-tank coupling coefficient `a` was applied
  identically day and night. At night the collector temperature `Tc`
  collapses to ambient (`isolar = 0`), so an un-isolated `a*(Tc-Tw)`
  term drains the tank back out through the idle collector loop at
  essentially the same rate it charges during the day — on top of the
  separate `UA_TANK_W_K` ambient-loss term, double-counting overnight
  losses. This is the second bug class from the same Rajasthan audit
  ("Barqawi's bidirectional a·(Tc−Tw) term let the tank drain heat
  through an idle collector overnight nearly as fast as it charged
  during the day").
- **Fix applied**: added `NIGHT_ISOLATION_FRACTION = 0.05`; the
  collector-coupling coefficient is gated to 5% of its daytime value
  whenever `Tc < Tw` (collector colder than tank), matching Rajasthan's
  fix exactly. Only the collector coupling is gated — the PCM-tank
  coupling `b` is an internal exchange, not a valved external loop, and
  is left untouched.
- **Observed effect (both fixes together)**: solar fractions now spread
  physically across roughly 20–80% (not pinned at 85–100%), complete
  cycles/year moved from 0–1 to tens/hundreds (physically plausible PCM
  freeze-melt cycling), and 41% of simulations now fall within the
  54–84% benchmark band (up from 0%). Mean Spearman ρ across clusters
  moved from **-0.151** to **+0.177** — still a weak-agreement, honestly
  reportable finding (not a data-fabrication target), but no longer an
  artifact of a broken solver. **Re-run required**: `10_physics_validation.py`
  → `09_recommendation_cards.py` (both already re-run to produce the
  current on-disk artifacts as of this fix).
- Remaining gap, explicitly not fixed here (a parameter-calibration
  question, not a bug): 59% of simulations still fall outside the
  54–84% band, split between above and below it depending on cluster —
  the tank/collector parameters (`M_W_KG`, `A_C_M2`, `COLLECTOR_EFF`,
  draw schedule) are stated literature-anchored assumptions, not
  empirically fit to this pipeline's own points, and calibrating them
  further would need real deployment data or a decision to match
  Rajasthan's own calibrated values — do not further hand-tune them
  just to force more runs into the benchmark band.

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
5. **K_FINAL=5 is hand-set, not selected by the Rajasthan-style tiered
   rule.** `05_cluster_tamilnadu.py` reports BIC/silhouette/Davies-Bouldin/
   Calinski-Harabasz for k=2..10 but does not compute bootstrap-ARI
   stability, so there is no data-driven tie-break when several k values
   sit in the accepted silhouette band. In the current run, k=6
   (silhouette 0.305) and k=9 (0.312) both score higher than the
   hard-coded k=5 (0.262) within that band. This is flagged, not
   changed, here — re-clustering at a different k would cascade through
   every downstream phase (feasibility, MCDM, physics, cards) and change
   the headline per-regime recommendations, which is a scientific
   decision for the project owner, not something to silently redo.
   Add a bootstrap-ARI pass (resample the 133 points with replacement,
   refit GMM, compare via Adjusted Rand Index against the full-data
   labels, repeat ~50x) if you want the same rigor Rajasthan's audit
   applied before finalizing k.
6. **No cross-phase provenance/fingerprint check.** Rajasthan's pipeline
   hard-fails (`SystemExit`) if Phase 6/7/8's input `cluster_profiles`
   doesn't match what's currently on disk, because sklearn's
   `GaussianMixture` cluster-index order is not guaranteed stable across
   separate re-runs. The Tamil Nadu scripts have no equivalent check —
   low risk today (nothing here indicates it has actually caused a
   mismatch), but a real gap if `05_cluster_tamilnadu.py` is ever re-run
   with different data/parameters without also re-running 07→10 in the
   same pass.
7. **MCDM criteria set is reduced to 5, not the framework doc's 8.**
   `08_mcdm_ranking.py` ranks only on Tm-fitness, latent heat (climate-
   relative), volumetric latent heat, thermal conductivity, and cycling
   confidence — `cost`, `corrosion`, and `supercooling` are dropped
   entirely rather than carried as always-near-zero-weight criteria the
   way Rajasthan does. This is a documented, deliberate scope reduction
   (the database has no real cost data and only one corrosion-relevant
   candidate), not an error, but it also means Rajasthan's dominant
   "supercooling drives 48–64% of the entropy weight and the physics
   model can't simulate it" finding cannot recur here — a different,
   narrower set of caveats applies to this pipeline's MCDM/physics
   disagreement instead.
8. **Absolute latent-heat floor (`LATENT_HEAT_ABSOLUTE_MIN_KJ_KG = 100`)
   is a Tamil-Nadu-only addition beyond the framework doc's Table 12**,
   which specifies only the relative `L ≥ 0.7 × L_required` rule. Given
   `L_required` is ~301–326 kJ/kg here, `0.7 × L_required` (≈211–228
   kJ/kg) already exceeds the 100 kJ/kg absolute floor, so this addition
   is currently a no-op — but it should be named explicitly as a
   deviation from the literal spec if the write-up quotes Table 12
   verbatim.

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
