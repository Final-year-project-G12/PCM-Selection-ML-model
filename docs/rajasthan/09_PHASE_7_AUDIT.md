# 09 — Phase 7 Audit: Physics-Based Validation of MCDM Rankings

> **This doc keeps the number 09 for continuity, but the SCRIPT was renumbered
> 2026-09-08 to `10_physics_validation.py`** (matching Tamil Nadu: physics = 10,
> recommendation cards = 09). References to `10_physics_validation.py`
> below mean `10_physics_validation.py`.

Script: `10_physics_validation.py` (was `10_physics_validation.py`). Phase 8
(`08_phase8_supercooling_sweep.py`) extends this with supercooling-penalty sensitivity testing.

⚠️ **NEWEST UPDATE (2026-09-19), re-run COMPLETE — supersedes every `0.105 / -0.190 / -0.091` and
`Tm_fitness (51.9/83.3/68.1%)` figure below.** `Tm_fitness` was fixed to score against each
cluster's achievable `Tm_target_capped_C` instead of the raw `Tm_target_C`. **Current on-disk
numbers** (per `outputs/recommendation_cards_rajasthan.md` and
`physics_validation_summary_rajasthan.txt`):
- Spearman rho (Borda vs. simulated SF): **-0.200 / -0.168 / +0.569** (clusters 0/1/2),
  p = 0.800/0.691/0.067. Clusters 0–1 remain NEGATIVE (rho≤0.4); Cluster 2 is now **PARTIAL**
  (0.4<rho≤0.8) — the first cluster in this project's Phase 7 history to clear the 0.4 threshold.
- Kendall's W: **1.0000 / 0.7679 / 0.4170** (not 0.900/0.750/0.555 as older sections below state).
- Dominant entropy-weighted criterion is **NOT uniformly Tm_fitness**: `vol_latent_heat` (47.5%,
  Cluster 0), `Tm_fitness` (80.5%, Cluster 1), `thermal_conductivity` (41.4%, Cluster 2).
- Seasonal sensitivity re-run the same day: **8/11 (cluster, season) cells now flip** the annual
  #1 pick — supersedes the "0/11 flip, seasonally robust" finding in the 2026-09-13 section below.
See the "UPDATE (2026-09-19)" section further down for the full before/after table and
`08_PHASE_6_AUDIT.md` for the MCDM-side detail. Everything between here and that section
(the 2026-09-13 banners, the historical ρ table, "Dominant Entropy-Weighted Criterion", and
"Cluster-Specific Interpretations") predates this fix and should be read as history, not current
state.

⚠️ **UNIFICATION UPDATE (2026-09-08): Phase 6 unified with Tamil Nadu — this Phase 7 result set is
STALE, re-run pending.** The unified `08_mcdm_ranking.py` now **caps supercooling's entropy weight
at 2× its Table-13 prior (0.16)**, directly addressing the overweighting this Phase 7/8 pair
diagnosed. The "supercooling dominates at 48–64% but the model can't simulate it" finding below was
the *evidence* for that cap; after the cap, supercooling blends to ≈0.12–0.16, and Tm_fitness is the
dominant criterion. Re-run `10_physics_validation.py` against the new `mcdm_full_rankings.csv` — the
post-cap Spearman rho vs the pre-cap `-0.385 / +0.125 / -0.097` is the actual test of whether the
fix helped.

⚠️ **CRITICAL UPDATE (2026-08-31): L_required Methodology Correction** — also folded into the pending
re-run. Phase 3's L_required uses SHARE_PCM=0.5 (combined sensible+latent), halving it. See
CLAUDE.md §3.1.

✅ **DELIVERY TEMPERATURE CORRECTION (2026-09-13): re-run complete, this doc's numbers above are
superseded again.** `T_DELIVERY_C` (used both in Phase 3's `Tm_target_C`/`L_required` and directly
inside `physics_lib.py`'s grey-box simulator as the delivery-success threshold) was `50.0°C`;
corrected to `60.0°C` to match Avargani et al. (2021)'s actual validated delivery temperature for
the cited 300 L/7h figure — see `05_PHASE_3_AUDIT.md`. Full pipeline re-run, **current on-disk
results:**

- Calibration medoid solar fractions: **58.1% / 59.9% / 57.9%** (clusters 0/1/2, RT47 calibration
  PCM) — down from the pre-fix ~64.9%/65.8%/63.8%, since the simulator now targets a harder-to-hit
  60°C delivery threshold. Still 100% inside the 54–84% benchmark band (target ~69%).
- PCM-mass sensitivity sweep: ranking of the top-5 candidates is **no longer stable across
  50–800 kg PCM mass** (was stable pre-fix) — at 400 kg the #3/#4 positions (RT50, RT45HC) swap
  order. Report `hours_target_met_per_year` or `mean_melt_fraction` as a supplementary rank target
  per this script's own printed conclusion, rather than treating 50 kg solar-fraction ranking alone
  as fully discriminating.
- Per-cluster Spearman rho (MCDM Borda rank vs. simulated solar-fraction rank): **0.105 / -0.095 /
  -0.091** (clusters 0/1/2) — cluster 0's reading now sits on an n=4 undersized candidate pool
  (Phase 5's raised L_required ceiling shrank it from 9 to 4 survivors), so read it with that
  caveat rather than as a like-for-like comparison against clusters 1/2's healthy pools.
- See `physics_validation_summary_rajasthan.txt` for the full caveat-aware per-cluster
  interpretation and `outputs/recommendation_cards_rajasthan.md` for the aggregated Phase 8 view.

✅ **TANK MASS DECOUPLED FROM AVARGANI + RE-CALIBRATED (2026-09-13, same pass as the delivery-
temperature fix above, re-run complete).** `M_W_KG` (the simulator's static tank water mass) had
been set to 300.0 kg as "the same volume as Avargani's 300L design basis" — but re-reading
Avargani et al. (2021) directly confirms their system is continuous-FLOW (water passes through
the collector and PCM bed while a draw is ongoing; there is no static tank in the sense `M_W_KG`
represents), so their 300L/7h figure is cumulative throughput, not a tank's resting capacity — a
different physical quantity that happened to share a number with this pipeline's own
`NIGHT_DRAW_TOTAL_L`. Re-derived from Eldokaishi et al. (2022)'s hybrid-PCM-SWH design study
(building on Abdelsalam et al. 2020, already cited in this pipeline's SHARE_PCM basis): a
literature-validated 50–240 L tank-volume range for a 1–8 m² collector, with an explicit A_c=4 m²
design case matching this script's own `A_C_M2` exactly. **`M_W_KG` corrected 300→200 kg**
(comfortably above this pipeline's own 46.3 L/hour peak draw — a naive 40–50L guess would not have
been). Dropping `M_W_KG` alone pushed calibration solar fraction out of the 54–84% band (45–52%);
re-calibrated by jointly sweeping `COLLECTOR_UL_WM2K` (2.5→2.0 W/m²K) and
`NIGHT_ISOLATION_FRACTION` (0.05→0.03), selecting the smallest joint adjustment that restored all
3 medoids in-band. **Current results:**
- Calibration medoid solar fractions: **64.3% / 65.8% / 64.0%** (clusters 0/1/2) — closer to the
  69% target than the pre-recalibration T_DELIVERY_C-only numbers (58.1/59.9/57.9%).
- 5-candidate spread at PCM_MASS_KG=50 kg: **~1.25 pp** (was ~0.9pp at the original M_W_KG=300 —
  differentiation did not degrade from the smaller tank).
- Energy-conservation self-test still passes (residual fraction ~9e-14, threshold 1e-3).
- Per-cluster Spearman rho: **0.105 / -0.190 / -0.091** (clusters 0/1/2) — still negative-band;
  cluster 1's magnitude grew slightly (was -0.095).
- PCM-vs-plain-tank comparator: still ~0.0% (tank-dominated finding unchanged in kind — M_W_KG=200
  is still 4x PCM_MASS_KG=50).
See `physics_lib.py`'s CALIBRATION docstring section (correction #4) for the full sweep numbers
and citation detail.

⚠️➡️✅ **DOWNSTREAM CONSEQUENCE, THEN FIXED (2026-09-13) — Level B seasonal PCM sensitivity was
DEGENERATE, root cause found and corrected.** The raised `L_required` (from the T_DELIVERY_C fix)
pushed seasonal `L_required` to 375–556 kJ/kg — but the REAL cause of "< 2 survivors" everywhere
was a separate bug: `11_seasonal_pcm_sensitivity.py` re-filtered each cluster's already-calibrated
survivor pool using a hardcoded `LATENT_HEAT_FRACTION=0.7`, stricter than every cluster's own Phase
5 calibrated kappa (0.0/0.5/0.3) and inconsistent with the pool it drew from. **Fixed:** the
seasonal floor now uses each cluster's own `calibrated_kappa` instead of a shared fixed 0.7.

**Result (2026-09-13 state, SUPERSEDED 2026-09-19 — see below): no longer degenerate.** At the time
of this fix, all 11 (cluster, season) cells had healthy survivor counts (4–11) and ranked cleanly —
**0/11 flip from the annual #1 pick** (was the "6-7/9 flip" finding pre-Fix-1, briefly "0/0/0 empty
tables" between Fix 1 and this correction). This was read as Rajasthan's delivery-temperature-
anchored `Tm_target` rule being seasonally robust.

**⚠️ This "0/11 flip, seasonally robust" reading did not survive the next fix.** The 2026-09-19
`Tm_fitness` scoring correction (scored against `Tm_target_capped_C` instead of raw `Tm_target_C`)
changed Cluster 0 and Cluster 2's annual Top-1 PCM, and a same-day re-run of
`11_seasonal_pcm_sensitivity.py` against the corrected Top-1 picks shows **8/11 (cluster, season)
cells now flip** (Cluster 0: 4/4, Cluster 1: 0/3 valid cells, Cluster 2: 4/4) — see
`outputs/seasonal_pcm_sensitivity_rajasthan.md`, the current on-disk result. **The "0/11,
seasonally robust" conclusion above is stale; 8/11 is current.**

**Re-examined against the DRL literature (2026-09-13): this null result is the RIGHT outcome for
Objective 1, and does not weaken Objective 3's case.** The original plan motivated O3's DRL
controller by "the optimal PCM ranking flips across seasons" — but that is not actually how this
class of paper motivates a learned controller. **Emami et al. (2025/2026)** (already in this
project's bibliography — `sources/Emami2026DRL_Solar_ORC_TES_summary.md`) motivates their DDPG
supervisory controller purely by real-time weather stochasticity: year-long solar irradiance
variability (transient/low-flux regimes, steep ramps up to ±100 W/m²/min) that a fixed-flow
passive baseline cannot track, with no PCM-material-selection instability involved at all. Heidari
et al. (2022) is reported (by the reviewing party, not yet independently verified against a source
extract in this repo — see caveat below) to motivate a comparable controller by stochastic
occupant hot-water-demand behavior plus weather variability, again with no PCM-ranking-flip
argument. **Neither precedent needs "the optimal material changes" to justify a learned
controller** — the controller's actual job is real-time charge/discharge/bypass decisions under
weather/demand variability that a fixed PCM choice and a fixed rule-based controller cannot react
to. That is a stronger and better-precedented justification for O3 than a PCM-ranking-flip story
would have been.

**Recommended framing for the paper:** "Objective 1's climate-region-aware PCM recommendation is
stable across seasons within a climatic regime (Section X) — the *material selection* decision
does not need to be adaptive. What DOES need to be adaptive is the *operating* decision: how much
to charge/discharge/bypass at a given hour, in response to real-time weather and demand
stochasticity the climate-cluster-level analysis cannot capture (Objective 3)." This cleanly
separates O1's job (climate-region-level material selection, done once per region) from O3's job
(hour-by-hour operational control, running continuously) — matching how Emami et al. and (reported)
Heidari et al. motivate their own controllers.

**Caveat — Heidari et al. (2022) is not yet a verified project source.** Per this project's own
grounding rule (CLAUDE.md §0), no claim from it should be cited in the actual paper until it has
been read directly and extracted to `sources/` — the characterization above is relayed from an
external reviewer's summary, not independently confirmed in this session. Emami et al., by
contrast, IS already an extracted project source and its framing above was checked directly against
that extract.

**Status:** not yet applied to the actual literature-review/methodology write-up — this is guidance
for that write-up, not a code or doc-numbers change. See `Objective1_Fixes_SourceVerified.md` Fix 6
for the full decision trail.

## UPDATE (2026-09-19): three fixes ported from Tamil Nadu, evaluated against this doc's own -0.190/0.105/-0.091-style baseline — full detail in `08_PHASE_6_AUDIT.md`, summarized here for the physics-validation side

**1. HDD18/CDD24 annualization fix** (`02b_build_daily_aggregates.py`) — verified pure rescale,
zero effect on anything downstream of clustering (survivor counts, rankings, this doc's ρ all
unaffected by this fix alone).

**2. `Tm_fitness` scored against `Tm_target_capped_C`, not raw `Tm_target_C`** — **NOT inert here**,
unlike Tamil Nadu (§2 above documents TN's identical fix having zero effect). Spearman ρ (MCDM
Borda rank vs. simulated solar fraction), before → after this fix (survivor pool unchanged, 4/8/11):

| Cluster | n | ρ before (raw target) | ρ after (capped target + asymmetric σ) | p (after) |
|---|---|---|---|---|
| 0 | 4 | +0.105 | **-0.200** | 0.800 |
| 1 | 8 | -0.190 | -0.168 | 0.691 |
| 2 | 11 | -0.091 | **+0.569** | 0.067 (vs. Copeland rank: ρ=0.633, p=0.036) |

Cluster 2 moved from a small negative reading into the best correlation seen anywhere in this
project's Phase 7 history (still not conventionally significant on the primary Borda comparison,
p=0.067, n=11 — but its Copeland-rank companion is nominally significant, p=0.036). Cluster 0
flipped sign in the other direction, more negative, but n=4 keeps this descriptive only (even
ρ=1.0 at n=4 gives two-sided p≈0.083 — no reading at this n could ever reach significance). None of
this should be read as "the fix improved validation" uniformly — it moved two of three clusters in
opposite directions, which is itself informative: the capped-target fix isn't neutral here the way
it was in Tamil Nadu, because Rajasthan's caps sit much further below the raw target than TN's do
(see `08_PHASE_6_AUDIT.md` for the numbers).

**3. `thermal_margin` 9th criterion, tested behind a flag, kept OFF.** Full per-cluster table in
`08_PHASE_6_AUDIT.md`. Summary: it made Cluster 0 (n=4) worse (ρ -0.200→-0.800, the opposite
direction from Tamil Nadu's Cluster 0 improvement), and made Cluster 2 (n=11) worse (ρ +0.569→+0.209,
dropping back below the 0.4 threshold it had just cleared without this criterion). Cluster 1 (n=8)
improved marginally but stayed ≤0.4 (-0.168→+0.096). Per the pre-agreed stopping rule — same one
Tamil Nadu's own investigation (§3-§5 below) used to close its case after one pass — this is not
iterated further. Top-1 with the criterion on is a different PCM in every cluster (Myristic
acid/NBR-1.0 / RT57HC / savE® OM55), so Rajasthan also does not reproduce TN's "same PCM in every
cluster" convergence pattern.

**Reconciled — 8/11 is the current, on-disk result** (confirmed against
`outputs/seasonal_pcm_sensitivity_rajasthan.md`): the seasonal-sensitivity "0/11 flip, seasonally
robust" finding documented earlier in this file (2026-09-13) was generated against the
pre-2026-09-19 Top-1 picks and is now stale. The same-day re-run of `11_seasonal_pcm_sensitivity.py`
(after the Tm_fitness capped-target fix changed Cluster 0 and Cluster 2's annual Top-1) shows
**8/11 (cluster, season) combinations flip** — the opposite of the "seasonally robust" conclusion.
**Open, not yet resolved in the write-up:** whether the O3-motivation reframing above (real-time
weather/demand variability, not PCM-ranking instability) still holds now that the seasonal-flip
finding is back, or whether the original "PCM ranking flips across seasons" motivation for O3
should be reinstated instead. This is a paper-framing decision, not a further code/data fix — flag
it before finalizing Objective 3's motivation section.

## Purpose

Phase 6 produces a consensus MCDM ranking (four methods, two aggregators, Monte Carlo stability). Phase 7 asks the critical question: **does a higher-MCDM-rank PCM actually deliver better simulated thermal performance under this cluster's real climate?** This validation makes the ranking falsifiable, not deferrable to future work.

## The Independent Check

- **Input**: Phase 6 MCDM rankings + Phase 5 feasibility survivors
- **Climate**: Real hourly NASA POWER weather for each cluster's medoid (2023–2025, whichever year is complete, <1% fill values)
- **Model**: Lumped-enthalpy grey-box simulator (Barqawi 2025, 3-phase PCM dynamics)
- **Output per PCM**: annual solar fraction, hours meeting delivery temperature, melt-fraction statistics, complete cycles
- **Correlation**: Spearman ρ between MCDM Borda rank and simulated solar fraction per cluster

## Model Class & Calibration (Critical Details)

### Why grey-box lumped, not EnergyPlus/CFD?

- EnergyPlus: no supported method to place a latent-heat PCM inside a tank node network
- CFD: overkill for single-objective PCM screening; lumped-enthalpy is appropriate fidelity for material selection
- This is a deliberate architectural decision, not an oversight

### Calibration findings (August 11, 2026)

Two bugs caught and fixed **during this script's own self-tests** (mandatory energy-conservation check):

1. **Backward-Euler solver bug**: Phase 1 closed-form Tw solve had spurious `+ dt·c·Tw_old` term, destabilizing at hourly timestep. Fixed by re-deriving algebraically; verified against numpy.linalg.solve to full float precision.

2. **Night-loss bug**: Barqawi's bidirectional `a·(Tc−Tw)` term let the tank drain heat through an idle collector overnight as fast as it charged during day. Real systems isolate the collector at night. **Fixed via NIGHT_ISOLATION_FRACTION = 0.05**, reducing collector coupling when Tc < Tw.

**Result after both fixes**: All three medoids land in 54–84% benchmark solar-fraction band (Phase 3's Avargani design basis). Phase 7 uses this calibrated model as-is.

### Assumptions (explicitly stated, not hidden)

| Parameter | Value | Justification |
|---|---|---|
| Tank water mass M_W | 300 kg | Avargani et al. (2021) design basis, reused throughout pipeline for consistency |
| Collector area A_c | 4.0 m² | Barqawi 2025 was unloaded (no household draw); sized up to 4.0 m² per Indian FPC sizing convention (~1.3–2 m²/100L of design draw) |
| Collector efficiency | 0.70 | Barqawi 2025; within 45–73% FPC band cited by Al-Mamun et al. 2023 |
| Collector overall loss U_L | 2.5 W/m²K | **Calibrated down from Barqawi's 20** — represents well-insulated collector; within Duffie–Beckman 3–8 W/m²K range |
| PCM–water HTC h_p base | 800 W/m²K | Barqawi 2025 |
| h_p scaling | By TC_solid / 0.2 | Deviation from Barqawi: allows thermal conductivity to differentiate candidates, not held fixed |
| PCM mass (fixed) | 50 kg | ASSUMED_PCM_MASS_KG from Phase 3/4; not independently optimized (each PCM gets same design, not co-optimized size) |
| Draw profile shape | Two Gaussians (morning ~07:00, evening ~19:00) | Informed by ASHRAE 90.2 Section 8.9.4 documented shape; exact hourly fractions are **reconstructed qualitatively**, not reproduced verbatim (exact table not retrievable) — flagged as reconstruction, not claim of exact reproduction |
| Daily draw total | 300 kg/day | Avargani et al. 2021; same citation as Phase 3 night-draw, but applied as full-day total here |
| Target delivery temp | 50°C | Pipeline-wide constant |

**⚠️ SUPERSEDED 2026-09-13 — this table reflects the original (pre-fix) calibration, not the
current on-disk state.** As already detailed in the "✅ DELIVERY TEMPERATURE CORRECTION" and "✅ TANK
MASS DECOUPLED..." banners near the top of this doc: `Target delivery temp` is now **60°C** (was
50°C); `Tank water mass M_W` is now **200 kg** (was 300 kg, wrongly reused from Avargani's
continuous-flow-through 300L/7h figure rather than a static tank capacity); `Collector overall loss
U_L` is now **2.0 W/m²K** (was 2.5, re-tuned as calibration — no longer independently
Duffie–Beckman-justified at this value); and `NIGHT_ISOLATION_FRACTION` is now **0.03** (was 0.05).
Current calibration medoid solar fractions: 64.3%/65.8%/64.0% (clusters 0/1/2). See
`physics_lib.py`'s CALIBRATION docstring for the full derivation and citation detail.

## Self-Tests: Both Pass

```
Energy conservation (constant solar, no draw, 48 hours):
  Residual: 1.638e-13 J  →  Pass (threshold: 0.1% of cumulative input)

Draw-profile integration (365 days):
  Daily total: 300.000 kg  →  Pass (expected 300.0 kg)
```

## Results: Per-Cluster Spearman ρ Against MCDM Borda Rank

**⚠️ HISTORICAL TABLE — the ρ values below are PRE-unification (raw, uncapped supercooling entropy
weight, n=9/15/17 survivors on the pre-2026-09-13 L_required basis). Kept as the diagnostic
baseline this audit was originally built against. For the CURRENT on-disk numbers (post-cap,
post-2026-09-13 T_DELIVERY_C/M_W_KG corrections, AND post-2026-09-19 Tm_fitness scoring fix,
n=4/8/11 survivors), see the "NEWEST UPDATE (2026-09-19)" banner at the top of this file:
rho = -0.200 / -0.168 / +0.569 (clusters 0/1/2), calibration SF = 64.0/65.8/64.3%. (The
0.105/-0.190/-0.091 reading that appeared here between 2026-09-13 and 2026-09-19 is itself now
superseded, not current.)**

| Cluster | n_candidates | Borda vs. Solar Fraction (PRE-cap, historical) | Notes |
|---------|---|---|---|
| **0** | 9 | **ρ = −0.385** | Weak negative. Kendall's W ≈ 0.34–0.39 (<0.6) — genuine method disagreement (GRA is the structural outlier in the unified run). |
| **1** | 15 | **ρ = +0.125** | Weak positive. Kendall's W ≈ 0.65 (moderate). |
| **2** | 17 | **ρ = −0.097** | Weak negative. Largest cluster. |

**Overall (pre-cap, historical) finding**: no cluster exceeds ρ=0.4. **Current (post-cap,
post-2026-09-13 corrections, AND post-2026-09-19 Tm_fitness fix) result: rho = -0.200 / -0.168 /
+0.569** — Clusters 0–1 still don't exceed ρ=0.4 (negative-band), but **Cluster 2 now does**
(0.4<ρ≤0.8, PARTIAL agreement) — the 2026-09-13 "still no cluster exceeds ρ=0.4" framing is no
longer accurate; see the "NEWEST UPDATE (2026-09-19)" banner at the top of this file.

## Dominant Entropy-Weighted Criterion Per Cluster

**PRE-unification Phase 6** identified supercooling as dominant (63.8% / 48.6% / 57.0%) — an
artifact of the Shannon-entropy formula overweighting a near-zero-ideal cost criterion.

**POST-unification (`08_mcdm_ranking.py` on-disk output, 2026-09-13 re-run, SUPERSEDED 2026-09-19)**
capped supercooling's entropy weight at 2× its Table-13 prior (0.16); it blended to ≈0.12–0.16, and
at that point `Tm_fitness` was the dominant criterion in every cluster — entropy weights 51.9%
(Cluster 0) / 83.3% (Cluster 1) / 68.1% (Cluster 2).

**CURRENT (post-2026-09-19 Tm_fitness scoring fix)**: the dominant criterion is **no longer
uniformly Tm_fitness** — `vol_latent_heat` (47.5%, Cluster 0), `Tm_fitness` (80.5%, Cluster 1),
`thermal_conductivity` (41.4%, Cluster 2), per `physics_validation_summary_rajasthan.txt`. The
"model can't simulate supercooling" caveat still holds for the residual supercooling weight, but
supercooling remains a minor driver of the MCDM ranking in every cluster. Phase 8's penalty sweep —
which *worsened* physics/MCDM agreement as k rose in every cluster, across all three corrections —
is the empirical evidence that supercooling was over-weighted pre-cap, i.e. that the cap is the
right direction.

## PCM-vs-Plain-Tank Comparator (Honest Negative Result)

Framework doc cites +30% (series) / +4–8% (other configs) solar-fraction gain from adding PCM vs. plain sensible-only tank.

**This study found**: ~0.0% difference (RT47 PCM vs. zero-latent "PCM" on same tank/weather).

**Root cause**: At PCM_MASS_KG = 50 kg (pipeline-consistent reuse from Phase 3) against 300 kg tank, the tank's own sensible capacity dominates. PCM-vs-PCM ranking (this phase's actual purpose) remains valid and non-tied; PCM-vs-plain-tank sensitivity should NOT be over-interpreted as evidence of flawed system design. Reported honestly, not tuned away.

## Known Caveats Inherited from Phase 6 (Carried Forward)

Every Phase 7 output carries these inherited caveats verbatim, never silently dropped:

1. **Cost always NaN**: Unavoidable — Phase 6 database limitation. No remedy here.
2. **Corrosion is binary proxy**: `2.0 if inorganic else 1.0`, not a measured rating. Cannot be independently verified by this simulation.
3. **Database status**: current 23 survivors (4/8/11 per cluster) tagged "COMPLETE — 55-row manufacturer database (+7 literature rows = 62 candidates total)" (Phase 6) — the 62-row pool itself is final; the survivor count changed with the 2026-09-13 T_DELIVERY_C/L_required correction, not with the database.
4. **Cluster instability, current numbers (post-2026-09-19)**: Kendall's W = 1.0000/0.7679/0.4170 (clusters 0/1/2) — Cluster 2 is the one below the 0.6 ambiguous threshold (was Cluster 0 historically, at W=0.388 on the pre-2026-09-13 pool; the intermediate 0.900/0.750/0.555 reading from 2026-09-13 is itself superseded). Low ρ in Clusters 0–1 may reflect pre-existing MCDM instability as much as physics disagreement — requires more data or method recalibration, not physics-model retuning.
5. **Supercooling cannot be validated**: supercooling is no longer the dominant entropy-weighted criterion in any cluster (current dominant criteria: vol_latent_heat 47.5%/Cluster 0, Tm_fitness 80.5%/Cluster 1, thermal_conductivity 41.4%/Cluster 2 — see "Dominant Entropy-Weighted Criterion Per Cluster" above), but **this physics model still deliberately does not simulate supercooling** (Barqawi's 3-phase model assumes ideal solid–liquid transition at Tm with no nucleation delay — see physics_lib.py for derivation), so any residual supercooling weight in a cluster's blend remains unvalidatable by this simulation regardless of its current size. Phase 8 tests the supercooling-penalty hypothesis directly via sensitivity analysis.

## Completion Report: What Was Actually Built (2026-08-11, Re-run 2026-08-14, Re-run again 2026-09-13)

Phase 7 was built and run deliberately against the pre-expansion ~25-row PCM database (pre-2026-08-12), not withheld pending database expansion. **Rationale**: the validation methodology itself needed to be built, tested, and debugged now rather than blocked indefinitely on a database-expansion task with no fixed completion date. Every output carried the caveat `PROVISIONAL — ~25-row database, not yet expanded to 40–60`. When the database was expanded to 55 rows (2026-08-12), Phases 5 and 6 were re-run (2026-08-14), and then Phase 7 was re-run against the fresh Phase 6 output — **39 survivors** at that point (vs. pre-expansion 20). **Superseded again 2026-09-13**: the T_DELIVERY_C (50→60°C) and M_W_KG (300→200kg) corrections raised `L_required` and shrank the κ-calibrated survivor pool to **23 (4/8/11 per cluster)** — this is the current on-disk state (see banners at top of this file).

### Bugs Caught & Fixed Before Trusting Any Result

Two bugs were caught by Phase 7's own mandatory self-tests (`self_test_energy_conservation()` and `self_test_draw_profile_integration()`) and fixed **before any real simulation result was trusted**:

1. **Backward-Euler solver bug**: A spurious `+ dt·c·Tw_old` term in the closed-form solve for water temperature in pre-melt/post-melt phases was destabilizing at hourly timestep, causing unbounded temperature blow-up. Fixed by re-deriving the 2×2 implicit system algebraically and verified against `numpy.linalg.solve` to full floating-point precision.

2. **Night-loss bug**: Barqawi's original bidirectional coupling term `a·(Tc−Tw)` allowed the tank to drain heat back through an idle collector overnight nearly as fast as it charged during the day — physically impossible (real systems have thermosiphon check valves or controller-gated pumps). **Fixed via `NIGHT_ISOLATION_FRACTION = 0.05`**, gating the collector coupling coefficient to 5% of its daytime value whenever Tc < Tw (collector colder than tank).

**Result after both fixes**: All three medoids land in 54–84% benchmark solar-fraction band. Energy conservation holds to machine precision (~1.6e-13 J residual). This calibrated model is used as-is for Phase 7 real experiment and Phase 8 penalty sweep.

**Historical note (2026-08-11 state) — superseded 2026-09-13.** `M_W_KG=300`/`NIGHT_ISOLATION_FRACTION=0.05` above were this calibration's ORIGINAL values; both were subsequently corrected (`M_W_KG`→200 kg, decoupled from Avargani's continuous-flow figure; `NIGHT_ISOLATION_FRACTION`→0.03, jointly re-tuned with `COLLECTOR_UL_WM2K`→2.0) — see the "✅ TANK MASS DECOUPLED..." banner near the top of this file and `physics_lib.py`'s CALIBRATION docstring. The pipeline still lands in the same 54–84% band post-fix (64.0–65.8%), just at the new parameter values.

## Cluster-Specific Interpretations

**⚠️ HISTORICAL (2026-08 state, n=9/15/17, pre-2026-09-13 corrections) — interpretations below were
originally written against that run, then partially updated for the 2026-09-13 corrections
(rho=0.105/-0.190/-0.091, Kendall's W=0.900/0.750/0.555, dominant criterion Tm_fitness uniformly at
51.9/83.3/68.1%). BOTH of those states are now superseded by the 2026-09-19 Tm_fitness scoring fix:
current numbers are rho = -0.200/-0.168/+0.569, Kendall's W = 1.0000/0.7679/0.4170, dominant
criteria vol_latent_heat(47.5%)/Tm_fitness(80.5%)/thermal_conductivity(41.4%) — see the "NEWEST
UPDATE (2026-09-19)" banner at the top of this file. The qualitative diagnoses below (MCDM
instability vs. structural physics-model scope limits) still apply in kind, but every specific
number quoted in each subsection below predates the 2026-09-19 fix.**

### Cluster 0 (ρ = −0.385 pre-unification; ρ = +0.105 as of 2026-09-13; current ρ = −0.200, still on an undersized n=4 pool)

MCDM and physics rankings were **negatively correlated** pre-unification — higher-ranked PCM by MCDM delivered **worse** simulated performance; the 2026-09-13 run flipped weakly positive (+0.105), and the current (2026-09-19) run flipped negative again (−0.200), still far below the 0.4 bar, on the same undersized n=4 pool. Two non-exclusive diagnoses, still applicable:

1. **MCDM ranking itself unstable/undersized**: current Kendall's W=1.0000 (perfect agreement given only n=4 candidates) but candidate_pool_status remains undersized. Low correlation against physics may reflect the small pool as much as a physics-model gap. **Fix indicated**: this cluster's pool is confirmed (2026-09-13 investigation, CLAUDE.md §3.4) to be genuinely climate-driven, not a calibration artifact — expanding it further would need new PCM candidates in the 42–70°C band with a lower melting point, not a κ change.

2. **Dominant criterion (current)**: `vol_latent_heat` (47.5% entropy weight), not Tm_fitness (was 51.9% at 2026-09-13) or supercooling (63.8% pre-cap). The physics model still cannot independently verify latent-heat-driven disagreement here any more precisely than it could supercooling — this caveat is structural, not tied to the specific dominant-criterion number.

### Cluster 1 (ρ = +0.125 pre-unification; ρ = −0.190 as of 2026-09-13; current ρ = −0.168)

**Was the best outcome of the three clusters pre-unification; has been the worst since 2026-09-13.** Current Kendall's W = 0.7679 (moderate-to-strong method agreement), n=8 (healthy pool, `in_band` at κ=0.5).

- Dominant criterion remains `Tm_fitness` across both the 2026-09-13 and current runs (83.3% → 80.5% entropy weight, the highest of the three clusters both times) — worth investigating whether Tm_fitness's Gaussian-target transform is driving this cluster's persistent disagreement, rather than supercooling (which was never dominant here post-cap).
- Phase 8's k-sweep still shows agreement worsening further as the supercooling penalty rises (−0.168 → −0.311), so supercooling over-weighting is not the current explanation for this cluster's disagreement.

### Cluster 2 (ρ = −0.097 pre-unification; ρ = −0.091 as of 2026-09-13; current ρ = +0.569, largest cluster)

**The only cluster to cross into PARTIAL agreement (0.4<ρ≤0.8) since the 2026-09-19 Tm_fitness fix** — a reversal from weakly-negative-to-uncorrelated in both earlier runs. Kendall's W = 0.4170 (the most ambiguous of the three clusters, <0.6; Borda and Copeland also disagree on Top-3 membership here — Copeland-vs-simulation rho=0.633, p=0.036, nominally significant).

- Dominant criterion is now `thermal_conductivity` (41.4%), not Tm_fitness (68.1% at 2026-09-13) or supercooling (57% pre-cap) — same caveat as Clusters 0/1: the physics model cannot validate supercooling at any residual weight.
- Candidate pool may still be heterogeneous enough (paraffins vs. fatty acids vs. inorganics) that a single MCDM ranking cannot capture the variation under this climate — the Borda/Copeland Top-3 disagreement is itself evidence of this.
- Phase 8's k-sweep shows agreement worsening (+0.569 → +0.387 at k≥0.1) in this cluster too — the direction is consistent with Clusters 0/1 even though the sign here is now positive.

## Code Quality & Documented Design Decisions

- **Provenance hard-fail check**: Confirms Phase 5 and Phase 6 outputs were built from the same cluster partition (prevents silent mismatch from separate re-runs of Phase 4).
- **Mass sensitivity sweep** (Phase 7, lines ~312–362): Tests whether PCM_MASS_KG=50kg is the right scale to see differentiation. Result: spread widens, ranking stable at 50–800 kg — confirms signal is real, not noise from mass underdimensioning.
- **Night-delivery test** (lines ~364–371): Validates ability to sustain 58–62°C overnight discharge (Avargani benchmark).
- **Explicit self-test mandatory before main experiment**: Energy conservation and draw-profile checks; failures block main run.

## Relationship to Phase 8

Phase 7 identifies supercooling as the dominant MCDM criterion but flags the model cannot simulate it. Phase 8 extends this by:
1. Implementing a supercooling penalty in physics_lib.py (proportional reduction to h_p in supercooled region)
2. Running sensitivity sweep across penalty strength k ∈ [0.0, 0.1, 0.2, 0.3]
3. Testing whether the penalty brings physics/MCDM agreement closer to zero or improves it

See `10_PHASE_8_AUDIT.md` for the full Phase 8 findings.

## Literature & References

**Barqawi 2025**: Model equations, h_c=1500 W/m²K, h_p=800 W/m²K, A_c/M_w defaults  
**Duffie & Beckman**: Flat-plate collector U_L range justification  
**Avargani et al. 2021**: 300L @ 60±2°C design basis  
**Al-Mamun et al. 2023**: FPC efficiency range (45–73%)

---

**Status**: Phase 7 script renumbered to `10_physics_validation.py` (2026-09-08). Pre-unification
physics validation found weak-to-negative correlation with MCDM, attributed to supercooling's
entropy-inflated MCDM weight (48–64%). The unified Phase 6 caps that weight at 0.16; a
`10_physics_validation.py` run against the post-cap `mcdm_full_rankings.csv` (2026-09-13, together
with the T_DELIVERY_C=60°C and M_W_KG=200kg corrections) still found no cluster exceeding ρ=0.4
(rho = 0.105/-0.190/-0.091). **A further fix, 2026-09-19 (`Tm_fitness` scored against
`Tm_target_capped_C` instead of raw `Tm_target_C`), changed the result again and is the current
on-disk state: rho = -0.200/-0.168/+0.569 (clusters 0/1/2), p=0.800/0.691/0.067.** Clusters 0–1
remain a genuine negative validation (≤0.4); **Cluster 2 crossed into PARTIAL agreement**
(0.4<ρ≤0.8) for the first time in this project's Phase 7 history. The dominant MCDM criterion is no
longer uniformly `Tm_fitness`: `vol_latent_heat` (47.5%, Cluster 0), `Tm_fitness` (80.5%, Cluster 1),
`thermal_conductivity` (41.4%, Cluster 2). Phase 8's k-sweep (agreement still worsens as the
supercooling penalty rises, in every cluster, including Cluster 2's now-positive baseline) remains
the supporting evidence that supercooling is over-weighted whenever its penalty is amplified, even
though neither the entropy cap nor the Tm_fitness fix was sufficient on its own to produce a
uniformly positive physics/MCDM correlation. The seasonal-sensitivity picture flipped alongside
this fix too: 8/11 (cluster, season) cells now flip the annual #1 pick (was 0/11 at 2026-09-13) —
see the "Level B seasonal PCM sensitivity" section above.
