# 09 — Phase 7 Audit: Physics-Based Validation

**Script**: **none.**

**Status**: **NOT IMPLEMENTED.** There is no physics-validation script of any name in
`era5-uttarakhand/`.

---

## Purpose Phase 7 would have served

Independently test the Phase 6 MCDM ranking against a physics-based simulation, rather than against
another ranking method. The MCDM stack ranks PCMs on properties; a grey-box tank model ranks them
on *simulated delivered performance*. Comparing the two orderings (Spearman ρ) is the only way
Objective 1 can claim its ranking is more than internally consistent.

For Uttarakhand this would carry extra weight, because Phase 5 and Phase 6 return an **identical
survivor set and an identical #1 PCM in all five clusters** (see `07_PHASE_5_AUDIT.md` and
`08_PHASE_6_AUDIT.md`). `08_mcdm_ranking.py`'s own diagnostic names Phase 7 as the place that
differentiation would have to appear:

> (a) State it as a finding: Uttarakhand's climate regimes differ more in solar reliability/cloud
> persistence than in delivery-relevant temperature, so a single PCM family serves the whole state
> under the corrected `Tm_target` rule — **differentiation would need to show up in Phase 7 physics
> simulation (solar fraction per regime), not in the candidate list itself.**

## Confirmation that no script exists

| Evidence | Detail |
|---|---|
| Directory listing | `era5-uttarakhand/` contains `00_unzip_accum`, `00a`, `00b`, `01`, `01b`, `02`, `02b`, `03`, `03b`, `04`, `04b`, `04c` ×2, `04d`, `05`, `05b`, `05c`, `05d`, `05_cluster_regions`, `06`, `07`, `07b`, `08`, `09`, `comparison_plots_uttarakhand`, `generate_objective1_plots`, `verify_01`…`verify_04`, `config`. **There is no `10_*.py`.** |
| `README.md` pipeline map | Jumps directly from "PHASE 6 — MULTI-CRITERIA RANKING" to "PHASE 8 — FINAL OUTPUT" |
| `README.md`, explicit note | "(Phase 7 — physics-based validation via a grey-box lumped enthalpy tank model — **has no script in this repo yet**; see 'What's genuinely still open' at the bottom.)" |
| `NEXT_STEPS.md` status table | "7. Physics-Based Validation \| Grey-box lumped enthalpy tank model, Spearman rho vs. MCDM rank \| **Not written.**" |
| `08_mcdm_ranking.py` closing text | Lists "A minimal grey-box physics validation run per cluster's Top-1" under "What's still genuinely optional beyond this" |

## What the source files say should be built

`README.md`, "Notes / known limitations":

> **Phase 7 (physics-based validation) has no script here.** A minimal single-PCM grey-box
> lumped-enthalpy-tank simulation per cluster, compared against published annual-solar-fraction
> benchmarks (54-84%), is enough to defensibly write "consistent with published benchmarks" — but
> it isn't required for Objective 1 to stand as a working framework, and is explicitly an accepted
> "future work" outcome if you don't get to it.

`NEXT_STEPS.md`, step 13:

> If time allows: a **minimal** physics check — one grey-box lumped PCM tank simulation for just
> the Top-1 PCM in 1-2 clusters, compared against published Table-16-style benchmarks (annual solar
> fraction 54-84%). A single calibration run per cluster is enough to write "consistent with
> published benchmarks" honestly. **If you can't fit this in, say explicitly in the paper that
> physics validation is future work — an accepted outcome per the plan doc, not a weakness you
> need to hide.**

`NEXT_STEPS.md`, "What's genuinely still open":

> **Physics validation (Phase 7)** is not written. If you have a spare half-day, a single-PCM
> grey-box run per cluster against the Table 16 benchmark ranges (annual solar fraction 54-84%) is
> enough to write "consistent with published benchmarks" honestly — full validation across every
> cluster is not required.

So the specification is consistent across three files: a **lumped-enthalpy grey-box tank model**,
run per cluster (or for 1–2 clusters at minimum), for at least the Top-1 PCM, with the annual solar
fraction checked against a published **54–84 %** band.

The only numeric anchor stated anywhere in `era5-uttarakhand/` is that **54–84 % band**, attributed
to "Table 16" of the plan document. The plan document is not present in the folder, so the band's
underlying references are **not available in the source files**.

## The one adjacent piece that does exist

`07b_charging_feasibility.py` is the closest thing in the repository to a physics calculation, and
its docstring explicitly marks the boundary:

> This is a **HEURISTIC PROXY, not a real collector thermal model.** A rigorous version needs the
> cluster's 5th-percentile daily insolation fed through an actual collector efficiency curve
> (`eta_th = F_R[S − U·(T_in − T_amb)/G]`, as several of your literature summaries already have) —
> **that's Phase 7 territory**, not something to improvise here under deadline pressure.

That efficiency-curve form is the most concrete statement anywhere in the folder of what the Phase 7
collector model would need. `07b` deliberately does not implement it. See `07_PHASE_5_AUDIT.md`.

## What the plot layer expected, and did not get

Two figure scripts were written against a Phase 7 that does not exist:

**`generate_objective1_plots.py`** declares
`PHYS_VAL = data/processed/pcm/physics_validation_results.csv` and its plot 11
("Agreement plot: physics rank vs consensus rank") checks for a
`hours_target_met_per_year` column. The file does not exist, so `p11()` falls through to its
degenerate branch and plots **TOPSIS rank against consensus rank** instead of simulated performance
against consensus rank. The committed `11_agreement_plot.png` and
`11_agreement_plot_interactive.html` are therefore **not** physics-validation figures despite their
titles saying "Simulated Performance vs MCDM Consensus Rank".

**`comparison_plots_uttarakhand.py`** declares the same `PHYS` path and its comparison 6 ("Physics
validation: hours_target_met vs MCDM rank") is skipped. That script never produced output at all —
see `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md` for its separate path defect.

**`generate_objective1_plots.py`'s plot 12** ("Tank temperature / melt-fraction profile") produces
`12_tank_temperature_melt_fraction.png` and `.html`, both committed. These are **synthetic
illustrations, not simulation output**:

```python
Ta   = 28 + 14*np.sin((hrs-6)*np.pi/12)
tank = Tm - 6 + 18*np.sin((hrs-6)*np.pi/12)
melt = np.clip((tank - Tm + 5)/10, 0, 1)
```

Hard-coded sinusoids over a 24-hour axis, with only `Tm` taken from the real per-cluster
`Tm_target_C`. The ambient amplitude (28 ± 14 °C) does not correspond to any Uttarakhand cluster
profile. **These figures must not be presented as physics-validation results.**

## Consequence for Objective 1's claims

| Claim | Supportable for Uttarakhand? |
|---|---|
| "The MCDM ranking was tested against an independent physics simulation" | **No** — no simulation was run |
| "Annual solar fraction is consistent with published 54–84 % benchmarks" | **No** — no solar fraction was computed |
| "Different regimes favour different PCMs" | **No** — and Phase 7 was the designated place to demonstrate it |
| "The framework is complete through Phase 6, with Phase 7 as declared future work" | **Yes** — this is exactly how all three source files frame it |

The last row is the honest position, and it is the position the source files themselves take. It
should be stated in the paper as a declared limitation, not omitted.

## Inputs Phase 7 would need (all present or recoverable)

| Requirement | Availability |
|---|---|
| Per-cluster medoid point | `09_recommendation_cards.py` already computes an approximate medoid |
| Real daily climate for that point | `daily_aggregates_uttarakhand.csv` — 45 × 3,653 point-days from `02b` |
| Daily GHI, Tmax, Tmin | `GHI_daily_kWh`, `Ta_max_true`, `Ta_min_true` — all present in that file |
| Candidate PCM properties | `pcm_database_uttarakhand.csv` — `Tm_C`, `latent_heat_kJ_kg`, `density_*`, `Cp_*`, `TC_W_mK` |
| Survivors to simulate | `feasibility_survivors_by_cluster.csv`, or the 15 Top-3 rows from `08` |
| Ranking to correlate against | `mcdm_topk_by_cluster.csv` / `mcdm_full_scores_by_cluster.csv` |

**Every input a Phase 7 script would need already exists on disk.** The blocker is the script
itself, not the data.

One caveat that would apply to any Phase 7 built on this pipeline: a collector model driven by
`era5_GHI` would inherit the magnitude anomaly documented in `04_PHASE_2_AUDIT.md` Part A.3. It
should be driven by the **NASA POWER** `GHI_daily_kWh` from `02b` instead, which is the source the
canonical signature column already uses.

## Problems / risks

1. **Phase 7 is absent**, so Objective 1's ranking has no external validation of any kind for
   Uttarakhand — only internal method agreement, which is itself poor (pooled TOPSIS-vs-GRA
   Spearman ρ = −0.930, see `08_PHASE_6_AUDIT.md`).
2. **Two committed figures carry physics-sounding titles without physics behind them.**
   `11_agreement_plot.png` silently plots TOPSIS-vs-consensus rank, and
   `12_tank_temperature_melt_fraction.png` is a hard-coded sinusoid. Both are titled as if they
   were simulation results. Neither should appear in a paper as validation evidence.
3. **The identical-across-clusters Phase 5/6 result has nowhere to be resolved.** `08`'s own
   diagnostic nominates Phase 7 as the place regime differentiation would show up; with Phase 7
   absent, the finding stands unresolved.
4. **The 54–84 % benchmark band has no traceable citation** inside `era5-uttarakhand/` — only a
   reference to "Table 16" of an absent plan document.

## Status

**NOT IMPLEMENTED — and correctly declared as such in three separate source files.** This is not a
silent omission; `README.md`, `NEXT_STEPS.md` and `08_mcdm_ranking.py` all name it, specify what a
minimal version would look like, and state that recording it as future work is an accepted outcome.
Every input a minimal implementation would need is already on disk.
