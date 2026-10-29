# Objective 1 — Complete Consolidated Summary

**Project:** Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating
**Group 12 | Amrita School of Engineering | Guide: Dr. T. Deepika**
**Scope of this file:** everything done under Objective 1 in the `PCM-Selection-ML-model` repo, followed by the Objective 2 statement.
**Compiled:** 2026-10-17 from the repo's own docs and on-disk outputs (see "Source map" at the end).

> **Read this first.** Several older files in this repo describe superseded states of the pipeline.
> In particular, `OBJECTIVE1_COMPLETE_EXPLANATION.md` (Uttarakhand, 25 PCMs, 5 criteria) and
> `docs/cross_region_results_comparison.md` (Assam K=4, unanimous RT44HC) are **out of date**.
> Rajasthan is the most complete and most recently corrected pipeline, so it is the reference
> state in this file. Numbers are not mixed across states.

---

## 1. What Objective 1 is

**O1 — Climate-Region-Aware PCM Recommendation Framework.** Turn 10 years of reanalysis climate data into
population-weighted climate regimes, derive PCM performance targets for each regime, and rank candidate PCMs
against those targets with an auditable, multi-method, uncertainty-aware pipeline. Then check the ranking
against an independent physics simulation.

**Research gap addressed:** RG1 (no climate-region-aware, data-driven PCM recommendation framework). In the
framework doc's own novelty list (N1–N6) it contributes population-weighted sampling (N6), a two-tier climate
signature (N2), an SWH-specific 42–70 °C melting band (N3), a four-method consensus ranking (N4) and a
falsifiable physics validation (N5).

**What O1 does not do:** no real-time control (that is O3), no design-parameter optimisation (that is O2),
no hardware (O4). It produces the *input* those objectives consume.

---

## 2. The pipeline at a glance

```
Phase 1   Data collection        population-weighted points + ERA5 + NASA POWER
Phase 2   Preprocessing & QA     ERA5-vs-POWER cross-check, quantile mapping, Hampel, MICE imputation
Phase 3   Climate signature      two-tier indices + Tm_target + L_required
Phase 4   Regime clustering      Gaussian Mixture Model (diag covariance), BIC-selected k
Phase 5   Feasibility filter     8 constraints on a 62-row PCM database + kappa calibration
Phase 6   MCDM ranking           TOPSIS + PROMETHEE II + VIKOR + GRA, Borda/Copeland, Kendall's W, Monte Carlo
Phase 7   Physics validation     lumped-enthalpy tank simulation, Spearman rho vs MCDM rank
Phase 8   Supercooling sweep     sensitivity of the physics/MCDM agreement
Deliver   Recommendation cards   one markdown card per climate regime
```

Each state has its own folder (`era5-rajasthan/`, `era5-tamilnadu/`, `era5-uttarakhand/`, `era5-assam/`) with the
same numbered scripts. Cross-state constants live once in [pcm_shared_config.py](pcm_shared_config.py).

---

## 3. Phase-by-phase: what was done and how

### Phase 1 — Data collection
- **Sampling.** `00a_build_population_grid.py` picks the smallest set of highest-population cells on ERA5's own 0.25° grid that covers ≥ 87.5 % of the state's population (WorldPop 2020 raster, GADM v4.1 boundary). Reason: a PCM recommendation should reflect where hot-water demand actually lives, not uninhabited cells. Rejected alternatives: uniform grid, district centroids.
- **Sun-event timing.** `00b_build_suntimes.py` computes exact sunrise / solar noon / sunset per point per day with the pvlib NREL SPA, so sampling is aligned to the sun and not to fixed clock hours (a fixed 06:00 UTC pull misses the solar peak by up to ~48 min at Tamil Nadu's longitude).
- **Elevation.** `00c_attach_elevation.py` attaches real per-point elevation from ERA5 geopotential (replaced flat proxies of 1200 m for Uttarakhand and 150 m for Tamil Nadu).
- **Downloads.** `01_download_era5_*.py` (CDS API, instant + accumulated variables as two separate requests, sun-event-aligned hour windows) and `01b_download_nasapower.py` (full hourly cache per point). `00_unzip_accum.py` fixes the CDS quirk where a ZIP arrives disguised as `.nc`.
- **Period:** 2016–2025 (10 years) for Rajasthan, Uttarakhand, Tamil Nadu and Assam.

### Phase 2 — Preprocessing and cross-source validation
- **Combine** (`02_combine_*.py`): unit conversion, pvlib solar geometry (SZA, azimuth, Ineichen clear-sky, CSI), match ERA5 and POWER to each sun-event instant within `MAX_MATCH_HOURS = 3`.
- **True daily integrals** (`02b_build_daily_aggregates.py`): trapezoidal integration of the hourly POWER cache for daily GHI, DTR, HDD18, CDD24, cloudy fraction. This is the "Tier 2" repair for the 3-rows-per-day sampling limit.
- **Cross-source agreement** (`03b_agreement_analysis.py`): MBE / RMSE / Pearson r between ERA5 and NASA POWER with a pre-registered decision rule (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW).
- **Cleaning** (`04_preprocess_*.py`): physical bounds, solar night-mask, persisted per-season ERA5→POWER quantile mapping, Hampel (MAD) outlier filter, hierarchical imputation ending in MICE, chronological 70 % train fit for MinMax scaling.
- **The headline integrity finding:** the ERA5 de-accumulation bug (see §6).

### Phase 3 — Climate signature
- **Two tiers.** Tier 1 = sun-event indices (e.g. `Ta_mean`, `T_noon_mean`, `GHI_noon_mean`, `kt_noon_mean`). Tier 2 = true daily-integral indices (`GHI_daily_kWh`, `kt_daily_mean`, `DTR_true`, `HDD18`, `CDD24`, `cloudy_frac`, `CCI`, `SAI`, `seasonality`, `monsoon_index`). Plus 5 interaction terms and a 4-component PCA on the temperature/elevation block (`PCA_N_COMPONENTS = 4` so every state has the same schema).
- **PCM-facing targets** derived from each regime's own climate:
  - `Tm_target_C = T_DELIVERY_C + DT_APPROACH_C` (indirect-system rule; currently 60 + 7 = **67 °C** in the shared config)
  - `Tm_target_capped_C`: the melting temperature a poor-insolation day can still reach (derived from the worst-month / 5th-percentile clearness)
  - `L_required_kJ_per_kg = (SHARE_PCM × Q_night) / m_PCM`, with `SHARE_PCM = 0.5`

### Phase 4 — Climate regime clustering
- **Gaussian Mixture Model** with **diagonal covariance** (full covariance was saturating membership probabilities to ~1.0 with so few samples). k chosen from BIC, silhouette, Davies–Bouldin, Calinski–Harabasz and bootstrap ARI. Latitude/longitude are excluded from the clustering matrix, so spatial coherence is an outcome, not an input.
- **Canonical relabelling** by ascending mean latitude, so cluster IDs are stable across runs.
- **Level B** (`05a`): re-clusters per point per season to detect regime shifts.
- **External check:** Köppen–Geiger (Beck et al. 2018) is wired in for Rajasthan: ARI = 0.2787, NMI = 0.3817 — read as "the GMM finds finer structure than Köppen's broad classes". NBC/ECBC Indian climate zones are still stubbed.

### Phase 5 — PCM feasibility filtering
- **Database:** 62 candidates = 55 manufacturer rows (14 Rubitherm RT, 7 PLUSS savE, 4 PCM Products/PlusICE, 5 PureTemp, 1 CrodaTherm, plus literature n-alkanes / fatty acids / composites) + 7 extra literature rows (Singh 2025). Missing properties filled by MICE with Random Forest + predictive-mean-matching blend and flagged as imputed. Expanded from the earlier 18–25 rows to reach the plan's 40–60 target.
- **Eight constraints, in order:** (1) melting window, (2) absolute band 42–70 °C, (3) latent-heat floor, (4) cycling ≥ 300 (flag only if unreported), (5) supercooling ≤ 8 K (flag only if unknown), (6) charging feasibility `Tm ≤ Tm_target_capped_C`, (7) corrosion veto, (8) safety exclusion. Constraints 7 and 8 are correct but inert here because every candidate is organic.
- **κ-calibration.** The latent-heat floor is `κ × L_required`. At the nominal κ = 0.7 the raised ceiling leaves **0** survivors in every Rajasthan cluster, so a per-cluster calibrated κ is searched. Policy decided 2026-09-17: **accept calibrated κ** (not rank-by-proximity).

### Phase 6 — MCDM ranking
- **Criteria (8 from the plan's Table 13):** Tm fitness (Gaussian against the *capped* target, σ = 4 K below / 2 K above), latent heat relative to `L_required`, volumetric latent heat (ρ·L), thermal conductivity, supercooling (inverse), cycling confidence (log-scaled), corrosion, cost. All normalised to [0, 1].
- **Weights:** 50/50 blend of Shannon-entropy weights and the plan's Table-13 priors. The supercooling entropy weight is **capped at 2× its prior (0.16)** because the raw entropy formula over-weights near-zero-ideal cost criteria. Honest note: the "AHP" half is the Table-13 priors used unmodified; no pairwise elicitation was done (`AHP_PAIRWISE_MATRIX = None`).
- **Methods:** TOPSIS, PROMETHEE II, VIKOR, GRA → Borda consensus, Copeland cross-check, Kendall's W, pairwise method agreement, and a Monte Carlo (Dirichlet-perturbed weights + property jitter) for Top-3 inclusion and Top-1 retention.
- **Provenance hard-fail** (`provenance_lib.py`): Phases 6/7/8 refuse to run if their cluster fingerprint does not match the current cluster profiles.

### Phase 7 — Physics-based validation
- `physics_lib.py` + `10_physics_validation.py`: backward-Euler, lumped-enthalpy PCM + water tank model driven by 10 years of real hourly weather at each regime's medoid point; cited draw profile; calibrated so the plain-tank solar fraction lands in the 54–84 % literature band (current medoids 64.0 / 65.8 / 64.3 %).
- **Self-tests must pass before any result is trusted:** energy conservation (~1e-13 relative residual) and draw-profile integration.
- Output: Spearman ρ between the MCDM Borda rank and the simulated annual solar fraction, per cluster.

### Phase 8 — Supercooling-penalty sweep and recommendation cards
- Sweep of a proportional heat-transfer penalty for supercooling (k ∈ {0.0, 0.1, 0.2, 0.3}) to test whether supercooling is over-weighted.
- `09_recommendation_cards.py` aggregates Phases 4/6/7 into one card per regime (identity, signature, targets, candidates screened, Top-3 with per-criterion contributions, physics context, caveats). It re-verifies cluster identity independently before writing.

---

## 4. Results by state

### 4.1 Rajasthan — the reference state (current on-disk results)

320 points, 87.5 % population coverage, **k = 3**, 62 candidates.

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Medoid | Udaipur (RJP_0132) | Nagaur (RJP_0192) | Jaipur (RJP_0083) |
| Points / population | 109 / 21.8 M | 83 / 14.0 M | 128 / 34.5 M |
| Character | cooler, low-monsoon, erratic solar | hot, monsoon-influenced, steady solar | cooler, low-monsoon, erratic solar |
| `Tm_target_capped_C` | 55.5 | 61.1 | 59.2 |
| `L_required` (kJ/kg) | 438 | 427 | 443 |
| Survivors at κ = 0.7 | 0 | 0 | 0 |
| Calibrated κ → survivors | 0.0 → **4** (undersized) | 0.5 → **8** | 0.3 → **11** |
| **#1 (Borda)** | savE® OM55 | PureTemp 60 | Palmitic-stearic acid / expanded graphite |
| #2 | Palmitic-stearic acid / expanded graphite | CrodaTherm 60 | n-Heptacosane (C27) |
| #3 | Myristic acid / NBR-0.5 | n-Heptacosane (C27) | PureTemp 58 |
| Kendall's W | 1.0000 | 0.7679 | 0.4170 (< 0.6) |
| MC Top-3 inclusion of #1 | 100 % | 96.2 % | 63.1 % |
| **Spearman ρ (MCDM vs physics)** | **−0.200** (negative) | **−0.168** (negative) | **+0.569** (partial) |
| Simulated solar fraction of Top-3 | 65.5 / 65.7 / 65.5 % | 66.3 / 66.1 / 66.4 % | 65.4 / 64.6 / 65.1 % |

Other points worth stating plainly:
- Cluster 2's Borda and Copeland disagree on Top-3 membership (Copeland swaps PureTemp 58 for savE® OM55).
- The simulated solar fractions of the Top-3 differ by under ~1 pp inside each cluster, so the physics model barely separates the candidates.
- Phase 8 sweep: raising the supercooling penalty *worsens* agreement in every cluster (ρ: −0.200→−1.000, −0.168→−0.311, +0.569→+0.387 at k = 0.1–0.3).
- Testing a 9th `thermal_margin` criterion made ρ worse in Clusters 0 and 2, so it was left off by default.
- Seasonal sensitivity (`11_seasonal_pcm_sensitivity.py`): **8 of 11** valid (cluster, season) cells flip the annual #1 pick — see §7 for why this matters.

### 4.2 Tamil Nadu (re-run 2026-09-16)
133 points (87.5 %), **k = 3** (auto-selected; a forced k = 5 was tried and reverted because it was the worst of 2–5 on silhouette and bootstrap ARI). κ-calibrated survivors **8 / 10 / 8** (n = 26). 9 criteria (Table 13 + `thermal_margin`), 4 methods, 5,000 MC draws. Kendall's W 0.848 / 0.477 / 0.780. Consensus #1 is `RT57HC` in all three clusters (flagged for an independent sanity check). Mean Spearman ρ = +0.169 (+0.381 / +0.103 / +0.024), none statistically significant (n = 8–10). Köppen raster missing in this copy.

### 4.3 Uttarakhand (older design basis)
45 points, **K = 5** (diag GMM), silhouette 0.28, 55-row database, `Tm_target` still the old **57 °C** (50 + 7). Survivors 29 / 30 / 29 / 27 / 29 (regime cap from `07b` genuinely lowers the target for Clusters 1 and 2 to 55.16 / 56.51 °C). #1 is PureTemp 58 in Clusters 0/2/3/4 and PureTemp 53 in Cluster 1. Kendall's W 0.708–0.842. Physics: 0 % of simulations land in the 54–84 % band (≈12–19 %), ρ from −0.338 to +0.169, none significant. This state was **not** moved to the 60 °C delivery basis.

### 4.4 Assam (own 11-phase variant)
129 points (87.8 %), **K = 3** GMM (full covariance, 5 features, BIC 1574.94, bootstrap ARI 0.629), `Tm_target` = 44 °C (delivery 50 °C, ΔT 6 K), 58-PCM database. Under the final K = 3 and strict filtering, **zero confirmed survivors in every cluster** (one conditional: n-Tetracosane C24), so formal MCDM was *not performed* and Monte Carlo *skipped*. The older K = 4 ranking (RT44HC unanimous) was retained only as a labelled historical artefact; compared against a 10-year sub-hourly physics run it gave ρ = −0.52 to −0.64, Top-1 agreement 0 % → verdict "not physically supported" (delivering at 50 °C favours PCMs melting nearer 51 °C than the 44 °C target).

### 4.5 Cross-state differences to keep in mind

| | Rajasthan | Tamil Nadu | Uttarakhand | Assam |
|---|---|---|---|---|
| Points | 320 | 133 | 45 | 129 |
| k | 3 | 3 | 5 | 3 (final) |
| Outlier handling | Hampel | Hampel | Hampel/MAD | IsolationForest |
| `Tm_target` basis | 67 °C | 67 °C | 57 °C | 44 °C |
| MCDM state | complete | complete | complete | not performed at K = 3 |
| Physics verdict | mixed | weak, n.s. | weak, n.s. | not supported (K = 4) |

---

## 5. The corrections that shaped the final numbers

Chronological record of design-basis fixes. Each was found by a specific diagnostic and documented in code.

| Date | Fix | Effect |
|---|---|---|
| 2026-08-11 | GMM cluster-index instability: canonical relabelling + provenance hard-fail | Phase 5 and 6 had disagreed on which PCMs belonged to which cluster |
| 2026-08-12 | PCM database 18/25 → 55 rows (62 evaluated) | Closed the row-count gap |
| 2026-08-31 | `L_required` redefined with `SHARE_PCM = 0.5` | The all-latent version (400–550 kJ/kg) zeroed out every single-material PCM. Basis: Zhao 2022, Huang 2020, Abdelsalam 2020, Kožełj 2021 (range 0.4–0.78). Option A adopted; Option B (calibration loop) rejected as over-engineering |
| 2026-09-08 | Phase 5 and 6 unified with Tamil Nadu | Same 8 constraints, same engine, supercooling entropy cap |
| 2026-09-13 | `T_DELIVERY_C` 50 → 60 °C | Avargani 2021's 300 L/7 h figure is validated at 60 ± 2 °C. `Tm_target` 57 → 67 °C; `L_required` 285–344 → 410–469 kJ/kg |
| 2026-09-13 | Simulator tank mass `M_W_KG` 300 → 200 kg, with `COLLECTOR_UL_WM2K` 2.5 → 2.0 and `NIGHT_ISOLATION_FRACTION` 0.05 → 0.03 | Avargani's 300 L is a flow-through volume, not a tank. 200 kg sits inside Eldokaishi 2022's 50–240 L range. The UL/isolation retune is calibration to restore the 54–84 % band, not an independent literature choice |
| 2026-09-13 | Seasonal-sensitivity script used a hard-coded κ = 0.7 | Fixed to use each cluster's own κ; the "degenerate" result disappeared |
| 2026-09-17 | GHI quantile mapping confirmed wired in; Cluster 0 n = 4 policy decided | Cluster 0's small pool is a real climate finding (only 4 of 62 clear the melting/charging gates regardless of κ), not a calibration artefact |
| 2026-09-19 | `Tm_fitness` scored against `Tm_target_capped_C`, not the raw 67 °C | Changed Top-1 in 2 of 3 Rajasthan clusters and moved ρ to −0.200 / −0.168 / +0.569 |

---

## 6. Bugs found and fixed (the integrity story)

1. **ERA5 de-accumulation** — the code diffed consecutive hours, but this pipeline's CDS pull already returns per-hour flux. It produced near-zero GHI (noon r ≈ 0.01 vs NASA POWER). Fix: `accum_to_flux()` (a clip, no diffing). Rajasthan noon r = 0.81; Uttarakhand r = 0.759, MBE +19.6 W/m². Caught only because ERA5 was cross-checked against an independent source.
2. **GMM covariance type** `full` → `diag`.
3. **VIKOR compromise-index sign inversion**, which silently reversed rankings.
4. **Entropy-weight inflation** for sparse or all-NaN criteria (the empty `cost` column was taking 64–75 % weight).
5. **Phase 7 closed-form solve** (caused temperature blow-up) and **phase-transition energy accounting** (silently dropped sensible-heat overshoot). Both caught by the module's mandatory self-tests.
6. **Uttarakhand regime-cap normalisation** (`poor_day_kt` was divided by `kt_mean`, so it could never separate clusters) and **Uttarakhand Phase 7** solver bugs (an earlier "92 % in band" figure was an artefact).
7. **HDD18/CDD24 annualisation** summed the whole 10-year record (≈10× too large). A pure rescale here, verified to leave survivor counts, ARI and Köppen scores unchanged.

---

## 7. What can and cannot be claimed

**Can claim (with the caveats attached):**
- A population-weighted, dual-source-validated climate dataset and a two-tier climate signature.
- Statistically selected climate regimes (not hand-drawn zones), with an external Köppen cross-check for Rajasthan.
- A transparent, four-method, uncertainty-aware ranking that reports its own disagreement (Kendall's W, Borda vs Copeland).
- A physics validation that was actually run, with self-tests, and reported honestly.

**Cannot claim:**
- That the ranking is *physics-confirmed*. Correct wording: "the MCDM ranking was physics-tested against the final 62-row database, with a negative result in two clusters and partial agreement in the third."
- That all Top-3 lists are equally well supported. Cluster 0's list rests on **n = 4**.
- That the weights were AHP-elicited. They were not.
- That the clustering is validated against NBC/ECBC. Only Köppen is wired in.
- Rajasthan's `Tm_target` of 67 °C is a delivery-plus-approach rule; it is unsourced for T_mains (`Ta_mean − 2.0` is a placeholder; a Kusuda–Achenbach style ground-temperature model is still to do) and σ = 4 K for Tm fitness is not from a paper.

**Open framing question for the paper (O3 motivation).** `CLAUDE.md` §3.4 records a 0/11 seasonal-flip result that was reframed as a positive finding ("O1's material choice is seasonally stable; O3 handles hour-by-hour variability"). That 0/11 figure predates the 2026-09-19 `Tm_fitness` fix. The current Rajasthan result is **8/11 flips**, so the earlier reframing no longer follows from the data. The O3 motivation needs to be re-decided before it is written up. Heidari et al. (2022) is not yet an extracted project source and should not be cited for specifics.

---

## 8. Known limitations and open items

- Cluster 0 (Rajasthan) pool is undersized (n = 4) and ends at κ = 0.0 (`insufficient_even_at_kappa_0`).
- The physics model separates candidates by under ~1 pp solar fraction, and the PCM-vs-plain-tank gain it reports is ~0 % against a cited +4–30 % literature range. Reported as-is, not tuned.
- Salt hydrates / inorganics are absent from the database, so the corrosion veto never fires. (Kenisarin & Mahkamov 2016 is the suggested source.)
- **Tamil Nadu follow-up not applied:** its `10_physics_validation.py` still hard-codes `T_DELIVERY_C = 50.0` while Phase 3 imports 60.0, giving a Phase 3 / Phase 7 mismatch on its next re-run. Uttarakhand and Assam hard-code their own constants and were never moved.
- Monsoon month definitions differ between `02_combine` (Jun–Aug) and `02b` (Jun–Sep); `avg_sdirswrf` is never divided by 3600.
- No `requirements.txt` or ERA5 product-version manifest; the `PCM_data/PCM_data/` folder is doubly nested.
- Literature review flags that pure GMM + MCDM may be classed as "MCDM" rather than "AI-enabled"; adding one supervised or hybrid ML layer (e.g. a surrogate predicting solar fraction, or Bayesian-optimised weights) is the recommended upgrade.
- Missing methodology citations: SPA (Reda & Andreas 2004), Ineichen & Perez 2002, pvlib (Holmgren 2018), TOPSIS/PROMETHEE/VIKOR/GRA originals, MICE (van Buuren 2011), quantile mapping (Cannon 2015).

---

## 9. Reproducing the Rajasthan run

```bash
cd era5-rajasthan
python run_all_rajasthan.py                              # full core chain, Phase 2 onward
python run_all_rajasthan.py --from 07_feasibility_filter.py   # re-run from Phase 5
```

Individual order after a constant change: `07_feasibility_filter.py` → `08_mcdm_ranking.py` → `10_physics_validation.py` → `09_recommendation_cards.py` → `11_seasonal_pcm_sensitivity.py` → `08_phase8_supercooling_sweep.py`. Seeds: `GaussianMixture(random_state=42)`, `run_monte_carlo(seed=42)`. Check the embedded fingerprints in the Phase 5–8 outputs against `cluster_profiles_rajasthan.csv` before trusting any number.

---

## 10. Source map

| Topic | File |
|---|---|
| Shared constants | `pcm_shared_config.py` |
| Rajasthan master + status | `docs/rajasthan/00_MASTER_OVERVIEW.md`, `12_FINAL_READINESS_REPORT.md` |
| Rajasthan final cards | `era5-rajasthan/outputs/recommendation_cards_rajasthan.md` |
| Phase audits | `docs/<state>/03_…` to `10_PHASE_8_AUDIT.md` |
| Method justification per script | `docs/METHODS.md` |
| Cross-state audits | `docs/audits/*_HIGH_IMPACT_AUDIT_2026-09-17.md` |
| Literature | `docs/Literature Review.md`, `docs/references.bib` |
| Superseded overviews | `OBJECTIVE1_COMPLETE_EXPLANATION.md`, `docs/cross_region_results_comparison.md` |

---
---

# Objective 2

> Develop an AI-driven design optimization model to determine the optimal PCM thickness, capsule arrangement, number of PCM capsules, and flow rate for maximizing thermal energy storage under location-specific climatic conditions.

**What O1 hands to O2:** a climate regime per location, its `Tm_target` / `Tm_target_capped_C`, and a Top-3 PCM shortlist with property sets (Tm, L, k, density, cp, supercooling). The shared grey-box thermal simulator (`physics_lib.py`) is the natural testbed for O2's search, as noted in the project constraints.
