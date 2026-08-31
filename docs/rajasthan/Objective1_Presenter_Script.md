# Objective 1 — Presenter Script & Reference Notes
### For the "What Was Actually Implemented" → "Physics Validation" slide sequence

> Each section below covers one slide: a spoken narration script (aim for the stated time),
> the numbers you must not fumble, and likely panel questions with suggested answers.
> **Critical: Know which state this slide describes.** This pipeline applies to four states in parallel:
> - **Rajasthan (furthest along):** 320 population-weighted points, 3 climate regimes, 2016–2025 (10 years)
> - **Tamil Nadu (in progress):** 222 named locations, 5 climate regimes, 2024–2025 (2 years)
> - **Uttarakhand (in progress):** state-specific parameters — see Uttarakhand docs
> - **Assam (in progress):** state-specific parameters — see Assam docs
> **Do not mix numbers across states.** Every number you quote must be attributed to its correct state.

---

## Slide 1 — "Objective 1: What Was Actually Implemented"
**Suggested time: 60–75 sec**

### Script
> "Objective 1 is a climate-region-aware PCM recommendation framework — not a single model,
> but an 8-stage pipeline. On the left, we go from raw sampling to climate regimes; on the
> right, from PCM feasibility screening through to the final recommendation cards.
>
> The key methodological point I want to flag up front: our comparison point in the
> literature is often 'Taguchi plus GRA' — an orthogonal-array design paired with one
> ranking method. We don't use one ranking method. We use four independent MCDM methods —
> TOPSIS, GRA, PROMETHEE II, and VIKOR — and we only trust a ranking when independent
> methods agree with each other. GRA is one of our four, not our only method."

### Numbers to have cold
- 8 phases total
- 4 independent MCDM methods (name all four in order: TOPSIS, GRA, PROMETHEE II, VIKOR)

### Anticipated Q&A
- **Q: "Why 4 methods instead of just picking the best one?"**
  A: Different MCDM methods make different implicit assumptions (e.g., TOPSIS assumes
  distance-to-ideal is meaningful; VIKOR allows compensatory trade-offs). A candidate that
  ranks well *only* under one method's assumptions is a red flag, not a robust winner —
  hence Borda/Copeland consensus and flagging disagreement rather than hiding it.
- **Q: "Isn't this just adding complexity for its own sake?"**
  A: No — cross-method disagreement caught real bugs during development (see the VIKOR
  sign-inversion bug on the MCDM results slide). A single-method pipeline would not have
  self-detected that.

---

## Slide 2 — "Multi-Criteria Ranking Formulae"
**Suggested time: 90 sec (this is a dense, formula-heavy slide — go slower)**

### Script
> "Two formulas anchor the feasibility side before ranking even starts. First, melting-point
> fitness — Tm is neither a pure benefit nor a pure cost criterion; being close to a target
> temperature is what matters, so we score it with a Gaussian centered on Tm-target.
>
> Second, the latent-heat requirement — this sets the feasibility floor. It comes from the
> overnight thermal load: draw rate times water's specific heat times the delivery-minus-mains
> temperature difference, divided by PCM mass. This is a ceiling requirement — a PCM has to
> clear this floor to be usable at all before we even rank it.
>
> For ranking itself: TOPSIS measures closeness to an ideal solution: geometric distance
> from the worst case over the sum of distances from best and worst. GRA computes a grey
> relational grade against an ideal reference. PROMETHEE II uses net outranking flow — how
> much a candidate outranks others minus how much it's outranked. VIKOR balances group
> utility and individual regret in a single compromise index.
>
> Weights come from a 50/50 blend of entropy — which is data-driven, rewarding criteria with
> real spread — and AHP, which reflects our guide's expert priors. And every ranking carries
> a confidence estimate: a 5,000-draw Monte Carlo perturbing both the weights and the PCM
> property values, giving us Top-3 inclusion probability and rank-reversal frequency, not
> just a single point-estimate ranking."

### Numbers to have cold
- σ (sigma) in the Gaussian Tm-fitness formula — **be ready to state its value and source**
  (per the Phase 6 audit: σ=4K, sourced to the framework doc §9.2, not independently
  literature-calibrated — say so plainly if asked, don't imply it's empirically fit)
- λ = 0.5 (entropy/AHP blend)
- **5,000-draw Monte Carlo is what this slide states — but flag internally**: the Phase 6
  audit documents the *actual Rajasthan run* used **N=1000**, not 5000, as a documented
  engineering tradeoff (5000 draws took 606s/cluster, "impractical for iteration"). If this
  slide is describing the same run as the audit, **1000 is the honest number to say out
  loud**, with 5000 mentioned only as "the framework doc's primary spec, with 1000 as its
  own documented fallback." Don't get caught stating 5000 as fact if a panelist asks "did
  you actually run 5000?"

### Anticipated Q&A
- **Q: "Why exp(-(Tm-target)²/2σ²) instead of a simple linear penalty?"**
  A: A Gaussian gives a smooth, differentiable falloff and matches the physical intuition
  that small deviations from target barely matter while large deviations should be
  strongly penalized — this is standard target-based fitness scoring in MCDM literature.
- **Q: "Where does σ=4K come from?"**
  A: It's sourced to the project's own framework document (§9.2), tied to the
  heat-exchanger approach temperature — **not independently calibrated from literature or
  data**. Say this plainly; the audit itself flags this as the honest answer.
- **Q: "Is AHP based on a real pairwise comparison you did?"**
  A: **No — be careful here.** Per the Phase 6 audit, `AHP_PAIRWISE_MATRIX = None`; the
  eigenvector-method AHP code exists but is never invoked. The run uses the framework
  doc's Table 13 indicative weights, unmodified except for corrosion's cluster-rescaling.
  If asked "did you do real AHP elicitation," the honest answer is **no, not yet** — this
  is a stated structural caveat, not something to gloss over.

---

## Slide 3 — "Full Pipeline Map (Phase 1–Phase 8)"
**Suggested time: 45–60 sec — this is a reference table, don't over-narrate it**

### Script
> "This table is the map we'll walk through slide by slide. Each row is one phase, its
> script, and its concrete output. I won't read every cell — the short version is: Phase 1
> is data acquisition, Phase 2 is preprocessing and validation, Phase 3 builds the climate
> signature, Phase 4 discovers climate regimes via clustering, Phase 5 builds the PCM
> database and screens for feasibility, Phase 6 ranks survivors, Phase 7 validates rankings
> against physics simulation, and Phase 8 packages the final recommendation cards."

### Numbers to have cold
- **Coverage %**: table says 90% — **know that the Rajasthan audit's actual constant is
  `COVERAGE_TARGET = 0.875` (87.5%)**. If this table is describing Rajasthan, correct
  yourself to 87.5% rather than repeat 90% under questioning. If it's genuinely Tamil
  Nadu's own (different) parameter, say so explicitly rather than let it look like a typo.
- **K_FINAL**: table says 5 — **the Rajasthan audit ground-truths k=3** via the documented
  3-tier selection cascade (bootstrap-ARI 0.8137 at k=3). Same instruction: know which
  state this slide is actually reporting before you're asked.

### Anticipated Q&A
- **Q: "Why does Phase 4 come before Phase 5 if clustering depends on the climate
  signature?"**
  A: Order is intentional — Phase 3 builds the per-point signature vector, Phase 4 clusters
  points into regimes using that signature, and Phase 5's feasibility screening is then run
  *per cluster*, using each cluster's aggregated climate profile (e.g., Tm_target_capped,
  L_required) rather than per individual point.
- **Q: "What's the actual difference between Phase 7 and Phase 8?"**
  A: Phase 7 is validation — it takes Phase 6's rankings and checks them against an
  independent grey-box physics simulation, computing Spearman rho between simulated
  performance and MCDM rank. Phase 8 is packaging — it re-imports Phase 6 to recompute the
  per-criterion contribution breakdown and generates the final markdown recommendation
  cards. Phase 7 is a check; Phase 8 is the deliverable.

---

## Slide 4 — "Data Acquisition I: Population-Weighted Sampling"
**Suggested time: 75 sec**

### Script
> "The first design decision was *where* to sample. Instead of a uniform grid, we used
> WorldPop's 2020 population raster, clipped to the state boundary via GADM, aggregated
> onto a quarter-degree grid. Critically, that grid is aligned to ERA5's own native grid
> origin — so every sampling point's cell center lands exactly on an ERA5 node. That gives
> a clean one-to-one mapping with zero interpolation error at download time, which a
> misaligned grid would not.
>
> We then rank cells by population and keep the minimum set covering our target share of
> the state's population — so we're not wasting sampling budget on empty desert or forest
> cells that would dilute a uniform average.
>
> For *when* to sample, every point gets its exact sunrise, solar noon, and sunset time for
> every day across our ten-year window, computed via the Solar Position Algorithm — accurate
> to a fraction of a degree, not a fixed clock-hour approximation."

### Numbers to have cold
- Grid resolution: 0.25°
- ERA5 grid origin: lat=90.0°, lon=−180.0°
- Coverage target — **see Slide 3's flag above; resolve 87.5% vs 90% before presenting**
- Date range: 2016–2025 (10 years)
- Algorithm name: Solar Position Algorithm (SPA) — **cite Reda & Andreas (2004)** if pressed
  for a source; the slide says "NREL Solar Position Algorithm," which is directionally
  correct (SPA was developed at NREL) but the actual literature citation is Reda & Andreas,
  *Solar Energy* 76(5), 2004 — have this ready if a panelist asks "which paper?"

### Anticipated Q&A
- **Q: "Why not just sample uniformly and weight the analysis afterward?"**
  A: Uniform sampling would spend equal budget on uninhabited and densely-populated cells,
  wasting API calls and download volume on climate data nobody's water heater will ever
  actually use. Population-weighting front-loads the budget onto where people live.
- **Q: "Doesn't grid-alignment bias your sample toward ERA5's resolution rather than true
  population density?"**
  A: The *aggregation* still uses the full-resolution 100m population raster — grid
  alignment only affects where the *cell boundaries* fall, so the population sums per
  cell are accurate; alignment just guarantees each selected cell maps to exactly one ERA5
  node instead of two nearby population cells silently colliding on the same node.
- **Q: "48 minutes at Tamil Nadu's longitude — where does that number come from?"**
  A: This is a known consequence of not using sun-event-aligned sampling — a fixed
  clock-hour grab (e.g., "download 12:00 UTC every day") can miss true solar noon by tens
  of minutes depending on longitude within a state/time-zone; have a source or your own
  calculation ready if pressed, since this specific figure isn't independently verified in
  the audit trail reviewed for this deck.

---

## Slide 5 — "Data Acquisition II: Dual-Source Climate Data"
**Suggested time: 75 sec**

### Script
> "We pull two independent climate sources for every point: ERA5 from Copernicus's Climate
> Data Store, and NASA POWER. ERA5 gives us instant variables like temperature, humidity,
> wind, and pressure, plus accumulated variables like solar radiation — and because we only
> download narrow windows around sunrise, noon, and sunset instead of the full 24 hours, we
> cut download volume substantially. NASA POWER gives us a full hourly cache for the same
> ten years, independently derived from MERRA-2, GEWEX surface radiation budget, and CERES
> — a genuinely independent data lineage from ERA5's model-driven radiation product.
>
> The reason for two sources, not one: a single-source pipeline would carry any processing
> fault silently downstream into clustering and ranking with nothing to catch it. We compute
> cross-source agreement — mean bias error, RMSE, Pearson r — at every matched sun-event
> timestamp, *before* any climate signature gets built."

### Numbers to have cold
- ERA5: instant + accumulated variable types
- NASA POWER: 87,660 hours/point (≈10 years hourly)
- File counts — **state-dependent**: the slide gives 240 NetCDF / 1330 JSON for a
  133-point run; the Rajasthan audit gives 240 NetCDF / **3200** JSON for a 320-point run
  (320 pts × 10 yrs). Match the file count to the point count you're actually presenting.
- Download volume reduction: "~75%" — flagged in the earlier content-audit as **not stated
  anywhere in the Phase 1 audit doc** — treat as an estimate you should be able to defend
  or soften if pressed ("substantially reduced" is a safer phrasing if you can't source
  the exact 75% figure).

### Anticipated Q&A
- **Q: "Why not just trust ERA5 alone — it's the more widely-used reanalysis product?"**
  A: ERA5 is model-driven (a physics-based reanalysis), while NASA POWER's radiation
  product draws on independent satellite-derived data (GEWEX-SRB, CERES) — different
  error characteristics. Relying on one source means a systematic bias in that source's
  radiation model propagates invisibly into every downstream stage.
- **Q: "You rejected PVGIS TMY — why not use a Typical Meteorological Year, which is the
  industry-standard approach for solar system sizing?"**
  A: TMY is a single synthetic representative year, which is appropriate for steady-state
  system sizing but discards inter-annual variability — and our pipeline explicitly wants
  to capture real year-to-year variation (e.g., in `kt_worst_month`, `cloudy_frac`) since
  those variability metrics directly feed the PCM feasibility targets.

---

## Slide 6 — "Preprocessing and Cross-Source Validation"
**Suggested time: 90 sec — this slide has your best "we caught a real bug" story, use it**

### Script
> "Before any of this data is trusted, it's checked and cross-verified. We compute the WMO
> standard reanalysis-validation triplet — Pearson r, mean bias error, and RMSE — between
> ERA5 and POWER at every matched timestamp. Outliers are removed with a Hampel filter — a
> rolling median plus a multiple of median-absolute-deviation, which is robust to the
> heavy-tailed, right-skewed nature of solar irradiance data in a way a simple z-score cutoff
> isn't. Missing values are filled with MICE imputation, which preserves cross-variable
> relationships like the temperature-humidity trade-off, rather than filling each column
> independently.
>
> We deliberately rejected Bland-Altman analysis, because it assumes two equally-reliable
> instruments — but ERA5 is a model and POWER is satellite-derived, they're not symmetric.
> We also rejected a simple IQR outlier rule, because it would flag genuine heat-wave
> extremes as errors rather than real signal.
>
> This validation step did its job for real: our first agreement run flagged near-zero
> correlation at solar noon — r of about 0.01. We traced that to a genuine bug in how ERA5's
> accumulated variables were being de-accumulated. After the fix, overall correlation came
> to r=0.81. That's not a hypothetical safeguard — that's the two-source check catching a
> real, would-have-been-silent error before it reached the clustering stage."

### Numbers to have cold
- Hampel filter window: 7-day rolling
- Pre-fix: r ≈ 0.01 at solar noon
- Post-fix: **r = 0.81 overall** (per `bias_decision_rajasthan.txt`, the precise figure is
  r=0.8102 — "0.81" is a safe rounding)
- ⚠️ **Be precise about what the bug actually was.** This slide's script text says "a
  genuine unit-conversion bug"; the underlying audit calls it an **accumulation-convention
  mismatch in `deaccumulate()`** — i.e., how ERA5's accumulated (forecast-type) fields were
  being converted from cumulative totals to per-timestep fluxes, not a simple unit
  conversion (like W/m² vs J/m²) in isolation. If a technically sharp panelist asks "what
  exactly was wrong," "unit conversion" is close enough for a general audience but you
  should be ready to say "deaccumulation" if pushed, since that's the audit's own term.

### Anticipated Q&A
- **Q: "How did you catch the deaccumulation bug — what tipped you off?"**
  A: The cross-source agreement check itself — r≈0.01 and a large negative MBE at solar
  noon specifically was the signal. A near-zero correlation at exactly the timestamp where
  solar radiation should be most reliable and most strongly correlated between two
  independent sources was the anomaly that triggered investigation.
- **Q: "Since you found and fixed one bug in ERA5's processing, how confident are you
  there isn't a similar unfixed issue elsewhere in the pipeline?"**
  A: This is a fair question — the honest answer is that this project has an active
  self-audit culture (documented, dated bug fixes appear at multiple later phases too —
  e.g., a VIKOR sign-inversion bug and an entropy-weight-inflation bug were both caught
  the same way, via cross-method or cross-source disagreement diagnostics). That's a
  process argument, not a guarantee — but it's evidence the pipeline is built to
  self-detect this class of error rather than assume correctness.
- **Q: "Why is a quantile-mapping correction for GHI bias mentioned elsewhere but not
  applied here?"** *(only answer if this comes up — don't volunteer it)*
  A: The r=0.81 agreement check and the quantile-mapping correction are two different,
  later steps. Quantile mapping corrects a smaller residual *bias* in GHI specifically
  (mean bias ~13 W/m² pre-correction), computed *after* the deaccumulation fix — it's
  currently a diagnostic-only correction, not yet applied to the dataset Phase 3 onward
  consumes. Worth stating as an open methodological item if directly asked, not as a flaw
  to hide.

---

## Slide 7 — "Two-Tier Climate Signature Construction"
**Suggested time: 75 sec**

### Script
> "Once the data is clean, we collapse each point's ten years of records into a compact
> climate signature. Tier 1 captures conditions at the three moments that actually matter
> physically for a PCM system: sunrise, solar noon, and sunset — when charging starts, when
> charging peaks, and when discharge begins. Tier 2 adds true daily totals computed by
> trapezoidal integration over the full hourly series — daily solar energy, temperature
> range, cloudy-day frequency — things three snapshot-in-time samples per day can't capture
> on their own. Finally, PCA compresses the correlated temperature-family variables into
> four components, removing redundancy before clustering.
>
> We rejected monthly averages — the common approach in the literature — because they
> smooth out short, sharp monsoon bursts that matter a great deal for storage sizing. We
> also rejected full time-series dynamic time warping as computationally infeasible at
> this scale."

### Numbers to have cold
- Tier 1: 3 sun-events (sunrise, noon, sunset)
- PCA: → 4 components
- ⚠️ Per the Rajasthan audit, the PCA block explains **>95% variance** with 4 retained
  components — data-determined, not a fixed target. If asked "why 4," the honest answer
  is "that's how many components were needed to reach the 95% variance-retained
  threshold for this state's data," not a number chosen in advance.

### Anticipated Q&A
- **Q: "Why sun-event timestamps instead of just using daily averages for everything?"**
  A: Because the downstream PCM targets are physically tied to specific moments — Tm_target
  and the interaction terms are built from sunrise/noon/sunset-specific stats, not daily
  means, because those are the moments a real system actually charges and discharges at.
- **Q: "Doesn't PCA make the clustering harder to interpret physically?"**
  A: Loadings and explained-variance ratio are retained and reported, not discarded — so
  each component can still be traced back to which raw physical variables it's summarizing.
  The signature isn't a black box; the PCA step compresses the *correlated temperature/
  elevation* block only, while interaction terms and PCM-facing targets stay in physical
  units, untouched by PCA.

---

## Slide 8 — "Climate Regime Clustering"
**Suggested time: 90 sec**

### Script
> "Rather than hand-drawing climate zones, we discover them empirically with a Gaussian
> Mixture Model, testing k from 2 up through the low double digits. The best k is chosen
> using a combination of BIC, silhouette score, and — importantly — bootstrap stability: we
> resample the data fifty times, refit, and measure how consistently the same points end up
> in the same cluster. We pick k only when the clustering both falls in the range we'd
> physically expect for a single state and clears a realistic silhouette band — not just
> whichever k minimizes BIC, because BIC alone tends to keep decreasing all the way to the
> edge of the scan without a meaningful interior minimum.
>
> We chose GMM over k-means or hierarchical clustering because climate doesn't change in
> sharp boundaries — a location near a transition zone can genuinely have partial membership
> in two regimes, which GMM's soft, probabilistic assignment captures and a hard-boundary
> method like k-means cannot.
>
> A separate seasonal analysis checks whether the same point's climate regime — and
> therefore its top PCM recommendation — shifts materially across the four seasons, which
> matters for a state with a strong monsoon contrast."

### Numbers to have cold
- k-scan range: 2–10 or 2–12 depending on level (Level A: 2–12 per the Rajasthan audit;
  this slide says 2–10 — **minor discrepancy, know which range your actual run used**)
- Bootstrap resamples: 50
- **Final k — this is the biggest state-dependent number in the whole deck.** Slide says
  K_FINAL=5; Rajasthan audit ground-truths **k=3** with bootstrap-ARI 0.8137. Know which
  state you're presenting and state the correct number for it.
- Covariance type: `diag`, not `full` — **have the reason ready** (see Q&A below) since
  this is a documented, real bug fix worth being able to explain if asked why "diag"
  specifically.

### Anticipated Q&A
- **Q: "Why diagonal covariance instead of full covariance — doesn't that throw away
  information about how features correlate within a cluster?"**
  A: At this dimensionality (35 standardized columns) with only ~100 points per cluster,
  full covariance requires far more parameters than data points — badly underdetermined.
  This was caught because membership probabilities were saturating to ~1.0 for
  essentially every point despite only moderate silhouette — a sign the covariance
  estimate was numerically extreme, not that the clusters were genuinely that
  well-separated. Diagonal covariance restored a realistic spread of membership
  confidence while barely moving silhouette (0.303 vs 0.309) — confirming the fix changed
  *how confidently* the model reports its answer, not *what* the answer is.
- **Q: "How do you know cluster 0 means the same thing every time you re-run this?"**
  A: It doesn't, by default — sklearn's GMM gives no guarantee cluster indices are stable
  across re-runs, even with a fixed random seed. This was caught when two downstream
  phases disagreed on which PCMs belonged to which cluster. The fix: clusters are
  canonically relabeled 0..k-1 by sorting on mean latitude (south to north) — a
  fit-independent ordering key — and a provenance fingerprint check now hard-fails any
  downstream phase whose input doesn't match the current cluster profile.
- **Q: "How do your clusters compare to an established climate classification, like
  Köppen-Geiger?"**
  A: We validate externally against Köppen-Geiger (Beck et al. 2018) — agreement is
  low-to-moderate (ARI≈0.19, NMI≈0.32). We read this as the GMM finding climate structure
  at a finer resolution than Köppen's broad classes capture within a single state, which
  is arguably the point of empirical clustering rather than applying a coarse
  classification directly — not evidence the clustering failed.

---

## Slide 9 — "PCM Database and Feasibility Screening"
**Suggested time: 75 sec**

### Script
> "Before any ranking happens, we build a trustworthy shortlist. Our database has
> fifty-five PCMs — thirty-one from manufacturer datasheets across five manufacturers,
> twenty-four from published literature — expanded this month from an earlier
> twenty-five-PCM version specifically to close the row-count gap against our own
> forty-to-sixty target. Where properties are missing, we fill them with a Random
> Forest–based estimator that respects known physical relationships, like how density and
> thermal conductivity and phase tend to move together.
>
> Then we apply eight hard filters — melting range, latent-heat floor, cycling stability,
> safety, and more — *before* any ranking method sees the data. The reason filtering comes
> first: ranking methods can be fooled. A PCM with a badly-wrong melting point but
> excellent latent heat could still score deceptively well on a weighted-sum ranking.
> Filtering first removes anything physically unworkable, so ranking only ever compares
> realistic candidates. And if fewer than five PCMs survive, the melting-window filter is
> relaxed gradually rather than leaving the cluster with an empty candidate pool."

### Numbers to have cold
- Database size: 55 PCMs (31 datasheet across 5 manufacturers + 24 literature), expanded
  2026-08-12 from an earlier 25-PCM version, now inside the 40–60 PCM target
- 8 hard filters
- Minimum survivor threshold before filter relaxation: 5

### Anticipated Q&A
- **Q: "55 PCMs — is that actually enough now, and are your results based on it?"**
  A: **Be precise here — the database expansion and the pipeline re-run are two separate
  things.** The database itself now meets our stated 40–60-row target (up from 25), closing
  that specific gap. But the ranking numbers on the following slides (Kendall's W, the
  Spearman rho physics-validation result) were generated *before* this expansion, against
  the older 25-PCM pool, and have not yet been regenerated against the expanded database —
  say so plainly if asked which numbers are current. Re-running Phases 5 through 8 against
  the 55-PCM database is the immediate next step, not a hypothetical future one.
- **Q: "How do you justify Random Forest imputation instead of just excluding PCMs with
  missing data?"**
  A: Missingness here is structural, not random — entire property columns are missing for
  100% of certain manufacturer product lines (e.g., thermal conductivity for the RT
  series). Excluding those PCMs would mean discarding a large fraction of the database
  over a systematic reporting gap, not a data-quality problem with those specific PCMs.
  Imputation is verified via a cross-manufacturer donor audit and physical-constraint
  checks post-imputation, not left unvalidated.

---

## Slide 10 — "PCM Property Data: Cleaning & Imputation"
**Suggested time: 75 sec**

### Script
> "This is the detail behind that 55-PCM database. Raw manufacturer datasheets come in with
> wildly inconsistent formatting — a regex parser handles every messy variant we found:
> ranges become midpoints, approximate values get their symbols stripped, genuinely
> non-numeric entries like 'under trial' are correctly left as missing rather than forced
> into a number.
>
> Before imputing anything, we diagnosed the missingness pattern by product line — and
> that's what revealed it's structural: certain thermal conductivity and specific heat
> columns are missing for 100% of one manufacturer's product line, and a different property
> is missing for 100% of another's. That pattern shaped every downstream choice.
>
> We impute using MICE with a Random Forest — each missing column predicted from every
> other column, cycling through eight rounds — refined with Predictive Mean Matching, so
> every filled cell is a distance-weighted blend of the three nearest *real, measured*
> donor values, never a raw synthetic model output. We then explicitly audit which
> manufacturer supplied each donor value, confirming genuine cross-manufacturer donation
> rather than a model quietly copying values within the same product line. Finally, we
> enforce physical constraints post-imputation and engineer derived features like
> volumetric energy density."

### Numbers to have cold
- MICE rounds: 8
- PMM donor pool: 3 nearest real values
- Structural missingness examples: TC_liquid/TC_solid/Cp_solid → 100% missing for RT-line;
  heat_storage_Wh_kg/TC_both → 100% missing for savE-line

### Anticipated Q&A
- **Q: "Predictive Mean Matching — why not just trust the Random Forest's direct
  prediction?"**
  A: A raw regression prediction can produce a value that was never actually observed in
  any real PCM — physically implausible interpolation. PMM constrains every imputed value
  to be a blend of values that were *actually measured* somewhere in the dataset, which is
  the standard defense against exactly that failure mode.
- **Q: "How do you know the donor audit actually worked, rather than just trusting the
  code?"**
  A: The audit is a logged, inspectable output — it records which manufacturer supplied
  each donor value per imputed cell, and confirms 100% cross-manufacturer donation for
  every column that was missing across an entire product line. That's a verification step
  with a persisted artifact, not a claim taken on faith.

---

## Slide 11 — "MCDM Ranking Engine: Results & Self-Audit Findings"
**Suggested time: 90 sec — expect the toughest questions on this slide**

### Script
> "This is where the four ranking methods actually run, and where our self-audit process
> caught three real bugs worth talking about honestly.
>
> First: a VIKOR sign inversion. An early version of the compromise-index formula had the
> wrong sign, which silently inverted the entire VIKOR ranking — we caught this because our
> pairwise method-agreement diagnostic showed VIKOR almost totally disagreeing with TOPSIS
> and PROMETHEE, a Spearman correlation as low as negative 0.86. That's not a subtle
> disagreement; that's a red flag a properly-designed diagnostic is supposed to catch.
>
> Second: entropy-weight inflation. A criterion with essentially no real data — cost, which
> is always missing in our current database — was, due to how the entropy formula handles
> near-empty columns, receiving the *highest possible* weight. In the first run that
> inflated cost's weight to over sixty percent in every cluster. We fixed this by zeroing
> the weight for any criterion with fewer than two real values, bypassing the entropy
> formula entirely for structurally-empty criteria.
>
> Third, a provenance check: we now fingerprint the cluster data going into this stage and
> hard-fail if it doesn't match what Phase 4 and Phase 5 actually produced — this exists
> because we found Phase 5 and Phase 6 disagreeing, cluster by cluster, on which PCMs
> belonged to which cluster, traced to cluster-index instability across separate re-runs.
>
> On the actual result: Kendall's W — our cross-method agreement statistic — comes in at
> 0.44, 0.54, and 0.59 across the three clusters. None of those clear the 0.8 threshold we'd
> call 'strong agreement.' That's an honest, reported result, not a number we're hiding —
> and it's consistent with a currently undersized PCM database, which we've flagged as
> provisional throughout."

### Numbers to have cold — **these are the most likely to be probed, know them exactly**
- Kendall's W: Cluster 0 = 0.4375 (n=5, flagged undersized), Cluster 1 = 0.536,
  Cluster 2 = 0.589
- W thresholds: >0.8 strong, <0.6 ambiguous
- Entropy-weight domination: Tm_fitness ~48–49% in clusters 0/1, supercooling 56.5% in
  cluster 2 — both above the pipeline's own 40% "near-total-domination" flag
- N_DRAWS actually used: **1000**, not 5000 (see Slide 2's flag — be consistent)
- AHP: **not actually elicited** — Table 13 priors used unmodified except corrosion rescaling

### Anticipated Q&A
- **Q: "A Kendall's W of 0.44 in one cluster sounds like your methods don't actually
  agree. Doesn't that undermine the whole four-method approach?"**
  A: It's an honest finding, not a failure to hide. Two things soften it: cluster 0's
  survivor pool is undersized (n=5) — low sample count naturally produces less stable
  rank agreement regardless of method quality — and this is exactly why we run Phase 7's
  independent physics validation rather than stopping at the MCDM stage. This 0.44 was
  measured against our earlier 25-PCM database; we've since expanded it to 55 PCMs and the
  re-run that would show whether cluster 0's agreement improves is our immediate next step,
  not yet done.
- **Q: "If cost has no real data, why include it as a criterion at all instead of dropping
  it?"**
  A: It's currently a structural placeholder, correctly weighted to zero by the fix
  described — but kept as a schema slot because cost data is a planned addition, not
  something permanently absent. Dropping the column entirely would require re-deriving
  weights and re-running when cost data does arrive; keeping the slot with a
  correctly-zeroed weight is a cleaner path forward.
- **Q: "You call your AHP weights 'guide-elicited priors' on the formula slide, but here
  you're saying AHP wasn't actually run — which is it?"**
  A: **Be very careful and precise here** — this is the sharpest inconsistency in the
  deck if a panelist cross-references both slides. The honest answer: the weights
  currently used *are* the framework document's Table 13 priors (which were originally
  set with guide input) — but the *project's own* pairwise-comparison AHP elicitation
  process, with its consistency-ratio check, exists as working code and has not yet been
  run on this specific database. Don't claim "real AHP elicitation happened" if directly
  asked to clarify — say the current run uses documented starting priors, with full AHP
  elicitation as a planned, not-yet-executed step.

---

## Slide 12 — "Physics Validation and Final Recommendations"
**Suggested time: 75 sec**

### Script
> "The last check before packaging results: does the top-ranked PCM actually perform well
> in a realistic simulation, not just on paper? We built a simplified tank model — water
> and PCM coupled through a heating coil — and simulate a full year of real, location-
> specific historical weather, not an idealized synthetic solar curve. It's solved with a
> numerically stable time-stepping method suited to the system's fast internal dynamics.
> Each climate regime's simulated performance is then compared against its MCDM ranking
> using a Spearman rank-correlation check.
>
> We rejected commercial tools like TRNSYS or EnergyPlus — they either require paid
> licenses or lack native PCM support, both impractical for the repeated, automated
> simulation runs this pipeline needs.
>
> The final output is a reproducible report per climate regime: the top-3 recommended
> PCMs, the confidence behind that ranking, and the simulated performance — generated
> automatically, with no manual editing."

### Numbers to have cold
- Output: top-3 PCMs per climate regime
- Validation metric: Spearman ρ (rank correlation) between MCDM rank and simulated
  performance
- ⚠️ **This slide's narration implies validation succeeded — check this before presenting
  as a success.** Per the Phase 6 audit's closing note: Phase 7 actually returned Spearman
  ρ ≤ 0.4 for **all three clusters** — a genuine **negative** validation result, at least
  partly attributable to the same undersized (25-PCM) database this figure was measured
  against. That database has since been expanded to 55 PCMs, but Phase 7 has not yet been
  re-run against it, so this ρ ≤ 0.4 figure is still the current, only-available number. If
  asked "did the physics validation confirm your rankings," **the honest answer is currently
  no** — say so, and frame it the way the audit does: not evidence the ranking is wrong, but
  evidence that expects re-checking once the (now-complete) database expansion is
  propagated through a fresh Phase 5–8 run.

### Anticipated Q&A
- **Q: "Did the physics simulation actually confirm your top-ranked PCMs perform best?"**
  A: **Not yet, and that should be said directly.** Spearman correlation between the
  simulated performance and the MCDM ranking came in at or below 0.4 across all three
  clusters — a weak, not-yet-confirming result. We read this alongside Phase 6's own
  self-flagged caveat that every ranking is provisional given the (then) 25-PCM database
  it ran on — the negative validation result and the undersized database are very plausibly
  linked, not independent problems. This is presented as an honest open finding. The
  database has since been expanded to 55 PCMs, meeting our stated target; re-running
  Phase 5 through 8 against it — not further database work — is now the immediate next
  step.
- **Q: "Why build your own tank simulation instead of validating against real installed
  systems?"**
  A: Real-world field data collection at this scale (multiple climate regimes, controlled
  comparison across PCM candidates) wasn't feasible within project scope — the grey-box
  simulation is a middle ground between pure ranking (no physical check at all) and full
  field deployment, and uses real historical weather rather than synthetic curves to stay
  as close to real conditions as the constraint allows.

---

## Cross-Slide Consistency Checklist (review before presenting)

- [ ] Confirm which state (Rajasthan vs. Tamil Nadu) this specific slide deck describes,
      and align every number accordingly: coverage % (87.5 vs 90), point count (320 vs
      133), K_FINAL (3 vs 5), file counts (3200 vs 1330 POWER JSONs)
- [ ] Decide whether to say "5,000-draw Monte Carlo" (the spec) or "1,000-draw" (what
      actually ran, per the audit) — pick one and be consistent across Slides 2 and 11
- [ ] Be ready to clearly separate "framework doc's stated priors" from "actually-elicited
      AHP weights" if Slides 2 and 11 are both discussed in the same session — this is the
      single most likely internal-consistency trap
- [ ] Decide how to frame Slide 12's validation outcome — presenting it as unqualified
      success does not match the audit's own ground-truthed Spearman ρ ≤ 0.4 result
