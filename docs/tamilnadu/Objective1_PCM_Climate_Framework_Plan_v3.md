**OBJECTIVE 1 --- IMPLEMENTATION PLAN**

**Climate-Region-Aware PCM Recommendation Framework**

*Clustering multi-year meteorological data and identifying Top-2 / Top-3
PCM candidates per climatic regime by multi-criteria decision-making,
validated against a physics-based thermal model*

**Climate-Adaptive Intelligent Control and Optimization of PCM Thermal
Storage for Solar Water Heating**

Group 12 · B.Tech Computer Science & Engineering (Final Year)

Amrita School of Engineering · Guide: Dr. T. Deepika

**Document version 3.0**

*Supersedes v2.0. Revised to match the data-collection pipeline as
actually built: four contrasting Indian states, population-weighted ERA5
grid points aligned to the ERA5 grid origin, and sun-event-aligned
temporal sampling with an independent NASA POWER cross-check. All
changes from v2.0 are listed in §0.*

0\. What Changed in Version 3.0

Version 2.0 was written before data collection began and assumed an
all-India, city-point, full-hourly ERA5 dataset. The pipeline that was
actually built --- on the guide\'s direction --- differs in three
structural ways: it covers four states rather than the whole country, it
samples population-weighted grid cells rather than named cities, and it
samples three astronomically computed sun-event instants per day rather
than all 24 hours. Two of those three changes improve the design. The
third requires a specific, non-optional repair, described below and in
§6.

  -------------------------------------------------------------------------
  **Item**        **v2.0 (planned)**    **v3.0 (as built, plus required
                                        repairs)**
  --------------- --------------------- -----------------------------------
  **Geographic    \~30 named cities     Four states: Rajasthan, Assam,
  scope**         spanning all of India Tamil Nadu, Uttarakhand. Depth over
                                        breadth --- four contrasting
                                        climate families sampled densely,
                                        rather than all of India sampled
                                        thinly. Stronger for validation,
                                        narrower for generalisation claims
                                        (§1.3).

  **Sampling      One ERA5 grid point   Population-weighted 0.25° cells
  unit**          per named city        aligned to the ERA5 grid origin,
                                        retaining the minimal set covering
                                        \~87.5 % of each state's
                                        population. This is a genuine
                                        methodological improvement and
                                        becomes novelty claim N6.

  **Temporal      All 24 hours, 10      Sunrise, solar noon and sunset per
  sampling**      years                 point per day, 2016--2025, computed
                                        by the pvlib SPA algorithm.
                                        Physically well aligned to the PCM
                                        charge--discharge cycle --- but
                                        insufficient on its own for
                                        daily-integral indices.

  **Second data   CERES satellite       NASA POWER hourly at identical
  source**        radiometry (planned,  points and instants, already
                  not obtained)         downloaded. This is a better
                                        cross-check than planned: two
                                        independent estimates of the same
                                        quantity at the same instant and
                                        location.

  **Bias          Quantile mapping of   ERA5-versus-POWER agreement
  correction**    ERA5 solar against    analysis at matched instants first
                  CERES                 (MBE, RMSE, correlation per season
                                        per point). Quantile mapping
                                        applied only if a systematic
                                        seasonal bias is demonstrated.
                                        Fixed-weight blending of the two
                                        remains rejected.

  **Climate       18 indices assuming   Restructured into two tiers (§6.2).
  signature**     full hourly input     Tier 1 sun-event indices come from
                                        the merged CSV. Tier 2
                                        daily-integral indices are
                                        recomputed from the NASA POWER
                                        hourly cache already on disk. No
                                        new ERA5 download is required.

  **Elevation**   elev_proxy index in   REQUIRED REPAIR. The pipeline
                  the signature         assumes a flat 300 m for solar
                                        geometry. That is materially wrong
                                        for Uttarakhand, which spans
                                        roughly 200 m to over 7,000 m.
                                        Per-point elevation must be
                                        attached from ERA5 surface
                                        geopotential or an SRTM DEM, and
                                        solar geometry recomputed for the
                                        mountain points.

  **Clustering    Discover the climate  Discover intra- and inter-state
  framing**       regions of India      regimes across four contrasting
                                        states. State identity becomes an
                                        external validation label rather
                                        than a result --- recovering the
                                        four state boundaries alone would
                                        be a trivial finding (§7.1).

  **Expected k**  5--7, checked against 6--10. Four states with expected
                  5--6 NBC zones        internal splitting: arid west
                                        versus semi-arid east in Rajasthan,
                                        terai versus mid-hills versus high
                                        Himalaya in Uttarakhand, coast
                                        versus interior versus Nilgiris in
                                        Tamil Nadu, valley versus hills in
                                        Assam.

  **Novelty       N1--N5                N6 added: population-weighted,
  claims**                              deployment-relevant
                                        regionalisation. Regimes are
                                        weighted by where people actually
                                        live, therefore by where solar
                                        water heaters would actually be
                                        installed.

  **Timeline**    16 weeks from zero    12 weeks remaining. Phases 1 and 2
                                        are substantially complete; the
                                        schedule is re-baselined in §12.
  -------------------------------------------------------------------------

*Table 0. Change log from v2.0 to v3.0. Two structural changes improve
the design; the elevation assumption and the sun-event-only merged
output require repair before Phase 3 can proceed.*

  -----------------------------------------------------------------------
  **The good news, stated plainly.** The sampling design is better than
  the one this plan originally specified. Population weighting means the
  regimes describe where installations would actually go, not where grid
  cells happen to fall. Sun-event sampling is not an arbitrary subsample
  --- sunrise is the coldest instant and therefore the solidification
  test, solar noon is the peak charging condition, and sunset is the
  start of the discharge period. Those are precisely the three instants a
  PCM store cares about. Say this explicitly in the paper; a reviewer
  will otherwise read three-samples-per-day as a shortcut rather than a
  design.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **The repair that cannot be skipped.** The merged CSV keeps only the
  three sun-event rows per point-day. Several signature indices --- daily
  GHI integral, true diurnal temperature range, heating and cooling
  degree days, cloudy-day fraction, consecutive-cloudy-day index ---
  cannot be computed from three instantaneous samples. They do not need a
  new ERA5 request: the NASA POWER raw cache at
  data/raw/nasapower/power\_{point_id}\_{year}.json already holds the
  full hourly series for every point and year. The merge step subsets it
  to sun events and discards the remaining 8,757 hours. Recovering the
  Tier 2 indices is a read over files already on disk (§6.2).

  -----------------------------------------------------------------------

Contents

*(In Word: Ctrl+A then F9, or right-click the table below and choose
\"Update Field\", to populate page numbers.)*

1\. Scope and Objective Decomposition

1.1 The objective, restated precisely

*Develop a climate-region-aware recommendation framework that clusters
ten years of population-weighted meteorological data across four
contrasting Indian states, and for each discovered climatic regime
identifies the Top-2/Top-3 suitable PCM candidates for solar domestic
hot water storage using multi-criteria decision-making, with an explicit
confidence measure and independent physics-based validation of the
resulting ranking.*

Decomposed into four verifiable sub-goals:

-   SG1 --- Assemble a ten-year, multi-point meteorological dataset for
    Rajasthan, Assam, Tamil Nadu and Uttarakhand, sampled at
    population-weighted locations and at instants aligned to the PCM
    charge--discharge cycle, with an independent second source for
    cross-validation.

-   SG2 --- Reduce each location's ten-year record to a compact,
    physically meaningful climate signature vector, then cluster those
    signatures into a small number of climate regimes spanning and
    subdividing the four states.

-   SG3 --- For each regime, filter a PCM database to a feasible
    candidate set and rank it by MCDM, returning an ordered Top-3 with a
    confidence measure rather than a single winner.

-   SG4 --- Validate the ranking against an independent physics-based
    thermal simulation, not against the MCDM's own scores.

1.2 What is deliberately out of scope

This objective is a regional, offline selector. It answers "what PCM
should be specified for a system installed in this climate regime?" ---
not "what should the controller do tomorrow?" and not "what should the
valve do right now?".

**Scope discipline.** The regional selector is the foundation: it
narrows a 40--60 PCM database to 2--3 candidates per regime, and only
those candidates ever enter the day-ahead layer or the DRL controller.
Building the regional layer first means the forecasting objective
inherits a validated shortlist instead of guessing one. Fusing the
layers now makes it impossible to attribute any result to either
mechanism.

Explicitly out of scope for Objective 1:

-   Time-series forecasting of any variable. Historical climatology is
    the input here, not a forecast.

-   Real-time control, charge/discharge policy, DRL. That is the control
    objective.

-   Hardware, sensors, embedded deployment.

-   PCM synthesis or experimental characterisation. Published
    thermophysical property data is consumed, not generated.

-   Field trials. Recorded as future work in §14.

-   Extension to the remaining Indian states. The framework is
    state-agnostic by construction, and §14.1 records the extension as
    future work, but no claim about states outside the four should
    appear in the paper.

1.3 Why these four states --- and how to defend the choice

The obvious reviewer question is why not all of India. The answer is
that these four states are not an arbitrary subset: they span the widest
climatic contrast available within India while allowing dense sampling
inside each.

  ------------------------------------------------------------------------------
  **State**         **Climate family**    **What it contributes to the study**
  ----------------- --------------------- --------------------------------------
  **Rajasthan**     Hot-dry / arid to     Highest solar availability, largest
                    semi-arid (Thar       diurnal range, lowest cloud
                    desert to eastern     persistence. The regime where charging
                    Aravalli)             is never the constraint and cycling
                                          stress is highest. Expected internal
                                          split: arid west versus semi-arid
                                          east.

  **Assam**         Warm-humid,           Lowest clearness index, longest
                    monsoon-dominated     consecutive-cloudy-day runs, highest
                    (Brahmaputra valley)  humidity stress. The regime that sets
                                          the latent-heat floor and drives
                                          salt-hydrate corrosion exclusion.
                                          Expected internal split: valley versus
                                          surrounding hills.

  **Tamil Nadu**    Warm-humid coastal    A north-east monsoon regime that is
                    and hot semi-arid     out of phase with the rest of the
                    interior, with a      country --- valuable because it breaks
                    montane exception     any assumption that Indian seasonality
                                          is uniform. Expected internal split:
                                          coastal, interior dry, and the
                                          Nilgiris.

  **Uttarakhand**   Cold / temperate,     The only cold regime in the study, and
                    extreme elevation     the one where mains water temperature
                    gradient              is lowest and the latent-heat
                                          requirement therefore highest.
                                          Expected internal split: terai plains,
                                          mid-hills, high Himalaya. Also the
                                          state where the flat-300 m elevation
                                          assumption does the most damage
                                          (§4.3).
  ------------------------------------------------------------------------------

*Table 1. The four study states. Between them they cover four of the
five or six NBC/ECBC climate zones --- hot-dry, warm-humid, composite
and cold --- which is what makes a four-state study defensible rather
than merely convenient.*

**State the limitation honestly.** The one NBC zone not represented is
temperate as classified in the Indian standard (the Bengaluru plateau
type). Say so in the limitations section rather than letting a reviewer
find it. Four zones out of five or six, sampled densely and
population-weighted, is a stronger dataset than six zones sampled at one
city each.

1.4 Deliverables

  ----------------------------------------------------------------------------
  **ID**    **Deliverable**                              **Status**
  --------- -------------------------------------------- ---------------------
  **D1**    Ten-year meteorological dataset for four     Substantially
            states at population-weighted points,        complete --- pipeline
            sun-event aligned, with matched NASA POWER   built and run
            cross-check                                  

  **D1b**   Tier 2 daily-aggregate table recovered from  REQUIRED --- see §4.3
            the NASA POWER hourly cache; per-point       
            elevation attached                           

  **D2**    PCM property database, 40--60 candidates in  Outstanding
            the corrected 42--70 °C band, every row      
            cited                                        

  **D3**    Climate signature feature matrix, one row    Outstanding
            per point, two-tier index set                

  **D4**    Cluster model with k-selection evidence,     Outstanding
            cross-checked against state identity,        
            Köppen--Geiger and NBC/ECBC zones            

  **D5**    MCDM ranking engine: entropy+AHP weights,    Outstanding
            four ranking methods (+ optional CoCoSo),    
            aggregation, Monte Carlo                     

  **D6**    Top-3 PCM table per regime with consensus    Outstanding --- the
            score, stability percentage and population   headline result
            coverage                                     

  **D7**    Physics validation: simulated annual solar   Outstanding
            fraction per feasible PCM versus MCDM rank,  
            Spearman ρ                                   

  **D8**    IEEE conference paper, 6--8 pages            Outstanding
  ----------------------------------------------------------------------------

*Table 2. Deliverables and current status. D6 is the headline result; D7
is what makes D6 defensible.*

2\. Response to the Critical Review

The external review claimed four errors. Each was checked against
primary sources. Summary verdicts first; reasoning follows.

  --------------------------------------------------------------------------
  **Claimed error**     **Verdict**   **Action taken**
  --------------------- ------------- --------------------------------------
  **#1 Undefined        Partially     Algorithm and validation are now
  clustering            correct       stated explicitly (they were implicit
  methodology --- use                 in v1.0). The specific prescriptions
  K-Means + STL,                      are rejected: GMM retained over
  silhouette \> 0.75**                K-Means, STL rejected, silhouette
                                      target set at a realistic 0.15--0.35.

  **#2 Undefined MCDM   Wrong as a    Method was already fully specified in
  method --- adopt      replacement   v1.0. Four-method consensus retained.
  AHP + CoCoSo**                      CoCoSo added as an optional fifth
                                      ranker. The review\'s supporting
                                      statistic could not be traced to any
                                      source.

  **#3 Ambiguous PCM    Wrong for     These are building passive-cooling
  criteria --- hot-arid this          targets. Replaced with an SWH-specific
  35--40 °C, tropical   application   melting-temperature rule anchored to
  humid 28--32 °C**                   delivery and collector temperatures
                                      (§6.3). The review did, however,
                                      correctly expose a sign error in the
                                      v1.0 rule.

  **#4 No validation    Partially     v1.0 already had physics validation.
  strategy ---          correct       EnergyPlus rejected as technically
  EnergyPlus + Monte                  incapable of the task; TRNSYS Type 860
  Carlo + field trial**               named as the optional cross-check.
                                      Monte Carlo retained at 5,000. Field
                                      trial moved to future work.
  --------------------------------------------------------------------------

*Table 2. Verdicts on the four claimed errors.*

2.1 Error #1 --- clustering methodology

**What is correct.** Algorithm choice does materially affect cluster
structure, and Objective 1 should state its algorithm, features, and
validation criteria explicitly rather than saying \"clusters data\".
v2.0 does so in §7.

**What is wrong --- the silhouette target.** The review cites an Indian
urban rainfall clustering study reporting DBSCAN 0.82, Spectral 0.80,
K-Means 0.70, OPTICS 0.44, and converts this into a \"\>0.75\"
acceptance bar. That study clusters daily gridded rainfall event
vectors, where dense well-separated groups occur naturally. It is a
different object from a multi-year climate signature vector per city.
For genuine climate-zone clustering over India, published silhouettes
are far lower: a criteria-based reclassification of Indian climate zones
reports a silhouette of 0.21 against −0.2 for the current NBC
classification, peaking at approximately 0.3 at k = 6 --- and that
result was considered a success because it outperformed the official
map. An Indian thermal-comfort clustering study reports an average
silhouette of 0.235.

  -----------------------------------------------------------------------
  **Why the \>0.75 bar is dangerous.** India\'s climate lies on a
  continuous gradient. There is no partition of Indian cities into
  climate regimes that produces a silhouette above 0.75 without either
  collapsing k to two or three, or selecting features specifically to
  manufacture separation. Adopting that threshold would push this project
  toward exactly the kind of result-shaping it is trying to avoid.
  Silhouette should be read against a null benchmark and alongside BIC,
  Davies--Bouldin and external agreement, not against an idealised
  absolute.

  -----------------------------------------------------------------------

**What is wrong --- STL.** STL (seasonal-trend decomposition by LOESS)
separates trend, seasonal and residual components of a raw time series.
The clustering object here is an aggregated 18-index signature per site;
seasonality and monsoon behaviour are already captured by named indices
(seasonality, monsoon_index, CCI, cloudy_frac). STL adds a preprocessing
stage that reintroduces the time-series representation the signature
abstraction deliberately removed. It is appropriate for forecasting
tasks, which this is not.

**K-Means versus GMM.** The literature consistently finds K-Means
produces crisper, higher-silhouette partitions on spherical clusters,
while GMM better represents overlapping, non-spherical, gradient
structure and yields soft membership probabilities. For a country where
a city may sit genuinely between two regimes, soft membership is a
feature, not a compromise: a site that is 60 % hot-arid and 40 %
composite can receive a membership-weighted PCM recommendation. GMM is
retained. K-Means is still fitted for k = 2...10 as a reported
comparison, which also answers the reviewer\'s concern directly.

2.2 Error #2 --- MCDM method

**The supporting statistic could not be traced.** The review states that
\"CoCoSo demonstrated 10.73 % rank stability in Monte Carlo tests vs 0 %
for TOPSIS under uncertainty\". No located source reports this. The
nearest real statements in the literature are a review noting roughly a
9 % rank-stability improvement over a closest competitor under a
10,000-trial Monte Carlo, and a study finding CoCoSo more stable with
respect to changes of alternatives than of criteria --- qualitative, and
not the quoted figure. Read literally, \"TOPSIS is 0 % stable\" is
false; TOPSIS is repeatedly shown stable under moderate perturbation.
This figure should not be cited.

**CoCoSo does not solve the target-based criterion.** CoCoSo
normalisation is strictly benefit/cost. Melting temperature in this
project is target-based --- closer to the optimum is better, in both
directions. To feed Tm into CoCoSo at all, the Gaussian fitness
transform of §9.2 must be applied first. The transform, not CoCoSo, is
what handles the physics. PROMETHEE II remains the method that expresses
this most naturally, through indifference and preference thresholds with
direct engineering meaning.

**Replacing four methods with one is a regression.** The defence of this
framework against method-induced bias is the agreement statistic across
independent ranking logics --- Kendall\'s W across TOPSIS, PROMETHEE II,
VIKOR and GRA, with Borda and Copeland aggregation and disagreement
reported rather than hidden. A single method, however modern, removes
that evidence. CoCoSo is nonetheless a legitimate, current, citable
method whose hybrid compensatory / non-compensatory aggregation is
genuinely different in kind from the other four, so it is worth adding
as a fifth ranker and reporting whether it changes the consensus. If it
does not, that is a robustness statement.

2.3 Error #3 --- PCM selection criteria

**The evidence base is building passive cooling.** The review\'s OM35 /
OM37 / n-eicosane recommendation and its \"30--40 °C ambient\" framing
trace to a roof passive-cooling study in Rupnagar using spherical
macro-encapsulated modules in an RC roof, and to a prior MCDM study
screening 26 PCMs for building space cooling. Those systems operate
across a 27--43 °C diurnal ambient cycle. This project delivers water at
approximately 50 °C from a storage tank. The PCM families, the melting
range and the selection rule are all different. The proposed profiles
--- hot-arid 35--40 °C, tropical humid 28--32 °C --- would produce a PCM
that cannot deliver usable hot water anywhere in India.

**The \"40 %\" claim is unverified.** The assertion that melting-point
alignment within ±3 °C yields a 40 % performance improvement, or that
alignment is \"40 % more important\" than latent heat, could not be
traced to a primary source in a solar hot water context. The figure
appears in the literature in unrelated forms --- PV
thermal-storage-potential variance, desalination yield gains. It should
not be cited. The defensible statement is weaker and sufficient: melting
temperature must lie between the mains inlet and the collector delivery
temperature, and mismatch degrades usable latent capacity, with no
universal constant.

**But the review exposed a real error.** v1.0 defined Tm_target =
T_delivery − ΔT_approach, giving 42--45 °C. That sign is wrong for
discharge. During discharge the PCM is the heat source and the water the
sink, so the PCM must sit above the delivery temperature by the approach
temperature, not below it. The corrected rule and its literature support
are in §6.3. This is the single most valuable outcome of the review.

2.4 Error #4 --- validation strategy

**EnergyPlus cannot do this.** EnergyPlus models PCM through
MaterialProperty:PhaseChange / PhaseChangeHysteresis with the conduction
finite difference algorithm --- but only as solid conduction layers
inside building surfaces. It models solar water heaters through
WaterHeater:Mixed / WaterHeater:Stratified on a plant loop. There is no
supported path to place a latent-heat PCM inside the water tank node
network; the two capabilities live in different modelling domains.
Specifying EnergyPlus here would produce either a building-envelope
result mislabelled as a hot water result, or nothing at all.

**What is appropriate.** A Python grey-box lumped enthalpy tank model,
calibrated against published experimental benchmarks, remains the
primary tool --- it is transparent, every line is explicable in a viva,
and it integrates directly with the ranking pipeline. TRNSYS Type 860, a
PCM-in-tank component built on the Type 60 water tank using the enthalpy
method with support for encapsulation geometry, hysteresis and
supercooling, is the correct optional cross-check if a licence is
available.

**Monte Carlo count and field trial.** Rank-inclusion probabilities
converge well before 5,000 draws; many published MCDM stability studies
use 1,000. Moving to 10,000 is not a material improvement and 5,000 is
retained. The 12--24 month field trial is not feasible within this
project and is recorded as future work.

3\. Closest Prior Work and Novelty Position

**Read this before you start.** A 2025 paper in Energies, "Comparative
Framework for Climate-Responsive Selection of Phase Change Materials in
Energy-Efficient Buildings", already does something close to the stated
objective: AHP-derived weights applied across COPRAS, VIKOR, TOPSIS,
MOORA and PROMETHEE II, over 16 PCM alternatives, for three climate
zones. It must be cited, and the contribution must be stated as a
difference from it.

That paper fixes three representative zones by hand (temperate 18 °C,
subtropical 23 °C, tropical/hot-desert 28 °C), derives AHP weights
(melting point 47.5 %, latent heat 25.7 %, volumetric latent heat 13.5
%, thermal conductivity 6.8 %, specific heat 3.3 %, density 3.3 %), runs
five MCDM methods, and reports that the methods agree.

  ----------------------------------------------------------------------------
  **\#**   **Their approach**     **Ours**
  -------- ---------------------- --------------------------------------------
  **N1**   Three climate zones    Climate regimes discovered by unsupervised
           chosen by hand from a  clustering of ten years of data across four
           textbook               contrasting Indian states, k selected by
           classification         statistical criteria and cross-checked
                                  against state identity, Köppen--Geiger and
                                  NBC/ECBC zones

  **N2**   One representative     A two-tier climate signature per point
           temperature per zone   combining sun-event-aligned instantaneous
                                  indices with daily-integral indices ---
                                  solar availability, cloud persistence,
                                  diurnal range and humidity stress

  **N3**   Building thermal       Solar domestic hot water, 42--70 °C melting
           comfort, 18--28 °C     range --- a different PCM family, and a Tm
           melting range          rule driven by delivery and collector
                                  temperature rather than comfort

  **N4**   Single best PCM        Top-3 with a consensus score across four
           reported per zone      (optionally five) MCDM methods and a Monte
                                  Carlo stability percentage

  **N5**   MCDM rankings compared MCDM ranking validated against an
           only against each      independent grey-box thermal simulation of
           other                  annual solar fraction, so the ranking is
                                  falsifiable

  **N6**   Zones treated as       Regimes derived from population-weighted
           uniform geographic     sampling covering \~87.5 % of each state's
           areas                  population, so each regime carries a
                                  population figure. A recommendation is
                                  therefore expressed as "this PCM serves N
                                  million people" rather than "this PCM serves
                                  this many square kilometres" --- deployment
                                  relevance, not just geographic coverage.
  ----------------------------------------------------------------------------

*Table 3. Novelty positioning. N1, N4, N5 and N6 are the strongest; lead
with those.*

**N5 deserves emphasis.** Almost the entire PCM-MCDM literature
validates a ranking by showing that several MCDM methods agree with each
other. That demonstrates internal consistency, not correctness ---
methods sharing the same weight vector and the same decision matrix will
usually agree. Physics validation is what converts a preference ordering
into a testable claim.

**N6 is the one nobody else has.** Population weighting is unusual in
the PCM selection literature and is not merely a sampling convenience.
It changes what the result means: a regime covering a large but sparsely
inhabited desert receives proportionally less influence than one
covering a dense river valley. Since the downstream purpose is
specifying domestic hot water systems, that is the correct weighting,
and it should be argued for explicitly rather than mentioned in passing.

4\. Phase 1 --- Data Collection (As Built)

This section documents the pipeline that exists, evaluates it against
what Phase 3 requires, and specifies the two repairs needed before
feature construction can begin.

4.1 The pipeline

  --------------------------------------------------------------------------------------
  **Stage**    **Script**                      **What it produces**
  ------------ ------------------------------- -----------------------------------------
  **Sampling   00a_build_population_grid.py    Downloads the GADM v4.1 admin-1 boundary
  design**                                     and the WorldPop 2020 UN-adjusted 100 m
                                               India raster, clips to the state,
                                               aggregates population onto a 0.25° grid
                                               aligned to the ERA5 grid origin, ranks
                                               cells by population and keeps the minimal
                                               set covering \~87.5 % of state
                                               population. Output:
                                               population_grid_points.csv with point_id,
                                               lat, lon, population, weight.

  **Temporal   00b_build_suntimes.py           For every point and every date from
  design**                                     2016-01-01 to 2025-12-31, computes exact
                                               UTC sunrise, solar noon and sunset using
                                               the pvlib SPA implementation, correctly
                                               handling refraction, orbital eccentricity
                                               and the cross-midnight UTC case. Output:
                                               suntimes.csv.

  **Primary    01_download_era5_rajasthan.py   ERA5 hourly reanalysis over the bounding
  source**                                     envelope of the population points,
                                               restricted to three padded UTC hour
                                               windows per day derived from
                                               suntimes.csv, with the instant/accum
                                               variable split and deaccumulation helper
                                               hours preserved.

  **Second     01b_download_nasapower.py       NASA POWER hourly point data
  source**                                     (ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN,
                                               T2M, RH2M, WS10M) for every point and
                                               every year. Note: this returns the FULL
                                               hourly series, not just sun-event hours.

  **Repair     00_unzip_accum.py               Detects and fixes CDS responses returned
  utility**                                    as ZIP despite an unarchived NetCDF
                                               request.

  **Merge**    02_combine_rajasthan.py         Nearest-neighbour snaps each point to the
                                               ERA5 grid, concatenates and
                                               deaccumulates, computes solar geometry,
                                               and for each (point_id, date, event) row
                                               selects the nearest-in-time ERA5 and NASA
                                               POWER readings, rejecting either if more
                                               than three hours from the true event
                                               instant. Output:
                                               climate_rajasthan_points.csv with
                                               era5\_\* and power\_\* columns side by
                                               side.
  --------------------------------------------------------------------------------------

*Table 4. The pipeline as built, replicated for Rajasthan, Assam, Tamil
Nadu and Uttarakhand. Every stage is resumable and status-tracked, which
is good engineering practice and worth one sentence in the paper.*

4.2 What the design gets right --- and how to say so

**Grid alignment.** Aggregating population onto a 0.25° grid aligned to
ERA5's own grid origin means each sampling point maps to a distinct ERA5
cell with no double-counting and no interpolation artefact. This is a
detail most papers get wrong by sampling city coordinates that fall two
to a cell. State it.

**Sun-event sampling is physically motivated.** Frame it as
charge--discharge-cycle-aligned sampling, because that is what it is:
sunrise is the coldest instant and therefore the test of whether the PCM
fully solidified overnight; solar noon is the peak charging condition;
sunset is the ambient condition at the start of the evening draw. A
uniform three-hourly subsample would be an arbitrary shortcut. This is
not.

**Two independent sources at matched instants.** ERA5 is a reanalysis
and NASA POWER is satellite-derived; they are genuinely independent
estimates of the same quantity at the same place and instant. This is a
stronger cross-check than the CERES comparison v2.0 planned, and it is
already downloaded.

**Static population raster.** Using one 2020 WorldPop snapshot for a
2016--2025 study period is a standard simplifying assumption ---
WorldPop does not publish a distinct India raster per year at this
resolution. Declare it in the limitations, do not defend it at length.

4.3 Two required repairs

Repair 1 --- Recover the daily-integral indices from the NASA POWER
cache

Three instantaneous samples per day cannot produce a daily energy
integral, a true diurnal range, or a degree-day count. The merged CSV
therefore cannot support Tier 2 of the climate signature. But the raw
NASA POWER JSON cache already contains the full hourly series for every
point and every year, and the merge step simply discards all but the
sun-event hours.

-   Write a new script --- 02b_build_daily_aggregates.py --- that reads
    data/raw/nasapower/power\_{point_id}\_{year}.json directly and
    produces a daily table: daily GHI integral, daily clear-sky
    integral, daily clearness index, true daily minimum and maximum
    temperature, daily mean temperature, daily mean relative humidity,
    daily mean wind speed.

-   From that daily table compute the Tier 2 indices of §6.2 ---
    GHI_daily_kWh, SAI, kt_daily_mean, kt_daily_std, cloudy_frac, CCI,
    HDD18, CDD24, DTR_true, seasonality, monsoon_index.

-   Cost: no new downloads, no CDS queue, a few hours of implementation.
    This is the single highest-value task remaining in the data phase.

-   If ERA5 daily integrals are wanted as well for consistency, that
    does require a new CDS request for all 24 hours at the same points.
    Treat it as optional --- the two-source cross-check at sun events
    already establishes whether ERA5 and POWER agree, and if they do,
    POWER alone is a defensible backbone for the daily tier.

Repair 2 --- Attach real elevation

  -----------------------------------------------------------------------
  **The flat 300 m assumption is not survivable in Uttarakhand.** The
  pipeline uses a uniform 300 m elevation for solar geometry because
  population points carry no elevation field. Across Rajasthan, Assam and
  coastal Tamil Nadu the resulting error is small. Uttarakhand spans
  roughly 200 m to over 7,000 m; at 2,500 m the air mass, clear-sky
  irradiance and boiling point all differ materially from the 300 m
  assumption, and the elev_proxy signature index is meaningless without
  it. A reviewer who knows the region will ask.

  -----------------------------------------------------------------------

-   Attach per-point elevation from ERA5 surface geopotential (the
    invariant z field, divided by 9.80665) or from an SRTM/Copernicus
    DEM sampled at the point coordinates. ERA5 geopotential is the
    lower-effort option and is consistent with the rest of the ERA5
    data.

-   Recompute pvlib solar geometry and clear-sky irradiance with the
    true elevation. This changes the clear-sky reference and therefore
    every clearness-index-derived index for the mountain points ---
    which is precisely why it matters.

-   Note in the limitations that ERA5 orography is a grid-cell mean and
    will smooth extreme terrain. That is an honest and sufficient
    caveat.

4.4 Quality control to report

Numbers a reviewer will expect, and which the pipeline can produce
cheaply:

-   Number of population points retained per state, and the population
    fraction actually covered (the \~87.5 % target may be met at
    different point counts per state). Report the count per state --- if
    any state falls below roughly 30 points, its internal structure may
    be under-resolved.

-   Sensitivity of the point set to the coverage threshold: rerun at 80
    % and 95 % and report whether the cluster structure changes. If it
    does not, that is a robustness result worth one sentence.

-   Percentage of rows where the nearest ERA5 or NASA POWER reading was
    rejected by the three-hour window, per source, per state.

-   The known 2016-01-01 NaN for accumulation-derived columns where a
    sun-event window touches hour 0 UTC. One day in ten years; report it
    and move on.

-   ERA5-versus-POWER agreement at matched instants --- MBE, RMSE and
    correlation for GHI, temperature, humidity and wind, per season, per
    state. This is Phase 2 and doubles as the bias-correction decision.

4.5 PCM property database --- unchanged and still outstanding

Target: 40--60 candidates in the 42--70 °C melting range. Below
approximately 45 °C the PCM cannot drive heat into water at the 50 °C
delivery target; above approximately 70 °C a flat-plate collector will
not reliably charge it. This is independent of the climate data and can
be built in parallel --- it is now the critical path item, since the
climate data is largely in hand.

  ------------------------------------------------------------------------
  **Family**      **Representative         **Notes**
                  candidates**             
  --------------- ------------------------ -------------------------------
  **Paraffins     RT42, RT44HC, RT50,      Mainstream SWH choice.
  (Rubitherm      RT55, RT58, RT64HC       Non-corrosive, low
  RT)**                                    supercooling, good cycling. k ≈
                                           0.2 W/m·K is the main weakness.

  **PLUSS savE OM OM42, OM45, OM48, OM55,  Indian supplier, commercially
  series**        OM65                     available, which matters for a
                                           deployment-oriented study. OM55
                                           is characterised as thermally
                                           stable across 45--60 °C and
                                           already used in domestic solar
                                           water heating.

  **Fatty acids   Lauric (\~44 °C),        Moderate latent heat, some
  and eutectics** myristic (\~54 °C),      corrosivity, mild odour.
                  stearic (\~69 °C), and   Eutectics allow Tm tuning
                  binary eutectics         between regimes.

  **Salt          Sodium acetate           High volumetric storage
  hydrates**      trihydrate (\~58 °C),    density. Strong supercooling
                  sodium thiosulfate       (needs a nucleator) plus
                  pentahydrate (\~48--49   phase-segregation and corrosion
                  °C)                      risk --- the corrosion veto
                                           will likely exclude these in
                                           the Assam regimes.

  **Sugar         Erythritol (\~118 °C)    EXCLUDED. Far above the usable
  alcohols**                               window for 50 °C domestic
                                           delivery. Listed so the
                                           exclusion is documented rather
                                           than silent.
  ------------------------------------------------------------------------

*Table 5. PCM families in the corrected 42--70 °C band. Record Tm,
latent heat, thermal conductivity, density, specific heat, cycling
stability, supercooling degree, corrosion class and cost, each with a
source citation. Where a value is unreported, record "not reported" and
let the Monte Carlo of §9.6 handle it --- never guess.*

5\. Phase 2 --- Preprocessing and Cross-Source Validation

Much of the classical preprocessing burden --- timezone handling,
deaccumulation, solar geometry, nearest-in-time matching --- is already
inside the pipeline. What remains is the validation that turns two
downloaded sources into a defensible single backbone.

5.1 The ERA5-versus-POWER agreement analysis

This replaces the CERES quantile-mapping step of v2.0 and is the more
direct comparison, because both sources are evaluated at the same point
and the same instant rather than at different resolutions.

1.  For each variable present in both sources (GHI, 2 m temperature,
    relative humidity, 10 m wind speed), compute mean bias error, RMSE
    and Pearson correlation between era5\_\* and power\_\* columns.

2.  Stratify by season and by state. A bias that appears only in the
    Assam monsoon season is a different finding from a uniform offset,
    and the two call for different responses.

3.  Stratify also by sun event. Sunrise and sunset GHI values are small
    and near-zero-crossing; disagreement there is expected and less
    consequential than disagreement at solar noon.

4.  Plot ERA5 against POWER as a scatter with the identity line, one
    panel per state per season. This single figure answers the
    data-quality question for a reviewer faster than any table.

5.2 The decision rule

  -----------------------------------------------------------------------
  **Finding**                   **Action**
  ----------------------------- -----------------------------------------
  **Agreement is close and      Use ERA5 as the primary backbone and
  unbiased (MBE small,          report the agreement as a validation
  correlation high)**           result. No correction needed. State that
                                two independent sources agree --- this is
                                a stronger position than a corrected
                                single source.

  **A systematic,               Apply quantile mapping of ERA5 GHI onto
  season-dependent bias appears the POWER distribution, fitted per season
  (most likely: ERA5            per state. Report MBE, RMSE and
  underestimating               correlation before and after.
  monsoon-season GHI in Assam   
  and coastal Tamil Nadu)**     

  **The two disagree severely   Do not average them. Investigate the
  with no interpretable         merge --- a nearest-in-time mismatch, a
  pattern**                     units error, or an
                                instant-versus-accumulation confusion is
                                far more likely than genuine disagreement
                                between two established datasets.
  -----------------------------------------------------------------------

*Table 6. Bias-correction decision rule. Deciding this from the data,
rather than assuming a correction is needed, is the defensible order.*

  -----------------------------------------------------------------------
  **Fixed-weight blending remains rejected.** Combining the sources as,
  for example, 0.6 × ERA5 + 0.4 × POWER has no derivation and would make
  the resulting dataset impossible to characterise. Either one source is
  the backbone with the other as validation, or one is quantile-mapped
  onto the other. There is no third defensible option.

  -----------------------------------------------------------------------

5.3 Remaining acceptance checks

-   Plot the mean seasonal cycle of noon GHI and noon temperature for
    every point, one panel per state, and inspect by eye. Unit and
    timezone errors are the most common silent failures and are obvious
    in these plots.

-   Verify that the sun-event times behave sensibly across the year: day
    length should be longest near the June solstice and shortest near
    December, with the amplitude larger in Uttarakhand than in Tamil
    Nadu. If it is not, something is wrong in the SPA call or the UTC
    handling.

-   Confirm the cross-midnight cases: eastern points with a summer
    sunrise falling at roughly 23:55 UTC on the previous calendar date
    should carry the true instant in time_utc. Spot-check a handful by
    hand.

-   Confirm no two population points snap to the same ERA5 cell. Grid
    alignment should guarantee this; verify it rather than assume it.

6\. Phase 3 --- Climate Signature Construction

Goal: reduce each point's ten-year record to one vector of roughly 20
numbers capturing everything relevant to PCM behaviour. This vector is
the object that gets clustered.

6.1 Design principle

Every index must answer the question "which PCM property does this
constrain, and by what physical mechanism?". If that sentence cannot be
completed, the index is removed. Twenty indices each defensible in one
sentence produce a better paper --- and better clusters --- than sixty
that arrived by automated generation.

6.2 The two-tier index set

The sampling design splits the signature naturally into two tiers with
different provenance. Both are required; neither is optional.

Tier 1 --- Sun-event indices, from the merged CSV

These come directly from climate\_{state}\_points.csv and are the
indices the sampling design was built for. Each is aggregated over the
ten years as a mean and a standard deviation, and where noted as a
percentile.

  --------------------------------------------------------------------------------
  **Index**               **Constrains**    **Mechanism**
  ----------------------- ----------------- --------------------------------------
  **T_sunrise_mean,       Tm lower bound,   The coldest instant of the day.
  T_sunrise_p05**         solidification    Determines whether the PCM fully
                                            re-solidifies overnight and therefore
                                            whether the next charge cycle starts
                                            from a full latent capacity.

  **T_noon_mean**         Tm, charging      Ambient at peak charging --- sets the
                                            collector loss term when it matters
                                            most.

  **T_sunset_mean,        Discharge rate, k Ambient at the start of the evening
  T_sunset_p95**                            draw. High sunset temperature reduces
                                            store losses but also reduces the
                                            discharge gradient.

  **diurnal_gradient**    Cycling stability T_noon minus T_sunrise. A proxy for
                                            diurnal range --- note it
                                            underestimates true DTR because peak
                                            air temperature lags solar noon by two
                                            to three hours, which is why DTR_true
                                            is retained in Tier 2.

  **kt_noon_mean**        Latent heat,      Clearness index at peak sun. The
                          charging quality  single best one-number descriptor of
                                            charging quality.

  **kt_noon_std**         Cycling           Day-to-day variability of peak
                          stability, k      charging --- partial-cycle stress.

  **GHI_noon_mean**       Latent heat,      Peak charging flux available per unit
                          storage mass      collector area.

  **GHI_sunset_mean**     Charging window   Residual irradiance at the start of
                                            discharge --- how much charging
                                            overlaps the draw period.

  **RH_sunrise_mean**     Corrosion,        Relative humidity at the coldest
                          encapsulation     instant, which is when condensation
                                            actually occurs. More directly
                                            relevant than a daily mean.

  **HSI_sunrise**         Corrosion class   Humidity stress at the
                                            condensation-critical instant. Drives
                                            the salt-hydrate exclusion in §8.

  **wind_noon_mean,       Thermal           Convective loss coefficient during
  wind_sunset_mean**      conductivity k    charging and during discharge
                                            respectively.

  **daylength_mean,       Storage sizing    Sunset minus sunrise, from
  daylength_amplitude**                     suntimes.csv at no extra cost.
                                            Charging window duration and its
                                            seasonal swing --- larger in
                                            Uttarakhand than in Tamil Nadu, and a
                                            real constraint on daily charge
                                            completion.
  --------------------------------------------------------------------------------

*Table 7. Tier 1, sun-event indices. These exist because of the sampling
design rather than in spite of it, and the paper should present them
that way.*

Tier 2 --- Daily-integral indices, recomputed from the NASA POWER hourly
cache

These cannot be computed from three instantaneous samples and must come
from the full hourly series already cached on disk (§4.3, Repair 1).

  ----------------------------------------------------------------------------
  **Index**           **Constrains**    **Mechanism**
  ------------------- ----------------- --------------------------------------
  **GHI_daily_kWh**   Storage sizing    Daily energy integral --- the number a
                                        designer actually uses to size a
                                        system. Not obtainable from noon
                                        irradiance alone.

  **SAI**             Storage capacity  Solar availability index: fraction of
                                        the theoretical clear-sky resource
                                        actually delivered over the day.

  **kt_daily_mean,    Latent heat,      Whole-day clearness and its
  kt_daily_std**      cycling           variability, distinct from the
                                        noon-instant value.

  **cloudy_frac**     Storage capacity  Fraction of days below a clearness
                                        threshold --- how often autonomy is
                                        exercised.

  **CCI**             Latent heat       Longest run of consecutive
                      floor, cost       low-clearness days. The autonomy the
                                        store must provide with no recharge,
                                        and the index expected to separate
                                        Assam most sharply.

  **DTR_true**        Cycling stability True daily maximum minus minimum.
                                        Complete cycles per year, therefore
                                        degradation rate.

  **Ta_mean, Ta_p95,  Tm feasibility    Whole-day temperature statistics; the
  Ta_p05**            window            extremes decide whether the PCM ever
                                        fully melts or fully solidifies.

  **HDD18**           Tm lower bound,   Water-heating demand intensity. Drives
                      L_required        the latent-heat requirement, and
                                        expected to dominate in Uttarakhand.

  **CDD24**           Discharge rate, k High ambient reduces losses but also
                                        reduces the discharge gradient.

  **seasonality**     Phase stability   Seasonal swing in charging conditions.

  **monsoon_index**   Regime identity   Separates monsoon-dominated from arid
                                        regimes, and --- given Tamil Nadu's
                                        north-east monsoon --- should also
                                        separate the two monsoon timings. This
                                        is the index most likely to produce an
                                        interesting result.
  ----------------------------------------------------------------------------

*Table 8. Tier 2, daily-integral indices. Recovering these from the
existing NASA POWER cache costs no downloads and is the highest-value
remaining data task.*

Static attributes

-   elevation --- from ERA5 surface geopotential or an SRTM DEM (§4.3,
    Repair 2). Constrains density and convection, and is expected to be
    the dominant splitting variable within Uttarakhand.

-   population and weight --- carried through from the sampling design.
    Used for reporting and for the population coverage figure on each
    recommendation card, NOT as clustering features.

  -----------------------------------------------------------------------
  **Do not put latitude and longitude in the clustering matrix.** It is
  tempting, and it is wrong. Including coordinates makes the algorithm
  cluster geography rather than climate, guarantees that the four states
  separate, and destroys the entire finding --- because the interesting
  result is precisely where climate regimes cross state boundaries or
  split within one. Coordinates are for plotting the map afterwards, not
  for fitting. Elevation is a physical variable and is legitimately
  included; coordinates are not.

  -----------------------------------------------------------------------

6.3 Derived PCM-facing quantities

  -----------------------------------------------------------------------
  **This section corrects v1.0 §6.3.** v1.0 stated Tm_target = T_delivery
  − ΔT_approach, giving 42--45 °C. The sign is wrong. During discharge
  the PCM is the heat source and the water the sink; heat flows from PCM
  to water only if the PCM sits above the water temperature. A PCM
  melting at 43 °C cannot deliver water at 50 °C.

  -----------------------------------------------------------------------

The corrected rule:

**Tm_target = T_delivery + ΔT_approach**

with T_delivery = 50 °C for Indian domestic use and ΔT_approach the
heat-exchanger approach temperature, typically 5--8 K. This yields
Tm_target ≈ 55--58 °C for an indirect system, or ≈ 50--53 °C for a
direct system where the PCM is encapsulated in the potable tank. State
which configuration is assumed; do not tune the rule per regime after
seeing results.

Two upper bounds constrain the same quantity from above: Tm must lie
below the collector delivery temperature achievable on a poor day in
that regime (which is what makes the target regime-dependent --- a
high-CCI Assam regime supports a lower Tm than an arid Rajasthan
regime), and below flat-plate stagnation temperature, which should be
checked and reported even though it is rarely binding.

Literature support for the corrected band

  ------------------------------------------------------------------------------
  **Source**              **Finding**                     **Implication here**
  ----------------------- ------------------------------- ----------------------
  **Zhao et al., J.       The phase change temperature    Directly brackets the
  Cleaner Prod., 2019**   range suitable for conventional corrected Tm_target
                          heating systems is 47.5--57.5   and confirms 42--45 °C
                          °C; a series tank--PCM          sits below the useful
                          arrangement raised solar        band
                          fraction by roughly 30 % over a 
                          single tank and 5--12 % over    
                          parallel                        

  **Avargani et al., J.   A paraffin PCM bed (0.3 m dia × A night-delivery
  Energy Storage, 2021**  0.6 m) sustained up to 300 L of benchmark for the
                          hot water at 60 ± 2 °C for 7 h  grey-box model, and
                          of operation                    evidence that \~60
                                                          °C-class paraffins
                                                          perform in this duty

  **SDHW                  Tanks operating at 58--60 °C    Confirms 50--65 °C as
  encapsulated-paraffin   and \~62 °C; evacuated-tube     the working PCM band
  studies**               manifold paraffin at \~67 °C    for domestic hot water
                          maintaining 55--60 % efficiency 

  **China SWH standard    Exit water required above 50    Independent
  practice**              °C; paired paraffins at 48--50  confirmation of the
                          °C (low) and 62--64 °C (high)   delivery-anchored rule
                                                          and of a two-PCM
                                                          cascade as a
                                                          legitimate design
                                                          option --- relevant if
                                                          Uttarakhand and
                                                          Rajasthan regimes
                                                          demand different Tm
  ------------------------------------------------------------------------------

*Table 9. Literature support for the corrected melting-temperature rule.
None of these is a building passive-cooling study --- that distinction
is the point of the correction.*

The latent-heat floor:

**L_required = Q_night / m_PCM , with Q_night = ṁ_draw · cp_water ·
(T_delivery − T_mains)**

**T_mains matters more in this study than in an all-India one.** Mains
water temperature tracks ground temperature, which tracks annual mean
air temperature with a lag. Across four states spanning Uttarakhand to
Tamil Nadu, T_mains varies enough to change L_required substantially
between regimes --- a cold Himalayan regime demands materially more
latent heat per litre delivered than a Tamil Nadu coastal regime.
Estimate T_mains per regime from Ta_mean with a standard lag correlation
and report the resulting L_required spread; it is a good result in its
own right.

6.4 Interaction terms and dimensionality

Five interaction terms, each named and justified:

-   GHI_daily_kWh × kt_daily_std --- charging energy weighted by its
    unreliability; high values mean a large but erratic resource

-   DTR_true × cloudy_frac --- cycling stress under intermittent
    charging, the worst case for phase stability

-   RH_sunrise_mean × (T_sunrise_mean − Tm_target) --- condensation risk
    at the store surface at the condensation-critical instant

-   wind_sunset_mean × (T_sunset_mean − T_delivery) --- convective loss
    driving potential during the evening draw

-   CCI × (1 − SAI) --- combined autonomy requirement

Then PCA on the correlated block only (Ta_mean, Ta_p95, Ta_p05,
T_sunrise_mean, T_noon_mean, HDD18, CDD24, elevation), retaining
components to 95 % variance --- typically three. Keep the solar,
variability and humidity indices out of the PCA: they carry the
discriminating signal. Report the component loadings; across these four
states they should be readable as roughly "heat", "altitude" and
"seasonal amplitude", which is itself a result.

Standardise all columns to zero mean and unit variance before clustering
--- Euclidean distance is meaningless otherwise when kWh/m²/day sits
beside a percentage.

**Rejected alternatives, unchanged.** DCCA remains rejected on
sample-size and interpretability grounds. A TabTransformer or
FT-Transformer encoder may be run as an optional ablation after the
engineered pipeline works; a negative result there is worth reporting.

7\. Phase 4 --- Climate Regime Clustering

This is the core of the objective and the part a reviewer scrutinises
hardest, because unsupervised clustering has no ground truth and is easy
to do badly. The four-state design changes what a good result looks
like.

7.1 What counts as a result here

  -----------------------------------------------------------------------
  **Recovering the four states is not a finding.** Rajasthan, Assam,
  Tamil Nadu and Uttarakhand are known to be climatically different. If
  the clustering returns four clusters that map one-to-one onto state
  boundaries, it has reproduced the sampling design and told you nothing.
  Report the adjusted Rand index against state identity precisely so that
  this can be seen and addressed: an ARI near 1.0 at k = 4 means k is too
  low. The result of interest is intra-state splitting and, more
  valuably, any cross-state merging.

  -----------------------------------------------------------------------

Three specific findings to look for, each of which would be a genuine
contribution:

-   Intra-state splitting --- arid west versus semi-arid east in
    Rajasthan; terai, mid-hills and high Himalaya in Uttarakhand;
    coastal, interior dry and Nilgiris in Tamil Nadu; Brahmaputra valley
    versus surrounding hills in Assam. Each split is a case where a
    single state-level PCM specification would be wrong.

-   Cross-state merging --- if, for example, interior Tamil Nadu
    clusters with eastern Rajasthan rather than with coastal Tamil Nadu,
    that directly demonstrates that administrative boundaries are the
    wrong unit for PCM specification. This is the strongest available
    argument for the whole framework.

-   Population-weighted regime size --- a regime covering a small area
    but a dense population deserves more design attention than a large
    sparse one. Report population per regime alongside point count; this
    is where novelty claim N6 pays off.

7.2 Two levels of clustering --- do both

  ------------------------------------------------------------------------
  **Level**      **What is clustered**     **What it gives you**
  -------------- ------------------------- -------------------------------
  **Level A ---  One signature vector per  Climate regimes spanning and
  spatial**      population point; cluster subdividing the four states.
                 the points across all     Answers: "which PCM for a
                 four states together      system installed at this
                                           location?"

  **Level B ---  For each point, one       Operating regimes within a
  temporal**     signature per season      location --- monsoon, dry,
                 across the ten years;     transition. The merged CSV
                 cluster those             already carries season and
                                           season_code, so this is nearly
                                           free. Answers: "does this
                                           location need a different PCM
                                           in July than in March?"
  ------------------------------------------------------------------------

*Table 10. Two-level clustering. Level A is required by the objective;
Level B is what makes it climate-aware rather than merely region-aware.*

**Level B is where the interesting result lives, and this dataset is
unusually good for it.** Tamil Nadu's north-east monsoon is out of phase
with the south-west monsoon that dominates Assam and Rajasthan. If Level
B shows the Top-3 flipping between seasons in Tamil Nadu but holding
steady in Rajasthan, that is direct empirical motivation for the
adaptive control objective --- generated by this objective, from your
own data, rather than asserted from the literature. Either outcome is a
result.

7.3 Algorithm choice: Gaussian Mixture over K-Means

GMM is the primary model. K-Means is fitted alongside for k = 2...10 and
reported as a comparison, which answers the algorithm-sensitivity
concern directly and turns it into a reported result rather than an
unexamined choice.

  ----------------------------------------------------------------------------
  **Consideration**   **K-Means**              **GMM (chosen)**
  ------------------- ------------------------ -------------------------------
  **Cluster shape**   Spherical, equal         Elliptical, per-component
                      variance assumed         covariance

  **Assignment**      Hard                     Soft --- membership
                                               probabilities

  **Silhouette**      Typically higher         Typically lower, because it
                      (crisper partitions)     does not force separation that
                                               is not there

  **Model selection** Heuristic (elbow)        Principled --- BIC / AIC

  **Downstream use**  A point belongs to       A transition-zone point that is
                      exactly one regime       60 % arid and 40 % semi-arid
                                               receives a membership-weighted
                                               PCM recommendation. With
                                               population-weighted sampling
                                               across sharp gradients such as
                                               the Uttarakhand terai--hill
                                               boundary, transition points are
                                               common and soft membership is
                                               not a nicety.
  ----------------------------------------------------------------------------

*Table 11. Algorithm comparison. The soft-membership column is why GMM
is retained despite scoring lower on silhouette.*

**Do not weight the GMM fit by population.** The sampling is already
population-weighted by construction --- densely inhabited areas
contribute more points. Applying population weights again during fitting
would double-count. Use population for reporting and for the
recommendation cards, not for the likelihood.

7.4 Choosing k, and what silhouette to expect

Fit k = 2...12 and report all of: BIC and AIC curves, mean silhouette,
Davies--Bouldin index, Calinski--Harabasz index, and bootstrap stability
(adjusted Rand index between clusterings of resampled data). Choose k
where the criteria agree, and state the disagreement where they do not.

**Expected k: 6--10.** Four states with plausible internal splits of two
to three each. If k selects at 4, check immediately whether the clusters
are simply the states --- and if they are, report the k = 6--8 solution
alongside as the informative one, with the model-selection evidence for
both. If k selects above 10, some clusters are likely singletons or
near-duplicates; inspect membership counts before accepting it.

  -----------------------------------------------------------------------
  **Realistic silhouette expectation: 0.15--0.35.** For data-driven
  climate zoning over India, published silhouettes peak near 0.3. A
  criteria-based reclassification of Indian climate zones reports 0.21
  against −0.2 for the current NBC classification, peaking around 0.3 at
  k = 6, and that was a successful result. An Indian thermal-comfort
  clustering study reports 0.235. Note one caveat specific to this
  design: because four contrasting states are sampled and the intervening
  territory is not, the between-state gaps are artificially clean and the
  silhouette may come out somewhat higher than an all-India study would
  produce. Do not present an inflated silhouette as evidence of superior
  method --- explain that it partly reflects the sampling frame.

  -----------------------------------------------------------------------

7.5 External validation --- the step that earns credibility

-   Adjusted Rand index against state identity. Expect substantial but
    imperfect agreement; perfect agreement means the clustering added
    nothing (§7.1).

-   Adjusted Rand index and normalised mutual information against
    Köppen--Geiger classes at the same points.

-   The same statistics against NBC/ECBC Indian climate zones,
    restricted to the four zones represented.

-   A map per state, points coloured by hard assignment and shaded by
    maximum membership probability, so ambiguous transition points are
    visible. Four state panels are a better figure than one national map
    here, because the states are not contiguous.

-   Elevation profile per cluster for Uttarakhand specifically --- if
    the mid-hills and high-Himalaya clusters do not separate on
    elevation, Repair 2 was not applied correctly.

**Interpreting agreement.** The target is substantial but imperfect
agreement with existing classifications, with every departure
explainable from the signature indices --- for example, two
Köppen-identical points separating because one has a much higher CCI and
therefore a higher autonomy requirement. That explanation is the paper's
most persuasive paragraph.

7.6 Regime characterisation

For each regime produce a profile card: medoid point (with its district
and elevation), member points and their state distribution, total
population covered, the full two-tier signature as mean ± standard
deviation, a one-line physical description, and the derived Tm_target
and L_required. These feed directly into Phase 5 and become the results
section.

8\. Phase 5 --- Feasibility Filtering

Before any ranking, hard-filter the PCM database per cluster. MCDM is a
compensatory method: a large advantage on one criterion can offset a
fatal deficiency on another. A PCM with an unreachable melting point and
outstanding latent heat can score well in TOPSIS and be physically
useless. Filtering first prevents that.

  --------------------------------------------------------------------------
  **Constraint**   **Rule (v2.0)**          **Justification**
  ---------------- ------------------------ --------------------------------
  **Melting        Tm ∈ \[Tm_target − 5,    Below the lower bound the store
  window**         Tm_target + 8\] °C       cannot drive heat into water at
                                            the delivery temperature; above
                                            the upper bound a flat-plate
                                            collector will not reliably
                                            charge it in that cluster\'s
                                            solar regime

  **Absolute       Tm ∈ \[42, 70\] °C       Outside this, the candidate is
  band**           regardless of cluster    not a solar domestic hot water
                                            PCM at all

  **Charging       Tm below the collector   The PCM must melt on a poor day,
  feasibility**    delivery temperature at  not only on a good one
                   the cluster\'s           
                   5th-percentile daily     
                   insolation               

  **Latent heat    L ≥ 0.7 × L_required for Below this the store cannot
  floor**          that cluster             supply the night draw within a
                                            plausible tank volume

  **Cycling        ≥ 300 cycles where       Roughly one cycle per day means
  stability**      reported; retain and     300 cycles is under one year of
                   flag where not reported  service

  **Corrosion      Exclude bare salt        Condensation-driven corrosion is
  veto**           hydrates where HSI       a documented failure mode in
                   exceeds the cluster 75th humid coastal installations
                   percentile unless        
                   encapsulation is         
                   specified                

  **Supercooling   Exclude candidates with  Supercooling means the store
  veto**           supercooling \> 8 K      holds energy it cannot release
                   unless a nucleating      --- the specific failure the
                   agent is specified       whole system exists to avoid

  **Safety**       Exclude toxic or highly  Non-negotiable for a household
                   flammable candidates for product
                   domestic installation    
  --------------------------------------------------------------------------

*Table 12. Feasibility constraints, updated for the corrected melting
band. Report how many of the original candidates survive per cluster ---
that number is itself informative and belongs in the results.*

If a cluster retains fewer than five candidates, relax the melting
window by 2 K and record that it was relaxed. If it retains more than
twenty-five, the constraints are too loose. Eight to twenty is a healthy
candidate set for MCDM.

9\. Phase 6 --- Multi-Criteria Ranking Engine

9.1 Criteria

  ----------------------------------------------------------------------------------
  **Criterion**     **Type**       **Indicative   **Note**
                                   weight**       
  ----------------- -------------- -------------- ----------------------------------
  **Melting-point   Target-based   0.24           Converted from \|Tm − Tm_target\|
  fitness**                                       to a fitness score --- see §9.2,
                                                  the step most implementations get
                                                  wrong

  **Latent heat L** Benefit        0.20           Ranked highest-priority property
                                                  in the PCM-SWH review literature

  **Volumetric      Benefit        0.12           What actually determines tank
  latent heat ρL**                                size; often diverges from L alone

  **Thermal         Benefit        0.13           Governs charge and discharge rate;
  conductivity k**                                weighted higher here than in
                                                  building studies because SWH has a
                                                  charging-rate constraint that
                                                  comfort applications do not

  **Cycling         Benefit        0.11           Service life
  stability**                                     

  **Supercooling    Cost           0.08           Energy that cannot be released is
  (inverse)**                                     energy not stored

  **Corrosion class Cost           0.06           Cluster-dependent: weight higher
  (inverse)**                                     in high-HSI clusters

  **Cost**          Cost           0.06           Weight honestly --- do not let a
                                                  data-poor criterion dominate
  ----------------------------------------------------------------------------------

*Table 13. Criteria set with indicative starting weights. Actual weights
come from §9.3; the sensitivity analysis in §9.6 must show the Top-3 is
not an artefact of them.*

9.2 The target-based criterion --- get this right

  -----------------------------------------------------------------------
  **The most common error in PCM MCDM papers.** Melting temperature is
  neither a benefit nor a cost criterion. Higher is not better and lower
  is not better; closer to target is better. Standard TOPSIS, VIKOR, GRA
  and CoCoSo normalisation all assume monotonic criteria and will
  silently produce plausible-looking nonsense if fed raw Tm.

  -----------------------------------------------------------------------

Convert Tm to a fitness score before it enters the decision matrix:

**f_Tm(i) = exp( − (Tm_i − Tm_target)² / (2σ²) ), σ ≈ 4 K**

This is a Gaussian fitness peaking at the target, decaying
symmetrically, bounded in (0, 1\], and now a proper benefit criterion.
Justify σ = 4 K from the heat-exchanger approach temperature. An
asymmetric form is physically better motivated --- the penalty for Tm
being too high (the PCM never melts on a poor day) is more severe than
for being slightly too low (the PCM melts early and delivers at a lower
temperature) --- so σ_upper \< σ_lower. Whichever is chosen, state it
and test the alternative in the sensitivity analysis.

**PROMETHEE II handles this more elegantly.** Define the criterion as
−\|Tm − Tm_target\| with a V-shape or Gaussian preference function,
indifference threshold q = 2 K and preference threshold p = 8 K. Those
thresholds have direct engineering meaning --- \"differences under 2 K
do not matter; differences over 8 K are decisive\" --- which is
precisely the kind of statement a thermal engineer can confirm or
contest. This is the strongest single argument for keeping PROMETHEE II
in the stack, and the reason adding CoCoSo does not remove the need for
it.

9.3 Weight determination

Combine an objective and a subjective source:

**w_j = λ · w_j\^entropy + (1 − λ) · w_j\^AHP, λ = 0.5**

**Entropy weights** are computed per cluster from that cluster\'s own
filtered decision matrix. A criterion on which all surviving candidates
are near-identical automatically receives low weight, which is correct
since it cannot discriminate. This makes weights cluster-specific --- a
feature worth pointing out in the paper.

**AHP weights** encode domain priority. Build the pairwise matrix with
the guide and, if possible, one thermal-engineering faculty member.
Record who provided the judgements and report the consistency ratio; it
must be below 0.10. If it is not, revisit the inconsistent comparisons
with the respondent rather than adjusting the matrix directly.

  -----------------------------------------------------------------------
  **Why the 0.5 / 0.5 blend is the right call.** Entropy weighting is
  data-driven and can be dominated by whichever criterion happens to have
  the most spread in the matrix. In the Oluah 2020 entropy+TOPSIS PCM
  study, thermal conductivity received 72.12 % of the total weight purely
  because that column varied most --- a result no thermal engineer would
  endorse as a statement of priority. In the building AHP studies, latent
  heat or melting point receives 47--57 %. Neither source alone is
  trustworthy; the blend anchors objective spread against expert priors,
  and reporting λ = 0, 0.5 and 1 in the sensitivity analysis shows
  whether the Top-3 depends on that choice. If the Top-3 is identical
  across all three, say so --- it is a strong robustness statement.

  -----------------------------------------------------------------------

9.4 The ranking methods

  -------------------------------------------------------------------------
  **Method**        **Output**             **Why it is in the stack**
  ----------------- ---------------------- --------------------------------
  **TOPSIS**        Closeness coefficient  Interpretable, precedented in
                    Ci ∈ \[0,1\]           PCM selection, gives a natural
                                           score for reporting

  **PROMETHEE II**  Net outranking flow φ  Non-compensatory pairwise
                    ∈ \[−1,1\]             preference; handles the
                                           target-based Tm criterion
                                           natively; resistant to rank
                                           reversal

  **VIKOR**         Compromise index Qi    Its formal conditions can return
                    plus                   a set of compromise solutions
                    acceptable-advantage   rather than one winner --- the
                    and                    cleanest principled
                    acceptable-stability   justification for reporting
                    tests                  Top-2/Top-3

  **GRA**           Grey relational grade  Robust to sparse and noisy
                    Γ ∈ \[0,1\]            property data, which this is;
                                           already precedented in the
                                           project references

  **CoCoSo          Three appraisal scores Hybrid compensatory /
  (optional, new in fused into a composite non-compensatory aggregation,
  v2.0)**           ki                     genuinely different in kind from
                                           the other four. Added as a fifth
                                           cross-check after the
                                           four-method consensus is working
                                           --- never as a replacement.
                                           Requires the §9.2 Tm transform
                                           first.
  -------------------------------------------------------------------------

*Table 14. Ranking methods. Roughly 120 lines of Python for the core
four; CoCoSo adds about 30 more. None needs a specialised library.*

9.5 Rank aggregation into a consensus Top-3

-   Compute Kendall\'s W (coefficient of concordance) across the
    rankings per cluster. W \> 0.8 means strong agreement and the
    consensus is safe. Low W is itself a finding: it identifies clusters
    where the PCM choice is genuinely ambiguous, and those deserve
    discussion rather than a forced answer. If W falls below roughly
    0.6, investigate the criterion definitions before trusting the
    consensus.

-   Aggregate by Borda count and cross-check with Copeland pairwise
    majority. Where they disagree, report both.

-   Report the consensus Top-3 with each method\'s individual rank
    alongside, so a reader can see the disagreement rather than having
    it hidden by the aggregate.

-   If CoCoSo is run, report the consensus both with and without it. If
    the Top-3 is unchanged, that is a robustness result and a direct
    answer to the reviewer.

**Borda(i) = Σ_m (n − rank_m(i)) Consensus Top-3 = argmax₃ Borda(i)**

9.6 Confidence via Monte Carlo

A Top-3 without a stability measure is a bare assertion. Quantify it:

-   Draw 5,000 perturbed scenarios. In each, perturb the criterion
    weights by a Dirichlet draw centred on the nominal weights
    (concentration chosen to give roughly ±20 % variation), and
    independently perturb each PCM property by Gaussian noise scaled to
    its reported measurement uncertainty (±5 % latent heat, ±10 %
    thermal conductivity, ±1 K melting point, wider for cost).

-   Re-run the full ranking pipeline for each draw.

-   For each PCM, report the proportion of draws in which it appears in
    the Top-3 --- \"RT55 appears in the Top-3 in 94 % of 5,000 perturbed
    scenarios.\"

-   Report alongside it: Top-1 retention rate, rank-reversal frequency,
    and Spearman ρ or Kendall τ of each perturbed ranking against the
    baseline. These four together are standard reported practice.

-   Report the full inclusion-probability distribution as a figure. It
    is one of the strongest results the paper can carry and costs only
    compute time.

**On the proposed 10,000 draws.** Not adopted. Inclusion probabilities
converge well before 5,000; many published MCDM stability studies use
1,000. If runtime turns out to be trivial, raising the count is harmless
but should not be presented as a methodological improvement.

**Missing data is handled here, cleanly.** Where a property is unknown,
sample it from the type-class distribution rather than imputing a point
value. PCMs with more missing data then show wider inclusion intervals,
which is the honest representation of what is known.

10\. Phase 7 --- Physics-Based Validation

  -----------------------------------------------------------------------
  **Do not skip this.** Everything up to §9 produces a preference
  ordering. Nothing in it establishes that a higher-ranked PCM actually
  performs better. Without this phase the paper says \"four MCDM methods,
  given the same weights and the same matrix, agreed with each other\"
  --- which is close to a tautology. This phase makes the claim
  falsifiable, and it is the difference between an undergraduate exercise
  and a publishable result.

  -----------------------------------------------------------------------

10.1 The simulation tool --- and why not EnergyPlus

  --------------------------------------------------------------------------
  **Tool**          **Verdict**   **Reasoning**
  ----------------- ------------- ------------------------------------------
  **Python grey-box PRIMARY       Transparent, every line explicable in a
  lumped enthalpy                 viva, integrates directly with the ranking
  tank model**                    pipeline, no licence. The appropriate and
                                  defensible tool at this scale.

  **TRNSYS Type     Optional      A PCM-in-tank component built on the Type
  860**             cross-check   60 water tank using the enthalpy method,
                                  supporting encapsulation geometry,
                                  hysteresis and supercooling. The right
                                  tool if a licence is available.

  **EnergyPlus**    REJECTED      Models PCM only as solid conduction layers
                                  inside building surfaces
                                  (MaterialProperty:PhaseChange with the
                                  CondFD algorithm), and models solar water
                                  heaters through a separate plant-loop
                                  water tank object. There is no supported
                                  path to place a latent-heat PCM inside the
                                  tank node network. Specifying it would
                                  produce a building-envelope result
                                  mislabelled as a hot water result.

  **CFD**           REJECTED      Out of scope and unnecessary. A
                                  well-calibrated lumped model validated
                                  against published experiment is
                                  appropriate; an elaborate model that is
                                  wrong is worse than a crude one honestly
                                  described.
  --------------------------------------------------------------------------

*Table 15. Validation tooling. The EnergyPlus row exists to record why
the review\'s proposal was not adopted.*

10.2 The experiment

-   Build the grey-box tank model with an enthalpy formulation for the
    phase change and a lumped water node, driven by the cluster medoid
    city\'s hourly weather.

-   Use a cited standard domestic hot water draw profile. Do not invent
    one.

-   Simulate a full year for every feasible PCM in that cluster, not
    only the Top-3 --- the full ordering is needed to correlate against.

-   Record: annual solar fraction (the primary metric), hours per year
    with delivery temperature met, mean melt fraction achieved, and
    number of complete cycles.

-   Compute Spearman ρ between the MCDM consensus rank and the simulated
    solar-fraction rank, per cluster.

Calibration benchmarks

Calibrate before trusting the model. If it produces results outside
these bands, fix the model before running the experiment.

  ------------------------------------------------------------------------
  **Benchmark**         **Published range**      **Use**
  --------------------- ------------------------ -------------------------
  **Annual solar        ≈ 54--84 %, typically    The band the annual solar
  fraction, SWH         around 69 %              fraction output must fall
  systems**                                      within

  **TRNSYS model vs     Within ±10 %             The accuracy target for
  experiment**                                   the grey-box model
                                                 against any published
                                                 case it is calibrated on

  **PCM-in-tank series  ≈ +30 % solar fraction   Sanity check that adding
  configuration gain**  over a plain tank;       PCM in the model improves
                        +5--12 % over parallel   solar fraction by a
                                                 plausible margin

  **Night-time          300 L at 60 ± 2 °C       Sanity check on discharge
  delivery**            sustained for \~7 h from duration and delivered
                        a 0.3 × 0.6 m paraffin   volume
                        bed                      

  **Flat-plate paraffin Maximum daily efficiency Upper bound on
  daily efficiency**    near 65 %                collector-side efficiency
                                                 in the model
  ------------------------------------------------------------------------

*Table 16. Calibration benchmarks drawn from published PCM-SWH
experiments.*

10.3 Interpreting the result --- all three outcomes are publishable

  -------------------------------------------------------------------------
  **Outcome**   **Meaning**         **What to write**
  ------------- ------------------- ---------------------------------------
  **ρ \> 0.8**  MCDM ranking        The strong result. Report ρ per cluster
                predicts physical   and state that MCDM is a valid low-cost
                performance         proxy for simulation --- which matters
                                    because simulation is expensive and
                                    MCDM is not.

  **0.4 \< ρ \< Partial agreement   The most likely outcome and still a
  0.8**                             good result. Identify which criteria
                                    drive the disagreement --- usually a
                                    weight over-rewarding latent heat while
                                    the simulation is conductivity-limited.
                                    Recommend a weight adjustment and show
                                    it improves ρ.

  **ρ \< 0.4**  MCDM ranking does   Also publishable, and more interesting
                not predict         than it feels. It means the criteria
                performance         set or the weights are wrong. Diagnose
                                    which, fix it, report both before and
                                    after. A negative result you diagnosed
                                    beats a positive result you did not
                                    earn.
  -------------------------------------------------------------------------

*Table 17. Validation outcomes. Decide now that whatever result appears
will be reported --- deciding afterwards is how results get quietly
reshaped.*

11\. Phase 8 --- Explanation and Final Output

For every cluster, produce a recommendation card. Six of these --- one
per cluster --- form the results section of the paper.

  -----------------------------------------------------------------------
  **Field**          **Content**
  ------------------ ----------------------------------------------------
  **Cluster          ID, medoid city, member cities, one-line physical
  identity**         description, mean maximum membership probability

  **Climate          The 18 indices, mean ± standard deviation
  signature**        

  **Derived          Tm_target (with the assumed system configuration
  targets**          stated), L_required, dominant constraint

  **Candidates       Number entering and surviving the feasibility
  screened**         filter, and whether the window was relaxed

  **Rank 1 / 2 / 3** PCM name, type, Tm, L, k, consensus Borda score,
                     per-method ranks (including CoCoSo if run), Monte
                     Carlo Top-3 inclusion probability

  **Criterion        Which criteria drove each PCM\'s score, as a signed
  contributions**    decomposition

  **Simulated        Annual solar fraction from Phase 7 for each of the
  performance**      three, and the cluster Spearman ρ

  **Caveats**        Missing properties, imputations, whether constraints
                     were relaxed, membership ambiguity
  -----------------------------------------------------------------------

*Table 18. Recommendation card schema.*

12\. Timeline --- Re-Baselined

Phases 1 and 2 are substantially complete: the sampling design, the
sun-event computation, both downloads and the merge all exist and have
been run for four states. That is roughly five weeks of the v2.0
schedule already banked. Twelve weeks remain.

12.1 Weeks 1--3: repairs and the PCM database

  ----------------------------------------------------------------------------
  **Week**   **Focus**                            **Exit criterion**
  ---------- ------------------------------------ ----------------------------
  **1**      Repair 1 --- write                   Daily aggregate table exists
             02b_build_daily_aggregates.py,       for all four states; every
             reading the NASA POWER hourly cache  point carries a real
             directly and producing the daily     elevation; point counts and
             table. Repair 2 --- attach per-point coverage fractions reported
             elevation from ERA5 surface          per state
             geopotential and recompute solar     
             geometry for the mountain points.    
             Report the QC numbers of §4.4.       

  **2**      Phase 2 cross-source validation.     Bias decision made and
             ERA5-versus-POWER MBE, RMSE and      documented; agreement figure
             correlation per season per state per drafted; acceptance checks
             sun event. Decide the                of §5.3 all pass
             bias-correction question by the §5.2 
             rule and record the decision. Build  
             the four-panel scatter figure.       

  **3**      PCM property database --- now the    D2 complete at 40+ cited
             critical path. Build to 40+ rows in  rows; Tm_target derivation
             the 42--70 °C band from datasheets   written
             and review papers, every row with a  
             source citation. Derive Tm_target    
             for both direct and indirect         
             configurations and write up the rule 
             with its literature support.         
  ----------------------------------------------------------------------------

*Table 19. Weeks 1--3. The two repairs are independent of each other and
of the PCM database, so this block parallelises well across a team.*

12.2 Weeks 4--12: signature, clustering, ranking, validation, writing

  ----------------------------------------------------------------------------
  **Week**   **Output**
  ---------- -----------------------------------------------------------------
  **4**      Climate signature construction. Both tiers implemented,
             interaction terms computed, PCA on the correlated block,
             correlation matrix inspected for redundancy. D3 complete: one row
             per point with a written justification for every index.

  **5**      Level A clustering. GMM and K-Means fitted for k = 2...12; BIC,
             silhouette, Davies--Bouldin, Calinski--Harabasz and
             bootstrap-stability curves; k chosen and justified, with the
             state-recovery check of §7.1 performed explicitly.

  **6**      Cluster validation and characterisation. ARI against state
             identity, Köppen--Geiger and NBC/ECBC; four state maps with
             membership shading; regime profile cards with population
             coverage; Uttarakhand elevation separation check.

  **7**      Level B temporal clustering for representative points in each
             state, with particular attention to the Tamil Nadu north-east
             monsoon versus the south-west monsoon states. Feasibility filter
             implemented; Tm_target and L_required derived per regime.

  **8**      MCDM core. TOPSIS and GRA working with target-based Tm handling;
             TOPSIS unit-tested against the Oluah 2020 fixture to three
             decimal places; entropy weights computed per regime.

  **9**      MCDM completion. PROMETHEE II and VIKOR implemented; AHP pairwise
             matrix collected from the guide with CR verified below 0.10;
             Borda and Copeland aggregation; Kendall's W per regime.

  **10**     Confidence and headline result. The 5,000-draw Monte Carlo; D6
             --- the Top-3 table per regime with inclusion probabilities,
             Top-1 retention, rank-reversal frequency and population coverage.
             This is the headline result.

  **11**     Thermal simulation. Grey-box enthalpy tank model implemented and
             calibrated against the Table 16 benchmarks, driven by each regime
             medoid point's weather; full-year simulation for every feasible
             PCM in every regime; D7 --- Spearman ρ per regime.

  **12**     Figures, draft and buffer. All figures finalised; full IEEE
             draft; reproducibility check --- clean clone, rerun end to end,
             confirm every number in the paper regenerates.
  ----------------------------------------------------------------------------

*Table 20. Weeks 4--12. Week 12 carries both the draft and the buffer,
which is tight --- if weeks 10--11 overrun, the optional items in §12.3
are what to drop.*

12.3 Critical path, parallelism, and what to drop

The critical path is now: daily-aggregate repair → signature →
clustering → MCDM → simulation → results. The ERA5 download, which
dominated the v2.0 critical path, is behind you.

-   The PCM database (week 3) is fully independent of the climate data
    --- assign it to a different team member and run it in parallel with
    weeks 1--2.

-   The AHP pairwise elicitation needs the guide's time. Book the slot
    in week 6, not week 9.

-   The thermal simulation depends only on the PCM database and one
    point's weather. Start it in week 7 if ahead; it remains the task
    most likely to overrun.

-   Drop in this order if time runs short: the CoCoSo fifth-ranker
    ablation first, the FT-Transformer encoder ablation second, the
    coverage-threshold sensitivity at 80 % and 95 % third. None is
    load-bearing. Do not drop the physics validation --- it is what
    makes the paper publishable.

-   Optional if ahead: a full 24-hour ERA5 request at the same points,
    giving ERA5-based Tier 2 indices for consistency with Tier 1. Only
    worth it if the Phase 2 analysis shows ERA5 and POWER disagreeing
    materially.

13\. Repository Structure and Tooling

The data-acquisition half of the repository already exists. What follows
maps the existing scripts onto the modelling modules still to be
written, so the two halves form one coherent project rather than a
pipeline with an analysis bolted on.

pcm-climate-framework/

\|\-- data/

\| \|\-- raw/boundary/ \# GADM v4.1 admin-1 (exists)

\| \|\-- raw/population/ \# WorldPop 2020 100m raster (exists)

\| \|\-- raw/era5/points/ \# sun-event NetCDF per state-year-month
(exists)

\| \|\-- raw/nasapower/ \# FULL hourly JSON per point-year (exists)

\| \|\-- processed/population_grid_points.csv (exists)

\| \|\-- processed/suntimes.csv (exists)

\| \|\-- processed/climate\_{state}\_points.csv (exists, Tier 1 source)

\| \|\-- processed/daily\_{state}.csv (Repair 1 - to build)

\| \|\-- processed/signature_matrix.csv (to build)

\| \`\-- pcm/pcm_database.csv \# 40-60 rows, 42-70 C band, every row
cited

\|\-- src/

\| \|\-- acquire/ config.py, 00a_build_population_grid.py,

\| \| 00b_build_suntimes.py, 01_download_era5.py,

\| \| 01b_download_nasapower.py, 00_unzip_accum.py (exist)

\| \|\-- preprocess/ 02_combine.py (exists),

\| \| 02b_build_daily_aggregates.py (Repair 1),

\| \| 02c_attach_elevation.py (Repair 2),

\| \| cross_source_validate.py (Phase 2)

\| \|\-- features/ indices_tier1.py, indices_tier2.py,

\| \| signature.py, tm_target.py

\| \|\-- cluster/ fit.py, select_k.py, validate.py

\| \|\-- pcmrank/ filter.py, weights.py, topsis.py, promethee.py,

\| \| vikor.py, gra.py, cocoso.py, aggregate.py, montecarlo.py

\| \|\-- sim/ tank_model.py, draw_profile.py, run_year.py

\| \`\-- viz/ state_maps.py, ranking_plots.py, agreement_scatter.py

\|\-- tests/ test_topsis.py, test_entropy.py, test_indices.py

\|\-- notebooks/ 01_explore.ipynb \... 07_results.ipynb

\|\-- results/ figures/, tables/, cards/

\|\-- environment.yml

\`\-- README.md

**One suggestion on structure.** The acquisition scripts are currently
named for Rajasthan. Since the same pipeline now runs for four states,
parameterise the state as an argument and keep one copy of each script
rather than four near-identical forks. A state column in every processed
output, plus a states.yml listing the four, will save a great deal of
reconciliation later and makes the framework's state-agnosticism visible
to a reader of the code.

Libraries: geopandas and rasterio for the population grid; xarray,
netCDF4 and cdsapi for ERA5; requests for NASA POWER; pandas and numpy;
pvlib for solar geometry, sun-event times and the clear-sky reference;
scikit-learn for GMM, K-Means, PCA and the cluster metrics; scipy for
Spearman and statistics; pymcdm for reference MCDM implementations ---
but write your own TOPSIS as well, because every line must be explicable
in a viva and the target-based criterion needs custom handling;
matplotlib and geopandas for the state maps.

13.1 The TOPSIS unit-test fixture

  -----------------------------------------------------------------------
  **Test your TOPSIS before you trust it.** A wrong TOPSIS still produces
  plausible-looking numbers. Normalisation and weighting errors are
  invisible in the output and only a numerical fixture catches them.

  -----------------------------------------------------------------------

Use Oluah, Akinlabi and Njoku (2020), Energy and Buildings 217,
"Selection of phase change material for improved performance of Trombe
wall systems using the entropy weight and TOPSIS methodology". It
publishes every intermediate matrix --- raw decision matrix, normalised
matrix, weighted normalised matrix, positive and negative ideal
solutions, separation measures and final closeness coefficients --- for
11 PCMs across 4 criteria.

  -----------------------------------------------------------------------
  **Assertion**                           **Expected value**
  --------------------------------------- -------------------------------
  **Entropy weight, thermal               ≈ 72.12 %
  conductivity**                          

  **Entropy weight, heat of fusion**      ≈ 2 %

  **Entropy weight, density**             ≈ 11 %

  **Entropy weight, cost**                ≈ 15 %

  **Best alternative closeness            Capric + palmitic eutectic, Pi
  coefficient**                           ≈ 0.951

  **Worst alternative closeness           n-octadecane, Pi ≈ 0.004
  coefficient**                           
  -----------------------------------------------------------------------

*Table 21. Unit-test assertions. Reproduce these to three decimal places
before running project data. The fixture uses four criteria and omits
melting temperature, so it validates the entropy and TOPSIS machinery
but NOT the target-based Gaussian transform --- test that separately
against a hand-computed example.*

**A second, incidental use.** The 72.12 % thermal-conductivity entropy
weight in this fixture is also the clearest available demonstration of
why entropy weighting alone is untrustworthy, and therefore why the 0.5
/ 0.5 entropy--AHP blend of §9.3 is justified. Cite it for both
purposes.

14\. Risk Register

  --------------------------------------------------------------------------------
  **Risk**              **Likelihood**   **Mitigation**
  --------------------- ---------------- -----------------------------------------
  **Clustering merely   High             Report ARI against state identity
  recovers the four                      explicitly and present the k = 6--8
  state boundaries**                     solution alongside k = 4 with
                                         model-selection evidence for both. If no
                                         intra-state structure exists at all, that
                                         is itself reportable --- but check the
                                         feature set first, since omitting Tier 2
                                         or elevation is the most likely cause.

  **Tier 2 repair       High             Do it in week 1. Without daily-integral
  skipped or deferred**                  indices the signature cannot describe
                                         storage sizing or autonomy, and the
                                         clustering will be driven almost entirely
                                         by temperature. This is the single most
                                         consequential outstanding task.

  **Scope creep back    High             Re-read §1.2. Those are separate
  into forecasting,                      objectives. Finish this one first.
  control or hardware**                  

  **PCM cost and        High             Sample from type-class distributions in
  cycling data                           the Monte Carlo rather than imputing
  unavailable**                          point values; or drop the criterion and
                                         renormalise weights, documenting the
                                         choice.

  **Elevation repair    Medium           Uttarakhand results become indefensible
  skipped**                              and the elev_proxy index meaningless.
                                         ERA5 surface geopotential is a single
                                         invariant field --- a few hours of work
                                         at most.

  **Reviewer objects to  Medium          §1.3 is the prepared answer: four of five
  four states rather                     or six NBC zones, sampled densely and
  than all India**                       population-weighted, versus six zones at
                                         one city each. State the missing
                                         temperate zone as a limitation before it
                                         is raised.

  **Silhouette inflated Medium           Anticipated in §7.4. Explain that
  by the non-contiguous                  between-state gaps are artificially clean
  sampling frame**                       because the intervening territory was not
                                         sampled. Do not present the inflated
                                         value as method superiority.

  **Same PCM wins every Medium           Likely means the Tm window is too wide or
  regime**                               the candidate set too narrow. If it
                                         persists across four genuinely
                                         contrasting states, that is a strong and
                                         commercially useful result --- state it.

  **Thermal simulation  Medium           Simplify to a single-node tank with an
  does not calibrate**                   effective heat capacity. A crude model
                                         honestly described beats an elaborate one
                                         that is wrong.

  **MCDM rank does not  Medium           This is a result, not a failure. See
  correlate with                         Table 17 --- diagnose and report.
  simulation**                           

  **Guide unavailable   Medium           Book the slot in week 6. Fall back to
  for AHP elicitation**                  entropy-only weights (λ = 1) with a
                                         sensitivity analysis over published
                                         subjective weight vectors.

  **A reviewer repeats  Medium           §2 is the prepared answer. Present the
  the earlier critical                   K-Means comparison and the CoCoSo
  review's objections**                  ablation as evidence the alternatives
                                         were tested rather than dismissed.
  --------------------------------------------------------------------------------

*Table 22. Risk register, updated for the four-state design.*

14.1 Recorded as future work

-   Extension of the framework to the remaining Indian states and to the
    temperate NBC zone not represented here. The pipeline is
    state-parameterised and the marginal cost per additional state is
    one download run.

-   A full 24-hour ERA5 request at the same population points, giving
    ERA5-derived daily-integral indices for consistency with the
    sun-event tier.

-   A 12--24 month instrumented field trial in two or three regimes,
    comparing measured against simulated annual solar fraction. Not
    achievable within a final-year timeline without a thermal
    laboratory, but the correct next step.

-   A two-PCM cascade specification for regimes where Level B temporal
    clustering shows the Top-3 flipping between seasons --- most likely
    in Tamil Nadu given its out-of-phase monsoon.

-   Time-varying population weighting, if WorldPop or a comparable
    product publishes annual India rasters at this resolution.

15\. References

IEEE format. Items marked \[V\] were located and verified during the
review-verification pass of §2; \[A\] are already available in the
project folder; \[P\] still need to be pulled; \[D\] are data-source and
software citations required by the methods section.

\[1\] \[V\] N. Ben Ali, B. Louhichi, W. H. Hassan, A. Alizadeh, A. A.
Hussein, W. Aich, K. Hajlaoui, and S. Aminian, \"Design of a Li-ion
battery cooling system incorporating PCM, heat pipes, and liquid
circuits using marine predator algorithm-enhanced ANN and multi-verse
optimization,\" Sci. Rep., vol. 16, no. 1, Art. no. 11796, 2026, doi:
10.1038/s41598-026-41155-5.

\[2\] \[V\] A. B. Huluka and S. Muthulingam, \"Integrated spherical
phase change modules in concrete roofs enhance thermal performance in
hot climates,\" Sci. Rep., vol. 15, no. 1, Art. no. 39845, 2025, doi:
10.1038/s41598-025-23490-1.

\[3\] \[V\] A. Binte Ahmed, M. M. Uddin Qureshi, M. M. Hussain Khan, A.
Dulmini, M. A. Haque Mollah, and R. Rois, \"Application of
seasonal-adjusted hybrid models for forecasting Discomfort Index in a
heat-prone region of Bangladesh,\" PLoS ONE, vol. 21, no. 3, Art. no.
e0344556, 2026, doi: 10.1371/journal.pone.0344556.

\[4\] \[V\] G. Velusamy, N. Kopparthi et al., \"Integrating machine
learning and trend analysis for rainfall forecasting: insights from
DBSCAN, spectral clustering, and climate variability assessments over
major cities in India,\" Int. J. Climatol., vol. 46, no. 4, Art. no.
e70239, 2026, doi: 10.1002/joc.70239.

\[5\] \[V\] P. J. Abass and S. Muthulingam, \"Selection and
thermophysical assessment of phase change materials (PCMs) for space
cooling applications in buildings,\" Numer. Heat Transf. A, Appl., vol.
86, no. 8, pp. 2423--2445, 2025, doi: 10.1080/10407782.2023.2292183.

\[6\] \[P\] M. B. Awan, Z. Ma, W. Lin, A. K. Pandey, and V. V. Tyagi,
\"A characteristic-oriented strategy for ranking and near-optimal
selection of phase change materials for thermal energy storage in
building applications,\" J. Energy Storage, vol. 57, Art. no. 106301,
2023, doi: 10.1016/j.est.2022.106301.

\[7\] \[P\] C. Oluah, E. T. Akinlabi, and H. O. Njoku, \"Selection of
phase change material for improved performance of Trombe wall systems
using the entropy weight and TOPSIS methodology,\" Energy Build., vol.
217, 2020, doi: 10.1016/j.enbuild.2020.109967. --- the TOPSIS unit-test
fixture.

\[8\] \[V\] M. Yazdani, P. Zaraté, E. K. Zavadskas, and Z. Turskis, \"A
combined compromise solution (CoCoSo) method for multi-criteria
decision-making problems,\" Manag. Decis., vol. 57, no. 9, pp.
2501--2519, 2019, doi: 10.1108/MD-05-2017-0458.

\[9\] \[P\] \"Comparative framework for climate-responsive selection of
phase change materials in energy-efficient buildings,\" Energies, vol.
18, no. 22, Art. no. 5982, 2025, doi: 10.3390/en18225982. --- the
closest prior work; read before designing the criteria set.

\[10\] \[V\] \"A criteria-based climate classification approach
considering clustering and building thermal performance: case of
India,\" Build. Environ., 2024, doi: 10.1016/j.buildenv.2024.112057. ---
source of the realistic silhouette expectation.

\[11\] \[V\] S. Dhruva, R. Krishankumar, D. Pamucar, E. K. Zavadskas,
and K. S. Ravichandran, \"Demystifying the stability and the performance
aspects of CoCoSo ranking method under uncertain preferences,\"
Informatica, 2023.

\[12\] \[V\] Y. Zhao et al., \"Study on a hybrid solar water heating
system with phase-change material storage tank,\" J. Cleaner Prod.,
2019. --- the 47.5--57.5 °C suitable phase-change band.

\[13\] \[V\] V. M. Avargani, B. Norton, M. Rahimi, and G. Karimi,
\"Integrating paraffin phase change material in the storage tank of a
solar water heater to maintain a consistent hot water output
temperature,\" J. Energy Storage, 2021. --- the 300 L at 60 ± 2 °C for 7
h benchmark.

\[14\] \[V\] EnergyPlus Engineering Reference, \"Conduction Finite
Difference Solution Algorithm\" and \"Water Thermal Tanks (includes
Water Heaters),\" US DOE / Lawrence Berkeley National Laboratory. ---
basis for the EnergyPlus rejection in §10.1.

\[15\] \[V\] India Meteorological Department, \"Supply of Meteorological
Data,\" IMD Data Supply Portal, dsp.imdpune.gov.in. --- basis for the
IMD access assessment in §4.1.

\[16\] \[A\] B. Singh, R. S. Rai, P. Yadav, S. Srivastava, and C. Yadav,
\"Application of phase change materials in solar water heating systems
--- a comprehensive review,\" Sol. Energy Mater. Sol. Cells, vol. 293,
Art. no. 113888, 2025.

\[17\] \[A\] G.-R. Chen, T.-W. Liao, C.-C. Hsieh, J. Barman, C.-Y.
Huang, and C.-F. J. Kuo, \"Using the Taguchi method and grey relational
analysis to optimize the parameter design of flat-plate collectors with
nanofluids and phase change materials in an integrated solar water
heating system,\" Energy Convers. Manage. X, vol. 26, Art. no. 100910,
2025. --- GRA precedent in the project references.

\[18\] \[A\] Y. Kou et al., \"A novel solar heating building integrated
heat pipes and PCMs: optimizing thermophysical properties and reducing
energy consumption,\" Build. Environ., vol. 285, Art. no. 113674, 2025.

\[19\] \[A\] L. Liu et al., \"The contribution of artificial
intelligence to phase change materials in thermal energy storage: from
prediction to optimization,\" Renew. Energy, vol. 238, Art. no. 121973,
2025.

\[20\] \[A\] F. A. Barqawi, \"Dynamic simulation of phase change
material-integrated solar water heating systems: a machine learning
approach to energy conversion optimization,\" Muthanna J. Eng. Technol.,
vol. 13, no. 3, pp. 1--14, 2025.

\[21\] \[D\] H. Hersbach et al., \"The ERA5 global reanalysis,\" Q. J.
R. Meteorol. Soc., vol. 146, no. 730, pp. 1999--2049, 2020, doi:
10.1002/qj.3803. --- the primary meteorological source.

\[22\] \[D\] NASA Langley Research Center, \"POWER (Prediction of
Worldwide Energy Resources) Hourly Data,\" NASA POWER Project. --- the
independent cross-check source.

\[23\] \[D\] I. Reda and A. Andreas, \"Solar position algorithm for
solar radiation applications,\" Sol. Energy, vol. 76, no. 5, pp.
577--589, 2004, doi: 10.1016/j.solener.2003.12.003. --- the SPA
algorithm used via pvlib for sun-event times.

\[24\] \[D\] W. F. Holmgren, C. W. Hansen, and M. A. Mikofski, \"pvlib
python: a python package for modeling solar energy systems,\" J. Open
Source Softw., vol. 3, no. 29, p. 884, 2018, doi: 10.21105/joss.00884.

\[25\] \[D\] WorldPop, \"Global High Resolution Population Denominators
Project --- India, 2020, UN-adjusted, 100 m,\" University of
Southampton, doi: 10.5258/SOTON/WP00660. --- the population weighting
source.

\[26\] \[D\] GADM, \"Database of Global Administrative Areas, version
4.1,\" 2022. --- state boundary source.

16\. Summary of Decisions

  -----------------------------------------------------------------------
  **Question**             **Answer**
  ------------------------ ----------------------------------------------
  **Is four states enough, Four is defensible and arguably better.
  or does it need to be    Rajasthan, Assam, Tamil Nadu and Uttarakhand
  all of India?**          cover four of the five or six NBC zones with
                           dense population-weighted sampling inside
                           each. State the missing temperate zone as a
                           limitation.

  **Is sun-event sampling  Not for Tier 1 --- it is
  a problem?**             charge--discharge-cycle-aligned and should be
                           argued for as a design choice. It is
                           insufficient for daily-integral indices, which
                           is what Repair 1 fixes.

  **Does the Tier 2 repair No. The NASA POWER raw cache already holds the
  need a new ERA5          full hourly series for every point and year;
  download?**              the merge step discards it. Reading it back is
                           a few hours of work and no queue time.

  **Does the flat 300 m    Yes, in Uttarakhand, which spans roughly 200 m
  elevation matter?**      to over 7,000 m. Attach real elevation from
                           ERA5 surface geopotential or an SRTM DEM and
                           recompute solar geometry. Not optional.

  **Should latitude and    No. That clusters geography rather than
  longitude be clustering  climate and guarantees the states separate,
  features?**              destroying the finding. Coordinates are for
                           the map; elevation is a legitimate physical
                           feature.

  **Should the GMM be      No. The sampling is already
  weighted by              population-weighted by construction; weighting
  population?**            again double-counts. Use population for
                           reporting and for the recommendation cards.

  **What does a good       Intra-state splitting and, ideally,
  clustering result look   cross-state merging. Recovering the four state
  like here?**             boundaries alone is not a finding --- report
                           ARI against state identity so this is visible.

  **How should ERA5 and    Not blended. Cross-validate at matched
  NASA POWER be            instants, then either use ERA5 as backbone
  combined?**              with POWER as reported validation, or
                           quantile-map ERA5 onto POWER if a systematic
                           seasonal bias is demonstrated.

  **What melting           Tm_target = T_delivery + ΔT_approach ≈ 50--58
  temperature should the   °C, not 42--45 °C. The v1.0 sign was wrong:
  framework target?**      the PCM must sit above the delivery
                           temperature to discharge into the water.

  **Switch from GMM to     No to both. GMM suits a continuous gradient
  K-Means, or add STL?**   and its soft membership matters at transition
                           points. STL decomposes raw time series; the
                           clustering object is an aggregated signature
                           vector.

  **Target silhouette      No. Expect 0.15--0.35, and note that this
  above 0.75?**            sampling frame may inflate the value because
                           the territory between states is not sampled.

  **Replace the four MCDM  No --- that is a regression. Keep the
  methods with CoCoSo?**   four-method consensus and add CoCoSo as an
                           optional fifth ranker.

  **Use EnergyPlus for     No. It cannot place a latent-heat PCM inside a
  validation?**            water tank node network. Use the Python
                           grey-box enthalpy model, optionally
                           cross-checked against TRNSYS Type 860.

  **How much time is       Twelve weeks. Phases 1 and 2 are substantially
  left?**                  complete --- roughly five weeks of the
                           previous schedule already banked.
  -----------------------------------------------------------------------

*Table 23. Decision summary for version 3.0.*

Immediate next actions, in order: write 02b_build_daily_aggregates.py
against the NASA POWER cache, because every Tier 2 index and therefore
most of the clustering signal depends on it; attach real per-point
elevation and recompute solar geometry for Uttarakhand; run the
ERA5-versus-POWER agreement analysis and record the bias decision; and
in parallel, on a different pair of hands, build the PCM database to 40+
cited rows in the 42--70 °C band, since that is now the critical path.
