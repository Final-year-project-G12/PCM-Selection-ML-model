# 11 — Final Readiness Report: Assam

## Current implementation status

**All 8 phases (Data Collection → Recommendation Cards) are implemented and have been run
end-to-end on real Assam data.** Phase 7 (Physics Validation) returns a genuine, honestly-reported
NEGATIVE result (all four clusters' Spearman rho ≤ 0.286) — this is a real finding, consistent
with the same pattern seen in Rajasthan, and it is not a defect to hide.

## Completed phases

- **Phase 1** (Data Collection) — complete: 128 population-weighted points, 87.5% Assam coverage,
  10 years ERA5 + NASA POWER.
- **Phase 2** (Preprocessing & Validation) — complete: ERA5+POWER merge, 4-season classification
  (Winter/Pre-Monsoon/Monsoon/Post-Monsoon), solar geometry via pvlib.
- **Phase 2.5** (Quality Control) — complete: IsolationForest-based outlier flagging + imputation;
  parquet output per-point; outliers flagged, never deleted.
- **Phase 3** (Climate Signature) — complete: 18-index two-tier signature, PCA on thermodynamic
  block, Tm_target=44°C uniform, Tsoil_mean≈Ta_mean documented fallback.
- **Phase 4** (Clustering) — complete: k=4 (GMM full covariance), BIC selection, 500-bootstrap
  ARI=0.716 (borderline stable), reproducibility via saved joblib models.
- **Phase 5** (Feasibility Filtering + PCM DB) — complete: 25-row database, 7-constraint filter,
  corrosion veto active and load-bearing for humid clusters, 6–8 survivors per cluster.
- **Phase 6** (MCDM Ranking) — complete: 5,000-draw MC (matches plan spec), RT44HC #1 all
  clusters, strong Kendall's W (0.807–0.845), criterion contributions added.
- **Phase 7** (Physics Validation) — complete: grey-box lumped-enthalpy model, backward Euler,
  real climate data, genuine negative result (mean rho=0.242).
- **Phase 8** (Recommendation Cards) — complete: one card per cluster with full analytical
  criterion contributions (explainability requirement, corrected from Tamil Nadu).

## Strongest components

1. **Full Phase 1–8 implementation in one consistent pipeline run.** Unlike Rajasthan which
   required discovering the GMM cluster-label instability across re-runs mid-Phase-7, Assam
   benefits from the fix already being in place (canonical relabeling by ascending latitude from
   the start).

2. **5,000-draw Monte Carlo — matching the plan spec.** Rajasthan used 1,000 draws (documented
   deviation). Assam correctly uses 5,000, providing a more robust uncertainty estimate for the
   Top-3 inclusion probabilities.

3. **Corrosion veto activated in load-bearing way.** The humidity-driven corrosion veto is not
   just present in code — it actually differentiates between Assam's clusters (clusters with
   higher HSI lose inorganic PCM candidates). This is a real climate-sensitive decision, not
   a cosmetic rule.

4. **Criterion Contributions implemented (explainability mandate).** The Tamil Nadu Phase 8
   script missed this plan doc requirement. Assam adds it: each PCM's card now shows the
   percentage contribution of each criterion to its overall score.

5. **Honest negative validation result reported at face value.** All four clusters: Spearman
   rho ≤ 0.286, p-values 0.49–0.69 (not statistically significant). The interpretation logged
   is "weak agreement — diagnose before trusting the MCDM ranking here" — the correct framing.

## Weakest components

1. **No ERA5-POWER agreement analysis**: Assam lacks the formal cross-source bias-correction
   decision that Rajasthan has. Phase 3+ consumes ERA5 GHI without a documented correction step.

2. **PCM database still undersized**: 25 rows vs. 40–60-row target. All Phase 5/6/7/8 outputs
   are provisional. The 6-candidate pools in Clusters 0/1 have insufficient diversity for
   Spearman rho to be meaningful (statistical power is low at n=6).

3. **Bootstrap ARI = 0.716, below the 0.75 threshold**: The k=4 partition is borderline stable.
   This should be reported, not concealed — it accurately reflects Assam's gradual climate
   transitions.

4. **No external classification validation**: Köppen-Geiger lookup not wired in. The k=4
   cluster solution is internally validated only.

5. **No orchestration script**: No `run_all_assam.py`. Pipeline must be run script-by-script.

## What can already be used in the thesis

The full Phase 1–8 methodology description, the 128-point population-weighted sampling strategy,
Assam's 4-cluster result with its BIC/silhouette justification, the corrosion-veto activation
narrative (a climate-sensitive PCM filtering decision), the 5,000-draw MC methodology, the Phase 7
physics validation methodology and its honest-negative result, and the criterion-contribution
explainability addition. All of this is real, defensible, and ready to write up — with the stated
caveats explicitly disclosed.

## What cannot yet be claimed

- That the current Top-3 PCM recommendation is final (provisional, PCM database undersized)
- That the clustering result is externally validated (Köppen-Geiger not wired in)
- That AHP pairwise elicitation informed the weights (it did not)
- That the MCDM ranking is confirmed by physics simulation (it is not — negative result, rho≤0.286)
- That the ERA5 solar radiation is corrected for cross-source bias (no agreement analysis run)

## Prerequisites for a final (non-provisional) result

1. **PCM database expansion to 40–60 rows** — the single blocking item
2. **Run ERA5-POWER agreement analysis** and document the bias-correction decision
3. **Wire in Köppen-Geiger external validation** for Phase 4
4. **Re-run Phases 5–8** after database expansion and agreement-analysis correction

## Final verdict

**READY WITH MINOR FIXES** for Phases 1–4 (data collection, preprocessing, signature, clustering) —
methodology is sound, BIC-justified k=4, 500-bootstrap ARI documented, reproducibility via saved
models. The ERA5-POWER agreement analysis gap is the main methodology-completeness item.

**READY, WITH THE NEGATIVE RESULT STATED PLAINLY** for Phase 7 — the simulation code is
correctly implemented (backward Euler, real climate data, calibration benchmark check), and the
negative rho is correctly reported. What is not ready is treating the negative result as final —
it likely reflects the database size, not a fundamental MCDM failure.

**NOT READY — a clearly-identified fix required** for Phases 5–8 as a *final* result. The PCM
database (25 rows, 6–8 survivors per cluster) is too small for the feasibility filter, MCDM, and
physics simulation to produce trustworthy final recommendations. Re-running after database expansion
is mandatory, not optional cleanup.
