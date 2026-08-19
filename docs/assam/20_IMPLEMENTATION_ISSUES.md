# 10 — Implementation Issues: Consolidated List

Ranked by severity. Fixed items are included as evidence of the project's self-audit process.

## Fixed, high-impact

1. **ERA5 deaccumulation bug (inherited fix from Rajasthan)**. The Rajasthan pipeline discovered
   that a naive `deaccumulate()` (diffing consecutive hours) produced near-zero GHI. Assam's
   `02_combine_assam.py` inherits the **fixed** version (`accum_to_flux()` — stateless clip, no
   diffing), applied from the start. This is not a new bug found in Assam — it is a benefit of
   developing Assam after the Rajasthan audit. The fix should be cited as a methodology benefit
   in any write-up.

2. **Uniform Tm_target with raw-latent-heat criterion (design correction)**. Because Tm_target=44°C
   is the same for all Assam clusters, using raw latent heat as a MCDM criterion would rank
   identically in every cluster (zero climate signal). `08_mcdm_ranking.py` correctly uses
   `latent_heat_margin_ratio = L / L_required` instead, making the criterion cluster-relative.
   This is documented explicitly in the script.

## Open, high-priority

3. **PCM database undersized**: 25 rows vs. 40–60-row / 42–70°C target. All Phase 5/6/7/8
   results are tagged provisional. With only 6–8 survivors per cluster, Spearman rho in Phase 7
   has insufficient statistical power (n_candidates=6 → 30 possible rank orderings; rho p-values
   are all 0.49–0.69, confirming no statistical significance).

4. **Phase 7 negative result (weak Spearman rho across all 4 clusters)**: rho = 0.257/0.257/0.286/0.167.
   This is not a code error — it is the correct, honest result. The likely causes are: (a) undersized
   PCM pool, (b) uniform Tm_target creating degenerate rank differences between clusters, (c) solar
   fraction ceiling effect (multiple PCMs cluster near 82–85%). These are data/database issues,
   not methodology issues.

5. **No ERA5-POWER agreement analysis for Assam**: Rajasthan had `03b_agreement_analysis.py`
   producing a documented bias-correction decision. Assam does not have this — no `bias_decision_assam.txt`
   exists. Phase 3+ consumes ERA5 GHI without a documented cross-source correction. This is an
   open scientific-rigor gap.

6. **External climate classification not implemented**: No Köppen-Geiger lookup for Assam.
   Cluster 4's external validity rests entirely on internal statistics (BIC, silhouette, ARI).
   The claim "k=4 produces interpretable climate regimes" is supported by geography knowledge
   but not by a quantitative external ARI check.

7. **Bootstrap ARI = 0.716, stable = False**: The k=4 partition does not meet the 0.75 stability
   threshold. This is an honest result that should be reported. The borderline stability is
   consistent with Assam's genuinely gradual climate transitions (especially Cluster 0 hill/valley
   boundary). It does not invalidate the result but sets the uncertainty correctly.

## Open, lower-priority

8. **No `00c_attach_elevation.py`**: Per-point elevation from ERA5 geopotential was not attached.
   The 100m default is reasonable for the Brahmaputra plains but underestimates elevation for
   Karbi Anglong / Dima Hasao hill points. This affects atmospheric pressure estimation and
   therefore the elev_proxy signature index.

9. **AHP pairwise elicitation not performed**: `AHP_PAIRWISE_MATRIX = None` in `08_mcdm_ranking.py`.
   The "AHP" component uses framework doc Table 13 priors unmodified. Must be stated explicitly
   in any methodology write-up.

10. **Literature PCM property gaps (NaN thermal conductivity/density for Singh2025 rows)**:
    `C22H46 (docosane-class paraffin)` and other literature additions have missing TC, density,
    specific heat in the source. Their criterion contributions are incomplete in Phase 8 output.

11. **Solar fraction exceeds 84% upper benchmark**: RT44HC, C22H46, savE® OM42 all exceed 84%
    solar fraction in Clusters 2 and 3. The published 54–84% band was derived from dry-climate
    SWH systems; it may not be the correct calibration reference for Assam's monsoon-modulated
    solar resource. This is an inherited model assumption, not a code error.

12. **No `run_all_assam.py` orchestration script**: Rajasthan had `run_all_rajasthan.py` to run
    the full pipeline in one invocation. Assam does not have an equivalent. Scripts must be run
    in order manually.

13. **`T_mains_est_C = Ta_mean − 2.0` is unsourced**: Inherited from Rajasthan and Tamil Nadu.
    Feeds `L_required_kJ_per_kg`, which determines the latent-heat feasibility floor. The −2.0 K
    offset has no cited source and is a persistent caveat across all four states.

14. **Incomplete Criterion Contributions for C22H46**: The `09_recommendation_cards.py` outputs
    a blank criterion row for C22H46 due to NaN property values. This is a data-completeness issue,
    not a script error.
