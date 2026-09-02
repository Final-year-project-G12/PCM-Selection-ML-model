# 21 — Research Gap Mapping (Assam)

## Important disambiguation (see also `01_PROJECT_CONTEXT.md`)

Two distinct novelty/gap systems exist in this project:

- **N1–N6** (framework doc §3, Table 3): Objective 1's own novelty positioning for the
  climate-signature/clustering/MCDM/validation pipeline.
- **RG1–RG5** (`prompt for extraction.txt` literature-summary template): Research gaps for the
  **broader, multi-objective project** (DRL control, hardware prototype, demand alignment, etc.).
  RG1–RG5 do not appear in the framework doc itself.

## Phase → N (novelty claim) mapping for Assam

| Phase | Primary N-claim(s) | How Assam implements it |
|---|---|---|
| 1 — Data Collection | N6 | 128 population-weighted points, 87.5% coverage, sun-event-aligned sampling |
| 2 — Preprocessing & Validation | (supports all) | ERA5 `accum_to_flux()` fix (inherited from Rajasthan); 4-season classification with IMD-standard Monsoon=Jun–Sep. Gap: no formal ERA5-POWER agreement analysis — N-claims' downstream trustworthiness is partially deferred |
| 3 — Climate Signature | N2, N3 | Two-tier 18-index signature; Tm_target=44°C in the corrected 42–70°C SWH band (not 18–28°C comfort band) |
| 4 — Regime Clustering | N1 | GMM k=4 with BIC/silhouette justification; 500-bootstrap ARI=0.716 reported honestly; **external classification not yet wired** — N1's "discovered, not hand-picked" claim rests on internal statistical measures only for Assam |
| 5 — Feasibility Filtering | N3 (partial) | Enforces 42–70°C band; corrosion veto **actively differentiates Assam from Rajasthan** (HSI > p75 triggers veto in humid clusters) — this is N3's strongest Assam-specific contribution |
| 6 — MCDM Ranking | N4 | Four-method consensus + 5,000-draw MC (matches plan spec); Kendall's W = 0.807–0.845 (strong agreement across all clusters); unanimous RT44HC #1 reflects uniform Tm_target, not a methodology failure |
| 7 — Physics Validation | N5 | Independently tested MCDM ranking against physics simulation — **genuine NEGATIVE result** (rho ≤ 0.286 all clusters). N5's correct framing: "the ranking WAS physics-tested, honestly, with a negative result attributable to the undersized PCM database (6–8 candidates per cluster is insufficient for Spearman rho to be meaningful)" |
| 8 — Recommendation Cards | (packaging) | All N1–N5 evidence packaged per cluster, including Phase 7 negative result and caveats; Criterion Contributions added (plan requirement that TN missed) |

## Phase → RG (broader project research gap) mapping

| Phase | Related RG | Nature of contribution |
|---|---|---|
| 1–2 (Data Collection, Validation) | RG5 | Validated climate data Assam provides as input to predictive-optimization-under-uncertainty components |
| 3–4 (Signature, Clustering) | RG5 | Population-weighted regime discovery — a climatic-uncertainty-aware framing directly relevant to RG5 |
| 5–6 (Feasibility, MCDM) | RG5 | 5,000-draw Monte Carlo uncertainty propagation over PCM properties/weights |
| 7 (Physics Validation) | RG4 (indirect) | Grey-box simulation is not real-world experiment, but is Objective 1's step toward RG4's experimental-validation direction |
| 8 (Recommendation Cards) | RG2, RG3 (feeding, not addressing) | Per-regime PCM recommendation is the input a hardware-prototype objective (RG2) and demand-alignment objective (RG3) would consume |
| — | RG1 | Not addressed by Objective 1 — real-time adaptive control is explicitly out of scope |

## Assam-specific novelty contribution not in other states

Assam adds one concrete N4/N5-adjacent contribution not present in Rajasthan or Tamil Nadu:

**Criterion Contributions (Analytical Decomposition)**: The `09_recommendation_cards.py` Assam
script implements percentage-breakdown of criteria contribution per PCM (min-max normalised
weighted-score decomposition). This directly satisfies the framework doc's Table 18 explainability
mandate that the Tamil Nadu Phase 8 implementation missed. This is a genuine methodology improvement
made during Assam's implementation that will need to be backported to Tamil Nadu's Phase 8.

## What this mapping does NOT claim

- That Objective 1 "solves" RG1–RG4 — it does not; only RG5 is directly addressed
- That the Phase 7 negative result is a methodology failure — it is an honest finding, correctly
  attributed to database size rather than MCDM design
- That the k=4 clustering result is externally validated — it is not; Köppen-Geiger is not wired
  in for Assam
