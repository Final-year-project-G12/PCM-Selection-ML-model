# 01 — Project Context

## Identity

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation Framework, Version 3.0
(supersedes v2.0), Group 12, B.Tech CSE Final Year, Amrita School of Engineering. Governing document:
`Objective1_PCM_Climate_Framework_Plan_v3.docx` (extracted in full for this audit — 16 numbered
sections, front matter, and an IEEE-style references list).

## Why four states

Section 1.3 (Table 1) names Rajasthan, Assam, Tamil Nadu, and Uttarakhand as the four target states,
chosen to span distinct climate archetypes (arid/semi-arid, humid subtropical/monsoon-heavy, coastal
tropical, and high-relief montane respectively) so the eventual multi-state clustering run has genuine
climate diversity to discover regimes across, rather than four samples of the same regime. This audit
covers **Rajasthan only** — the first of the four to reach Phase 6.

## Scope decomposition (§1.1–1.2)

Sub-goals SG1–SG4 (climate signature construction, regime discovery, PCM feasibility+ranking,
physics validation) are explicitly bounded: out of scope for Objective 1 are hardware prototyping,
DRL control, and real-time operation — those belong to later project objectives that consume this
objective's output (the per-regime PCM recommendation) as an input, not to Objective 1 itself.

## Deliverables (§1.4, Table 2, D1–D8)

Corresponds closely to the Phase 1–8 pipeline stages: D1 validated climate dataset, D2 climate
signature + PCA, D3 regime clusters + external validation, D4 PCM feasibility-survivor set, D5
MCDM ranking + Monte Carlo confidence, D6 physics-validated solar-fraction ranking, D7
recommendation cards, D8 (implicit) the write-up/methodology section itself, which this
documentation set is designed to directly support.

## Response to prior critical review (§2)

The v3.0 document is explicitly a **correction pass** over v1.0/v2.0, responding to methodology
review on four points: clustering methodology (§2.1 — commits to GMM as primary, K-Means only as a
reported comparison baseline, confirmed in `05_cluster_rajasthan.py`), MCDM method (§2.2 — commits to
a four-method stack, not a single TOPSIS-only ranking, confirmed in `08_mcdm_ranking_rajasthan.py`),
PCM selection criteria (§2.3 — corrects the melting-point band to 42–70°C from an earlier, apparently
wider or misaligned band), and validation strategy (§2.4 — adds Phase 7 physics-based validation as a
non-optional step, explicitly framed as "what makes the result publishable, not skippable as future
work").

## Closest prior work and novelty position (§3, Table 3)

Six novelty claims, **N1–N6** — this project's own framing of what it contributes beyond existing
PCM-SWH literature:

| ID | Claim |
|---|---|
| N1 | Discovered climate regimes (GMM clustering) vs hand-picked climate zones |
| N2 | Two-tier climate signature (sun-event + daily-integral) vs a single representative temperature |
| N3 | Corrected 42–70°C SWH-specific PCM band vs 18–28°C building-thermal-comfort band (a common confusion in adjacent literature) |
| N4 | Top-3 + explicit method-agreement/consensus reporting vs a single declared "winner" PCM |
| N5 | Physics-validated ranking (Phase 7) vs a self-referential MCDM-only result |
| N6 | Population-weighted sampling/regime discovery vs uniform-grid or arbitrary-city sampling |

## Important disambiguation: N1–N6 vs RG1–RG5

**Do not conflate these two systems** — they come from different documents and serve different
purposes:

- **N1–N6** (above) are the framework doc's own novelty positioning, specific to Objective 1's
  climate-signature/clustering/MCDM/validation pipeline.
- **RG1–RG5** (research gaps: RG1 no real-time adaptive control, RG2 no integrated PCM–AI–hardware
  prototype, RG3 poor alignment with household demand, RG4 limited real-world experimental
  validation, RG5 no predictive optimization under climatic uncertainty) come from a **separate**
  artifact, `prompt for extraction.txt`, the template used to generate every paper summary in
  `PCM-Selection-ML-model/Sources/`. Every one of the 21 literature summaries scores itself against
  RG1–RG5 in its own "Direct Relevance to My Project" section. RG1–RG5 belong to the **broader,
  multi-objective project** (this climate/PCM-selection objective plus the downstream DRL-control and
  hardware-prototype objectives), not to the Objective-1 framework doc's own phase structure.
- `18_RESEARCH_GAP_MAPPING.md` in this documentation set maps phases against **both** systems
  explicitly, keeping them separate, because the framework doc itself never states RG1–RG5 and a
  phase→RG mapping that implies otherwise would misattribute a claim this document doesn't make.

## Phase numbering — authoritative source

Confirmed directly from the framework doc (§4–§11), no phase-numbering assumption was needed:

| Phase | Name |
|---|---|
| 1 | Data Collection (As Built) |
| 2 | Preprocessing and Cross-Source Validation |
| 3 | Climate Signature Construction |
| 4 | Climate Regime Clustering |
| 5 | Feasibility Filtering |
| 6 | Multi-Criteria Ranking Engine |
| 7 | Physics-Based Validation |
| 8 | Explanation and Final Output |

There is no "Phase 0" in the framework doc; §0 is a version-3.0 changelog, not a phase. The
pipeline's own `phases.md` and script comments informally call the sampling-design step (population
grid, sun times, elevation) "Phase 0" because it precedes and feeds Phase 1's actual data download —
this documentation set keeps that informal label only where useful for describing implementation
order, and always defers to the framework doc's Phase 1–8 numbering for anything phase-labeled.

## How this documentation set was produced

Every phase audit in this set was built by (1) reading the actual pipeline source files in full —
not skimmed, not inferred from filenames — (2) cross-checking every claimed behavior against the
actual data files on disk (row counts, column headers, sample values), (3) reading the framework
doc's own methodology text for the corresponding phase, and (4) checking the project's literature
folder (`Sources/`) for what is and is not actually supported by a citable source. Where code
comments/docstrings recorded a bug that was found and fixed, this is reported as a finding, not
smoothed over — the project's own commit history of self-corrections (accum_to_flux, GMM covariance
type, VIKOR sign, entropy weight) is itself evidence of a working, self-auditing methodology and is
presented that way throughout this documentation set.
