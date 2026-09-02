# 01 — Project Context

## Identity

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation Framework, Version 3.0
(supersedes v2.0), Group 12, B.Tech CSE Final Year, Amrita School of Engineering. Governing document:
`Objective1_PCM_Climate_Framework_Plan_v3.docx`. This documentation covers the **Assam** state
pipeline — the third of four states, implemented after Rajasthan and Tamil Nadu.

## Why four states

Section 1.3 (Table 1) names Rajasthan, Assam, Tamil Nadu, and Uttarakhand as the four target states,
chosen to span distinct climate archetypes: arid/semi-arid (Rajasthan), humid subtropical/monsoon-heavy
(Assam), coastal tropical (Tamil Nadu), and high-relief montane (Uttarakhand). Assam's climate is the
project's most distinctive — extreme monsoon dominance, high annual humidity, and intra-state climate
variability from the Brahmaputra valley floodplain to the hill districts (Karbi Anglong, Dima Hasao).

## Scope decomposition (§1.1–1.2)

Sub-goals SG1–SG4 (climate signature construction, regime discovery, PCM feasibility+ranking, physics
validation) are explicitly bounded: out of scope for Objective 1 are hardware prototyping, DRL control,
and real-time operation.

## Deliverables (§1.4, Table 2, D1–D8)

| Deliverable | Corresponds to | Assam status |
|---|---|---|
| D1 | Validated climate dataset | Complete — `climate_assam_points.csv` |
| D2 | Climate signature + PCA | Complete — 18-index signature, PCA loadings |
| D3 | Regime clusters + external validation | Partially complete — k=4 clusters, external classification not yet wired |
| D4 | PCM feasibility-survivor set | Complete — 25-row database, 6–8 survivors per cluster |
| D5 | MCDM ranking + MC confidence | Complete — 5,000-draw MC, RT44HC #1 all clusters |
| D6 | Physics-validated solar-fraction ranking | Complete — Phase 7 run, negative result |
| D7 | Recommendation cards | Complete — with Criterion Contributions |
| D8 | Methodology write-up | This documentation set |

## Novelty positions (§3, Table 3)

| ID | Claim | Assam implementation |
|---|---|---|
| N1 | Discovered climate regimes vs hand-picked zones | GMM k=4 with BIC justification |
| N2 | Two-tier climate signature | 18 indices: Tier 1 sun-event + Tier 2 daily-integral |
| N3 | Corrected 42–70°C SWH-specific PCM band | Applied in feasibility filter |
| N4 | Top-3 + method-agreement reporting | Borda + Copeland + Kendall's W per cluster |
| N5 | Physics-validated ranking | Phase 7 implemented; negative result |
| N6 | Population-weighted sampling | 128 pts, 87.5% population coverage |

## Important disambiguation: N1–N6 vs RG1–RG5

**Do not conflate these two systems:**
- **N1–N6** are the framework doc's own novelty positioning for Objective 1.
- **RG1–RG5** (no real-time control, no integrated prototype, poor demand alignment, limited
  experimental validation, no predictive optimization) are from a separate artifact used to score
  literature summaries, belonging to the broader multi-objective project, not to this phase.

## Phase numbering — authoritative source

| Phase | Name |
|---|---|
| 1 | Data Collection |
| 2 | Preprocessing and Cross-Source Validation |
| 2.5 | Quality Control (Assam: IsolationForest-based) |
| 3 | Climate Signature Construction |
| 4 | Climate Regime Clustering |
| 5 | Feasibility Filtering (+ PCM database build) |
| 6 | Multi-Criteria Ranking Engine |
| 7 | Physics-Based Validation |
| 8 | Explanation and Final Output |

## Assam-specific contextual notes

**Monsoon dominance**: Assam receives the highest annual rainfall of the four states. The pipeline
explicitly models four seasons — Winter (Dec–Feb), Pre-Monsoon (Mar–May), Monsoon (Jun–Sep),
Post-Monsoon (Oct–Nov) — with Monsoon spanning 4 months, longer than Rajasthan's 3-month mapping.

**Humidity-Solar Interaction (HSI)**: The climate signature includes a dedicated HSI index, which
is not just informational but **operationally load-bearing** in Phase 5: clusters where HSI > global
p75 trigger the corrosion veto that excludes inorganic PCMs from the feasibility filter.

**Tsoil_mean approximation**: Soil temperature data was not downloaded for Assam. The pipeline uses
Ta_mean (annual mean surface temperature) as the fallback, the standard approximation for shallow
soil temperature. This is explicitly documented and user-approved.

**Tm_target uniformity**: Unlike Rajasthan (which used regime-specific capped Tm_target), Assam
uses a **uniform Tm_target = 44°C** across all four clusters (T_delivery=50°C − ΔT_approach=6°C,
Indian domestic standard). This means raw latent heat carries zero regime-specific information, which
is why Phase 6 uses the **latent_heat_margin_ratio** (L / L_required) as its latent-heat criterion,
not raw L.
