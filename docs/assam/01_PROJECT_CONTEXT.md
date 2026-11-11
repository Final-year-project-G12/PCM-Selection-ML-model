# 01 — Project Context

## Identity

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation Framework, Version 3.0
(supersedes v2.0), Group 12, B.Tech CSE Final Year, Amrita School of Engineering. Governing document:
`Objective1_PCM_Climate_Framework_Plan_v3.docx`. This documentation covers the **Assam** state
pipeline — the third of four states, updated through the audited **11-phase implementation architecture**.

## Why Four States

Section 1.3 (Table 1) names Rajasthan, Assam, Tamil Nadu, and Uttarakhand as the four target states,
chosen to span distinct climate archetypes: arid/semi-arid (Rajasthan), humid subtropical/monsoon-heavy
(Assam), coastal tropical (Tamil Nadu), and high-relief montane (Uttarakhand). Assam's climate is the
project's most distinctive — extreme monsoon dominance, high annual humidity, and intra-state climate
variability from the Brahmaputra valley floodplain to the hill districts (Karbi Anglong, Dima Hasao).

## Scope Decomposition (§1.1–1.2)

Sub-goals SG1–SG4 (climate signature construction, regime discovery, PCM feasibility + ranking, physics
validation) are explicitly bounded: out of scope for Objective 1 are physical hardware prototyping, DRL control,
and real-time grid operation.

## Deliverables Status (Complete 11-Phase Scope)

| Deliverable | Description | Authoritative Assam Status |
|---|---|---|
| **D1: Spatial & Climate Grid** | Population-weighted sampling & climate series | **Complete** — 80 points (`ASP_0001`–`ASP_0080`), 87.8% population coverage, 10-year hourly series |
| **D2: Climate Signature & QC** | 19-index signature + QC | **Complete with implementation limitation** — 19 indices across 80 points; all computed from event-sampled data (no working Tier-2 integration — see N2 below); 3σ + IsolationForest QC; 292,160 daily aggregate rows (80 × 3,652 days; 2016-01-02 to 2025-12-31) |
| **D3: Regime Clustering** | Discovered climate regimes | **Locked Final** — $K=3$ GMM (diagonal covariance, 5 features); BIC@K=3 = 1030.97; mean bootstrap ARI = 0.6939, median ARI = 0.7255; cluster sizes 27/28/25; medoids: ASP_0003, ASP_0036, ASP_0080 |
| **D4: SWH System Sizing** | Thermal energy storage specification | **Complete** — 50 kg PCM, 100 kg water, 100 L/day demand; nominal $T_m^{\text{target}} = 56.0^\circ\text{C}$, capped by regime to 49.659°C (C0), 49.395°C (C1), and 49.437°C (C2) |
| **D5: Curated PCM Database** | Deduplicated database with strict provenance | **Locked Final** — 58 PCMs × 41 properties (`pcm_database_final.csv`); strict $C_{p,\text{avg}}$ (no single-phase fallback) |
| **D6: Feasibility Screening** | Physics/safety/corrosion constraint vetoes | **Governed Final** — $K=3$: $n_{\text{confirmed}} = [2, 2, 2]$ and $n_{\text{deployable survivors}} = [5, 5, 5]$ |
| **D7: MCDM Ranking Engine** | Multi-method ranking & consensus | **Performed for final K=3 pipeline** — consensus ranking generated; Top-1 = **savE® OM49** in all three clusters |
| **D8: Monte Carlo Analysis** | Weight and property uncertainty propagation | **Performed** — 5,000 draws per cluster across all 3 clusters; 15 PCM-cluster result rows, all COMPLETED; mean Spearman ρ = 0.7601; savE® OM49 Top-1 retention = 86.6% |
| **D9: Sub-Hourly Physics Validation** | 10-year dynamic tank simulation | **Complete for final K=3 set** — `physics_validation_assam.csv` contains 15 rows (5 deployable PCMs × 3 medoids: ASP_0003, ASP_0036, ASP_0080); do not confuse with the legacy K=4 file `physics_validation_results_assam.csv` |
| **D10: Validation Comparison** | MCDM vs. physics performance comparison | **NOT VERIFIED for final K=3** — existing comparison CSV/report are stale (2026-09-17), contain old medoids ASP_0012/ASP_0092/ASP_0028 and a 9-PCM comparison. Re-run the comparison against the final 15-row K=3 physics output before reporting correlations or a scientific verdict |
| **D11: Consolidation & Manifest** | Thesis-ready outputs and verification | **Complete** — master manifest (31 entries), 10 thesis tables, 10 thesis figures; retain the manifest status only for the artifacts it actually verifies, and separately flag stale comparison/figures |

## Novelty Positions (§3, Table 3)

| ID | Claim | Assam Implementation Reality |
|---|---|---|
| **N1** | Discovered climate regimes vs hand-picked zones | GMM $K=3$ (diagonal covariance on 5 features); BIC@K=3 = 1030.97; bootstrap mean ARI = 0.6939 and median ARI = 0.7255 |
| **N2** | Two-tier climate signature | Not actually implemented as two tiers: `04b_climate_signature.py` computes all 19 indices from the event-sampled physical file alone and does not read `daily_aggregates_assam.csv` / `tier2_signature_assam.csv`, even though `02b_build_daily_aggregates_assam.py` produces both. Flagged as a genuine implementation gap, not a doc error — see `02_DATA_SOURCES_AND_VARIABLES.md` and `16_CLIMATE_SIGNATURE.md`. |
| **N3** | Corrected 42–70°C SWH-specific PCM band | Enforced in feasibility screening against the 58-record database |
| **N4** | Multi-method agreement & consensus | Final K=3 MCDM was performed; savE® OM49 is consensus Top-1 in all three clusters. The separate historical K=4 benchmark evaluated TOPSIS, GRA, PROMETHEE II, VIKOR, Borda, Copeland, and Kendall's W; keep historical and final K=3 results clearly distinguished. |
| **N5** | Dynamic physics validation | Final K=3 sub-hourly physics validation covers 5 deployable PCMs across 3 final medoids (15 result rows). The MCDM-versus-physics comparison for this final K=3 run is not yet verified because the on-disk comparison output is stale. |
| **N6** | Population-weighted sampling | 80 population-weighted points covering 87.8% of Assam's population; do not retain the obsolete 129-point / 128-point descriptions. |

## Important Disambiguation: N1–N6 vs RG1–RG5

**Do not conflate these two systems:**
- **N1–N6** are the framework document's own novelty positioning for Objective 1.
- **RG1–RG5** (no real-time control, no integrated prototype, poor demand alignment, limited
  experimental validation, no predictive optimization) are from a separate literature taxonomy used to score
  the broader multi-objective project, not this specific pipeline.

## Complete Phase Mapping (Phases 1–11)

| Phase | Authoritative Name | Script(s) | Status |
|---|---|---|---|
| **1** | Spatial Grid & Data Collection | `00a`, `00b`, `01`, `01b` | Complete — 80-point grid |
| **2** | Preprocessing & Cross-Source Validation | `02`, `02b`, `03b` | Complete — 292,160 daily aggregate rows |
| **2.5** | Quality Control & Outlier Flagging | `04` | Complete |
| **3** | Climate Regime Clustering (Locked $K=3$) | `04b`, `05` | Locked Final — diagonal covariance |
| **4** | Solar Water Heating Design Specification | `05` | Complete — nominal target 56°C with regime-specific caps |
| **5** | Curated PCM Property Database | `06` | Locked Final — 58 records × 41 columns |
| **6** | Feasibility Filtering Engine | `07` | Governed Final — 2 confirmed and 5 deployable survivors per cluster |
| **7** | Multi-Criteria Decision Making (MCDM) | `08` | Performed — K=3 consensus Top-1 savE® OM49 |
| **8** | Monte Carlo Uncertainty Propagation | `08b_monte_carlo_stability_final.py` | Performed — 5,000 draws per cluster |
| **9** | Sub-Hourly 10-Year Physics Validation | `10` | Complete — 15 rows, 5 PCMs × 3 final medoids |
| **10** | MCDM vs. Physics Validation Comparison | `10_validation_comparison.py` | Pending revalidation — existing comparison output is stale for K=3 |
| **11** | Final Outputs Audit & Consolidation | `consolidate_final_outputs.py`, `generate_phase11_figures.py`, `final_project_verification.py` | Consolidation artifacts exist; stale figures/comparison must be explicitly identified |

## Key Contextual Findings for Assam

1. **Monsoon Dominance & Seasonality**: Assam receives the highest rainfall among the four study states. Preprocessing explicitly tracks four seasons: Winter (Dec–Feb), Pre-Monsoon (Mar–May), Monsoon (Jun–Sep), and Post-Monsoon (Oct–Nov).
2. **Humidity-Solar Interaction (HSI)**: The climate signature features a dedicated HSI index ($HSI = RH_{\text{mean}} \times GHI_{\text{daily}}$), capturing high-humidity charging stress.
3. **PCM Target Temperature**: The current nominal $T_m^{\text{target}}$ is 56.0°C, with regime-specific capped targets of 49.659°C (C0), 49.395°C (C1), and 49.437°C (C2). Do not reuse the obsolete 44.0°C target as the current configuration.
4. **Independent Physics Validation**: The final K=3 physics run evaluates five deployable PCMs at medoids ASP_0003, ASP_0036, and ASP_0080. The earlier 8-PCM/K=4 comparison and its negative-correlation verdict are historical and are not valid evidence for the current K=3 MCDM–physics relationship. That relationship remains unverified until Phase 10 is regenerated.
