# 01 — Project Context

## Identity

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation Framework, Version 3.0
(supersedes v2.0), Group 12, B.Tech CSE Final Year, Amrita School of Engineering. Governing document:
`Objective1_PCM_Climate_Framework_Plan_v3.docx`. This documentation covers the **Assam** state
pipeline — the third of four states, implemented after Rajasthan and Tamil Nadu, fully updated
through the audited **11-phase architecture**.

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
| **D1: Spatial & Climate Grid** | Population-weighted sampling & climate series | **Complete** — 129 points (`ASP_0001`–`ASP_0129`), 87.8% population coverage, 10-year hourly series |
| **D2: Climate Signature & QC** | 18-index signature + QC | **Complete** — 18 indices across 129 points; IsolationForest QC; 467,367 daily rows |
| **D3: Regime Clustering** | Discovered climate regimes | **Locked Final** — $K=3$ GMM (full covariance, 5 features); min BIC=1574.94; medoids: ASP_0012, ASP_0092, ASP_0028 |
| **D4: SWH System Sizing** | Thermal energy storage specification | **Complete** — 50 kg PCM, 100 kg water, 100 L/day demand, $T_m^{\text{target}} = 44.0^\circ\text{C}$ ($T_{\text{del}}=50^\circ\text{C}$, $\Delta T=6\text{ K}$) |
| **D5: Curated PCM Database** | Deduplicated database with strict provenance | **Locked Final** — 58 PCMs × 41 properties (`pcm_database_final.csv`); strict $C_{p,\text{avg}}$ (no single-phase fallback) |
| **D6: Feasibility Screening** | Physics/safety/corrosion constraint vetoes | **Governed Final** — $K=3$: $n_{\text{confirmed}} = [0, 0, 0]$, 1 conditional candidate (`n-Tetracosane C24`); historical $K=4$ survivor set: 8 PCMs |
| **D7: MCDM Ranking Engine** | Multi-method ranking & consensus | **Governed Final** — $K=3$: **NOT PERFORMED** ($n_{\text{confirmed}}=0$); historical pre-audit $K=4$ ranking preserved as benchmark |
| **D8: Monte Carlo Analysis** | Weight and property uncertainty propagation | **Governed Final** — $K=3$: **SKIPPED** ($n_{\text{draws}}=0$); historical pre-audit $K=4$ 5,000-draw stability preserved |
| **D9: Sub-Hourly Physics Validation**| 10-year dynamic tank simulation | **Complete** — $\Delta t=300\text{ s}/150\text{ s}$, 4-state enthalpy model, 24 runs, First-Law error = 0.0000%, 100% spin-up convergence |
| **D10: Validation Comparison** | MCDM vs. physics performance comparison | **Complete** — Dual-level assessment; Spearman $\rho = -0.52$ to $-0.64$; verdict: **NOT PHYSICALLY SUPPORTED** |
| **D11: Consolidation & Manifest** | Thesis-ready outputs and verification | **Complete** — Master manifest (31 entries), 10 thesis tables, 10 thesis figures; master verification passes 100% |

## Novelty Positions (§3, Table 3)

| ID | Claim | Assam Implementation Reality |
|---|---|---|
| **N1** | Discovered climate regimes vs hand-picked zones | GMM $K=3$ (full covariance on 5 features) with global BIC minimum (1574.94) and bootstrap ARI (0.6289) |
| **N2** | Two-tier climate signature | 18 indices: Tier 1 sun-event statistics + Tier 2 daily-integral indices |
| **N3** | Corrected 42–70°C SWH-specific PCM band | Enforced in feasibility screening against the 58-row database |
| **N4** | Multi-method agreement & consensus | Historical $K=4$ benchmark evaluated TOPSIS, GRA, PROMETHEE II, VIKOR, Borda, Copeland, Kendall's W |
| **N5** | Dynamic physics validation | 10-year sub-hourly numerical simulation across 8 historical PCMs and 3 final medoids; revealed negative agreement |
| **N6** | Population-weighted sampling | 129 points covering 87.8% of Assam's population (correcting stale pre-audit text referencing 128 points) |

## Important Disambiguation: N1–N6 vs RG1–RG5

**Do not conflate these two systems:**
- **N1–N6** are the framework document's own novelty positioning for Objective 1.
- **RG1–RG5** (no real-time control, no integrated prototype, poor demand alignment, limited
  experimental validation, no predictive optimization) are from a separate literature taxonomy used to score
  the broader multi-objective project, not this specific pipeline.

## Complete Phase Mapping (Phases 1–11)

| Phase | Authoritative Name | Script(s) | Status |
|---|---|---|---|
| **1** | Spatial Grid & Data Collection | `00a`, `00b`, `01`, `01b` | Complete |
| **2** | Preprocessing & Cross-Source Validation | `02`, `02b`, `03b` | Complete |
| **2.5**| Quality Control & Outlier Flagging | `04` | Complete |
| **3** | Climate Regime Clustering (Locked $K=3$) | `04b`, `05` | Locked Final |
| **4** | Solar Water Heating Design Specification | `05` | Complete |
| **5** | Curated PCM Property Database | `06` | Locked Final |
| **6** | Feasibility Filtering Engine | `07` | Governed Final |
| **7** | Multi-Criteria Decision Making (MCDM) | `08` | Governed Final |
| **8** | Monte Carlo Uncertainty Propagation | `08` | Governed Final |
| **9** | Sub-Hourly 10-Year Physics Validation | `10` | Complete |
| **10**| MCDM vs. Physics Validation Comparison | `10_validation_comparison.py` | Complete |
| **11**| Final Outputs Audit & Consolidation | `consolidate_final_outputs.py`, `generate_phase11_figures.py`, `final_project_verification.py` | Complete |

## Key Contextual Findings for Assam

1. **Monsoon Dominance & Seasonality**: Assam receives the highest rainfall among the four study states. Preprocessing explicitly tracks four seasons: Winter (Dec–Feb), Pre-Monsoon (Mar–May), Monsoon (Jun–Sep), and Post-Monsoon (Oct–Nov).
2. **Humidity-Solar Interaction (HSI)**: The climate signature features a dedicated HSI index ($HSI = RH_{\text{mean}} \times GHI_{\text{daily}}$), capturing high-humidity charging stress.
3. **Uniform Approach Temperature**: With a standard delivery requirement of $T_{\text{delivery}} = 50.0^\circ\text{C}$ and heat-exchanger approach $\Delta T = 6.0\text{ K}$, $T_m^{\text{target}} = 44.0^\circ\text{C}$ applies uniformly across all Assam regimes.
4. **Independent Physics Reality**: Dynamic simulation proves that PCMs melting closer to the delivery temperature ($T_m \approx 51^\circ\text{C}$, e.g. `savE OM48`) achieve superior solar fraction and hot-water delivery compared to the $44^\circ\text{C}$ target prioritized by historical MCDM Gaussian scoring.
