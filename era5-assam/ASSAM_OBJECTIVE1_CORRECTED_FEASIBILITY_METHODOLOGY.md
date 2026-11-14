# Assam Objective 1 — Corrected Feasibility Methodology (Phase B) — Specification and Preliminary Evidence

**Status: specification and preliminary model-derived evidence only. Nothing was implemented, no production script, dataset, configuration or historical result was modified, Phase B was not executed, and the Stage E runs were not touched. Awaiting your explicit approval.**

Written for: the project owner and reviewers who must approve the methodology before implementation.

Companion document: [ASSAM_OBJECTIVE1_FEASIBILITY_FILTER_ROOT_CAUSE_AUDIT.md](ASSAM_OBJECTIVE1_FEASIBILITY_FILTER_ROOT_CAUSE_AUDIT.md) (corrections to that audit are listed in its new Addendum).

Isolated outputs (CSV + scripts used here): [ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY_assets/](ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY_assets/) — a new folder; the scripts read project data and write only there.

## Evidence labels used throughout

| Label | Meaning |
|---|---|
| **[FACT]** | Verified by reading source code or data in this session |
| **[LIT]** | Literature/regulatory knowledge **not** reproduced from a file in this repository; to be verified against the cited source before use |
| **[ENG]** | Engineering assumption chosen for this specification; not measured |
| **[MODEL]** | Result of the screening model of §6 run on the real hourly medoid weather; not experimentally validated |
| **[OPEN]** | Unresolved; needs your decision or new evidence |

---

## 1. Executive summary

**Verdict: NOT READY for implementation until you approve the decisions in §12.** The specification itself is complete and can be implemented as soon as they are approved; what is not ready is any claim of a final candidate list, because the evidence needed for a FEASIBLE label does not exist.

1. **The 49 is a regression, not a method [FACT].** The 2026-10-04 frozen Assam Phase 6 run, produced on byte-identical inputs (same SHA-256 for the PCM database, cluster profiles and design specification), returned **0 → 1 → 5** survivors across relaxation rounds 0/1/2 in every cluster. The Phase 6 script as it is now on disk (edited 2026-11-13 07:26) returns 49 because `survives_all` no longer contains the melting window, latent-heat, charging, cycling or supercooling criteria. The downstream physics-validation script still hard-requires exactly 5 survivors.
2. **Rajasthan is not a clean template for D1 [FACT].** Its archived survivors are all Tm ≤ 52.3 °C against a 57 °C target, i.e. the charging ceiling pulled them below the delivery-capable range, and Rajasthan itself later changed its delivery basis (50 → 60 °C, 2026-09-13). Its documented counts (4/7/5) and archived counts (9/14/16 and 9/15/17) are reconciled in §4.6: the archive is the κ-calibrated companion run, which is itself a survivor-count-targeting procedure (8–20 band) and must not be copied.
3. **Recommended design pathway: evaluate both pathways as separate, never-mixed tracks, with Path A (direct delivery, Tm ≥ 50 + 6 = 56 °C) as the primary track and Path B (preheating, auxiliary boost required) as a labelled secondary track.** The 51–64 °C window inherited from Rajasthan is thermodynamically inconsistent with the project's own 6 K approach for Tm < 56 °C.
4. **Medoids verified [FACT].** ASP_0003 (C0), ASP_0036 (C1), ASP_0080 (C2) are the Objective 1 distance-sum medoids (and also the centroid-nearest points); the Objective 2 hourly weather equals Objective 1's raw NASA POWER records at these points exactly (87 648 hourly rows each, maximum difference 0).
5. **Charging/discharge feasibility is mainly a design-variable question, not a material property [MODEL].** With the provisional nominal design (A = 4 m², m = 50 kg, 20 mm PCM half-thickness) only **18 / 24 / 17** candidates (C0/C1/C2) meet the duty reliability criterion; at the most favourable in-range corner (A = 6 m², 10 mm) every evaluable candidate except one passes; at the pessimistic corner none does. A material screen must therefore reject only what fails at the favourable corner.
6. **Latent heat: derived, not assumed [MODEL/ENG].** On the Objective 1 duty (two 50 L draws, PCM share 0.5, 50 kg) the minimum specific latent heat is 137 kJ/kg on a mean-demand day and 173 kJ/kg on a p95 (cold-month) day, 172/216 kJ/kg at 80 % usable fraction. At the Objective 2 baseline demand (200 L/day) the same share needs 69–86 kg of a 200 kJ/kg PCM, which is at or above the largest constructible PCM mass in the Objective 2 geometry (74.7 kg inline, 68.8 kg staggered [FACT]).
7. **Evidence is the binding limit.** No candidate has material-specific corrosion evidence; only 7 of 58 have reported cycle counts; the safety flag is manufacturer-reported for 7 and imputed for 48. Hence **FEASIBLE = 0 in every cluster** (this is a finding, not a failure of the method). Preliminary counts per cluster, emerging from the rules (not targeted):

| Cluster | FEASIBLE | CONDITIONAL | UNVERIFIED | REJECTED | by pathway | nominal-design pass among CONDITIONAL |
|---|---|---|---|---|---|---|
| C0 | 0 | 51 | 5 | 2 | A: 26 conditional; B: 25 conditional, 5 unverified, 2 rejected | 18 of 51 |
| C1 | 0 | 51 | 5 | 2 | A: 26 conditional; B: 25 conditional, 5 unverified, 2 rejected | 24 of 51 |
| C2 | 0 | 51 | 5 | 2 | A: 26 conditional; B: 25 conditional, 5 unverified, 2 rejected | 17 of 51 |

The status counts are equal across clusters because the REJECTED/UNVERIFIED gates are evaluated at the favourable design corner and by property availability, which do not depend on the cluster; the cluster dependence is in the nominal-design pass counts above (C1 differs most) and in the required collector-area-to-mass ratio of §4.2. I did not force equality; I report below exactly why it arises.

---

## 2. Root cause of the 49-candidate result

(Full candidate-level evidence is in the companion audit; additions from this round are marked.)

| Finding | Evidence | Label |
|---|---|---|
| Current survivor rule keeps only: Tm band 42–70, no inorganic-HSI veto (0 inorganics), no reported-flammability FAIL, k present | `07_feasibility_filter_final.py` lines 430–440 | FACT |
| Window, latent requirement, charging ceiling, cycling, supercooling computed but unused | lines 120–127, 173–181, 221–225, 294–302 | FACT |
| **NEW: the same inputs gave 0 → 1 → 5 on 2026-10-04.** Frozen Phase 6 file `objective1_frozen/pcm/pcm_feasibility_by_cluster.csv` (522 rows = 3 clusters × 3 rounds × 58) has the column `survives_c1_c2_c3_c6`; round 2 window [47, 68] °C; survivors savE OM46, n-Tricosane C23, RT50, Paraffin/EG 85.56 %, savE OM49. | SHA-256 of `pcm_database_final.csv`, `cluster_profiles_assam.csv`, `swh_design_specification.csv` identical to the current files | FACT |
| **NEW: downstream still encodes 5.** `10_physics_validation.py` lines 788–799 raise `ValueError` unless exactly 5 survivors exist and hard-code relaxation round 2 | the 49-row file would make Phase 10 fail | FACT |
| **NEW: the frozen run's "confirmed feasible" (OM46, OM49) rested on a category-only corrosion PASS** (`Not_Applicable_Organic_PCM` for all 58 rows) | frozen audit file | FACT |
| **NEW: 08/08b treat CONFIRMED/CONDITIONAL/INFEASIBLE as "audit classifications only"; only `survives_all` decides eligibility, and the MCDM label is DEPLOYABLE_SURVIVOR** | `08_mcdm_ranking_final.py` lines 296–330, `08b…` lines 215–226 | FACT |
| **NEW: flash point and maximum operating temperature exist in the source dataset** (`PCM_Properties_cleaned_mice_pmm_detailed.csv`: flash point reported for 16 of 55 rows, maximum operating temperature reported for 21) but `06_build_pcm_database_final.py` drops them, so Phase B never used them | source CSV vs `pcm_database_final.csv` columns | FACT |
| Medoid IDs hard-coded in 08/08b (`CLUSTER_MEDOIDS`), a third definition ("highest membership") used for the cards' "Representative point" | 08 line 61; 09 lines 185–190 | FACT |

Root cause (unchanged, now with history): a combination of (i) a rewritten survivor rule that dropped criteria, (ii) cluster-invariant clauses, (iii) permissive/incorrect handling of missing and imputed properties, (iv) discarded safety evidence, (v) absence of a physical model for the temperature criteria, and (vi) no provenance guard between phases.

---

## 3. Rajasthan–Assam comparison matrix (Rajasthan read-only)

| # | Topic | Rajasthan (source read) | Assam current | Same vulnerability in Rajasthan? | Resolution here |
|---|---|---|---|---|---|
| 1 | Survivor definition | c1∧c2∧c3∧c6 blocking (c4/c5 flag only; c7/c8 inert) (`07_feasibility_filter.py` 354–360) | c2 ∧ ¬c7 ∧ ¬c8 ∧ data-complete | No | Restore blocking physical criteria, derived per §4 |
| 2 | Cluster dependence | cluster `tm_target`, `tm_capped`, `l_required`, HSI enter `evaluate()` | clauses are candidate-only | No | Cluster-specific charging, duty, demand |
| 3 | Delivery/approach basis | Tm_target = T_delivery + 7 K; **corrected 2026-09-13 to 60 + 7 = 67 °C** (`04b_climate_signature.py` correction 6); archived outputs predate this (Tm_target 57) | 50 + 6 = 56 °C | Partly: archive contradicts current code | Derive window from 50 + approach; no −5 K lower slack (§4.1) |
| 4 | Charging ceiling | `Tm_target_capped_C = Ta + (kt_worst_month/kt_mean)(Tm_target − Ta)`; **area-agnostic** | identical formula copied (`04c_charging_ceiling_assam.py` states "EXACTLY") but not enforced | **Yes** (area-agnostic heuristic) | Replace by hourly-weather duty screen at declared A/m, δ (§4.2) |
| 5 | Latent requirement | `L_required = SHARE_PCM·Q_night/m_PCM`, SHARE 0.5 (lit. range 0.4–0.78), Q_night = 300 L × cp × (T_del − T_mains), T_mains = Ta − 2 (flagged TODO); documents "Option B: use as a ranking criterion, avoid double gating" | same functional form; spec L_req 125.7–131.4 for 100 L/day; never compared | Partly (placeholder T_mains) | Derive mass-dependent storage requirement (§4.3); drop κ |
| 5b | κ | 0.7 fixed; κ-calibration companion steps κ down until **8–20 survivors** | none | **Yes — count-targeting** | Not adopted |
| 6 | Cycling / supercooling | ≥300 / ≤8 K, flag-not-exclude if NaN; MICE-imputed counted as data | evidence-aware (imputed = UNKNOWN) | Rajasthan weaker | Keep Assam's evidence rule; Reported failures reject |
| 7 | Corrosion | veto only bare salt hydrates in high-HSI clusters; inert on data | same + name-pattern PASS (26 rows) | Inert, not wrong | No PASS without material-specific evidence (§4.4) |
| 8 | Safety | flags only; refuses to exclude on unqualified Yes/No (docstring C) | FAIL on Reported "Yes" (5 savE) | No | Use flash point / max operating temperature; Yes alone ≠ exclusion |
| 9 | Missing properties | NaN → `flag_*`, retained | k missing → rejected | Rajasthan retains silently | Missing → UNVERIFIED, never PASS, never silent reject |
| 10 | Medoid definition | nearest to cluster mean in z-space (`physics_lib.find_medoid`) | distance-sum medoid (`10_physics_validation`) | n/a | Keep Assam definition; verified identical result (§5) |
| 11 | Downstream labels | Top-3 with caveat section (imputed properties); 08 reads only the calibrated survivors file | DEPLOYABLE_SURVIVOR for any `survives_all` | Caveats exist but no hard status | Four statuses with enforced propagation (§4.5) |
| 12 | Provenance | `provenance_lib` fingerprints; 09 cross-phase identity precondition | none | No | Fingerprint + stale-output refusal (§11) |
| 13 | Physical validation | `physics_lib` calibrated simulator, A_c = 4 m², tank 200 kg | `10_physics_validation.py`, A_c = 2 m² | n/a | Screening ≠ validation |

---

## 4. Resolution of D1–D5 (and Rajasthan reconciliation)

### 4.1 D1 — PCM temperature basis

**Thermodynamic consistency of Path A [FACT + ENG].** Direct delivery of water at 50 °C from PCM discharge needs a heat-transfer driving difference at the outlet end; the project's design constant is a 6 K approach. Therefore the PCM must melt/freeze at **Tm ≥ 50 + 6 = 56 °C**. On charging, the collector-loop fluid must be ≥ Tm + 6 K (**≥ 62 °C at Tm = 56**), and the tank/PCM limits cap Tm ≈ 69 °C if the 75 °C system limit is kept. The window [Tm_target − 5, Tm_target + 8] = [51, 64] inherited from Rajasthan therefore contains Tm = 51–55.9 °C materials that are **not** direct-delivery capable at 6 K. A material at Tm = 51 °C corresponds to a 1 K approach, which would need a much larger heat exchanger. The boundary moves with the approach, so it is reported as a sensitivity rather than hidden:

| Approach (K) | Tm needed for direct delivery (°C) | Candidates (of 58) with 50 + approach ≤ Tm ≤ 70 |
|---|---|---|
| 2 | 52 | 39 |
| 3 | 53 | 35 |
| 4 | 54 | 32 |
| 5 | 55 | 28 |
| 6 | 56 | 26 |
| 7 | 57 | 24 |
| 8 | 58 | 24 |

**Path B [MODEL].** Candidates with Tm < 56 °C can preheat water to T_pre = Tm − 6 K at best. Fraction of the 50 °C load that preheat to T_pre can cover (mains from the medoid weather, 10 years):

| Cluster | Tm (°C) | T_pre (°C) | Preheat fraction of load (mean) | (coldest 10 % of days) | Auxiliary energy per 50 kg draw (kWh) |
|---|---|---|---|---|---|
| C0 | 44 | 38 | 0.63 | 0.71 | 0.70 |
| C0 | 47 | 41 | 0.72 | 0.78 | 0.52 |
| C0 | 49 | 43 | 0.78 | 0.83 | 0.41 |
| C0 | 52 | 46 | 0.88 | 0.90 | 0.23 |
| C0 | 55 | 49 | 0.97 | 0.98 | 0.06 |
| C1 | 44 | 38 | 0.61 | 0.70 | 0.70 |
| C1 | 47 | 41 | 0.71 | 0.77 | 0.52 |
| C1 | 49 | 43 | 0.77 | 0.82 | 0.41 |
| C1 | 52 | 46 | 0.87 | 0.90 | 0.23 |
| C1 | 55 | 49 | 0.97 | 0.97 | 0.06 |
| C2 | 44 | 38 | 0.61 | 0.70 | 0.70 |
| C2 | 47 | 41 | 0.71 | 0.78 | 0.52 |
| C2 | 49 | 43 | 0.77 | 0.83 | 0.41 |
| C2 | 52 | 46 | 0.87 | 0.90 | 0.23 |
| C2 | 55 | 49 | 0.97 | 0.98 | 0.06 |

Path B candidates therefore **cannot deliver 50 °C independently**; a downstream boost (electric or other auxiliary, or a second higher-temperature stage) is required. The project has not defined such a device [FACT: no auxiliary/heater/boost entry in `system_assam_fpc.yaml` or `pcm_bed_system_assam.yaml`]. Until it does, no Path B candidate can receive a status above CONDITIONAL.

**Comparison with Rajasthan [FACT].** Rajasthan's archived survivors have Tm 44–52.3 °C against Tm_target 57 °C, i.e. its method also ended in the preheat range through the charging ceiling and window relaxation (and Rajasthan later re-based delivery to 60 °C). That is a precedent for the vulnerability, not a method to reuse.

**Recommendation (D1): option 3 — evaluate both pathways, as separate tracks.** Reasons: (i) the project's success criterion is delivery ≥ 50 °C with zero tolerance, which only Path A can meet without auxiliary heat; (ii) Path B is physically meaningful (preheat fractions 0.63–0.97) but belongs to a different system definition; (iii) mixing them in one ranking would compare materials that perform different jobs. Each candidate is assigned to exactly one pathway by its Tm; rankings, MCDM, Monte Carlo and cards are produced per pathway; labels are never pooled.

### 4.2 D2 — Cluster-specific solar charging

**Verified medoids and weather: see §5.** Rajasthan's check (same worst-month clearness formula, no area or mass) is an area-agnostic heuristic; Assam copied it. Collector assumptions found: Objective 1 physics validation A_c = 2.0 m², F_R(τα) = 0.72, F_R U_L = 4.5 W/m²K (assumed); Objective 2 first test 2 m², M3 baseline 5 m² (FR(τα) 0.75); **Rajasthan physics uses 4 m²**. Your 4 m² / 50 kg are therefore treated as provisional design assumptions; sensitivity covers A = 2–6 m² and m = 25–100 kg (area/mass 0.02–0.24 m²/kg).

**Levels of evidence.**
1. *Necessary-condition screen* (energy balance, hourly Hottel–Whillier–Bliss, inlet = Tm + 6 K): can the collector deliver the melt energy at all? [MODEL]
2. *Detailed thermal feasibility* (this work's quasi-steady Stefan charge/discharge model with state carried over day to day, including recharge after the evening and morning draws): can it be delivered at the rate heat transfer allows? [MODEL, screening-level]
3. *Experimentally validated performance*: none exists. [OPEN]

**Energy screen, base A = 4 m², m = 50 kg, reference PCM L = 200 kJ/kg, starting from the day's mains temperature (pessimistic) — fraction of days a full melt is achievable; worst calendar month; required collector area per kg of PCM on the median day, on the median day of the worst month and on the 90th-percentile day:**

| Cluster | Tm (°C) | Days | Worst month | A/m median day (m²/kg) | A/m worst-month median | A/m p90 day |
|---|---|---|---|---|---|---|
| C0 | 47 | 0.88 | 0.81 | 0.038 | 0.057 | 0.091 |
| C0 | 52 | 0.82 | 0.68 | 0.044 | 0.067 | 0.116 |
| C0 | 56 | 0.78 | 0.57 | 0.049 | 0.076 | 0.146 |
| C0 | 60 | 0.72 | 0.43 | 0.056 | 0.088 | 0.185 |
| C0 | 64 | 0.65 | 0.25 | 0.063 | 0.101 | 0.238 |
| C1 | 47 | 0.81 | 0.68 | 0.040 | 0.050 | 0.159 |
| C1 | 52 | 0.75 | 0.61 | 0.046 | 0.058 | 0.232 |
| C1 | 56 | 0.71 | 0.55 | 0.052 | 0.065 | 0.334 |
| C1 | 60 | 0.66 | 0.53 | 0.059 | 0.074 | 0.517 |
| C1 | 64 | 0.61 | 0.48 | 0.067 | 0.084 | 0.929 |
| C2 | 47 | 0.85 | 0.72 | 0.040 | 0.062 | 0.109 |
| C2 | 52 | 0.79 | 0.59 | 0.046 | 0.074 | 0.144 |
| C2 | 56 | 0.74 | 0.44 | 0.052 | 0.085 | 0.190 |
| C2 | 60 | 0.68 | 0.31 | 0.059 | 0.098 | 0.253 |
| C2 | 64 | 0.60 | 0.17 | 0.068 | 0.116 | 0.353 |

The base A/m = 0.08 m²/kg covers the median day (0.05) but not the p90 day (0.15–0.33), and C1 has the heaviest dull-day tail. Operating-point check (hours/day with irradiance above the critical value / hours above 10 W/m²; instantaneous efficiency at 600 W/m²):

| Cluster | Tm (°C) | Hours G ≥ G_crit / G ≥ 10 | η at 600 W/m² |
|---|---|---|---|
| C0 | 47 | 8.5 / 11.8 | 0.54 |
| C0 | 56 | 7.7 / 11.8 | 0.48 |
| C0 | 64 | 7.0 / 11.8 | 0.42 |
| C1 | 47 | 8.3 / 11.8 | 0.55 |
| C1 | 56 | 7.4 / 11.8 | 0.48 |
| C1 | 64 | 6.6 / 11.8 | 0.42 |
| C2 | 47 | 8.4 / 11.7 | 0.54 |
| C2 | 56 | 7.6 / 11.7 | 0.47 |
| C2 | 64 | 6.8 / 11.7 | 0.41 |

**Threshold derivation (not invented).** The duty is two draws per day after the sun has set or before it has risen. A day is *adequate* when the PCM delivers its full duty share on both draws (the melt fraction needed is whatever that share requires, `x_req = duty/(m L)`, not 100 %). Reliability is then the fraction of days adequate. Rajasthan's own charging ceiling is defined on the **worst calendar month's mean clearness**, i.e. design for the median day of the worst month. The threshold derived from that convention is therefore **R_wm ≥ 0.5** (median day of the worst month is adequate). This is the only threshold used for gating; **0.25, 0.75 and 0.9 are shown as sensitivity** and the decision belongs to you (D2 in §12). The sensitivity table in §9 shows how counts move with it.

### 4.3 D3 — Latent-heat requirement

**Rajasthan [FACT].** `L_required = (SHARE_PCM · Q_night)/m_PCM`, SHARE = 0.5, with Q_night over 300 L and a note that a hard gate is not prescribed and that the property should preferably be a ranking criterion rather than double-gated. Its κ = 0.7 gate and the κ = 0.5/0.6/0.5 values in the archive are calibration outputs of the 8–20 survivor band, not physics.

**Assam's own definition [FACT].** The Phase 4 specification's `Q_required` equals 0.5 × (100 L × cp × (50 − T_mains)) per day, spread over 50 kg: i.e. share 0.5 of the **night plus morning** demand, the same form as Rajasthan (my earlier audit called this ambiguous; it is not — see Addendum).

**Derived requirement [MODEL/ENG].** Duty per day D = 2 draws × share × M_w × c_p × (50 − T_mains). A candidate stores m·L·x_use. Requirement: m·L·x_use·η ≥ D. Specific latent heat (kJ/kg) and total usable storage (MJ) are separated: the same L is enough or not enough depending on m, demand basis and share. Cluster C0 shown (C1/C2 within ±0.1 MJ):

| Demand basis | Share | Usable fraction | D per day (MJ) | L_min for 50 kg, mean / p95 day (kJ/kg) | Candidates below L_min (of 58) | PCM mass for L = 200 kJ/kg, mean / p95 day (kg) |
|---|---|---|---|---|---|---|
| Obj1 spec 50+50 L | 0.4 | 1 | 5.5 | 110 / 138 | 0 / 1 | 27 / 35 |
| Obj1 spec 50+50 L | 0.4 | 0.8 | 5.5 | 137 / 173 | 1 / 14 | 34 / 43 |
| Obj1 spec 50+50 L | 0.5 | 1 | 6.9 | 137 / 173 | 1 / 14 | 34 / 43 |
| Obj1 spec 50+50 L | 0.5 | 0.8 | 6.9 | 172 / 216 | 14 / 34 | 43 / 54 |
| Obj1 spec 50+50 L | 0.7 | 1 | 9.6 | 192 / 242 | 26 / 48 | 48 / 60 |
| Obj1 spec 50+50 L | 0.7 | 0.8 | 9.6 | 240 / 302 | 48 / 58 | 60 / 76 |
| Obj1 spec 50+50 L | 1 | 1 | 13.7 | 275 / 346 | 58 / 58 | 69 / 86 |
| Obj1 spec 50+50 L | 1 | 0.8 | 13.7 | 343 / 432 | 58 / 58 | 86 / 108 |
| Obj2 baseline 100+100 kg | 0.4 | 1 | 11.0 | 220 / 276 | 35 / 58 | 55 / 69 |
| Obj2 baseline 100+100 kg | 0.4 | 0.8 | 11.0 | 275 / 346 | 58 / 58 | 69 / 86 |
| Obj2 baseline 100+100 kg | 0.5 | 1 | 13.7 | 275 / 346 | 58 / 58 | 69 / 86 |
| Obj2 baseline 100+100 kg | 0.5 | 0.8 | 13.7 | 343 / 432 | 58 / 58 | 86 / 108 |
| Obj2 baseline 100+100 kg | 0.7 | 1 | 19.2 | 384 / 484 | 58 / 58 | 96 / 121 |
| Obj2 baseline 100+100 kg | 0.7 | 0.8 | 19.2 | 481 / 605 | 58 / 58 | 120 / 151 |
| Obj2 baseline 100+100 kg | 1 | 1 | 27.5 | 549 / 691 | 58 / 58 | 137 / 173 |
| Obj2 baseline 100+100 kg | 1 | 0.8 | 27.5 | 687 / 864 | 58 / 58 | 172 / 216 |

Consequences: (i) at Objective 1's 100 L/day and share 0.5 the requirement is mild (137 kJ/kg mean-day, 1 candidate below; 173 kJ/kg p95-day, 14 below) — the earlier L_req = 125.7 kJ/kg lies below the mean-day figure because it used the regional-mean mains temperature (19.96 °C) rather than the medoid's (17.2 °C); (ii) the requirement scales with demand and share and **cannot be turned into a fixed κ threshold** — it is a mass-sizing relation; (iii) at Objective 2's 200 L/day, a 50 kg bed cannot supply a 0.5 share from any candidate (L_min 275 kJ/kg), and the mass needed is near the geometric maximum. κ = 0.5 is **not retained**: it came from a survivor-count calibration in another state.

Specification: a candidate is *REJECTED on capacity* only if even the largest constructible mass (74.7 kg, inline [FACT]) at the most favourable share (0.4) and full usable fraction cannot cover D on the p95 day; otherwise capacity is evidence in the dynamic duty screen. Under that rule no candidate is rejected on capacity at the Objective 1 basis; one (Myristic acid/NBR-1.0, L = 128 kJ/kg, capacity ratio 0.93 at nominal) fails the duty screen at the favourable corner for reasons including capacity.

### 4.4 D4 — Safety and corrosion

**Provenance of the flags [FACT].** Flammability: manufacturer-reported for 7 rows (5 "Yes", all savE OM42/48/50/55/65; 2 "No", savE OM46/OM49); imputed (MICE) for 48; missing for 3. Flash point (source CSV): reported for 16 of 55 manufacturer/literature rows, imputed for the rest; all 16 reported values are 155–225 °C (RT42 186, RT44HC >180, RT45HC 155, RT47 >180, RT50 >200, RT55 >200, RT57HC 170, RT60 >200, RT64HC >190, RT65 >200, RT70HC 225, savE OM42/48/50/55/65 200). savE OM46 and OM49 have imputed flash points (206/170) and the Objective 2 evidence register says "needs a sourced value before any safety use". Maximum operating temperature is reported for 21 rows. Corrosion: no column, no coupon data for any candidate.

**Operating conditions considered [FACT/ENG].** PCM temperatures up to the 75 °C tank limit (Objective 2 configuration); enclosed capsules in a water loop; no ignition source inside the vessel; containment material not yet specified.

**Categories (mandatory exclusion marked):**

| Category | Rule | Mandatory exclusion? |
|---|---|---|
| Confirmed acceptable | sourced SDS/flash point **and** enclosure/fire-safety review completed | no (none exist today) |
| Conditional on controls / containment | reported flash point ≥ 93 °C [LIT: below the GHS flammable-liquid cut-off] with temperature limit, enclosure and fire-protection controls; or manufacturer flag "No" with flash point still unsourced; fatty acids (known Cu/Al attack [LIT, uncited in repo]) need compatible containment | no |
| Confirmed unacceptable | reported flash point ≤ the maximum PCM temperature (75 °C), or reported toxic/highly-flammable classification, or a **demonstrated** incompatibility with the specified containment | **yes** |
| Unknown / insufficient evidence | flags imputed or missing | no, but blocks FEASIBLE and is shown on every label |

Result on the 58 [FACT/MODEL]: safety — 18 conditional on controls, 40 unknown, **0 confirmed unacceptable, 0 confirmed acceptable**; corrosion — 10 conditional on containment (fatty acids), 48 unknown, 0 confirmed. Consequently the five savE rows rejected by the current script are **not** excluded by this methodology: a reported "Yes" with a 200 °C flash point is a combustible solid/liquid, not a confirmed unacceptable hazard in a 75 °C-capped enclosed system. This replaces a veto on a flag that cannot grade severity (Rajasthan's own docstring C reaches the same conclusion). None of the 21 reported maximum operating temperatures lies below the charging fluid temperature Tm + 6 K; four (RT47 65 °C, RT44HC 70, RT50 70, RT42 72) lie below the 75 °C system limit and RT45HC equals it, a **design/controller constraint for Objective 2**, not a Phase B exclusion.

### 4.5 D5 — Downstream classification and propagation

| Status | Definition (all four mandatory criteria sets evaluated: pathway thermodynamics, cluster duty screen, evidence, safety/corrosion) |
|---|---|
| **FEASIBLE** | passes the duty screen at nominal **and** under the lowest reported conductivity (for imputed k); cycling Reported ≥ 300; supercooling Reported ≤ 8 K; corrosion confirmed for the specified containment; safety confirmed acceptable; Tm, L, ρ, k all Reported |
| **CONDITIONAL** | not rejected; passes the duty screen at the favourable corner (reason says whether also at nominal); at least one evidence item unresolved |
| **UNVERIFIED** | a physics input needed to evaluate a mandatory criterion is missing or imputed (ρ missing, L imputed) |
| **REJECTED** | fails a mandatory criterion with adequate data: Tm outside 42–70 °C; Reported cycles < 300; Reported supercooling > 8 K; confirmed unacceptable safety/corrosion; fails the duty screen even at the favourable corner (cluster-specific) |

Propagation rules: (1) status is computed per (candidate, cluster, pathway) and carried as a column in every output; (2) **only FEASIBLE and CONDITIONAL enter MCDM**, with the status printed beside every rank; UNVERIFIED are listed but not ranked; REJECTED never enter; (3) `eligibility` may take only the values FEASIBLE, CONDITIONAL, UNVERIFIED, REJECTED — **DEPLOYABLE_SURVIVOR is retired**; (4) MCDM rank, score or Monte Carlo stability can never change a status; (5) Monte Carlo runs per pathway on the ranked set and reports status with every row; (6) physics validation takes the ranked set of a pathway and reports status; its expected-count assertion (`len == 5`) is replaced by "at least one" with the actual count printed; (7) recommendation cards print status, reason, pathway and an evidence-gap line for every PCM; a card may use the word "recommended" only for FEASIBLE and must say "conditional recommendation, evidence pending" otherwise; (8) a candidate rejected in one cluster is not silently dropped in another.

### 4.6 Rajasthan's historical survivor counts reconciled [FACT]

| Source | Count | Explanation |
|---|---|---|
| `07_feasibility_filter.py` docstring | 4 / 7 / 5 | primary run, κ = 0.7 fixed, 62-row database, re-verified 2026-09-07 |
| archived `objective1/feasibility_survivors_by_cluster.csv` (2026-09-07 copy) | 9 / 14 / 16 | this file carries `breakeven_kappa`, `calibrated_kappa` (0.5, 0.6, 0.5): **it is the κ-calibrated companion output saved under the canonical name**. Recomputing κ = 0.7 from its own columns (c1, c2, c6, c7, breakeven_kappa ≥ 0.7) gives **4 / 7 / 5**, exactly the docstring |
| top-level archive (2026-09-12 copy) | 9 / 15 / 17 | same file type, produced after cluster profiles were re-run (profile fingerprint changed 2554… → 2542…; L_required 312.8 → 312.3 etc.); `pcm_database_rajasthan.csv` in that copy has 55 rows while the survivors file has 62 candidates |

So there is no contradiction between the documented and archived counts; there are three different runs (fixed κ, calibrated κ, calibrated κ after re-clustering). **Caveats:** the original Rajasthan processed folder holds no per-run outputs (`era5-rajasthan/data/processed` contains only grid and by-location data), the 62- vs 55-row database discrepancy is unresolved, and Rajasthan's later delivery re-basing (60 °C) postdates both archives. My Rajasthan-rule replay on Assam data (0 → 1 → 5) is **parity evidence for Assam only**: its inputs are byte-identical to the 2026-10-04 frozen Assam run, which gave the same 0/1/5 and the same five names. It is **not** parity evidence for Rajasthan, whose archived inputs I cannot reproduce.

---

## 5. Verified Objective 1 medoids [FACT]

Method in the repository: `10_physics_validation.derive_true_medoids` — minimum sum of Euclidean distances in the five standardised GMM features (GHI_mean, Ta_mean, DTR, RH_mean, wind_mean), within each cluster. I recomputed it from `climate_signatures_raw.csv` and `cluster_assignments_assam.csv`, together with the other two definitions found in the repository:

| Cluster | Points | Distance-sum medoid (Assam O1 definition) | Nearest to centroid (Rajasthan definition) | Highest membership probability (09 cards, TN O2) | Lat, Lon | Elevation (m) |
|---|---|---|---|---|---|---|
| C0 | 27 | ASP_0003 | ASP_0003 | ASP_0048 | 26.125, 91.625 | 75 |
| C1 | 28 | ASP_0036 | ASP_0036 | ASP_0051 | 27.375, 95.125 | 150 |
| C2 | 25 | ASP_0080 | ASP_0080 | ASP_0038 | 26.625, 90.375 | 377 |

The hard-coded IDs in `08_mcdm_ranking_final.py` / `08b_…` equal the distance-sum medoids and lie in their own clusters. The cards' "Representative point" uses a different rule and prints different points (ASP_0048/0051/0038). Source weather: the Objective 2 hourly package at these points versus Objective 1 raw NASA POWER JSON (`data/raw/nasapower/power_<id>_<year>.json`), 2016–2025:

| Cluster | Package point | Hourly rows joined | max |ΔGHI| (W/m²) | max |ΔT| (K) |
|---|---|---|---|---|
| C0 | ASP_0003 | 87648 | 0 | 0 |
| C1 | ASP_0036 | 87648 | 0 | 0 |
| C2 | ASP_0080 | 87648 | 0 | 0 |

Conclusion: the Objective 2 weather files are the same data as Objective 1's raw records at the verified medoids. (Note: the Objective 2 design-basis file quotes ERA5-cell coordinates for the medoids; the weather used is NASA POWER at the grid-point coordinates above.)

---

## 6. Physical equations and assumptions

All items [MODEL] unless noted. IST hourly rows; hour 7 = 06:30–07:30 and hour 19 = 18:30–19:30 are the two draws [FACT: project demand convention]. Mains: T_mains(d) = max(5, mean daily T_amb − 6) [FACT: Objective 1 `10_physics_validation.py`].

1. **Collector (Hottel–Whillier–Bliss), hourly:** Q_col = A·max(0, F_R(τα)·G − F_R U_L·(T_in − T_a)) · η_loop, G ≥ 10 W/m², T_in = Tm + ΔT_app (loop fluid held at the PCM temperature plus the approach). F_R(τα) = 0.72, F_R U_L = 4.5 W/m²K [ENG, from Objective 1, assumed not measured]; GHI used directly on the collector plane (no tilt transposition) [ENG, project convention]. Charging window: hours 8–18 (5–7 % of daily GHI energy lies outside it; conservative).
2. **Charging (quasi-steady Stefan, both faces of a slab of half-thickness δ):** liquid film thickness φ = max((x − x_ref)·δ, 0.5 mm); heat-transfer limit Q_rate = A_face·k_l·ΔT_app/φ with A_face = m/(ρ_s δ); Q = min(Q_col, Q_rate); dx = Q·Δt/(m·L). Natural convection and sensible heat ignored (conservative for rate, optimistic for energy). x_ref resets at the start of each charge phase.
3. **Discharge (draw hours):** frozen-film thickness φ = max((x_ref,d − x)·δ, 0.5 mm); Q_rate = A_face·k_s·(Tm − SC − T̄_w)/φ with T̄_w = (T_mains + T_out)/2; delivered per 15 min = min(Q_rate·Δt, duty/4, remaining latent); T_out = 50 °C (Path A) or Tm − ΔT_app (Path B). Supercooling SC from the database (derived value or 0).
4. **Duty per draw:** Path A: share × M_w × c_p × (50 − T_mains); Path B: min of that and M_w c_p (T_pre − T_mains). Share 0.5 [LIT: Zhao 2022; Huang 2020; Abdelsalam 2020; Koželj 2021 via Rajasthan 04b, range 0.4–0.78; the primary papers were not re-read here]. M_w = 50 kg per draw (Objective 1 spec).
5. **Reliability:** R_all = fraction of days (after a 5-day spin-up, 2016–2025 = 3 647 days) on which both draws are fully covered; R_wm = the lowest monthly fraction over the 12 calendar months.
6. **Imputed conductivity bounds** [FACT/ENG]: lowest/highest *reported* conductivities in the database, solid 0.16–0.33, liquid 0.10–0.27 W/mK, are used as pessimistic/optimistic values for candidates whose k is imputed or missing.
7. **Not modelled (declared):** sensible heat of the solid/liquid PCM, overnight and tank losses, flow-rate/heat-exchanger effectiveness, capsule curvature (cylinder/sphere versus slab: capsule outer diameter 20–60 mm [FACT, M1], PCM radius ≈ 8–28 mm, so δ = 10–30 mm spans it), supercooling nucleation delay, charging while the draw is active. These make the screen a **necessary-condition screen**, not a performance prediction.

---

## 7. Cluster-specific feasibility framework

For each pathway and each of the three clusters independently:

1. Read the verified medoid hourly weather and daily mains temperature (§5).
2. Assign the pathway from Tm and the declared approach (§4.1).
3. Evaluate mandatory band/evidence criteria (§8).
4. Run the duty screen (§6) at nominal, favourable and (for imputed k) lowest-reported-k scenarios; compute R_all and R_wm.
5. Derive storage capacity from the cluster demand (§4.3).
6. Assign a status (§8) and write a reason string and evidence provenance.
7. Report per cluster; never pool clusters; emit a named warning if two clusters produce identical status vectors and say which gate was responsible.

## 8. Candidate classification rules (precise)

For candidate c, cluster k, pathway p (by Tm):

1. **REJECTED** if any: Tm ∉ [42, 70] °C (SWH operating band, retained from Rajasthan c2); cycles Reported < 300; supercooling Reported > 8 K; safety/corrosion *confirmed unacceptable*; physics inputs evaluable but R_wm < 0.5 at the **favourable corner** (A = 6 m², δ = 10 mm, share 0.4, F_R(τα) 0.80, F_R U_L 3.5, k at the reported maximum for imputed k).
2. **UNVERIFIED** if not rejected and a physics input is missing/imputed (ρ missing, L imputed).
3. **FEASIBLE** only if R_wm ≥ 0.5 at nominal **and** at the lowest reported k, and every evidence item is confirmed (cycling Reported ≥ 300, supercooling Reported ≤ 8 K, corrosion confirmed for the specified containment, safety confirmed acceptable, k/ρ/L Reported).
4. **CONDITIONAL** otherwise, with the reasons: *design-dependent* (passes only at the favourable corner), *k-dependent*, and each evidence gap.

## 9. Sensitivity analysis [MODEL]

Number of evaluable candidates and of those meeting R_wm ≥ 0.5 per scenario (C0 / C1 / C2); Path A: Tm ≥ 56 °C (26 candidates), Path B: Tm < 56 °C. Entries are C0 / C1 / C2. Each row changes one factor from nominal.

| Scenario | A: evaluable | A: pass R_wm ≥ 0.5 | B: evaluable | B: pass R_wm ≥ 0.5 |
|---|---|---|---|---|
| nominal: A=4 m2, m=50 kg, FR(ta)=0.72, FRUL=4.5, approach 6 K, delta 20 mm, share 0.5, eta_loop 1.0, k as in database | 26 / 26 / 26 | 9 / 13 / 8 | 30 / 30 / 30 | 9 / 11 / 9 |
| favourable corner: A=6, delta 10 mm, share 0.4, FR(ta)=0.80/FRUL=3.5, k_max (imputed k only) | 26 / 26 / 26 | 26 / 26 / 26 | 30 / 30 / 30 | 29 / 29 / 29 |
| pessimistic corner: A=2, delta 30 mm, share 0.7, FR(ta)=0.65/FRUL=5.0, eta_loop 0.85, k_min | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |
| A = 2 m2 | 26 / 26 / 26 | 0 / 2 / 0 | 30 / 30 / 30 | 2 / 9 / 2 |
| A = 5 m2 | 26 / 26 / 26 | 9 / 15 / 10 | 30 / 30 / 30 | 9 / 11 / 10 |
| A = 6 m2 | 26 / 26 / 26 | 10 / 16 / 11 | 30 / 30 / 30 | 10 / 12 / 10 |
| m = 25 kg | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |
| m = 100 kg | 26 / 26 / 26 | 26 / 26 / 26 | 30 / 30 / 30 | 30 / 30 / 30 |
| FR(ta)=0.80, FRUL=3.5 | 26 / 26 / 26 | 15 / 18 / 17 | 30 / 30 / 30 | 11 / 12 / 12 |
| FR(ta)=0.65, FRUL=5.0 | 26 / 26 / 26 | 0 / 6 / 0 | 30 / 30 / 30 | 5 / 9 / 6 |
| approach 4 K | 32 / 32 / 32 | 0 / 2 / 0 | 24 / 24 / 24 | 1 / 3 / 1 |
| approach 8 K | 24 / 24 / 24 | 11 / 16 / 11 | 32 / 32 / 32 | 14 / 16 / 14 |
| PCM half-thickness 10 mm | 26 / 26 / 26 | 22 / 23 / 22 | 30 / 30 / 30 | 21 / 25 / 23 |
| PCM half-thickness 30 mm | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |
| PCM share 0.4 | 26 / 26 / 26 | 20 / 22 / 21 | 30 / 30 / 30 | 20 / 25 / 21 |
| PCM share 0.7 | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |
| PCM share 1.0 | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |
| loop efficiency 0.85 | 26 / 26 / 26 | 6 / 12 / 5 | 30 / 30 / 30 | 9 / 10 / 9 |
| imputed k at lowest reported | 26 / 26 / 26 | 1 / 2 / 1 | 30 / 30 / 30 | 2 / 3 / 2 |
| imputed k at highest reported | 26 / 26 / 26 | 14 / 17 / 14 | 30 / 30 / 30 | 15 / 19 / 17 |
| Objective 2 baseline demand 200 L/day (two 100 kg draws), m = 50 kg | 26 / 26 / 26 | 0 / 0 / 0 | 30 / 30 / 30 | 0 / 0 / 0 |

Threshold sensitivity (counts of CONDITIONAL candidates meeting each reliability level; nominal / lowest-reported-k / favourable corner):

| R_wm threshold | Path | candidates (C0/C1/C2) | nominal pass | k_min pass | favourable-corner pass |
|---|---|---|---|---|---|
| 0.25 | A | 26 / 26 / 26 | 11 / 16 / 12 | 2 / 3 / 2 | 26 / 26 / 26 |
| 0.25 | B | 25 / 25 / 25 | 10 / 12 / 11 | 3 / 4 / 3 | 25 / 25 / 25 |
| 0.5 | A | 26 / 26 / 26 | 9 / 13 / 8 | 1 / 2 / 1 | 26 / 26 / 26 |
| 0.5 | B | 25 / 25 / 25 | 9 / 11 / 9 | 2 / 3 / 2 | 25 / 25 / 25 |
| 0.75 | A | 26 / 26 / 26 | 2 / 0 / 0 | 0 / 0 / 0 | 26 / 26 / 26 |
| 0.75 | B | 25 / 25 / 25 | 6 / 5 / 6 | 1 / 1 / 1 | 25 / 25 / 25 |
| 0.9 | A | 26 / 26 / 26 | 0 / 0 / 0 | 0 / 0 / 0 | 26 / 16 / 26 |
| 0.9 | B | 25 / 25 / 25 | 2 / 0 / 2 | 1 / 0 / 1 | 24 / 19 / 25 |

Reading: the outcome is controlled by the PCM half-thickness (10 mm → 22–23 of 26 Path A candidates pass; 30 mm → none), by the mass (25 kg → none, 100 kg → all), by the collector area and optical/thermal parameters, by the share of the load the PCM is asked to carry (0.4 → 20–22; 0.7 → none) and by the approach (4 K → 0–2 of 26 at nominal; 8 K → 11–16). Material identity matters mostly through conductivity: with imputed k at the lowest reported value only 1–2 Path A candidates still pass. Cluster identity matters mostly through the dull-day tail (C1 passes more candidates at nominal but needs the largest collector for the p90 day). These sensitivities are why the screen rejects only at the favourable corner.

## 10. Candidate-level preliminary results

**Counts (see §1 for the table).** FEASIBLE 0 in every cluster (no corrosion evidence; 7 of 58 reported cycle counts). Path A: 26 candidates, all CONDITIONAL in all clusters. Path B: 32 candidates; C0/C1/C2 each have 25 CONDITIONAL, 5 UNVERIFIED (two eutectics with ρ and k missing; three paraffin/HDPE composites with imputed L) and 2 REJECTED (RT42 below 42 °C; Myristic acid/NBR-1.0 fails the duty screen at the favourable corner in all three clusters, R_wm 0.03 / 0.33 / 0.16). Nominal-design pass among CONDITIONAL: C0 18, C1 24, C2 17.

**What this does and does not say.** It says that on the current database no candidate can be called FEASIBLE and that no candidate other than two is excluded by any evidence-backed physical criterion. It does **not** say that 51 candidates are good: CONDITIONAL means "not excluded, evidence pending", and the nominal-design pass counts show how few meet the duty at the provisional design. The five candidates preselected from the old Phase B (savE OM46, n-Tricosane C23, RT50, Paraffin/EG 85.56 %, savE OM49) are all Path B, CONDITIONAL; savE OM49 passes at nominal in all clusters (R_wm 0.90 / 0.78 / 0.91, independent of the k bounds because its k is reported); savE OM46 passes only at the favourable corner/thin layer (R_wm 0.00 nominal, 0.76–0.85 at δ = 10 mm), because its reported liquid conductivity (0.10 W/mK) limits charging.

Per-cluster candidate tables (Appendix A–C). Legend: Path A = direct delivery, B = preheat (auxiliary required); L, k (solid/liquid W/mK), ρ show the evidence status (Rep = reported, Imp = imputed); Cyc C = reported ≥ 300 cycles, U = unknown; SC D = supercooling derived from two reported temperatures, U = unknown; Corr/Safety CC = conditional on containment/controls, U = unknown; *Charge* = mean liquid fraction before the evening draw / R_wm (nominal design); *Discharge* = mean coverage of evening / morning duty (nominal); *Cap* = m·L/D at nominal (50 kg, share 0.5, mean day).

## 11. Validation tests and acceptance criteria

| # | Test | Exact expected behaviour / acceptance criterion |
|---|---|---|
| V1 | Medoid selection | `derive_true_medoids` on the frozen assignments returns ASP_0003/0036/0080; hard-coded IDs removed from 08/08b and read from one function; test fails if any ID is not the distance-sum medoid of its cluster or the cluster of an ID differs |
| V2 | Weather source | hourly package equals raw NASA POWER at each medoid: row count 87 648, max abs difference 0 for GHI and T |
| V3 | Cluster-specific evaluation | two synthetic clusters with different weather give different R_wm; identical status vectors raise a named warning naming the gate |
| V4 | Pathway separation | no candidate appears in both pathways; Tm ≥ 50 + approach ⇒ A, else B; changing the approach moves the boundary monotonically; rankings/MCDM/Monte Carlo files are separate per pathway |
| V5 | Latent-heat/capacity | D = 2 · share · M_w · c_p · (50 − T_mains) reproduces the §4.3 table to 1e-9 on fixed inputs; capacity rejection only when the largest constructible mass at share 0.4 cannot cover the p95 day |
| V6 | Charging energy balance | with kinetics switched off (very large k), cumulative melt energy = ∫ Q_col dt to 1e-9 relative; with zero GHI, melt energy 0 |
| V7 | Charging frequency | R_all and R_wm on a hand-built 3-day weather series equal the hand count |
| V8 | Melt fraction | 0 ≤ x ≤ 1 at every step; x never increases during discharge or decreases during charge |
| V9 | Discharge energy balance | delivered energy = Δx·m·L to 1e-9; delivered ≤ duty and ≤ remaining latent |
| V10 | Numerical stability | halving Δt (15 → 7.5 min) changes R_wm by ≤ 0.01; φ_min 0.5 → 0.25 mm changes R_wm by ≤ 0.01; no NaN/Inf for all 58 × 3 × scenarios |
| V11 | Missing properties | ρ missing ⇒ UNVERIFIED; L imputed ⇒ UNVERIFIED; k missing ⇒ evaluated with reported bounds and never FEASIBLE; no missing value ever becomes PASS |
| V12 | Safety/corrosion evidence | name patterns never create PASS; reported flash point ≤ 75 °C ⇒ REJECTED; flammability "Yes" alone ⇒ not REJECTED; corrosion without evidence ⇒ UNKNOWN |
| V13 | Status propagation | `eligibility` ∈ {FEASIBLE, CONDITIONAL, UNVERIFIED, REJECTED}; DEPLOYABLE_SURVIVOR cannot occur; REJECTED/UNVERIFIED never ranked; status identical in feasibility → MCDM → Monte Carlo → physics → cards for every (PCM, cluster, pathway); MCDM rank never alters status |
| V14 | Stale-output detection | every output stores the script SHA-256 and input fingerprints; Phase 7 refuses a Phase 6 file older than its script or with different input fingerprints; a deliberately stale file must hard-fail |
| V15 | Rajasthan–Assam parity | (a) the adapted function with the 2026-10-04 Assam inputs reproduces the frozen 0/1/5 and the five names exactly; (b) with Rajasthan's archived inputs it reproduces the archived survivor table **only if** its input fingerprint matches, otherwise the test reports "inputs differ" and does not claim parity |
| V16 | Regression pin of today's defect | current 49/9 outcome documented as a failing characterisation test before correction |
| V17 | Count non-targeting | no code path references a target survivor count; a static test fails if `MIN_SURVIVORS`, κ calibration or an expected-count assertion appears |
| V18 | Sensitivity report | scenario table of §9 produced for every run and stored with the outputs |

## 12. Remaining uncertainties and decisions requiring your approval

**Uncertainties [OPEN].** All collector parameters are unmeasured; the reference design (A, m, δ) is provisional; k is imputed for 39 of 58 and ρ missing for 2; flash point and corrosion evidence do not exist for most candidates; containment material not specified; no auxiliary-boost device defined for Path B; the screen uses a slab/Stefan quasi-steady approximation and ignores sensible heat and losses; the reliability threshold is derived from Rajasthan's worst-month convention, not from a measured duty requirement; demand basis (100 vs 200 L/day) changes capacity conclusions materially.

**Decisions for you:**

| ID | Decision | My recommendation |
|---|---|---|
| A1 | Pathway scope | evaluate both, separately; Path A primary |
| A2 | Approach ΔT used for the pathway boundary | 6 K (existing constant), sensitivity 4–8 K shown |
| A3 | Reliability gate | R_wm ≥ 0.5 (median day of worst month), sensitivity 0.25/0.75/0.9 |
| A4 | Reference design for the screen (A, m, δ, share) | nominal A = 4 m², 50 kg, δ = 20 mm, share 0.5; favourable corner for rejection |
| A5 | Demand basis | Objective 1 100 L/day for Phase B; report 200 L/day sensitivity |
| A6 | Safety rule | flash point/max-operating-temperature rule of §4.4; "Yes" alone not an exclusion |
| A7 | Corrosion | no PASS without material-specific evidence; specify the containment material |
| A8 | Retire DEPLOYABLE_SURVIVOR and adopt the four statuses | yes |
| A9 | Add the dropped columns (flash point, max operating temperature, volume expansion) to the database build | yes |
| A10 | Auxiliary-boost definition for Path B | needed before any Path B status above CONDITIONAL |

## 13. Recommended implementation sequence (after approval)

1. Characterisation tests (V16) pinning today's behaviour; freeze inputs with fingerprints.
2. New database columns (A9) in a **new** output file (no overwrite); evidence taxonomy module with V11/V12.
3. Medoid and weather module (V1/V2) replacing hard-coded IDs.
4. Pathway module and duty-screen module (V4–V10), isolated, with the scenario table (V18).
5. Classification and status module (V13), then Phase 7/8/8b/9/10 adaptations (retire DEPLOYABLE_SURVIVOR, per-pathway runs, removal of the "exactly 5" assertion).
6. Provenance/staleness guard (V14), Rajasthan parity test (V15) and non-targeting test (V17).
7. Run Phase B into a **new** output directory; compare with the frozen 2026-10-04 and 2026-11-13 outputs; review before anything is promoted.

## 14. Final summary

- **Recommended design pathway:** both pathways, separately; Path A (direct delivery, Tm ≥ 56 °C) as the primary track; Path B (preheat) secondary and always labelled "auxiliary boost required".
- **Justified feasibility methodology:** §§4–8: pathway from thermodynamics; cluster-specific hourly duty screen with rate limits and carried-over state; capacity from demand-derived storage; evidence-graded safety/corrosion; four statuses with enforced propagation; reject only on failed mandatory criteria evaluated at the favourable corner.
- **Counts supported by evidence:** FEASIBLE 0/0/0; CONDITIONAL 51/51/51; UNVERIFIED 5/5/5; REJECTED 2/2/2 (C0/C1/C2). Nominal-design passes among CONDITIONAL: 18 / 24 / 17.
- **Decisions needed:** A1–A10 above.
- **Verdict: NOT READY** — approval-gated; the specification is implementable immediately after you approve A1–A10, and no FEASIBLE label is possible until corrosion, cycling and safety evidence is sourced.

---

## Appendix A — Cluster C0 (ASP_0003) candidate-level evaluation

| ID | Candidate | Tm | Path | L (kJ/kg) | k S/L | ρ | Cyc | SC | Corr | Safety | Charge | Discharge | Cap | Status | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT42 | 40.5 | B | 165 Rep | 0.207/0.15 Imp | 880 Rep | U | D | U | CC | 0.95 / 0.00 | 1.00 / 0.92 | 1.20 | **REJECTED** | Tm 40.5 C outside the 42-70 C SWH band |
| 2 | RT44HC | 42.5 | B | 250 Rep | 0.306/0.174 Imp | 800 Rep | U | D | U | CC | 0.98 / 0.95 | 1.00 / 0.99 | 1.82 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 3 | Myristic-Palmitic eutectic (58/42) | 42.6 | B | 169.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.24 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 4 | savE OM42 | 44 | B | 199 Rep | 0.19/0.1 Rep | 903 Rep | C | D | U | CC | 0.83 / 0.00 | 0.99 / 0.84 | 1.45 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 5 | n-Docosane (C22) | 44.5 | B | 249 Rep | 0.307/0.173 Imp | 880 Rep | U | U | U | U | 0.97 / 0.87 | 1.00 / 0.99 | 1.81 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 6 | RT47 | 44.5 | B | 160 Rep | 0.206/0.15 Imp | 880 Rep | U | D | U | CC | 0.94 / 0.00 | 0.99 / 0.90 | 1.17 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 7 | Lauric acid (C12) | 44.8 | B | 184 Rep | 0.181/0.147 Imp | 960 Imp | U | U | CC | U | 0.91 / 0.00 | 0.99 / 0.91 | 1.34 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 8 | RT45HC | 45 | B | 240 Rep | 0.306/0.174 Imp | 880 Rep | U | D | U | CC | 0.97 / 0.82 | 1.00 / 0.99 | 1.75 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 9 | savE OM46 | 47 | B | 177 Rep | 0.2/0.1 Rep | 917 Rep | C | D | U | CC | 0.80 / 0.00 | 0.99 / 0.78 | 1.29 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 10 | n-Tricosane (C23) | 47.5 | B | 232 Rep | 0.307/0.171 Imp | 797 Imp | U | U | U | U | 0.96 / 0.79 | 1.00 / 0.98 | 1.69 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 11 | RT50 | 48 | B | 160 Rep | 0.208/0.146 Imp | 880 Rep | U | D | U | CC | 0.92 / 0.00 | 0.99 / 0.87 | 1.17 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 12 | Paraffin/Expanded graphite (85.56% paraffin) | 48.8 | B | 161 Rep | 0.193/0.139 Imp | 937 Imp | U | U | U | U | 0.89 / 0.00 | 0.99 / 0.84 | 1.17 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 13 | savE OM49 | 49 | B | 224 Rep | 0.33/0.27 Rep | 816 Rep | C | D | U | CC | 0.97 / 0.90 | 1.00 / 0.98 | 1.63 | **CONDITIONAL** | design OK at nominal; evidence gaps: corr,safety |
| 14 | Paraffin/HDPE PCM3 | 49.9 | B | 187 Imp | 0.205/0.143 Imp | 952 Imp | U | U | U | U | 0.88 / 0.00 | 0.99 / 0.88 | 1.36 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 15 | savE OM50 | 50 | B | 189 Rep | 0.21/0.14 Rep | 961 Rep | C | D | U | CC | 0.88 / 0.00 | 0.99 / 0.88 | 1.38 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 16 | PlusICE A50 | 50 | B | 190 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.95 / 0.41 | 0.99 / 0.95 | 1.38 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 17 | Paraffin/HDPE PCM6 | 50.3 | B | 187 Imp | 0.206/0.143 Imp | 953 Imp | U | U | U | U | 0.88 / 0.00 | 0.99 / 0.88 | 1.36 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 18 | Paraffin/HDPE PCM2 | 50.6 | B | 187 Imp | 0.208/0.146 Imp | 954 Imp | U | U | U | U | 0.88 / 0.00 | 0.99 / 0.88 | 1.36 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 19 | savE OM48 | 51 | B | 165 Rep | 0.2/0.1 Rep | 960 Rep | C | D | U | CC | 0.75 / 0.00 | 0.98 / 0.69 | 1.20 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 20 | PlusICE A52 | 52 | B | 220 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.95 / 0.62 | 0.99 / 0.96 | 1.60 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 21 | n-Tetracosane (C24) | 52 | B | 255 Rep | 0.306/0.173 Imp | 799 Rep | U | D | U | U | 0.95 / 0.81 | 0.99 / 0.98 | 1.86 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 22 | Paraffin/Expanded graphite (92% paraffin) | 52.2 | B | 170 Rep | 0.185/0.129 Imp | 871 Imp | U | U | U | U | 0.87 / 0.00 | 0.98 / 0.83 | 1.24 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 23 | Palmitic-Stearic eutectic (64.2/35.8) | 52.3 | B | 181.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.32 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 24 | Myristic acid (C14) | 53 | B | 199 Rep | 0.216/0.25 Imp | 990 Rep | U | U | CC | U | 0.95 / 0.60 | 0.99 / 0.96 | 1.45 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 25 | PureTemp 53 | 53 | B | 225 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.89 / 0.06 | 0.99 / 0.93 | 1.64 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 26 | RT54HC | 53.5 | B | 200 Rep | 0.199/0.172 Imp | 850 Rep | U | D | U | U | 0.92 / 0.17 | 0.99 / 0.94 | 1.46 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 27 | RT55 | 54 | B | 170 Rep | 0.197/0.147 Imp | 880 Rep | U | D | U | CC | 0.89 / 0.00 | 0.98 / 0.86 | 1.24 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 28 | n-Pentacosane (C25) | 54 | B | 238 Rep | 0.305/0.173 Imp | 801 Rep | U | U | U | U | 0.94 / 0.64 | 0.99 / 0.96 | 1.73 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 29 | Myristic acid/NBR-1.0 | 54.1 | B | 128 Rep | 0.171/0.138 Imp | 860 Imp | U | D | CC | U | 0.92 / 0.00 | 0.97 / 0.73 | 0.93 | **REJECTED** | duty R_wm 0.03 < 0.5 even at favourable corner |
| 30 | Myristic acid/NBR-0.5 | 54.6 | B | 142 Rep | 0.176/0.138 Imp | 865 Imp | U | D | CC | U | 0.90 / 0.00 | 0.98 / 0.77 | 1.03 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 31 | savE OM55 | 55 | B | 188 Rep | 0.16/0.1 Rep | 935 Rep | C | D | U | CC | 0.74 / 0.00 | 0.98 / 0.74 | 1.37 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 32 | Palmitic-stearic acid/Expanded graphite | 55.2 | B | 176 Rep | 0.181/0.139 Imp | 856 Imp | U | D | CC | U | 0.87 / 0.00 | 0.98 / 0.85 | 1.28 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 33 | n-Hexacosane (C26) | 56.5 | A | 256 Rep | 0.299/0.177 Imp | 770 Rep | U | U | U | U | 0.93 / 0.77 | 0.99 / 0.97 | 1.86 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 34 | RT57HC | 56.5 | A | 240 Rep | 0.304/0.174 Imp | 900 Rep | U | D | U | CC | 0.91 / 0.37 | 0.99 / 0.95 | 1.75 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 35 | PlusICE A58 | 58 | A | 215 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.93 / 0.53 | 0.98 / 0.94 | 1.57 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 36 | PureTemp 58 | 58 | A | 225 Rep | 0.25/0.15 Rep | 890 Rep | U | U | U | U | 0.86 / 0.04 | 0.98 / 0.91 | 1.64 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 37 | RT60 | 58 | A | 160 Rep | 0.192/0.147 Imp | 880 Rep | U | D | U | CC | 0.87 / 0.00 | 0.97 / 0.81 | 1.17 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 38 | n-Heptacosane (C27) | 59 | A | 236 Rep | 0.303/0.18 Imp | 773 Rep | U | U | U | U | 0.92 / 0.61 | 0.98 / 0.95 | 1.72 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 39 | CrodaTherm 60 | 59.8 | A | 217 Rep | 0.29/0.17 Rep | 922 Rep | U | U | U | U | 0.87 / 0.05 | 0.98 / 0.91 | 1.58 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 40 | Palmitic acid/Expanded graphite (80/20) | 60.9 | A | 148 Rep | 0.191/0.138 Imp | 860 Imp | U | D | CC | U | 0.86 / 0.00 | 0.96 / 0.75 | 1.08 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 41 | PureTemp 60 | 61 | A | 220 Rep | 0.25/0.15 Rep | 960 Rep | U | U | U | U | 0.82 / 0.01 | 0.97 / 0.87 | 1.60 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 42 | RT65 | 61.5 | A | 150 Rep | 0.183/0.147 Imp | 880 Rep | U | D | U | CC | 0.86 / 0.00 | 0.96 / 0.76 | 1.09 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 43 | n-Octacosane (C28) | 61.6 | A | 253 Rep | 0.307/0.226 Imp | 910 Rep | U | U | U | U | 0.91 / 0.73 | 0.98 / 0.95 | 1.84 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 44 | PlusICE A62 | 62 | A | 205 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.90 / 0.31 | 0.97 / 0.92 | 1.49 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 45 | RT62HC | 62.5 | A | 230 Rep | 0.304/0.173 Imp | 850 Rep | U | D | U | U | 0.88 / 0.17 | 0.97 / 0.92 | 1.68 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 46 | Palmitic acid (C16) | 62.6 | A | 198 Rep | 0.214/0.162 Imp | 989 Rep | U | U | CC | U | 0.82 / 0.00 | 0.97 / 0.84 | 1.44 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 47 | PureTemp 63 | 63 | A | 206 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.82 / 0.00 | 0.97 / 0.85 | 1.50 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 48 | RT64HC | 64 | A | 250 Rep | 0.303/0.263 Imp | 880 Rep | U | D | U | CC | 0.91 / 0.74 | 0.98 / 0.94 | 1.82 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 49 | Paraffin wax (generic) | 64 | A | 173.6 Rep | missing | 916 Rep | U | U | U | U | 0.87 / 0.00 | 0.96 / 0.84 | 1.26 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 50 | n-Nonacosane (C29) | 64 | A | 240 Rep | 0.303/0.257 Imp | 790 Rep | U | U | U | U | 0.92 / 0.75 | 0.98 / 0.94 | 1.75 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 51 | n-Triacontane (C30) | 65.4 | A | 251 Rep | 0.308/0.23 Imp | 910 Imp | U | U | U | U | 0.89 / 0.64 | 0.97 / 0.93 | 1.83 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 52 | savE OM65 | 67 | A | 188 Rep | 0.19/0.13 Rep | 924 Rep | C | D | U | CC | 0.75 / 0.00 | 0.95 / 0.75 | 1.37 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 53 | Stearic acid (C18) | 67.9 | A | 259 Rep | 0.25/0.172 Imp | 965 Rep | U | U | CC | U | 0.82 / 0.05 | 0.97 / 0.89 | 1.89 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 54 | PureTemp 68 | 68 | A | 213 Rep | 0.3/0.251 Imp | 811 Imp | U | U | U | U | 0.89 / 0.56 | 0.96 / 0.90 | 1.55 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 55 | n-Hentriacontane (C31) | 68 | A | 242 Rep | 0.25/0.177 Imp | 930 Rep | U | U | U | U | 0.83 / 0.05 | 0.96 / 0.89 | 1.76 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 56 | RT69HC | 69 | A | 230 Rep | 0.304/0.259 Imp | 940 Rep | U | D | U | U | 0.88 / 0.51 | 0.96 / 0.90 | 1.68 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 57 | n-Dotriacontane (C32) | 69.5 | A | 170 Rep | 0.191/0.139 Imp | 848 Imp | U | U | U | U | 0.79 / 0.00 | 0.94 / 0.74 | 1.24 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 58 | RT70HC | 70 | A | 260 Rep | 0.303/0.177 Imp | 880 Rep | U | D | U | CC | 0.83 / 0.20 | 0.96 / 0.89 | 1.89 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |

## Appendix B — Cluster C1 (ASP_0036)

| ID | Candidate | Tm | Path | L (kJ/kg) | k S/L | ρ | Cyc | SC | Corr | Safety | Charge | Discharge | Cap | Status | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT42 | 40.5 | B | 165 Rep | 0.207/0.15 Imp | 880 Rep | U | D | U | CC | 0.94 / 0.04 | 0.99 / 0.92 | 1.26 | **REJECTED** | Tm 40.5 C outside the 42-70 C SWH band |
| 2 | RT44HC | 42.5 | B | 250 Rep | 0.306/0.174 Imp | 800 Rep | U | D | U | CC | 0.96 / 0.84 | 0.99 / 0.97 | 1.91 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 3 | Myristic-Palmitic eutectic (58/42) | 42.6 | B | 169.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.29 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 4 | savE OM42 | 44 | B | 199 Rep | 0.19/0.1 Rep | 903 Rep | C | D | U | CC | 0.81 / 0.00 | 0.98 / 0.84 | 1.52 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 5 | n-Docosane (C22) | 44.5 | B | 249 Rep | 0.307/0.173 Imp | 880 Rep | U | U | U | U | 0.94 / 0.80 | 0.99 / 0.97 | 1.90 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 6 | RT47 | 44.5 | B | 160 Rep | 0.206/0.15 Imp | 880 Rep | U | D | U | CC | 0.92 / 0.01 | 0.98 / 0.89 | 1.22 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 7 | Lauric acid (C12) | 44.8 | B | 184 Rep | 0.181/0.147 Imp | 960 Imp | U | U | CC | U | 0.89 / 0.03 | 0.98 / 0.90 | 1.40 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 8 | RT45HC | 45 | B | 240 Rep | 0.306/0.174 Imp | 880 Rep | U | D | U | CC | 0.94 / 0.78 | 0.99 / 0.96 | 1.83 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 9 | savE OM46 | 47 | B | 177 Rep | 0.2/0.1 Rep | 917 Rep | C | D | U | CC | 0.78 / 0.00 | 0.97 / 0.77 | 1.35 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 10 | n-Tricosane (C23) | 47.5 | B | 232 Rep | 0.307/0.171 Imp | 797 Imp | U | U | U | U | 0.93 / 0.75 | 0.98 / 0.95 | 1.77 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 11 | RT50 | 48 | B | 160 Rep | 0.208/0.146 Imp | 880 Rep | U | D | U | CC | 0.89 / 0.00 | 0.97 / 0.86 | 1.22 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 12 | Paraffin/Expanded graphite (85.56% paraffin) | 48.8 | B | 161 Rep | 0.193/0.139 Imp | 937 Imp | U | U | U | U | 0.86 / 0.00 | 0.97 / 0.82 | 1.23 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 13 | savE OM49 | 49 | B | 224 Rep | 0.33/0.27 Rep | 816 Rep | C | D | U | CC | 0.94 / 0.78 | 0.98 / 0.95 | 1.71 | **CONDITIONAL** | design OK at nominal; evidence gaps: corr,safety |
| 14 | Paraffin/HDPE PCM3 | 49.9 | B | 187 Imp | 0.205/0.143 Imp | 952 Imp | U | U | U | U | 0.85 / 0.01 | 0.97 / 0.87 | 1.43 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 15 | savE OM50 | 50 | B | 189 Rep | 0.21/0.14 Rep | 961 Rep | C | D | U | CC | 0.84 / 0.01 | 0.97 / 0.86 | 1.44 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 16 | PlusICE A50 | 50 | B | 190 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.92 / 0.65 | 0.97 / 0.92 | 1.45 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 17 | Paraffin/HDPE PCM6 | 50.3 | B | 187 Imp | 0.206/0.143 Imp | 953 Imp | U | U | U | U | 0.85 / 0.01 | 0.97 / 0.86 | 1.43 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 18 | Paraffin/HDPE PCM2 | 50.6 | B | 187 Imp | 0.208/0.146 Imp | 954 Imp | U | U | U | U | 0.85 / 0.01 | 0.97 / 0.86 | 1.43 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 19 | savE OM48 | 51 | B | 165 Rep | 0.2/0.1 Rep | 960 Rep | C | D | U | CC | 0.73 / 0.00 | 0.95 / 0.69 | 1.26 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 20 | PlusICE A52 | 52 | B | 220 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.91 / 0.69 | 0.97 / 0.93 | 1.68 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 21 | n-Tetracosane (C24) | 52 | B | 255 Rep | 0.306/0.173 Imp | 799 Rep | U | D | U | U | 0.91 / 0.75 | 0.98 / 0.95 | 1.95 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 22 | Paraffin/Expanded graphite (92% paraffin) | 52.2 | B | 170 Rep | 0.185/0.129 Imp | 871 Imp | U | U | U | U | 0.83 / 0.00 | 0.96 / 0.82 | 1.30 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 23 | Palmitic-Stearic eutectic (64.2/35.8) | 52.3 | B | 181.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.39 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 24 | Myristic acid (C14) | 53 | B | 199 Rep | 0.216/0.25 Imp | 990 Rep | U | U | CC | U | 0.91 / 0.67 | 0.97 / 0.92 | 1.52 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 25 | PureTemp 53 | 53 | B | 225 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.85 / 0.34 | 0.97 / 0.91 | 1.72 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 26 | RT54HC | 53.5 | B | 200 Rep | 0.199/0.172 Imp | 850 Rep | U | D | U | U | 0.89 / 0.57 | 0.96 / 0.91 | 1.53 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 27 | RT55 | 54 | B | 170 Rep | 0.197/0.147 Imp | 880 Rep | U | D | U | CC | 0.85 / 0.00 | 0.95 / 0.83 | 1.30 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 28 | n-Pentacosane (C25) | 54 | B | 238 Rep | 0.305/0.173 Imp | 801 Rep | U | U | U | U | 0.90 / 0.70 | 0.97 / 0.93 | 1.82 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 29 | Myristic acid/NBR-1.0 | 54.1 | B | 128 Rep | 0.171/0.138 Imp | 860 Imp | U | D | CC | U | 0.88 / 0.00 | 0.94 / 0.72 | 0.98 | **REJECTED** | duty R_wm 0.33 < 0.5 even at favourable corner |
| 30 | Myristic acid/NBR-0.5 | 54.6 | B | 142 Rep | 0.176/0.138 Imp | 865 Imp | U | D | CC | U | 0.85 / 0.00 | 0.94 / 0.76 | 1.08 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 31 | savE OM55 | 55 | B | 188 Rep | 0.16/0.1 Rep | 935 Rep | C | D | U | CC | 0.71 / 0.00 | 0.94 / 0.73 | 1.43 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 32 | Palmitic-stearic acid/Expanded graphite | 55.2 | B | 176 Rep | 0.181/0.139 Imp | 856 Imp | U | D | CC | U | 0.83 / 0.00 | 0.95 / 0.83 | 1.34 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 33 | n-Hexacosane (C26) | 56.5 | A | 256 Rep | 0.299/0.177 Imp | 770 Rep | U | U | U | U | 0.89 / 0.72 | 0.97 / 0.93 | 1.95 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 34 | RT57HC | 56.5 | A | 240 Rep | 0.304/0.174 Imp | 900 Rep | U | D | U | CC | 0.87 / 0.65 | 0.96 / 0.92 | 1.83 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 35 | PlusICE A58 | 58 | A | 215 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.88 / 0.64 | 0.96 / 0.90 | 1.64 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 36 | PureTemp 58 | 58 | A | 225 Rep | 0.25/0.15 Rep | 890 Rep | U | U | U | U | 0.82 / 0.15 | 0.95 / 0.88 | 1.72 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 37 | RT60 | 58 | A | 160 Rep | 0.192/0.147 Imp | 880 Rep | U | D | U | CC | 0.83 / 0.00 | 0.93 / 0.79 | 1.22 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 38 | n-Heptacosane (C27) | 59 | A | 236 Rep | 0.303/0.18 Imp | 773 Rep | U | U | U | U | 0.87 / 0.67 | 0.96 / 0.91 | 1.80 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 39 | CrodaTherm 60 | 59.8 | A | 217 Rep | 0.29/0.17 Rep | 922 Rep | U | U | U | U | 0.83 / 0.27 | 0.94 / 0.88 | 1.66 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 40 | Palmitic acid/Expanded graphite (80/20) | 60.9 | A | 148 Rep | 0.191/0.138 Imp | 860 Imp | U | D | CC | U | 0.81 / 0.00 | 0.92 / 0.73 | 1.13 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 41 | PureTemp 60 | 61 | A | 220 Rep | 0.25/0.15 Rep | 960 Rep | U | U | U | U | 0.78 / 0.04 | 0.94 / 0.84 | 1.68 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 42 | RT65 | 61.5 | A | 150 Rep | 0.183/0.147 Imp | 880 Rep | U | D | U | CC | 0.81 / 0.00 | 0.91 / 0.74 | 1.14 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 43 | n-Octacosane (C28) | 61.6 | A | 253 Rep | 0.307/0.226 Imp | 910 Rep | U | U | U | U | 0.86 / 0.67 | 0.95 / 0.91 | 1.93 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 44 | PlusICE A62 | 62 | A | 205 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.86 / 0.58 | 0.94 / 0.87 | 1.56 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 45 | RT62HC | 62.5 | A | 230 Rep | 0.304/0.173 Imp | 850 Rep | U | D | U | U | 0.83 / 0.57 | 0.94 / 0.88 | 1.75 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 46 | Palmitic acid (C16) | 62.6 | A | 198 Rep | 0.214/0.162 Imp | 989 Rep | U | U | CC | U | 0.78 / 0.02 | 0.93 / 0.81 | 1.51 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 47 | PureTemp 63 | 63 | A | 206 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.78 / 0.02 | 0.93 / 0.82 | 1.57 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 48 | RT64HC | 64 | A | 250 Rep | 0.303/0.263 Imp | 880 Rep | U | D | U | CC | 0.86 / 0.67 | 0.95 / 0.90 | 1.91 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 49 | Paraffin wax (generic) | 64 | A | 173.6 Rep | missing | 916 Rep | U | U | U | U | 0.82 / 0.02 | 0.92 / 0.81 | 1.32 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 50 | n-Nonacosane (C29) | 64 | A | 240 Rep | 0.303/0.257 Imp | 790 Rep | U | U | U | U | 0.86 / 0.66 | 0.94 / 0.89 | 1.83 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 51 | n-Triacontane (C30) | 65.4 | A | 251 Rep | 0.308/0.23 Imp | 910 Imp | U | U | U | U | 0.84 / 0.63 | 0.94 / 0.89 | 1.92 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 52 | savE OM65 | 67 | A | 188 Rep | 0.19/0.13 Rep | 924 Rep | C | D | U | CC | 0.70 / 0.00 | 0.90 / 0.72 | 1.43 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 53 | Stearic acid (C18) | 67.9 | A | 259 Rep | 0.25/0.172 Imp | 965 Rep | U | U | CC | U | 0.77 / 0.49 | 0.92 / 0.86 | 1.98 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 54 | PureTemp 68 | 68 | A | 213 Rep | 0.3/0.251 Imp | 811 Imp | U | U | U | U | 0.83 / 0.57 | 0.92 / 0.85 | 1.63 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 55 | n-Hentriacontane (C31) | 68 | A | 242 Rep | 0.25/0.177 Imp | 930 Rep | U | U | U | U | 0.77 / 0.48 | 0.92 / 0.85 | 1.85 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 56 | RT69HC | 69 | A | 230 Rep | 0.304/0.259 Imp | 940 Rep | U | D | U | U | 0.82 / 0.56 | 0.92 / 0.85 | 1.75 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 57 | n-Dotriacontane (C32) | 69.5 | A | 170 Rep | 0.191/0.139 Imp | 848 Imp | U | U | U | U | 0.73 / 0.00 | 0.89 / 0.71 | 1.30 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 58 | RT70HC | 70 | A | 260 Rep | 0.303/0.177 Imp | 880 Rep | U | D | U | CC | 0.77 / 0.54 | 0.92 / 0.85 | 1.98 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |

## Appendix C — Cluster C2 (ASP_0080)

| ID | Candidate | Tm | Path | L (kJ/kg) | k S/L | ρ | Cyc | SC | Corr | Safety | Charge | Discharge | Cap | Status | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT42 | 40.5 | B | 165 Rep | 0.207/0.15 Imp | 880 Rep | U | D | U | CC | 0.95 / 0.00 | 1.00 / 0.93 | 1.24 | **REJECTED** | Tm 40.5 C outside the 42-70 C SWH band |
| 2 | RT44HC | 42.5 | B | 250 Rep | 0.306/0.174 Imp | 800 Rep | U | D | U | CC | 0.97 / 0.95 | 1.00 / 0.99 | 1.89 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 3 | Myristic-Palmitic eutectic (58/42) | 42.6 | B | 169.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.28 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 4 | savE OM42 | 44 | B | 199 Rep | 0.19/0.1 Rep | 903 Rep | C | D | U | CC | 0.83 / 0.00 | 0.99 / 0.86 | 1.50 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 5 | n-Docosane (C22) | 44.5 | B | 249 Rep | 0.307/0.173 Imp | 880 Rep | U | U | U | U | 0.96 / 0.88 | 1.00 / 0.98 | 1.88 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 6 | RT47 | 44.5 | B | 160 Rep | 0.206/0.15 Imp | 880 Rep | U | D | U | CC | 0.93 / 0.00 | 0.99 / 0.90 | 1.21 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 7 | Lauric acid (C12) | 44.8 | B | 184 Rep | 0.181/0.147 Imp | 960 Imp | U | U | CC | U | 0.91 / 0.00 | 0.99 / 0.91 | 1.39 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 8 | RT45HC | 45 | B | 240 Rep | 0.306/0.174 Imp | 880 Rep | U | D | U | CC | 0.96 / 0.85 | 1.00 / 0.98 | 1.81 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 9 | savE OM46 | 47 | B | 177 Rep | 0.2/0.1 Rep | 917 Rep | C | D | U | CC | 0.80 / 0.00 | 0.99 / 0.79 | 1.34 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 10 | n-Tricosane (C23) | 47.5 | B | 232 Rep | 0.307/0.171 Imp | 797 Imp | U | U | U | U | 0.95 / 0.82 | 0.99 / 0.98 | 1.75 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 11 | RT50 | 48 | B | 160 Rep | 0.208/0.146 Imp | 880 Rep | U | D | U | CC | 0.92 / 0.00 | 0.98 / 0.88 | 1.21 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 12 | Paraffin/Expanded graphite (85.56% paraffin) | 48.8 | B | 161 Rep | 0.193/0.139 Imp | 937 Imp | U | U | U | U | 0.89 / 0.00 | 0.98 / 0.84 | 1.21 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 13 | savE OM49 | 49 | B | 224 Rep | 0.33/0.27 Rep | 816 Rep | C | D | U | CC | 0.96 / 0.91 | 0.99 / 0.97 | 1.69 | **CONDITIONAL** | design OK at nominal; evidence gaps: corr,safety |
| 14 | Paraffin/HDPE PCM3 | 49.9 | B | 187 Imp | 0.205/0.143 Imp | 952 Imp | U | U | U | U | 0.88 / 0.00 | 0.98 / 0.89 | 1.41 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 15 | savE OM50 | 50 | B | 189 Rep | 0.21/0.14 Rep | 961 Rep | C | D | U | CC | 0.87 / 0.00 | 0.98 / 0.88 | 1.43 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 16 | PlusICE A50 | 50 | B | 190 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.94 / 0.46 | 0.99 / 0.95 | 1.43 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 17 | Paraffin/HDPE PCM6 | 50.3 | B | 187 Imp | 0.206/0.143 Imp | 953 Imp | U | U | U | U | 0.87 / 0.00 | 0.98 / 0.88 | 1.41 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 18 | Paraffin/HDPE PCM2 | 50.6 | B | 187 Imp | 0.208/0.146 Imp | 954 Imp | U | U | U | U | 0.88 / 0.00 | 0.98 / 0.88 | 1.41 | **UNVERIFIED** | latent heat is imputed, not reported: storage capacity unverified |
| 19 | savE OM48 | 51 | B | 165 Rep | 0.2/0.1 Rep | 960 Rep | C | D | U | CC | 0.76 / 0.00 | 0.98 / 0.71 | 1.24 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 20 | PlusICE A52 | 52 | B | 220 Rep | 0.18/0.18 Rep | 810 Rep | U | U | U | U | 0.93 / 0.61 | 0.99 / 0.96 | 1.66 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 21 | n-Tetracosane (C24) | 52 | B | 255 Rep | 0.306/0.173 Imp | 799 Rep | U | D | U | U | 0.94 / 0.81 | 0.99 / 0.97 | 1.92 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 22 | Paraffin/Expanded graphite (92% paraffin) | 52.2 | B | 170 Rep | 0.185/0.129 Imp | 871 Imp | U | U | U | U | 0.86 / 0.00 | 0.98 / 0.84 | 1.28 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 23 | Palmitic-Stearic eutectic (64.2/35.8) | 52.3 | B | 181.7 Rep | missing | missing | U | U | CC | U | 0.50 / 0.00 | 0.00 / 0.00 | 1.37 | **UNVERIFIED** | solid density missing: charge/discharge screen cannot be evaluated; thermal conductivity missing |
| 24 | Myristic acid (C14) | 53 | B | 199 Rep | 0.216/0.25 Imp | 990 Rep | U | U | CC | U | 0.94 / 0.59 | 0.98 / 0.95 | 1.50 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 25 | PureTemp 53 | 53 | B | 225 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.88 / 0.12 | 0.98 / 0.93 | 1.70 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 26 | RT54HC | 53.5 | B | 200 Rep | 0.199/0.172 Imp | 850 Rep | U | D | U | U | 0.91 / 0.27 | 0.98 / 0.93 | 1.51 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 27 | RT55 | 54 | B | 170 Rep | 0.197/0.147 Imp | 880 Rep | U | D | U | CC | 0.88 / 0.00 | 0.98 / 0.85 | 1.28 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 28 | n-Pentacosane (C25) | 54 | B | 238 Rep | 0.305/0.173 Imp | 801 Rep | U | U | U | U | 0.92 / 0.64 | 0.99 / 0.96 | 1.80 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 29 | Myristic acid/NBR-1.0 | 54.1 | B | 128 Rep | 0.171/0.138 Imp | 860 Imp | U | D | CC | U | 0.91 / 0.00 | 0.96 / 0.75 | 0.97 | **REJECTED** | duty R_wm 0.16 < 0.5 even at favourable corner |
| 30 | Myristic acid/NBR-0.5 | 54.6 | B | 142 Rep | 0.176/0.138 Imp | 865 Imp | U | D | CC | U | 0.89 / 0.00 | 0.97 / 0.78 | 1.07 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 31 | savE OM55 | 55 | B | 188 Rep | 0.16/0.1 Rep | 935 Rep | C | D | U | CC | 0.74 / 0.00 | 0.97 / 0.75 | 1.42 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 32 | Palmitic-stearic acid/Expanded graphite | 55.2 | B | 176 Rep | 0.181/0.139 Imp | 856 Imp | U | D | CC | U | 0.86 / 0.00 | 0.97 / 0.85 | 1.33 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 33 | n-Hexacosane (C26) | 56.5 | A | 256 Rep | 0.299/0.177 Imp | 770 Rep | U | U | U | U | 0.92 / 0.73 | 0.98 / 0.95 | 1.93 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 34 | RT57HC | 56.5 | A | 240 Rep | 0.304/0.174 Imp | 900 Rep | U | D | U | CC | 0.89 / 0.42 | 0.98 / 0.94 | 1.81 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 35 | PlusICE A58 | 58 | A | 215 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.91 / 0.55 | 0.98 / 0.93 | 1.62 | **CONDITIONAL** | design OK at nominal; evidence gaps: cyc,SC,corr,safety |
| 36 | PureTemp 58 | 58 | A | 225 Rep | 0.25/0.15 Rep | 890 Rep | U | U | U | U | 0.85 / 0.05 | 0.97 / 0.90 | 1.70 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 37 | RT60 | 58 | A | 160 Rep | 0.192/0.147 Imp | 880 Rep | U | D | U | CC | 0.86 / 0.00 | 0.96 / 0.81 | 1.21 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 38 | n-Heptacosane (C27) | 59 | A | 236 Rep | 0.303/0.18 Imp | 773 Rep | U | U | U | U | 0.90 / 0.59 | 0.98 / 0.94 | 1.78 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 39 | CrodaTherm 60 | 59.8 | A | 217 Rep | 0.29/0.17 Rep | 922 Rep | U | U | U | U | 0.86 / 0.07 | 0.97 / 0.90 | 1.64 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 40 | Palmitic acid/Expanded graphite (80/20) | 60.9 | A | 148 Rep | 0.191/0.138 Imp | 860 Imp | U | D | CC | U | 0.84 / 0.00 | 0.95 / 0.75 | 1.12 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 41 | PureTemp 60 | 61 | A | 220 Rep | 0.25/0.15 Rep | 960 Rep | U | U | U | U | 0.81 / 0.01 | 0.97 / 0.87 | 1.66 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 42 | RT65 | 61.5 | A | 150 Rep | 0.183/0.147 Imp | 880 Rep | U | D | U | CC | 0.84 / 0.00 | 0.95 / 0.76 | 1.13 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 43 | n-Octacosane (C28) | 61.6 | A | 253 Rep | 0.307/0.226 Imp | 910 Rep | U | U | U | U | 0.90 / 0.64 | 0.97 / 0.93 | 1.91 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 44 | PlusICE A62 | 62 | A | 205 Rep | 0.22/0.22 Rep | 910 Rep | U | U | U | U | 0.89 / 0.37 | 0.96 / 0.90 | 1.55 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 45 | RT62HC | 62.5 | A | 230 Rep | 0.304/0.173 Imp | 850 Rep | U | D | U | U | 0.86 / 0.26 | 0.97 / 0.91 | 1.73 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 46 | Palmitic acid (C16) | 62.6 | A | 198 Rep | 0.214/0.162 Imp | 989 Rep | U | U | CC | U | 0.81 / 0.01 | 0.96 / 0.83 | 1.49 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 47 | PureTemp 63 | 63 | A | 206 Rep | 0.25/0.15 Rep | 920 Rep | U | U | U | U | 0.81 / 0.01 | 0.96 / 0.84 | 1.55 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 48 | RT64HC | 64 | A | 250 Rep | 0.303/0.263 Imp | 880 Rep | U | D | U | CC | 0.89 / 0.67 | 0.97 / 0.92 | 1.89 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,corr,safety |
| 49 | Paraffin wax (generic) | 64 | A | 173.6 Rep | missing | 916 Rep | U | U | U | U | 0.85 / 0.01 | 0.95 / 0.83 | 1.31 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 50 | n-Nonacosane (C29) | 64 | A | 240 Rep | 0.303/0.257 Imp | 790 Rep | U | U | U | U | 0.89 / 0.67 | 0.97 / 0.92 | 1.81 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 51 | n-Triacontane (C30) | 65.4 | A | 251 Rep | 0.308/0.23 Imp | 910 Imp | U | U | U | U | 0.87 / 0.56 | 0.96 / 0.91 | 1.89 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 52 | savE OM65 | 67 | A | 188 Rep | 0.19/0.13 Rep | 924 Rep | C | D | U | CC | 0.74 / 0.00 | 0.94 / 0.75 | 1.42 | **CONDITIONAL** | design: favourable corner only; evidence gaps: corr,safety |
| 53 | Stearic acid (C18) | 67.9 | A | 259 Rep | 0.25/0.172 Imp | 965 Rep | U | U | CC | U | 0.80 / 0.10 | 0.95 / 0.88 | 1.95 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 54 | PureTemp 68 | 68 | A | 213 Rep | 0.3/0.251 Imp | 811 Imp | U | U | U | U | 0.87 / 0.51 | 0.95 / 0.88 | 1.61 | **CONDITIONAL** | design OK at nominal; needs imputed k; evidence gaps: cyc,SC,corr,safety |
| 55 | n-Hentriacontane (C31) | 68 | A | 242 Rep | 0.25/0.177 Imp | 930 Rep | U | U | U | U | 0.81 / 0.09 | 0.95 / 0.87 | 1.83 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 56 | RT69HC | 69 | A | 230 Rep | 0.304/0.259 Imp | 940 Rep | U | D | U | U | 0.85 / 0.47 | 0.95 / 0.88 | 1.73 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
| 57 | n-Dotriacontane (C32) | 69.5 | A | 170 Rep | 0.191/0.139 Imp | 848 Imp | U | U | U | U | 0.77 / 0.00 | 0.93 / 0.73 | 1.28 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,SC,corr,safety |
| 58 | RT70HC | 70 | A | 260 Rep | 0.303/0.177 Imp | 880 Rep | U | D | U | CC | 0.81 / 0.25 | 0.95 / 0.87 | 1.96 | **CONDITIONAL** | design: favourable corner only; evidence gaps: cyc,corr,safety |
