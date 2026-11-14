# Assam Objective 1 — Phase B (PCM Feasibility Filter) Root-Cause Audit

**Status: read-only audit. No correction implemented. Phase B not executed. Awaiting your review and explicit approval.**

Written for: the project owner, who will decide which corrections to approve.

> **ADDENDUM (2026-11-13, second round) — see [ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY.md](ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY.md).** The body below is the first audit and is left unchanged. The second round verified items the first audit flagged as unverified and **corrects or supersedes** the following statements:
>
> 1. **L_req "definitional ambiguity" (first audit §4.1(b)) is withdrawn.** The Phase 4 `Q_required` is 0.5 × the daily 100 L demand (night plus morning draws), the same form as Rajasthan's `SHARE_PCM·Q_night/m_PCM`. What remains true: it uses the regional-mean mains temperature (19.96 °C for C0) instead of the medoid's (17.2 °C), a 9 % difference, and a fixed L threshold cannot represent a mass-dependent storage requirement.
> 2. **Medoids are now verified** (ASP_0003/0036/0080 are the Objective 1 distance-sum and centroid-nearest points; the hourly weather equals Objective 1's raw NASA POWER records, max difference 0). The first audit's caveat is closed.
> 3. **Rajasthan's 4/7/5 versus 9/14/16 is reconciled:** the archive is the κ-calibrated companion run (κ = 0.5/0.6/0.5); fixed κ = 0.7 recomputed from the archived columns gives 4/7/5. (First audit §9 said I had not reconciled them.)
> 4. **History added:** the frozen 2026-10-04 Assam Phase 6 output, on SHA-256-identical inputs, returned 0 → 1 → 5 survivors (round-2 window 47–68 °C); `10_physics_validation.py` still hard-requires exactly 5. The first audit's regression evidence (§5.6) cited only an older copy.
> 5. **Safety evidence is stronger than the first audit stated:** the source PCM dataset contains manufacturer flash points (200 °C for the five savE rows rejected by the current script) and maximum operating temperatures that `06_build_pcm_database_final.py` drops. A reported flammability "Yes" with a 200 °C flash point is not a confirmed unacceptable hazard in a 75 °C-capped system.
> 6. **Temperature option A of the first audit (window 51–64 °C) is superseded:** it is thermodynamically inconsistent with the 6 K approach for Tm < 56 °C. The corrected specification separates a direct-delivery pathway (Tm ≥ 50 + approach) from a preheat pathway (auxiliary boost required).
> 7. **The first audit's charging and discharge numbers (§4.2, §4.3) are first-order screens superseded** by the sequential rate-limited model with carried-over state (methodology §6, §9). Its classification (§7, options A/B) is superseded by the four-status rules of methodology §4.5 and §8.

## 0. Scope, guarantees and how this was done

| Item | Statement |
|---|---|
| Production code | **Not modified, not executed.** The only file created is this report. |
| Existing Assam outputs | Preserved (read only). Nothing overwritten or deleted. |
| Rajasthan / Tamil Nadu / Uttarakhand | Read only (Rajasthan `07_feasibility_filter.py` and its archived outputs). Not touched. |
| How "49" was verified | I re-implemented the **current Assam survivor Boolean in memory** (scratch script outside the project) and applied it to `pcm_database_final.csv`: it reproduces exactly the 49 survivors and the same 9 rejections as `pcm_feasibility_by_cluster.csv` (row-by-row match = True). |
| Rajasthan comparison run | The **Rajasthan rule set was replayed in memory on Assam inputs** (no files written) to show what Rajasthan's own method gives on Assam data. |
| Weather/medoids | Objective 2 NASA POWER weather package for the three Assam regime points ASP_0003 (C0), ASP_0036 (C1), ASP_0080 (C2), 2016–2025, 3 652 days each (read through the existing validated loader). The package names these as the regime representative points; I did not independently re-derive that they are the Objective 1 cluster medoids. **No Rajasthan weather or numbers used.** |
| Physics checks in §4 | First-order, assumption-labelled **necessary-condition screens**, not a thermal simulation and not a validation. Collector parameters F_R(τα)=0.75, F_R·U_L=4.5 W/m²K are the project's **provisional, unmeasured** values. |

Source files examined (all read only):
- Assam: `PCM-Selection-ML-model/era5-assam/07_feasibility_filter_final.py` (current, 25 950 B, modified 2026-11-13 07:26), `08_mcdm_ranking_final.py`, `04c_charging_ceiling_assam.py`, `06_build_pcm_database_final.py`, `data/processed/{pcm,feasibility,design,clustering}/…`
- Older Assam copy: `PCM-Selection-ML-model-TN-latest/era5-assam/07_feasibility_filter_final.py` (14 443 B, 2026-11-12) and its `final_outputs/tables/table06_feasibility_survivors.csv`
- Rajasthan: `PCM-Selection-ML-model/era5-rajasthan/07_feasibility_filter.py`, archived outputs in `Objective---2/data/objective1/rajasthan/`

---

## 1. Root cause (one-paragraph answer)

**49 is not a scientifically derived number. It is the count of candidates that do not trip four narrow vetoes, and those vetoes are identical for every cluster.** The survivor rule (`07_feasibility_filter_final.py` lines 430–440) is

```
survives_all = (status_tm != "FAIL") and (status_corrosion != "FAIL") and (status_safety != "FAIL") and (status_data_complete == "PASS")
```

where `status_tm` is only the **absolute band 42 ≤ Tm ≤ 70 °C** (lines 205–213), corrosion FAIL needs an **inorganic** PCM in a high-HSI cluster (the database has **0** inorganic rows, so it can never fire), safety FAIL is "manufacturer-reported flammability = Yes" (5 savE® rows), and `status_data_complete` needs a thermal-conductivity value (3 literature rows lack one). That removes 1 + 5 + 3 = 9 of 58. **Every criterion the script computes that is actually physical or cluster-specific — the Tm window [51, 64] °C (computed at lines 120–127, 173–181, never used), the latent-heat requirement (always PASS for any L > 0, lines 221–225), the cluster charging ceiling (computed at lines 294–302, never used), cycling and supercooling (computed, never used) — is excluded from `survives_all`.** Consequently the result is global in effect, the same 49 appear in C0, C1 and C2 (verified: no output column differs across clusters), and the 49 are labelled "DEPLOYABLE_SURVIVOR" by Phase 7 although **0** of them is confirmed feasible and all 49 are `CONDITIONALLY_FEASIBLE`.

Root cause category (Task 5): **a combination, dominated by (i) criteria computed but not wired into the survivor rule, (ii) global rather than cluster-specific screening in effect, (iii) permissive/incorrect missing-property and severity handling, and (iv) no physical-feasibility model behind the temperature criteria.** It is *not* a coding slip in one line; it is a structural regression from the Rajasthan logic (and from the earlier Assam version) described in §5.

---

## 2. Task 1 — Tracing the number 49

### 2.1 Where 49 comes from
- Pool: `pcm_database_final.csv` → **58** unique candidates (31 manufacturer, 27 literature; 55 of 58 have at least one imputed property).
- Loop: `for _, prof in profiles.iterrows()` (line 142) then `for relax_round in range(MAX_RELAX_ROUNDS + 1)` (line 164) then `for _, pcm in pcm_db.iterrows()` (line 186). Survivors per round: `n_survive = sum(r["survives_all"])` (line 474). Stop condition `n_survive >= MIN_SURVIVORS_TARGET (5)` (line 507).
- Because `survives_all` does not depend on `window_lo/window_hi`, round 0 already returns 49 ≥ 5, the loop breaks immediately, and the "relaxation" machinery is **dead code**: every cluster reports `Relaxation Round: 0`, `Final Tm Window: [51.0, 64.0]` (printed, but not applied).
- Outputs: `data/processed/feasibility/pcm_feasibility_by_cluster.csv` (174 rows = 3 × 58), `pcm_feasibility_summary.csv`, `data/processed/pcm/feasibility_survivors_assam.csv`, `data/preprocessed/feasibility_filter_report.txt`. Phase 7 (`08_mcdm_ranking_final.py`) takes `survives_all == True` as its eligibility list (reads at line ~305), so 49 × 3 = 147 rows go to MCDM.

### 2.2 The nine rejected candidates (identical in all three clusters)

| # | Candidate | Tm (°C) | Rejected by | Evidence / note |
|---|---|---|---|---|
| 1 | RT42 | 40.5 | `status_tm` FAIL | below absolute band 42 °C |
| 3 | Myristic–Palmitic eutectic (58/42) | 42.6 | `status_data_complete` FAIL | TC missing, ρ·H missing (literature row) — **not a physical failure** |
| 23 | Palmitic–Stearic eutectic (64.2/35.8) | 52.3 | `status_data_complete` FAIL | same |
| 49 | Paraffin wax (generic) | 64.0 | `status_data_complete` FAIL | TC missing (ρ·H derived) |
| 4 | savE® OM42 | 44.0 | `status_safety` FAIL | flammability "Yes", status Reported |
| 15 | savE® OM50 | 50.0 | `status_safety` FAIL | same |
| 19 | savE® OM48 | 51.0 | `status_safety` FAIL | same |
| 31 | savE® OM55 | 55.0 | `status_safety` FAIL | same |
| 52 | savE® OM65 | 67.0 | `status_safety` FAIL | same |

(58 − 9 = 49.) Phase 6's own report line "Infeasible: 6" counts the six with an explicit FAIL status (RT42 + five savE®); the three data-incomplete rows are "Conditional" there yet are still removed from `survives_all` — an internal inconsistency.

### 2.3 Exact Boolean condition of every criterion (current code)

| Criterion | Code lines | Exact condition | In `survives_all`? |
|---|---|---|---|
| Tm | 205–213 | PASS iff `42.0 <= Tm <= 70.0` (absolute band only) | **Yes** (as `!= FAIL`) |
| Tm window [Tm_target−5, +8] = [51, 64] with ±2 K relaxation | 120–127, 173–181 | computed into `window_lo/window_hi`; **never compared with any Tm** | **No (dead)** |
| Latent heat | 221–225 | PASS iff `L` notna and `L > 0`; else UNKNOWN. **`L_req` (125.7/131.4/131.1 kJ/kg) is read (145–152) but never compared.** | No |
| Data completeness | 227–236 | PASS iff TC notna and `TC_status != "Missing"` and ρ·H notna | **Yes** |
| Cycling | 241–257 | PASS iff status "Reported" and ≥ 300; FAIL iff Reported and < 300; else UNKNOWN | No (informs class label only) |
| Supercooling | 262–289 | PASS iff value known and (status Reported or both Tm statuses Reported) and ≤ 8 K; FAIL > 8 K; else UNKNOWN | No |
| Charging feasibility (c6) | 294–302 | `"pass"` iff `Tm <= Tm_target_capped_C` (49.4–49.7 °C); never an input of `survives_all` | **No** (39 of 49 survivors have c6 = fail) |
| Corrosion | 307–365 | FAIL iff `is_inorganic and HSI_sunrise > p75` (never true: 0 inorganic); 6c PASS by name pattern (26 rows); 6b CONDITIONAL; else UNKNOWN | Yes as `!= FAIL` (inert) |
| Safety | 370–421 | FAIL iff status Reported and flammability ∈ {yes, highly/extremely flammable, toxic}; PASS iff Reported and {no, non-flammable, low}; else UNKNOWN | Yes as `!= FAIL` |
| Overall class | 446–467 | any FAIL → INFEASIBLE; any UNKNOWN/CONDITIONAL → CONDITIONALLY_FEASIBLE; else CONFIRMED | label only; **not** linked to `survives_all` |

### 2.4 Global or cluster-specific, and why 49 / 49 / 49
The loop runs per cluster, but the per-cluster inputs only enter through: `L_req` (unused), `HSI_sunrise` (feeds only the inert inorganic veto), `Tm_target_capped_C` (feeds only the unused c6). Every active clause is a function of the **candidate** alone. Result: screening is cluster-specific in form and **global in effect**. I checked by pivoting the audit CSV: all ten status columns, `survives_all` and `overall_feasibility` are identical across clusters for all 58 candidates. (Even the unused c6 happens to be identical only because the three ceilings 49.39 / 49.44 / 49.66 °C fall in the gap between the candidate Tm values 49.0 and 50.0 °C.)

### 2.5 The 49 survivors and their criterion values
All 49 are listed in the candidate table (Appendix A; rows whose "Current Phase B" column says *survives*). Key facts about them:

| Property of the 49 survivors | Count |
|---|---|
| Overall class `CONDITIONALLY_FEASIBLE` | **49** (confirmed feasible: **0**) |
| Cycling evidence Reported (≥300) | 2 (47 UNKNOWN: imputed or unreported) |
| Supercooling evidence Reported | 20 (29 UNKNOWN) |
| Corrosion PASS | 24 (all by rule 6c, name pattern, see §5.3); 17 UNKNOWN; 8 CONDITIONAL |
| Safety PASS (Reported non-flammable) | 2 (savE® OM46, OM49); 47 UNKNOWN |
| Charging c6 = **fail** (Tm above 49.4–49.7 °C ceiling) | **39** |
| Tm **outside** the declared [51, 64] °C window | **21** |
| Tm ≤ 50 °C (cannot by latent release alone raise water to the 50 °C delivery target) | 12 |
| Latent heat below 0.7 × L_req | 0 (min L in the whole pool is 128 kJ/kg; the requirement is non-discriminating, see §4) |

---

## 3. Task 2 — Rajasthan vs Assam (Rajasthan read-only)

| Aspect | Rajasthan `07_feasibility_filter.py` | Assam (current) | Assessment / Assam-specific justification |
|---|---|---|---|
| Global vs cluster | Per cluster: `tm_target`, `tm_capped`, `l_required`, `HSI` read from each cluster's profile and used in `evaluate()` (lines 502–531); survivors differ by cluster in its archived output (9 / 14 / 16) | Loop per cluster, but active clauses are candidate-only → identical | **Defect.** Cluster inputs must change the result. |
| Melting-temperature compatibility | c1 window [Tm_target−5, +8] **applied** (line 288), relaxed ±2 K up to 4 rounds until ≥5 survivors; c2 absolute band [42, 70] applied (291) | c2 applied; c1 computed, **never applied**; relaxation loop never relaxes anything | **Defect.** Restore, but see the Assam-specific conflict with the charging ceiling (§4.1, §6). |
| Charging feasibility | c6: `Tm <= Tm_target_capped_C`, in the blocking set (lines 354–357) | computed, not in `survives_all` | **Defect** (Rajasthan: blocking). Assam adaptation: replace the area-agnostic heuristic by an hourly-weather collector screen (§4.2, §6). |
| Latent heat / storage capacity | c3 `L >= κ·L_req`, κ = 0.7, blocking; unreported L → flagged, retained (302–308) | PASS iff L>0; `L_req` unused | **Defect.** Note: on Assam's `L_req` (126–131 kJ/kg) the κ = 0.7 floor excludes nothing (min L = 128), so c3 is formally restored but non-binding. |
| Thermal conductivity | Not a gate | Gate via "data complete" (TC/ρH) | Assam's gate is a **ranking-computability** rule disguised as feasibility. Missing k should make a candidate *Unverified*, not *rejected*; k enters the discharge screen (§4.3). |
| Cycling | c4: ≥ 300; **flag, not exclude** if unreported; MICE-imputed counts as data | UNKNOWN unless *Reported*; not gating | Assam is stricter on evidence quality (imputed ≠ evidence) — **keep**, justified: imputed cycle counts are model estimates. Reported < 300 should be a hard FAIL. |
| Supercooling | c5: ≤ 8 K; flag if unknown | same logic, evidence-aware | Keep evidence-awareness. 26 "Reported" supercooling values are **derived** from two reported temperatures (`supercooling_is_derived = True`), not measured supercooling — label as derived. |
| Corrosion | c7: veto only bare **salt hydrates** in a high-HSI cluster; "inert on current data" (documented) | Same veto (inert, 0 inorganic) **plus** rule 6c name-pattern PASS for 26 rows | 6c contradicts Assam's own header rule "No category-based PASS" (lines 11–13). Rajasthan never awards a PASS from category. |
| Safety | c8: **flag** unverified; explicitly refuses to exclude on an unqualified Yes/No flammability because it cannot distinguish "highly flammable" from ordinary combustibility (docstring point C) | **FAIL** on Reported flammability = Yes (5 savE® rows rejected) | Assam is **stricter than Rajasthan on a field that cannot support a severity decision**. Evidence should be flash point / hazard class; until present → Conditional, not Rejected. |
| Missing properties | NaN L/cycles/supercooling → `flag_*`, retained; missing safety → flagged | TC missing → rejected; others UNKNOWN (retained); safety missing → UNKNOWN | Rule must be uniform: missing never PASSes and never *silently* rejects except where it blocks computation of a mandatory physical test (then *Unverified*). |
| Final feasible population | `survives_all` = c1∧c2∧c3∧c6 (+c7, c8 non-excluded); **plus** documented κ-calibration companion run | c2 ∧ ¬c7 ∧ ¬c8 ∧ data-complete | Assam omits four of Rajasthan's six blocking criteria. |
| Provenance | Fingerprints the upstream profile (`provenance_lib`) | none; MCDM outputs predate Phase 6 outputs (see §9) | Add. |

**Replay of Rajasthan's rule set on Assam data (in memory, κ = 0.7 and κ = 1.0 give the same result):** round 0 window [51, 64] → **0** survivors in all three clusters; round 1 [49, 66] → **1** (savE® OM49); round 2 [47, 68] → **5** (savE® OM46, n-Tricosane C23, RT50, Paraffin/Expanded-graphite 85.56 %, savE® OM49) and the loop stops. Identical for C0, C1, C2 because the three ceilings (49.39–49.66 °C) separate the same candidates. So applying Rajasthan's own logic faithfully gives **0 → 1 → 5**, not 49, but the five survivors all have Tm ≤ 49 °C — below the 50 °C delivery target (see §4.1).

---

## 4. Task 3 — Cluster-specific physical feasibility (Assam medoid points)

### 4.1 Demand and the temperature design-basis conflict
Data: medoid regime points from the Objective 2 weather package (hourly 2016–2025).

| | C0 (ASP_0003) | C1 (ASP_0036) | C2 (ASP_0080) |
|---|---|---|---|
| Mains temperature, mean / p05 / p95 (°C) | 17.2 / 8.7 / 22.9 | 18.7 / 10.7 / 25.0 | 18.3 / 9.9 / 24.0 |
| Coldest-month mean mains (°C) | 9.2 | 11.2 | 10.4 |
| Evening 50 L draw to 50 °C: mean / p95 (kWh) | 1.91 / 2.40 | 1.82 / 2.28 | 1.84 / 2.33 |
| Phase 4 spec `Q_required` (kWh/day) and `L_req` (kJ/kg for 50 kg) | 1.746 / 125.7 | 1.825 / 131.4 | 1.820 / 131.1 |
| Daily GHI mean / worst month (kWh/m²) | 4.31 / 3.54 (Jan) | 4.10 / 3.62 (Feb) | 4.17 / 3.36 (Jan) |

Observations: (a) the spec's `Q_required` equals SHARE_PCM·100 L·ΔT with the **regional-mean** mains (19.96 °C for C0), whereas the medoid-point mains is 17.2 °C, so the medoid evening demand is ~9 % higher than the spec's for C0; (b) numerically the spec's required energy equals the **whole evening draw** (50 L) — PCM is implicitly asked to supply 100 % of it at 100 % efficiency by latent heat alone, which is a definitional ambiguity to resolve; (c) all 58 candidates have L ≥ 128 kJ/kg, so the latent-capacity requirement never discriminates on this definition.

**The design-basis conflict (needs your decision).** The Phase 4 design constants give Tm_target = 50 + 6 = **56 °C** (window [51, 64]), while the Phase B charging ceiling (`Tm_target_capped_C`, 04c, worst-month clearness) is only **49.66 / 49.39 / 49.44 °C**. The two ranges are disjoint, which Rajasthan's docstring (lines 391–400) already documents; Rajasthan resolved it by relaxing the window downward, which for Assam ends at 47–49 °C, **below** the 50 °C delivery target. Candidates with Tm ≤ 50 °C cannot, by latent release alone, deliver water at 50 °C; they can only pre-heat. Whether pre-heating is the intended role of the PCM is a project decision, not something the code can settle. (Cross-reference: M3 Stage D found that the OM46/OM49 beds melt fully every sunny day and discharge very little — consistent with, but not proof of, this tension.)

### 4.2 Charging: can the collector melt the PCM within the available hours?
Method: hourly Hottel–Whillier–Bliss useful gain with the provisional F_R(τα) = 0.75, F_R·U_L = 4.5 W/m²K, GHI used directly (as the project does), inlet temperature = Tm + 6 K approach, irradiance cut-off 10 W/m². Energy to melt 50 kg of a reference PCM (L = 200 kJ/kg, c_s = 2 kJ/kgK) starting from the day's mains temperature; the same collector area is assumed to serve this alone (an optimistic simplification: no competing tank load, no losses).

Collector area is **not defined in Objective 1** (Phase 3 first-test 2 m², M3 baseline 5 m²), so both are shown. Columns: useful-gain hours per day, mean / p10; useful energy per m² per day, mean / p10 (kWh/m²); fraction of days a full melt is achievable at 2 m² (all days / worst calendar month); same at 5 m².

| Cluster | Tm (°C) | Hours (mean / p10) | E_u kWh/m² (mean / p10) | 2 m²: days / worst-month | 5 m²: days / worst-month |
|---|---|---|---|---|---|
| C0 | 48 | 8.5 / 7 | 2.03 / 0.84 | 0.57 / 0.05 | 0.92 / 0.87 |
| C0 | 52 | 8.2 / 6 | 1.88 / 0.72 | 0.48 / 0.00 | 0.89 / 0.83 |
| C0 | 56 | 7.9 / 6 | 1.73 / 0.60 | 0.41 / 0.00 | 0.86 / 0.79 |
| C0 | 60 | 7.5 / 6 | 1.60 / 0.49 | 0.32 / 0.00 | 0.82 / 0.70 |
| C0 | 64 | 7.1 / 5 | 1.46 / 0.40 | 0.26 / 0.00 | 0.78 / 0.59 |
| C1 | 48 | 8.3 / 6 | 1.93 / 0.46 | 0.53 / 0.17 | 0.85 / 0.75 |
| C1 | 52 | 7.9 / 5 | 1.78 / 0.36 | 0.44 / 0.00 | 0.82 / 0.71 |
| C1 | 56 | 7.5 / 5 | 1.65 / 0.27 | 0.38 / 0.00 | 0.78 / 0.65 |
| C1 | 60 | 7.1 / 4 | 1.52 / 0.19 | 0.31 / 0.00 | 0.75 / 0.61 |
| C1 | 64 | 6.8 / 3 | 1.39 / 0.12 | 0.25 / 0.00 | 0.71 / 0.57 |
| C2 | 48 | 8.4 / 6 | 1.94 / 0.69 | 0.53 / 0.03 | 0.89 / 0.83 |
| C2 | 52 | 8.1 / 6 | 1.79 / 0.58 | 0.45 / 0.00 | 0.86 / 0.78 |
| C2 | 56 | 7.7 / 6 | 1.65 / 0.46 | 0.37 / 0.00 | 0.83 / 0.70 |
| C2 | 60 | 7.3 / 5 | 1.51 / 0.36 | 0.31 / 0.00 | 0.79 / 0.59 |
| C2 | 64 | 7.0 / 5 | 1.38 / 0.28 | 0.24 / 0.00 | 0.74 / 0.47 |

Required collector area for one full melt of 50 kg (C0, Tm 56 °C): **2.2 m² on a mean day, 6.5 m² on a p10 day**; C1: 2.3 m² mean, **14.3 m²** on a p10 day (dull monsoon-type days); C2: 2.3 / 8.3 m². Conclusions: (i) the worst-month ceiling in 04c is an **area-agnostic heuristic** and, for a 5 m² array, is conservative (at Tm = 56 °C a full melt is still achievable on 86 % of days in C0, 79 % in the worst month); for 2 m² it is not (41 % of days, ~0 % in the worst month). (ii) Clusters differ materially, especially in the dull-day tail (C1 worst), so cluster-specific results *should* differ. (iii) Any charging criterion must state the reference collector-area-to-PCM-mass ratio; this is currently undefined.

### 4.3 Discharge: can enough heat be delivered in the draw period?
Method (bounding, not a simulation): quasi-steady Stefan solidification in a slab of half-thickness δ, driven by ΔT = Tm − 50 °C, over the 1 h evening draw (the weather package spreads the 19:00 draw over 18:30–19:30), conductivity from the database. Needs Tm > 50 °C and a known k, else the screen cannot be evaluated. δ = 10/20/30 mm are illustrative only.

| | C0 | C1 | C2 |
|---|---|---|---|
| Candidates with Tm > 50 °C (can reach 50 °C by latent release) | 42 of 58 | 42 | 42 |
| …delivering ≥ 100 % of the evening need within 1 h at δ = 10 mm | 25 | 26 | 26 |
| …at δ = 20 mm | 6 | 8 | 8 |
| …at δ = 30 mm | 0 | 0 | 0 |
| k missing (screen not evaluable) | 3 | 3 | 3 |

Discharge rate, not latent capacity, is the binding limit; candidates with Tm within a few kelvin of 50 °C release only a small fraction of their latent heat in the draw hour. 39 of the 58 k values are imputed (MICE), so this screen carries that uncertainty.

### 4.4 Supercooling, cycling, corrosion, safety (evidence status, whole pool)
- Cycling: Reported 7, Imputed 48, Missing 3. Only 7 candidates have measured cycle counts.
- Supercooling: Reported 26 (all 26 derived from Tm_melt − Tm_freeze, not measured), Imputed 29, Missing 3.
- Corrosion: **no** candidate has material-specific compatibility test evidence (the database has no `corrosion_evidence` column; the script reads it with a NaN default, lines 310 and 356). The 26 "PASS" are name-pattern assertions (rule 6c).
- Safety: only 2 candidates (savE® OM46, OM49) have *Reported* non-flammable; 5 savE® rows are Reported "Yes"; 48 are Imputed; 3 missing. No flash point, GHS class or toxicity data exist anywhere in the database.

---

## 5. Task 5 — Root cause with code and candidate evidence

### 5.1 Criteria computed but not wired into the survivor rule (primary)
`survives_all` (lines 430–440) omits: the Tm window, the latent requirement, charging c6, cycling and supercooling. Evidence: 21 survivors have Tm outside the window the script itself prints; 39 survivors fail the script's own c6; the relaxation loop (line 164) cannot change `n_survive`.

### 5.2 Global screening in effect
All active clauses are candidate-only, so no cluster input can change the result (§2.4). This is exactly why 49 / 49 / 49 appears.

### 5.3 Missing-property and evidence handling (secondary)
- `status_data_complete` rejects 3 literature rows for missing k although k is not a feasibility criterion (it is a ranking input).
- Rule 6c (lines 329–339) grants corrosion PASS to 26 candidates by name pattern (`"RT"` prefix, `n-…cosane`, `"Paraffin wax"`): category-based, contradicting lines 11–13. RT-line products are proprietary formulations, and a generic paraffin wax record has no material-specific coupon test.
- Safety: unqualified "Yes" is treated as a hazard veto (5 rejections), unlike Rajasthan and without severity data.

### 5.4 No physical feasibility model behind the temperature criteria (secondary)
The window [Tm_target − 5, + 8] and the ceiling are heuristics; Tm_target = 56 °C and ceiling ≤ 49.7 °C are mutually exclusive, and nothing in Phase B reconciles them against hourly charging or draw-period discharge (§4).

### 5.5 Not the cause
- Not a bug in the replay: independent in-memory re-implementation reproduces 49 exactly.
- Not caused by C-specific data: with inorganics absent, HSI is irrelevant.
- Not a Rajasthan-copy error as such: Rajasthan's logic, applied faithfully, gives 0 → 1 → 5 on Assam data.

### 5.6 History (regression evidence)
An earlier Assam copy (`PCM-Selection-ML-model-TN-latest/era5-assam`, 2026-11-12, different design basis: Tm_target 40.6 °C, L_req 251 kJ/kg) has a `07_feasibility_filter_final.py` that applies a 42–52 °C Tm band and `L >= L_req` as FAIL statuses, and its `final_outputs/tables/table06_feasibility_survivors.csv` carries per-criterion pass columns (melting window, latent-heat floor 0.7 × L_req = 176 kJ/kg, cycling, supercooling, corrosion, safety) and lists **8** survivors (cluster 0 only in that file). The current Assam script drops the latent-heat comparison and the window from `survives_all`. I did not rerun the older version.

---

## 6. Task 6 — Recommended corrected method (for approval; no counts targeted)

1. **Make every criterion either gating or explicitly informational, in code.** A declared list `GATING = [...]`, `INFORMATIONAL = [...]`; a test fails if a computed criterion is in neither.
2. **Screen per cluster with cluster inputs that matter:** cluster `L_req`, cluster charging screen, cluster mains/demand; emit an explicit warning if two clusters return identical survivor sets and name the reason.
3. **Temperature criterion — decision D1 (yours):** choose the design basis: (A) delivery-capable PCM (Tm in [51, 64] °C, relaxation never below 50 °C), (B) Rajasthan-style sequential relaxation with ceiling (gives pre-heating candidates, Tm 47–49 °C, for Assam), or (C) a combined rule where the window is kept and the charging screen replaces the ceiling.
4. **Charging criterion — replace the area-agnostic ceiling** by the hourly collector screen of §4.2 at a declared reference area-to-PCM-mass ratio; report the charge-completion frequency and worst-month frequency per cluster; threshold chosen by you (decision D2), shown with sensitivity, not tuned to a count.
5. **Latent criterion:** keep κ·L_req (κ = 0.7 from Rajasthan) with the cluster `L_req`, but first fix the `L_req` definition (D3: whole evening draw vs SHARE_PCM share; regional-mean vs medoid mains). Expected: non-binding.
6. **Discharge criterion:** add the Tm > 50 °C + rate screen (§4.3) as informational until geometry is fixed (Objective 2 owns geometry), then gating.
7. **Evidence classes (§7 below):** Rejected only on a *mandatory failed* criterion with evidence; missing/imputed evidence → Conditional or Unverified; nothing "DEPLOYABLE" unless Feasible or explicitly approved Conditional.
8. **Safety (D4):** change flammability "Yes" from FAIL to Conditional unless flash point / GHS evidence is added; add those columns to the database before using safety as a veto.
9. **Corrosion:** remove rule 6c PASS; require `corrosion_evidence` (material-specific) for PASS; keep the HSI-veto logic for future salt hydrates.
10. **Missing k:** moves the candidate to Unverified (cannot evaluate discharge), not to Rejected.
11. **Provenance:** write script hash + input fingerprints into every output; Phase 7 refuses a Phase 6 output older than its script or inputs.

---

## 7. Task 4 — Candidate classification (illustrative application of the proposed rules)

Definitions used (explicit, evidence-based; no count targeted):
- **Rejected** — fails a mandatory criterion with data: Tm outside [42, 70]; Tm outside the declared window; (Option B) Tm above the cluster ceiling; `L < 0.7·L_req`; Reported cycles < 300; Reported supercooling > 8 K.
- **Unverified** — passes the above but a mandatory input is not Reported (Tm or L not Reported, or k Missing).
- **Feasible** — passes all mandatory criteria **and** cycling, supercooling, corrosion, safety, k are all Reported/PASS with material-specific evidence.
- **Conditional** — passes all mandatory criteria; ≥ 1 of cycling, supercooling, corrosion, safety, k lacks *Reported* evidence.

Temperature option A = window [51, 64] °C without relaxation; option B = Rajasthan sequential relaxation + ceiling (stops at round 2, window [47, 68] °C for all three clusters). The charging-frequency screen is shown as **information** (not gating) because decision D2 (threshold, reference area) is open.

Counts (Feasible / Conditional / Unverified / Rejected), out of 58:

| Cluster | Option A | Option B |
|---|---|---|
| C0 | 0 / 30 / 2 / 26 | 0 / 5 / 0 / 53 |
| C1 | 0 / 30 / 2 / 26 | 0 / 5 / 0 / 53 |
| C2 | 0 / 30 / 2 / 26 | 0 / 5 / 0 / 53 |

These are **outputs of the rules**, not targets. In both options **no candidate is Feasible**, because no candidate has material-specific corrosion evidence and almost none has Reported cycle and safety evidence; that is the honest state of the database. The class differences between clusters here are zero under these two options because the cluster-dependent inputs (ceilings, `L_req`) do not separate any pair of candidates; cluster dependence enters through the charging/discharge screens (§4), which therefore must become gating (D2) for the clusters to be distinguishable.

Option A's 32 admitted rows and Option B's 5 are listed in Appendix A (last two columns, shown for C0; C1 and C2 are identical).

---

## 8. Missing-data classification (whole pool of 58)

| Property | Reported | Imputed (MICE/derived) | Missing | Consequence in current code | Proposed treatment |
|---|---|---|---|---|---|
| Tm (melting) | 58 | 0 | 0 | gate | gate |
| Latent heat | 55 | 3 (Paraffin/HDPE PCM2, PCM3, PCM6) | 0 | not gated | gate; imputed → *Unverified* |
| Thermal conductivity | 16 | 39 | 3 | missing → rejected | missing → *Unverified*; imputed → flagged |
| Cycles tested | 7 | 48 | 3 | UNKNOWN | imputed ≠ evidence; Reported < 300 → Rejected |
| Supercooling | 26 (all derived) | 29 | 3 | UNKNOWN | derived → label; > 8 K Reported → Rejected |
| Flammability | 7 (5 Yes, 2 No) | 48 | 3 | Yes → FAIL | Yes → Conditional; flash point / GHS needed |
| Corrosion (material-specific) | 0 | – | 58 | 26 PASS by name, rest UNKNOWN/CONDITIONAL | no PASS without evidence |
| Toxicity / flash point / encapsulation | none in database | – | all | not considered | add columns |

---

## 9. Provenance and consistency findings

- On-disk timestamps (2026-11-13): MCDM outputs 03:21 → Phase 6 outputs 07:21 → current Phase 6 script 07:26 → recommendation cards 07:48/07:50. The Phase 7 eligibility summary reports **2 confirmed / 50 conditional / 6 infeasible**, the current Phase 6 report reports **0 / 52 / 6** — the downstream outputs do **not** correspond to the current Phase 6 output, and the Phase 6 output predates its own script by 5 minutes (I cannot prove the output matches the current script without running it; the in-memory replay of the survivor Boolean does match the CSV).
- `verify_03_feasibility_assam.py` still labels "Total Evaluated Candidates (25)" and "Latent Floor" — stale text from an earlier pipeline.
- Rajasthan's docstring quotes 4 / 7 / 5 survivors while the archived copy in `Objective---2/data/objective1/rajasthan/` shows 9 / 14 / 16. I used Rajasthan's **logic**, not its counts, and did not reconcile the two.
- The Phase 6 report states "Confirmed Feasible: 0" while the Phase 7 gate names all 49 "DEPLOYABLE_SURVIVOR".

---

## 10. Proposed tests and acceptance criteria

| # | Test | Acceptance criterion |
|---|---|---|
| T1 | Static wiring test: every criterion computed in the script is in `GATING` or `INFORMATIONAL` | no undeclared criterion; `survives_all` equals the AND of `GATING` |
| T2 | Window is applied: synthetic candidate with Tm 45 °C and Tm 70 °C under window [51, 64] | both not survivors; relaxation round changes survivor count monotonically |
| T3 | Cluster sensitivity: two synthetic clusters with different `L_req` / charging inputs | survivor sets differ where the inputs differ; if identical, a named warning is emitted |
| T4 | Parity test: the Assam-adapted function run with Rajasthan inputs and Rajasthan parameters | reproduces the archived Rajasthan survivor table row-for-row |
| T5 | Missing never passes: NaN k / L / cycles / flammability | never PASS; k-missing → *Unverified*; no silent rejection |
| T6 | No category-only PASS: corrosion PASS requires material-specific evidence | candidates with only a name pattern → *Conditional* |
| T7 | Safety: flammability "Yes" without severity evidence | *Conditional*, not *Rejected*; Reported toxic/highly flammable → *Rejected* |
| T8 | Partition: classes sum to the pool size per cluster; Feasible ⇒ every evidence field Reported | exact equality |
| T9 | Charging screen reproduces §4.2 grid values for the three medoid points | agreement within 1e-9 on fixed inputs |
| T10 | Provenance: output carries script hash and input fingerprints; Phase 7 rejects a stale Phase 6 file | stale input → hard error |
| T11 | Regression pin of the **current** behaviour for audit: 58 → 49, nine named rejections | documents the defect before correction (expected to change after approved correction) |
| T12 | Sensitivity report: κ ∈ {0.7, 1.0}, collector area ∈ {2, 5} m², δ ∈ {10, 20, 30} mm | included in output; no tuning to a survivor count |

Acceptance for the corrected Phase B as a whole: T1–T10 pass; the summary reports class counts per cluster with reasons; no candidate is called Feasible without Reported evidence for every field; the audit trail lists, for each rejected candidate, the exact failed criterion and value.

---

## 11. Decisions I need from you before any correction

| ID | Decision | Why it matters |
|---|---|---|
| D1 | Temperature design basis: delivery-capable (window 51–64 °C), pre-heating (Rajasthan-style 47–49 °C), or combined | Decides which physical role the PCM plays; the current design constants are mutually inconsistent |
| D2 | Charging screen: reference collector area-to-PCM-mass ratio and the acceptance frequency (all days / worst month) | The ceiling heuristic is area-agnostic |
| D3 | Definition of `L_req` (full evening draw vs share; medoid vs regional-mean mains) | Latent requirement is currently non-binding and ambiguous |
| D4 | Safety rule: Conditional vs Rejected for unqualified flammability; add flash point / GHS columns | 5 savE® rows are currently rejected on weak evidence; OM55 is a numerical benchmark only |
| D5 | Whether "Unverified/Conditional" candidates may proceed to Phase 7 and under what label | Currently all are labelled DEPLOYABLE |

Nothing has been implemented. I will wait for your review.

---

## Appendix A — All 58 candidates (cluster-independent evidence; C0 classes)

Columns: current Phase B outcome; evidence status (cycles / supercooling / corrosion / flammability-status / k); "Raj" = survives Rajasthan's rule replay (round 2, identical in C0–C2); charging = fraction of days a full melt from mains is achievable at **5 m²** (C0 / C1 / C2; hourly-weather screen of §4.2; informational); proposed class under option A and option B (C0; C1 and C2 identical).

| # | Candidate | Tm | L | Current Phase B | Cycles | Supercool. | Corrosion | Flammab. | k (W/mK) | Raj | Charge 5 m² C0/C1/C2 | Class A | Class B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT42 | 40.5 | 165 | REJECTED: Tm<42 | Imputed(1564) | Reported(0.5) | PASS | Yes/Imputed | 0.1785/Imputed | - | 0.97 / 0.92 / 0.95 | Rejected | Rejected |
| 2 | RT44HC | 42.5 | 250 | survives | Imputed(1216) | Reported(0) | PASS | No/Imputed | 0.24/Imputed | - | 0.93 / 0.87 / 0.91 | Rejected | Rejected |
| 3 | Myristic-Palmitic eutectic (58/42) | 42.6 | 169.7 | REJECTED: data-incomplete (k missing) | Missing | Missing | CONDITIONAL | Unknown/Missing | Missing | - | 0.96 / 0.91 / 0.94 | Rejected | Rejected |
| 4 | savE OM42 | 44 | 199 | REJECTED: safety FAIL (flammability=Yes, Reported) | Reported(2000) | Reported(1) | UNKNOWN | Yes/Reported | 0.145/Reported | - | 0.93 / 0.87 / 0.91 | Rejected | Rejected |
| 5 | n-Docosane (C22) | 44.5 | 249 | survives | Imputed(1440) | Imputed(0.2) | PASS | No/Imputed | 0.24/Imputed | - | 0.92 / 0.85 / 0.89 | Rejected | Rejected |
| 6 | RT47 | 44.5 | 160 | survives | Imputed(1665) | Reported(0) | PASS | No/Imputed | 0.178/Imputed | - | 0.95 / 0.90 / 0.93 | Rejected | Rejected |
| 7 | Lauric acid (C12) | 44.8 | 184 | survives | Imputed(2000) | Imputed(0) | CONDITIONAL | Yes/Imputed | 0.164/Imputed | - | 0.94 / 0.88 / 0.92 | Rejected | Rejected |
| 8 | RT45HC | 45 | 240 | survives | Imputed(1216) | Reported(0) | PASS | No/Imputed | 0.24/Imputed | - | 0.92 / 0.85 / 0.89 | Rejected | Rejected |
| 9 | savE OM46 | 47 | 177 | survives | Reported(2000) | Reported(2) | UNKNOWN | No/Reported | 0.15/Reported | Y | 0.93 / 0.86 / 0.90 | Rejected | Conditional |
| 10 | n-Tricosane (C23) | 47.5 | 232 | survives | Imputed(1216) | Imputed(2.6) | PASS | No/Imputed | 0.239/Imputed | Y | 0.90 / 0.83 / 0.88 | Rejected | Conditional |
| 11 | RT50 | 48 | 160 | survives | Imputed(1675) | Reported(0) | PASS | Yes/Imputed | 0.177/Imputed | Y | 0.93 / 0.87 / 0.91 | Rejected | Conditional |
| 12 | Paraffin/Expanded graphite (85.56% paraffin) | 48.8 | 161 | survives | Imputed(2000) | Imputed(2.2) | UNKNOWN | Yes/Imputed | 0.166/Imputed | Y | 0.93 / 0.87 / 0.91 | Rejected | Conditional |
| 13 | savE OM49 | 49 | 224 | survives | Reported(1200) | Reported(3.5) | UNKNOWN | No/Reported | 0.3/Reported | Y | 0.90 / 0.82 / 0.87 | Rejected | Conditional |
| 14 | Paraffin/HDPE PCM3 | 49.9 | 187 | survives | Imputed(2000) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.174/Imputed | - | 0.91 / 0.84 / 0.89 | Rejected | Rejected |
| 15 | savE OM50 | 50 | 189 | REJECTED: safety FAIL (flammability=Yes, Reported) | Reported(2000) | Reported(0) | UNKNOWN | Yes/Reported | 0.175/Reported | - | 0.89 / 0.82 / 0.86 | Rejected | Rejected |
| 16 | PlusICE A50 | 50 | 190 | survives | Imputed(1526) | Imputed(1.1) | UNKNOWN | Yes/Imputed | 0.18/Reported | - | 0.91 / 0.84 / 0.88 | Rejected | Rejected |
| 17 | Paraffin/HDPE PCM6 | 50.3 | 187 | survives | Imputed(2000) | Imputed(1) | UNKNOWN | Yes/Imputed | 0.1745/Imputed | - | 0.91 / 0.84 / 0.88 | Rejected | Rejected |
| 18 | Paraffin/HDPE PCM2 | 50.6 | 187 | survives | Imputed(2000) | Imputed(1.4) | UNKNOWN | Yes/Imputed | 0.177/Imputed | - | 0.91 / 0.84 / 0.88 | Rejected | Rejected |
| 19 | savE OM48 | 51 | 165 | REJECTED: safety FAIL (flammability=Yes, Reported) | Reported(2000) | Reported(3) | UNKNOWN | Yes/Reported | 0.15/Reported | - | 0.92 / 0.85 / 0.89 | Conditional | Rejected |
| 20 | PlusICE A52 | 52 | 220 | survives | Imputed(1447) | Imputed(2.8) | UNKNOWN | Yes/Imputed | 0.18/Reported | - | 0.88 / 0.80 / 0.85 | Conditional | Rejected |
| 21 | n-Tetracosane (C24) | 52 | 255 | survives | Imputed(1368) | Reported(2.7) | PASS | Yes/Imputed | 0.2395/Imputed | - | 0.86 / 0.78 / 0.83 | Conditional | Rejected |
| 22 | Paraffin/Expanded graphite (92% paraffin) | 52.2 | 170 | survives | Imputed(2000) | Imputed(3) | UNKNOWN | Yes/Imputed | 0.157/Imputed | - | 0.89 / 0.81 / 0.86 | Conditional | Rejected |
| 23 | Palmitic-Stearic eutectic (64.2/35.8) | 52.3 | 181.7 | REJECTED: data-incomplete (k missing) | Missing | Missing | CONDITIONAL | Unknown/Missing | Missing | - | 0.90 / 0.83 / 0.87 | Unverified | Rejected |
| 24 | Myristic acid (C14) | 53 | 199 | survives | Imputed(1686) | Imputed(1.8) | CONDITIONAL | Yes/Imputed | 0.233/Imputed | - | 0.89 / 0.82 / 0.86 | Conditional | Rejected |
| 25 | PureTemp 53 | 53 | 225 | survives | Imputed(1686) | Imputed(1.9) | UNKNOWN | Yes/Imputed | 0.2/Reported | - | 0.86 / 0.78 / 0.83 | Conditional | Rejected |
| 26 | RT54HC | 53.5 | 200 | survives | Imputed(1474) | Reported(0) | PASS | Yes/Imputed | 0.1855/Imputed | - | 0.89 / 0.81 / 0.85 | Conditional | Rejected |
| 27 | RT55 | 54 | 170 | survives | Imputed(2000) | Reported(0) | PASS | Yes/Imputed | 0.172/Imputed | - | 0.89 / 0.82 / 0.87 | Conditional | Rejected |
| 28 | n-Pentacosane (C25) | 54 | 238 | survives | Imputed(1404) | Imputed(3.4) | PASS | Yes/Imputed | 0.239/Imputed | - | 0.85 / 0.77 / 0.82 | Conditional | Rejected |
| 29 | Myristic acid/NBR-1.0 | 54.1 | 128 | survives | Imputed(2000) | Reported(4.9) | CONDITIONAL | Yes/Imputed | 0.1545/Imputed | - | 0.90 / 0.82 / 0.87 | Conditional | Rejected |
| 30 | Myristic acid/NBR-0.5 | 54.6 | 142 | survives | Imputed(2000) | Reported(4.1) | CONDITIONAL | Yes/Imputed | 0.157/Imputed | - | 0.89 / 0.81 / 0.86 | Conditional | Rejected |
| 31 | savE OM55 | 55 | 188 | REJECTED: safety FAIL (flammability=Yes, Reported) | Reported(2000) | Reported(1) | UNKNOWN | Yes/Reported | 0.13/Reported | - | 0.86 / 0.78 / 0.83 | Conditional | Rejected |
| 32 | Palmitic-stearic acid/Expanded graphite | 55.2 | 176 | survives | Imputed(2000) | Reported(0.3) | CONDITIONAL | Yes/Imputed | 0.16/Imputed | - | 0.85 / 0.77 / 0.82 | Conditional | Rejected |
| 33 | n-Hexacosane (C26) | 56.5 | 256 | survives | Imputed(1404) | Imputed(0.3) | PASS | Yes/Imputed | 0.238/Imputed | - | 0.80 / 0.73 / 0.77 | Conditional | Rejected |
| 34 | RT57HC | 56.5 | 240 | survives | Imputed(1404) | Reported(0) | PASS | Yes/Imputed | 0.239/Imputed | - | 0.82 / 0.75 / 0.79 | Conditional | Rejected |
| 35 | PlusICE A58 | 58 | 215 | survives | Imputed(1581) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.22/Reported | - | 0.82 / 0.75 / 0.79 | Conditional | Rejected |
| 36 | PureTemp 58 | 58 | 225 | survives | Imputed(1620) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.2/Reported | - | 0.80 / 0.73 / 0.77 | Conditional | Rejected |
| 37 | RT60 | 58 | 160 | survives | Imputed(2000) | Reported(0) | PASS | Yes/Imputed | 0.1695/Imputed | - | 0.88 / 0.79 / 0.84 | Conditional | Rejected |
| 38 | n-Heptacosane (C27) | 59 | 236 | survives | Imputed(1404) | Imputed(0) | PASS | Yes/Imputed | 0.2415/Imputed | - | 0.79 / 0.72 / 0.76 | Conditional | Rejected |
| 39 | CrodaTherm 60 | 59.8 | 217 | survives | Imputed(1533) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.23/Reported | - | 0.83 / 0.75 / 0.80 | Conditional | Rejected |
| 40 | Palmitic acid/Expanded graphite (80/20) | 60.9 | 148 | survives | Imputed(2000) | Reported(0.1) | CONDITIONAL | Yes/Imputed | 0.1645/Imputed | - | 0.82 / 0.75 / 0.78 | Conditional | Rejected |
| 41 | PureTemp 60 | 61 | 220 | survives | Imputed(1695) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.2/Reported | - | 0.79 / 0.72 / 0.76 | Conditional | Rejected |
| 42 | RT65 | 61.5 | 150 | survives | Imputed(2000) | Reported(0) | PASS | Yes/Imputed | 0.165/Imputed | - | 0.85 / 0.77 / 0.82 | Conditional | Rejected |
| 43 | n-Octacosane (C28) | 61.6 | 253 | survives | Imputed(1581) | Imputed(0) | PASS | No/Imputed | 0.2665/Imputed | - | 0.76 / 0.69 / 0.71 | Conditional | Rejected |
| 44 | PlusICE A62 | 62 | 205 | survives | Imputed(1581) | Imputed(0) | UNKNOWN | Yes/Imputed | 0.22/Reported | - | 0.79 / 0.72 / 0.75 | Conditional | Rejected |
| 45 | RT62HC | 62.5 | 230 | survives | Imputed(1404) | Reported(0.5) | PASS | Yes/Imputed | 0.2385/Imputed | - | 0.77 / 0.70 / 0.73 | Conditional | Rejected |
| 46 | Palmitic acid (C16) | 62.6 | 198 | survives | Imputed(1695) | Imputed(0.5) | CONDITIONAL | Yes/Imputed | 0.188/Imputed | - | 0.80 / 0.72 / 0.76 | Conditional | Rejected |
| 47 | PureTemp 63 | 63 | 206 | survives | Imputed(1510) | Imputed(1) | UNKNOWN | Yes/Imputed | 0.2/Reported | - | 0.78 / 0.71 / 0.75 | Conditional | Rejected |
| 48 | RT64HC | 64 | 250 | survives | Imputed(1404) | Reported(1.5) | PASS | Yes/Imputed | 0.283/Imputed | - | 0.73 / 0.66 / 0.68 | Conditional | Rejected |
| 49 | Paraffin wax (generic) | 64 | 173.6 | REJECTED: data-incomplete (k missing) | Missing | Missing | PASS | Unknown/Missing | Missing | - | 0.80 / 0.73 / 0.76 | Unverified | Rejected |
| 50 | n-Nonacosane (C29) | 64 | 240 | survives | Imputed(1404) | Imputed(1.7) | PASS | Yes/Imputed | 0.28/Imputed | - | 0.74 / 0.67 / 0.69 | Conditional | Rejected |
| 51 | n-Triacontane (C30) | 65.4 | 251 | survives | Imputed(1581) | Imputed(2.3) | PASS | No/Imputed | 0.269/Imputed | - | 0.70 / 0.65 / 0.66 | Rejected | Rejected |
| 52 | savE OM65 | 67 | 188 | REJECTED: safety FAIL (flammability=Yes, Reported) | Reported(2000) | Reported(1) | UNKNOWN | Yes/Reported | 0.16/Reported | - | 0.66 / 0.62 / 0.61 | Rejected | Rejected |
| 53 | Stearic acid (C18) | 67.9 | 259 | survives | Imputed(1695) | Imputed(0) | CONDITIONAL | Yes/Imputed | 0.211/Imputed | - | 0.66 / 0.61 / 0.61 | Rejected | Rejected |
| 54 | PureTemp 68 | 68 | 213 | survives | Imputed(1404) | Imputed(0.7) | UNKNOWN | No/Imputed | 0.2755/Imputed | - | 0.71 / 0.65 / 0.66 | Rejected | Rejected |
| 55 | n-Hentriacontane (C31) | 68 | 242 | survives | Imputed(1695) | Imputed(0) | PASS | Yes/Imputed | 0.2135/Imputed | - | 0.68 / 0.63 / 0.63 | Rejected | Rejected |
| 56 | RT69HC | 69 | 230 | survives | Imputed(1551) | Reported(1) | PASS | Yes/Imputed | 0.2815/Imputed | - | 0.68 / 0.63 / 0.63 | Rejected | Rejected |
| 57 | n-Dotriacontane (C32) | 69.5 | 170 | survives | Imputed(2000) | Imputed(1.6) | PASS | Yes/Imputed | 0.165/Imputed | - | 0.66 / 0.61 / 0.61 | Rejected | Rejected |
| 58 | RT70HC | 70 | 260 | survives | Imputed(1404) | Reported(0) | PASS | Yes/Imputed | 0.24/Imputed | - | 0.61 / 0.58 / 0.56 | Rejected | Rejected |
