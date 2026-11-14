# Assam Objective 1 — Pre-Implementation Technical Audit of the Corrected Feasibility Methodology

**READ-ONLY AUDIT. Nothing was implemented, Phase B was not run, no existing file was modified, and no Stage E process was touched. This audit is not user approval of anything.**

Written for: the project owner, who must decide which parts of the corrected methodology may proceed to implementation.

Evidence labels: **[FACT]** read from source/data in this session; **[CALC]** independent calculation by this audit (scratch, outside the project); **[INFER]** inference from facts; **[ASSUME]** assumption; **[LIT]** literature/regulatory knowledge not reproduced from a repository file; **[REC]** recommendation.

---

## 1. Executive verdict

**NOT READY for implementation.** The classification framework (four statuses, pathway separation, evidence grading, status propagation, tests) is sound in structure, and several parts verified exactly (energy conservation, class recomputation, mass bound, hand-checked reliability arithmetic, medoids). But the physical screen that sits behind the nominal-design results has defects that change its numbers, and three inputs are not defensible as stated.

Findings that must be resolved first (details in §3):

| # | Severity | Finding |
|---|---|---|
| F-01 | **CRITICAL** | The discharge model resets the frozen-layer thickness at the start of every draw, so the morning draw ignores the shell frozen by the previous evening draw. A multi-front reference model gives lower reliability: C0 nominal passes fall from 18 to 13 of 56; mean R_all overstated by 0.24. |
| F-02 | HIGH | The 900 s time step is not converged: R_wm changes by up to 0.079 versus 112.5 s (and 0.016 at 225 s); the nominal pass count changes from 18 to 16. The methodology's own acceptance test V10 (≤ 0.01) would fail. |
| F-03 | HIGH | Geometry: the nominal PCM half-thickness of 20 mm corresponds to an 80 mm capsule when matched by surface-to-volume ratio; the Objective 2 capsule range (20–60 mm) corresponds to 4–14 mm. The methodology's statement that δ = 10–30 mm "spans" the Objective 2 range is wrong. At in-range thicknesses the nominal passes rise from 9 to 20–22 of 26 (Path A, C0). The wall/convective resistance is also omitted. |
| F-04 | HIGH | The nominal collector area of 4 m² is Rajasthan's value. Assam's own Objective 1 physics uses 2.0 m² (and Objective 2 uses 5 m²). At A = 2 m² the nominal passes fall to 2 / 11 / 2 (from 18 / 24 / 17). |
| F-05 | HIGH | Objective 1 (100 kg water + 50 kg PCM + 2 m², 100 L/day) and Objective 2 (500 L tank + external bed, 200 L/day) are different systems. At 200 L/day nothing passes at nominal even with 74.7 kg; PCM share 0.5 cannot be carried by 50 kg. |
| F-06 | HIGH | R_wm ≥ 0.5 has no independent engineering justification, is mislabelled "median day" in the methodology, the "Rajasthan-consistent" claim is an analogy not an equivalence, and R_wm is in practice a January test (55 of 56 candidates). |
| F-07 | HIGH | The headline nominal-pass counts (18 / 24 / 17) overstate robustness: 15 / 19 / 14 of them depend on an imputed conductivity (fail at the lowest reported k); only 3 / 5 / 3 survive that test, before the F-01 correction. |
| F-08 | HIGH | The Path A discharge model has no hot-end outlet-temperature constraint. If the PCM had to heat the draw to ≥ 50 °C, only 7 of 26 Path A candidates satisfy the ε–NTU condition at nominal geometry (share 1.0). The system arrangement (series/parallel, where the PCM sits) is undefined. |

Decision matrix (§13): **A1** approve with conditions; **A2** approve with conditions; **A3 HOLD**; **A4 HOLD**; **A5 HOLD**; **A6** approve with conditions; **A7** approve with conditions; **A8** approve with conditions; **A9** approve; **A10** approve with conditions.

Minimum decisions needed from you: A5 (demand basis and which system Objective 1 represents), A4 (reference system: collector area, PCM mass), A3 (reliability requirement or demotion of R_wm to a diagnostic), and permission to specify and test the corrected Stefan formulation of §4.

---

## 2. Files and code inspected

[FACT] All read-only. Scratch scripts and outputs of this audit are in `C:\Users\hp\AppData\Local\Temp\claude\0189d894ab42dcbb\de02d978-3298-4895-a1f7-5f4bfe890c04\scratchpad\` (`audit_core.py`, `audit_run1.py` … `audit_run5b.py`, outputs in `obj1_audit\`); they were not copied into the project because this task asks for one new report only.

| Area | Path |
|---|---|
| Methodology and audit | `era5-assam/ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY.md`; `era5-assam/ASSAM_OBJECTIVE1_FEASIBILITY_FILTER_ROOT_CAUSE_AUDIT.md` |
| Implementation under audit | `…_METHODOLOGY_assets/obj1_method_analysis.py` (screening model), `obj1_method_classify.py` (classification), `make_method_md.py`; CSVs `candidate_status_C0/1/2.csv`, `sequential_results_all_scenarios.csv`, `energy_screen_sensitivity.csv`, `area_per_kg_required.csv`, `scenario_sensitivity_counts.csv`, `threshold_sensitivity.csv`, `latent_requirement_sensitivity.csv`, `preheat_fraction.csv`, `pathway_boundary_by_approach.csv`, `collector_operating_point.csv`, `status_counts.json`, `verification.json` |
| Assam Objective 1 | `era5-assam/07_feasibility_filter_final.py`, `08_mcdm_ranking_final.py`, `08b_monte_carlo_stability_final.py`, `09_recommendation_cards.py`, `10_physics_validation.py`, `06_build_pcm_database_final.py`, `04c_charging_ceiling_assam.py`; `data/processed/pcm/pcm_database_final.csv`, `data/processed/clustering/*`, `data/processed/design/swh_design_specification.csv`; `PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` |
| Frozen history | `Objective---2/objective2-assam/phase0_input_package/objective1_frozen/*`, `o2_design_basis_assam.yaml`, `PHASE0_AUDIT_REPORT.md` |
| Objective 2 | `objective2-assam-fpc/configs/{system_assam_fpc,pcm_bed_system_assam,pcm_input_evidence_assam,regimes_assam}.yaml`, `src/pcm_geometry.py`, `src/io_weather.py`, `results/m3_stageE/` (listing only) |
| Rajasthan (read-only) | `era5-rajasthan/07_feasibility_filter.py`, `04b_climate_signature.py`, `physics_lib.py`, `config.py`, `08_mcdm_ranking.py`, `09_recommendation_cards.py`; archived outputs in `Objective---2/data/objective1/rajasthan/` |

Not available [FACT]: Rajasthan original per-run processed outputs (`era5-rajasthan/data/processed` holds only grid and by-location data); text of the PDF literature in `Sources/pdfs/` (not text-searched); any SDS, coupon test or collector test report; any definition of the containment material or of an auxiliary heater.

---

## 3. Critical findings

Each finding lists: severity; file; function/equation/line; what it does; why correct or problematic; independent evidence; recommendation; whether implementation approval is required. Line numbers refer to the `…_assets/` copies.

### F-01 CRITICAL — Frozen-shell resistance is reset at each draw (state carry-over)
- **File/line:** `obj1_method_analysis.py` lines 177 (`xr = x.copy()`), 180 (`film = np.maximum((xr - x) * par["delta"], S_MIN)`).
- **What it does:** the discharge resistance is the thickness frozen *since the start of the current draw*. The state is only the scalar liquid fraction x; the arrangement of solid and liquid layers is not stored.
- **Why problematic:** after the evening draw the wall-adjacent layer is a solid shell of thickness (x_pre − x_post)δ. Nothing melts it overnight, so the morning draw starts with that shell in place; the code starts it at 0 (floored at 0.5 mm). Morning heat transfer is therefore overestimated. The evening draw (which follows a daytime charge) is modelled correctly.
- **Evidence [CALC]:** an explicit multi-front reference model (`audit_core.sim_stack`) was run on the same weather, same nominal inputs, all 56 evaluable candidates in C0. Nominal passes (R_wm ≥ 0.5): **18 (screen) vs 13 (reference)**; flips to fail: PlusICE A52 (0.62 → 0.00), Myristic acid C14 (0.60 → 0.00), PlusICE A58 (0.53 → 0.45), PureTemp 68 (0.56 → 0.49), RT69HC (0.51 → 0.43). Path A passes 9 → 6; Path B 9 → 7. Mean R_wm overstatement 0.066 (max 0.62); mean R_all overstatement 0.24; mean morning coverage overstatement 0.020 (max 0.14). Sentinels in all three clusters are in `stefan_stack_vs_vector.csv`; for example savE OM46 R_all is 0.495 in the screen and 0.000 in the reference (morning coverage never reaches 100 %).
- **Recommendation [REC]:** replace the scalar-plus-snapshot scheme by an explicit interface stack or a 1-D enthalpy solution (§4.4). All "nominal" counts and "design OK at nominal" reasons in the methodology must be regenerated. The REJECTED gate (favourable corner) is much less exposed (55 of 56 pass at that corner) but must be re-tested.
- **Approval required:** yes (model change).

### F-02 HIGH — Time step not converged
- **File/line:** line 165 (`DT = 900.0`), explicit film from the start of each step (lines 180–183, 194–197).
- **Evidence [CALC]:** C0, nominal, 58 candidates, R_wm versus Δt = 112.5 s: 900 s max |Δ| = 0.079 (2 pass/fail flips, 18 vs 16 passes); 450 s 0.039; 225 s 0.016. Not converged at 225 s. V10 in the methodology requires ≤ 0.01 for a halving.
- **Recommendation:** adopt Δt ≤ 112.5 s or an implicit/event-based update; re-run the convergence test as written in V10 and report the finest step used. **Approval required:** yes.

### F-03 HIGH — Geometry mapping and omitted external resistance
- **File/line:** methodology §6 item 7 ("PCM radius ≈ 8–28 mm, so δ = 10–30 mm spans it"); `obj1_method_analysis.py` line 161 (`Af = m/(ρ δ)`).
- **Why problematic [FACT/INFER]:** a slab of half-thickness δ has surface-to-volume 1/δ and latent energy ρLδ per unit wall area. A cylinder of PCM radius R heated at its outer wall has surface-to-volume 2/R and ρLR/2 per unit wall area. Energy-consistent equivalence is therefore δ = R/2. Objective 2 capsule outer diameter 20–60 mm (`src/pcm_geometry.py` docstring) with 1 mm wall gives R ≈ 9–29 mm, i.e. δ_slab ≈ 4.5–14.5 mm. The nominal δ = 20 mm corresponds to R = 40 mm (an 80 mm capsule). For inward melting of a cylinder the true resistance ln(R/(R−s))/(2πk) exceeds the slab value for thick films, so the slab is optimistic at large s.
- **Evidence [CALC]:** nominal Path A passes (C0/C1/C2) versus δ: 20 mm → 9/13/8; 14 mm → 20/22/21; ≤ 10 mm → 22/23/22 (saturated). The wall and convective resistance on the water side is not modelled; 1/h with h = 200 W/m²K equals a 1 mm film at k = 0.2, i.e. comparable to the 0.5 mm floor.
- **Recommendation:** map capsule diameter to the equivalent δ explicitly, include R_ext = R_wall + 1/h as a declared parameter with a sensitivity range, and state that Phase B uses a slab approximation of a cylinder. **Approval required:** yes.

### F-04 HIGH — Reference collector area inconsistent with the Assam system
- **File:** methodology §4.2, `obj1_method_analysis.py` BASE (A = 4.0). **[FACT]** Assam Objective 1 physics uses A_c = 2.0 m², F_R(τα) = 0.72, F_R U_L = 4.5, water mass 100 kg, PCM 50 kg, UA_pcm = 375 W/K (`10_physics_validation.py` docstring lines 8–16, constants lines 99–111). Rajasthan uses A_c = 4.0 m² (`physics_lib.py` line 446). Objective 2: 2 m² first test, 5 m² M3 baseline.
- **Evidence [CALC]:** nominal passes at A = 2 m² (100 L/day, 50 kg): C0/C1/C2 = 2/11/2 (Path A 0/2/0) versus 18/24/17 at 4 m².
- **Recommendation:** state which system Phase B represents (A5) and use its own area as the nominal; carry A = 2…6 m² only as sensitivity. **Approval required:** yes.

### F-05 HIGH — Objective 1 and Objective 2 are different systems; demand basis changes the answer
See §7. **[CALC]** At 200 L/day (two 100 kg draws), m = 50 kg: 0 candidates pass at nominal and at the favourable corner (all clusters). With the largest constructible mass 74.7 kg: 0 pass at nominal, 40/42/40 pass at the favourable corner. At 100 L/day the favourable corner passes 55 of 56. **Approval required:** yes (A5).

### F-06 HIGH — R_wm ≥ 0.5 is a convention, mislabelled, and January-dominated
See §6. **Approval required:** yes (A3).

### F-07 HIGH — Headline nominal-pass counts overstate robustness
- **File:** methodology §1 item 5 and §10; `obj1_method_classify.py` lines 70–76.
- **Evidence [CALC]:** of 18/24/17 nominal-pass CONDITIONAL candidates, those that fail at the lowest reported conductivity although their k is imputed: 15/19/14. Robust to k: 3/5/3. 3 nominal passes per cluster also rest on an imputed solid density. These figures are before the F-01 correction (C0: 18 → 13).
- **Recommendation:** report nominal passes as "with reported k only / with imputed k" and do not use the unqualified counts in summaries. **Approval required:** no (reporting), yes for rules.

### F-08 HIGH — Path A discharge has no hot-end outlet constraint
- **File/line:** lines 172–183: ΔT_dis uses the *mean* water temperature, `Twm = 0.5 (Tw + Tout)`; coverage is energy-based.
- **Why problematic [INFER]:** "direct 50 °C delivery" requires T_out ≥ 50 at the hot end. For an isothermal PCM wall at T_f = Tm − SC, ε–NTU gives T_out = T_f − (T_f − T_w) e^(−NTU), NTU = UA/(ṁ c_p), UA = A_face k_s/φ. T_out ≥ 50 requires φ ≤ φ_max = A_face k_s /[−ṁ c_p ln((T_f−50)/(T_f−T_w))].
- **Evidence [CALC]:** C0, mains 17.2 °C, 50 kg draw, 50 kg PCM, δ = 20 mm: n-Hexacosane (Tm 56.5, k_s 0.299 imputed): energy-limited shell thickness for one full-draw duty 10.7 mm versus φ_max = 9.1 mm (fails); across 26 Path A candidates only 7 satisfy φ_D ≤ φ_max when the PCM must heat the whole draw (share 1.0); 21 of 26 at share 0.5 (but then the outlet temperature depends on an undefined mixing/series arrangement).
- **Recommendation:** define the flow arrangement and add the ε–NTU outlet condition (or state that Path A is only a Tm-threshold check, not a delivery proof). **Approval required:** yes.

### F-09 MEDIUM — Approach parameter plays two roles
- **File/line:** line 196 (`Qr = Af * kl * appr / film`), line 191 (`Tin = Tm + appr`); sensitivity scenarios `S:appr=4/8`.
- **Problem:** the same 6 K is (i) the pathway boundary (Tm ≥ 50 + approach), (ii) the collector return temperature, and (iii) the conduction driving difference for charging. Raising it to 8 K *increases* charging rate and passes (Path A: 11/16/11 vs 9/13/8) although it shrinks Path A to 24 candidates, which is not physical. Charging ΔT is the loop-fluid-to-PCM difference and should come from a heat-exchanger UA, not from the approach used for the boundary.
- **Recommendation:** separate approach (boundary) from ΔT_charge(UA). **Approval required:** yes.

### F-10 MEDIUM — Collector model assumptions
See §5. Tilt is ignored (conservative, +5–8 % annual, +14–26 % January); the quadratic loss term is ignored (non-conservative, −2 to −45 % on the worst-month fraction depending on cluster and a₂); η_loop = 1 and ideal control are optimistic.

### F-11 MEDIUM — Path A upper bound not applied
- **File/line:** `obj1_method_classify.py` line 56 (band 42–70 °C).
- **Evidence [CALC]:** the methodology (§4.1) says the 75 °C system limit caps Tm near 69 °C, but classification uses 70 °C. Charging fluid Tm + 6: n-Dotriacontane 75.5 °C, RT70HC 76.0 °C (> 75), RT69HC 75.0 °C (= 75). Pathway boundary on freezing temperature T_f = Tm − SC: no Path A candidate is below 56 °C (checked, none affected).
- **Recommendation:** apply Tm + approach ≤ T_max,system (state T_max,system) in Path A. **Approval required:** yes.

### F-12 MEDIUM — Inconsistent handling of imputed and missing properties
- **File/line:** `obj1_method_classify.py` lines 61–66 (UNVERIFIED only for missing ρ or imputed L).
- **Evidence [FACT/CALC]:** solid density is imputed for 14 and missing for 2 of 58; conductivity is imputed for 39 and missing for 3. Imputed L → UNVERIFIED; imputed ρ and imputed k → evaluated and CONDITIONAL; missing k with ρ present (Paraffin wax generic) → CONDITIONAL, whereas the two eutectics (k and ρ missing) → UNVERIFIED. All three inputs enter A_face, k_s, k_l or capacity, so the rule is inconsistent. No candidate is promoted to FEASIBLE (verified: FEASIBLE requires Reported k, cycling, corrosion evidence, none exists).
- **Recommendation:** one rule: any imputed/missing physics input caps the status at UNVERIFIED unless the screen passes at the bounds of that input (state the bounds). **Approval required:** yes.

### F-13 MEDIUM — Safety and corrosion rules cite evidence not in the repository
See §8.

### F-14 MEDIUM — The rejection gate is nearly non-discriminating
- **File/line:** `obj1_method_classify.py` lines 67–68 (REJECTED if R_wm < 0.5 at the favourable corner).
- **Evidence [CALC]:** the favourable corner combines six favourable choices at once (A = 6 m², δ = 10 mm, share 0.4, FR(τα) 0.80, F_R U_L 3.5, k at the maximum reported for imputed k). 55 of 56 candidates pass; one is rejected. It is a design-envelope check, which is acceptable, but then equal status counts across clusters (51/5/2 each) are a property of that gate plus property availability, not evidence of cluster-specific feasibility. Nominal-pass differences by cluster are real: 7 candidates differ between clusters.
- **Recommendation:** state this plainly in the methodology and report cluster-specific results as the required A/m, not as status counts. **Approval required:** no.

### F-15 LOW/MEDIUM — Mass bound conditional on provisional constraints
74.74 kg (inline) and 68.76 kg (staggered) were **reproduced exactly** [CALC] from the stated constraints (H ≤ 1.20 m, V_cav ≤ 250 L, W_design 0.40 m, pitch 1.3 t, L_c 0.40 m, wall 1 mm). They assume ρ_pcm = 910 kg/m³: 62.4 kg at 760, 72.3 at 880, 78.8 at 960 kg/m³ (inline). The methodology presents 74.7 kg as a fact without these caveats.

### F-16 LOW — Provenance columns missing from the per-cluster CSVs
The CSVs carry per-property status strings but no `source_type` (manufacturer/literature), no data-source detail and no cluster-point ID. The required "evidence provenance" per candidate is only partly present.

### F-17 LOW — Numerical floor and dead code
`S_MIN` = 0.5 mm is a numerical lower bound on the resistance, not a conduction path. Its effect is immaterial for 0.25–2 mm (R_wm unchanged; 5.9 % median, 8.9 % maximum of charging energy transferred at the floor) and material at 5 mm (max change 0.18). Line 199 (`elif h == 8: xref = x.copy()`) is unreachable. Both are low risk but the floor must be reported.

### F-18 LOW — Spin-up and attribution
The 5-day spin-up is not demonstrated adequate by test: moving the cut from 5 to 30 days changes R_wm by up to 0.044, but this is mostly removal of early-January days (8 % of the January sample). State convergence (x is driven by a daily cycle) is plausible but untested. The "day d" success mixes the morning draw (fed by day d−1 charge) and the evening draw (fed by day d charge); this is not stated in the methodology.

### F-19 LOW — PCM share citations are second-hand
SHARE_PCM = 0.5 (range 0.4–0.78) comes from Rajasthan's `04b_climate_signature.py` comments citing Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021; the primary papers were not read in this audit. Rajasthan's own text says the property should preferably be a ranking criterion and not double-gated.

### F-20 Verified (no defect)
- Energy conservation of the screening model: maximum |Δ(x·m·L) − (Q_charge − Q_delivered)| = **0.0075 J** over 10 years, 58 candidates, 3 clusters. [CALC]
- Independent re-classification (fresh code from the rules and the scenario results): **0 mismatches** over 174 (candidate, cluster) rows; counts 51/5/2 per cluster reproduced; the methodology's appendix tables contain exactly 2/51/5 (REJECTED/CONDITIONAL/UNVERIFIED) in each cluster. [CALC]
- Hand-checkable reliability example (§6) agrees with the implementation expression. [CALC]
- Water-heating and latent-heat arithmetic (§11) reproduces the Phase 4 specification value 125.74 kJ/kg. [CALC]
- Medoids and weather source (previous round) remain valid. [FACT]
- **The 49-survivor defect is prevented only on paper:** V16/V17/V13/V14 are specified, none is implemented, and `10_physics_validation.py` still requires exactly 5 survivors. [FACT]

---

## 4. Stefan model audit (high priority)

### 4.1 Representation [FACT, `obj1_method_analysis.py` 161–204]

| Symbol | Definition | Implementation |
|---|---|---|
| x | liquid mass fraction (= liquid thickness fraction) of a half-slab | scalar per element, initial 0.5 (line 168) |
| δ | half-thickness (m), both faces heated | `par["delta"]` |
| A_face | total contact area = m/(ρ_s δ) | line 161 |
| x_ref | x at the start of the charging window | `xref`, reset at hour 8 (line 189) |
| x_r | x at the start of a draw | `xr`, reset at hours 7 and 19 (line 177) |
| φ_c | charging liquid film = max((x − x_ref)δ, s_min) | line 194 |
| φ_d | discharge solid shell = max((x_r − x)δ, s_min) | line 180 |
| s_min | 0.5 mm | line 165 |
| Q_rate,c | A_face k_l ΔT_app / φ_c | line 195 |
| Q_rate,d | A_face k_s (T_m − SC − T̄_w)/φ_d | lines 174, 181 |

### 4.2 Equation-by-equation

| Item | Assessment |
|---|---|
| Melt fraction and energy | dx = Q Δt/(m L); exact; energy residual ≤ 0.0075 J. **Correct.** |
| Charging film φ_c = (x − x_ref)δ | Correct when the wall-adjacent layer at 08:00 is a solid shell, which holds after any draw. The film then equals the thickness melted since x_ref. It is not valid if the wall-adjacent layer at 08:00 is liquid (no draw, partial charge). **Correct for the normal sequence.** |
| Discharge shell φ_d = (x_r − x)δ | Correct only if the wall-adjacent layer at the start of the draw is liquid. True for the evening draw after a charge; **false for the morning draw** (shell from the evening draw persists). **Defect F-01.** |
| `max(·, s_min)` | A numerical lower bound on the resistance. It prevents division by zero at t = 0 (physical quasi-steady flux is unbounded as φ → 0) and caps the flux at k ΔT / 0.5 mm (2.4 kW/m² at ΔT = 6 K, k = 0.2). It does not represent a conduction path; the real early-time limit is transient conduction and the wall/convection resistance, which are omitted. Effect on results immaterial for ≤ 2 mm, so it does not "artificially increase" rates materially; the omitted external resistance is the larger concern (F-03). |
| Geometry consistency | Charge and discharge use the same A_face and the same slab; k_l for charge, k_s for discharge. Consistent. Density change on melting is ignored (A_face uses ρ_s); imputed ρ enters A_face. |
| Phase-front reference | x_ref and x_r are *snapshots*, not front positions. With up to four interfaces possible after a partial cycle, a scalar cannot recover the wall-adjacent layer. |
| Daily carry-over | x persists across hours and days (correct), but the layer arrangement is lost (F-01). |
| Timestep | explicit; not converged at 900 s (F-02). |
| Sensible heat | ignored; charging from a solid below T_m, supercooling nucleation delay and superheating are not represented. This is declared, and is optimistic for energy and conservative for rate. |

### 4.3 Rajasthan comparison [FACT]
`physics_lib.simulate_pcm_swh_year` (lines 640–800) does not use a Stefan film. It uses a lumped three-phase energy model (Barqawi 2025 equations): sensible solid, isothermal melt with Q̇ = H_p A_p max(0, T_w − T_m), sensible liquid, with a constant PCM–water coefficient H_p = 800 W/m²K (Barqawi), scaled by candidate conductivity, and S/V = 100 m⁻¹ (`H_P_WM2K_BASE`, `SURFACE_TO_VOLUME_RATIO_PER_M`, lines ~480–484). It therefore has no moving-front resistance, no carry-over issue, and no resistance growth. Assam's own Objective 1 physics uses (UA)_pcm = 375 W/K (`10_physics_validation.py`, h = 150 W/m²K, A = 2.5 m²). At the nominal slab (δ = 20 mm, 50 kg, k = 0.2, film 5 mm) the quasi-steady conductance is 111 W/K [CALC], i.e. about one third of Assam's own validation value and about one fortieth of the Rajasthan-equivalent (800 W/m²K × 5.9 m²). The Stefan screen is more physical in having a moving front but is **inconsistent with both existing treatments**, and none is validated.

### 4.4 Corrected mathematical formulation (specification only)
Let the half-slab coordinate z run from the wall (z = 0) to the centre (z = δ) and keep an ordered list of layers (phase, thickness) with Σ thickness = δ.

1. **Charging** (collector available, wall at T_w,c = T_m + ΔT_ch). Wall-adjacent layer j = 1.
 - If layer 1 is solid, insert a liquid layer of thickness 0 at the wall.
 - Flux q = min(Q_col/A_face, (T_w,c − T_m)/(R_film + R_ext)), R_film = φ_1/k_l, R_ext = R_wall + 1/h.
 - Front advance dφ_1 = q Δt/(ρ L), taken from layer 2 (solid); remove layers of zero thickness; merge adjacent layers of the same phase.
2. **Discharging** (draw hour). Insert a solid layer of thickness 0 if layer 1 is liquid; flux q = (T_m − SC − T_w,wall)/(φ_1/k_s + R_ext); the layer-2 liquid freezes; **do not reset** φ_1 between draws; it changes only by freezing (growth) or by melting during charging.
3. **Outlet constraint** for Path A: T_out = T_f − (T_f − T_w,in) exp(−UA/(ṁ c_p)) with UA = A_face/(φ_1/k_s + R_ext), requiring T_out ≥ 50 °C (or the declared target) throughout the draw.
4. **Time stepping:** implicit or adaptive; Δt chosen by a convergence test (§11, V10); declare s_min only as a numerical floor with sensitivity.
5. **Preferred alternative:** a 1-D enthalpy (finite-volume) solution in the true capsule geometry (cylinder), which removes the layer bookkeeping and the slab approximation; the stack model is the cheap reference for it.

The reference stack model used in this audit (`audit_core.sim_stack`) implements items 1–2 with R_ext = 0 and Δt = 900 s; its results should be treated as a bias estimate, not as a final model.

---

## 5. Collector model audit

### 5.1 Equation and units [FACT/CALC]
Q_col = A · max(0, F_R(τα) G − F_R U_L (T_in − T_a)) · η_loop. Units: m² · (W/m² − W/m²K · K) = W; dimensionally consistent. Hourly G is treated as constant over the hour; the 10 W/m² cut-off is from the project configuration (`system_assam_fpc.yaml`).

### 5.2 Assumptions table

| Item | Value | Status | Comment |
|---|---|---|---|
| F_R(τα) | 0.72 (Obj 1), 0.75 (Obj 2) | ASSUMED, not measured | range 0.65–0.80 planned |
| F_R U_L | 4.5 W/m²K | ASSUMED, not measured | range 3.5–5.0; no quadratic term |
| η_loop | 1.0 nominal | ASSUMED | no pipe, heat-exchanger, pump or control losses |
| A | 4 m² nominal | Rajasthan value | Assam O1 = 2.0, O2 = 5 (F-04) |
| T_in = T_m + ΔT_app | 62 °C at T_m = 56 | design assumption | collector return at the PCM temperature plus approach; flow rate and HX not modelled |
| GHI on collector | horizontal, no tilt | ASSUMED (project convention) | see below |
| Charging window | 08:00–18:00 rows | ASSUMED | 4.9–6.6 % of daily GHI energy lies outside |
| Ideal control | collector runs whenever net gain > 0 | ASSUMED | optimistic |

### 5.3 Quantified biases [CALC] (T_m = 56 °C, A = 4 m², energy screen from mains temperature)
- **Tilt (isotropic sky, Erbs split, south-facing, tilt = latitude):** annual plane irradiation +5.8 / +7.7 / +5.9 % (C0/C1/C2); January +20 / +26 / +19 %. Worst-month fraction of days with a full melt rises from 0.57 / 0.55 / 0.44 to 0.72 / 0.59 / 0.65. Using horizontal GHI is therefore **conservative** and is worst for the worst-month metric.
- **Quadratic loss (a₂ = 0.01–0.02 W/m²K², ΔT ≈ 37 K at T_in = 62 °C):** mean useful energy −5 % / −11 %; worst-month fraction 0.57 → 0.45 → 0.35 (C0), 0.44 → 0.35 → 0.24 (C2). Ignoring it is **non-conservative**.
- **Charging window:** ignoring 5–7 % of GHI energy is a small conservative bias.
- **Loop efficiency and control:** η_loop = 0.85 reduces nominal passes (Path A 6/12/5 vs 9/13/8). Ideal control and η = 1 are optimistic.
- Net direction is unknown without measured data. The assumed values are not experimentally validated for Assam and the screen must not present them as such.

### 5.4 Can the collector charge at 62–76 °C? [CALC]
Hours per day with G above the critical value G_crit = F_R U_L (T_in − T_a)/F_R(τα): C0 7.7 h (of 11.8 h with G ≥ 10) at T_m = 56 °C, 7.0 h at 64 °C; instantaneous efficiency at 600 W/m² about 0.48 at 56 °C and 0.42 at 64 °C. Yes in the model; it depends on the unmeasured parameters.

### 5.5 Rajasthan comparison [FACT]
`physics_lib.py` lines 448–461: Tc = T_amb + 0.70 G / U_L with U_L = 2.0 W/m²K (Barqawi's stagnation-style relation, **calibrated down** from 20 to 2.5 and then 2.0 to reproduce a 54–84 % solar-fraction benchmark), A_c = 4 m², no inlet-temperature dependence, no tilt-related code found (searched `tilt`, `beta`), no charging-window restriction. Rajasthan's collector is a calibrated heuristic; Assam's HWB form is more physical in having T_in dependence. Neither is validated; the Rajasthan collector must not be copied.

### 5.6 Recommended sensitivity ranges [REC]
F_R(τα) 0.60–0.80; F_R U_L 3.5–6.0 with a₂ 0–0.02 W/m²K²; η_loop 0.80–1.00; tilt: horizontal and latitude-tilt both reported; A: 2, 4, 5, 6 m² with the system's own value as nominal; window 06:00–19:00 sensitivity.

---

## 6. Reliability metric audit

### 6.1 What is implemented [FACT, lines 200–204]
`ok[d] = (cov_m[d] ≥ 0.999) & (cov_e[d] ≥ 0.999)`; spin-up `sp = 5` days discarded; `R_all = ok[sp:].mean(0)`; `R_wm = min over calendar months m of ok[sp:][month == m].mean(0)`, pooled over all ten years for each calendar month.

So **R_wm is the lowest monthly fraction of successful days**, not a median-day statistic. The methodology text (§4.2, §8 comments, decision A3, classification script line 13 and the reason string at line 68) describes it as "the median day of the worst calendar month is adequate". That equivalence is not valid: the fraction being ≥ 0.5 does not imply the median-irradiance day succeeds, because success also depends on carried state and mains temperature. Counterexample [CALC]: GHI = [1,2,3,4,5], ok = [T,T,F,F,T] has fraction 0.6 while the median-GHI day fails.

### 6.2 Hand-checkable example [CALC]
Eight days, spin-up 1 day. ok = [T, T, F, T | T, T, F, F]; months = [Jan ×4 | Feb ×4]. After discarding day 1: Jan days 2–4 = [T, F, T] → 2/3 = 0.667; Feb days 5–8 = [T, T, F, F] → 2/4 = 0.500. R_all = 4 successes / 7 days = 0.571; R_wm = min(0.667, 0.500) = 0.500. The implementation expression returns exactly these values.

### 6.3 Behaviour on the real data [CALC]
- The worst calendar month is **January for 55 of 56 evaluable candidates** (December for 1), so R_wm is in practice a January test.
- Pooling years hides variability: the worst *individual* month-year fraction is ≥ 0.5 for only 4 of 56 candidates (median 0.0), against 18 for the pooled R_wm.
- Spin-up: 5-day cut versus 30 or 90 days changes R_wm by ≤ 0.044 (mostly a sampling effect, F-18).

### 6.4 Threshold R_wm ≥ 0.5
- **Not independently justified.** The methodology says it is derived from Rajasthan's convention. Rajasthan's charging ceiling (`04b_climate_signature.py`, lines 440–485) uses the **worst calendar month's mean clearness**, a *resource* statistic, not a success fraction, and it has no reliability threshold. The link is an analogy, so it is a screening convention.
- Engineering reading: 0.5 means half the days of the worst month fail the PCM duty. For a hot-water supply with no defined auxiliary this is not a service-reliability target. It can only be defended as a coarse necessary-condition screen.
- **Sensitivity [CALC] (nominal-pass counts, C0/C1/C2; before F-01 correction):** Path A: 0.25 → 11/16/12; **0.50 → 9/13/8**; 0.75 → 2/0/0; 0.90 → 0/0/0. Path B: 0.25 → 10/12/11; 0.50 → 9/11/9; 0.75 → 6/5/6; 0.90 → 2/0/2. At the favourable corner Path A passes 26/26/26 at 0.25–0.75 and 0.90 → 26/16/26; Path B 25/25/25 at 0.25–0.75, 0.90 → 24/19/25. The result depends strongly on the threshold at nominal and weakly at the favourable corner.
- **Recommendation [REC]:** do not fix the threshold. Report R_all and R_wm as diagnostics, use the favourable-corner pass as the only rejection test until a duty requirement exists (for example from the Objective 2 solar-fraction target), and correct the wording.

---

## 7. Demand-basis audit

### 7.1 Are Objectives 1 and 2 the same system? [FACT]
- Objective 1 (`10_physics_validation.py`, `swh_design_specification.csv`): 100 L/day as two 50 kg draws at 07:00 and 19:00, water 100 kg, PCM 50 kg in the tank, A_c = 2.0 m², UA_pcm = 375 W/K.
- Objective 2 (`o2_design_basis_assam.yaml`, `system_assam_fpc.yaml`, `pcm_bed_system_assam.yaml`): baseline 200 L/day as two 100 kg draws (ASSUMPTION), sensitivity 100/150/200 L/day, 500 L mixed tank, external PCM bed with its own pump, A_c = 5 m² (M3 baseline), PCM mass a design variable (4.6–384.7 kg raw; ≤ 74.7 kg constructible inline). The design-basis file itself states that the Objective 1 survivors "must be re-screened for the 200 L/day basis".
- They are therefore **not the same SWH system**. [INFER]

### 7.2 Candidate screening outcome at the two demands [CALC]
Nominal and favourable corners, 56 evaluable candidates, R_wm ≥ 0.5. Entries: C0 / C1 / C2.

| Set | Nominal passes | Favourable-corner passes |
|---|---|---|
| 100 L/day, 50 kg | 18 / 24 / 17 | 55 / 55 / 55 |
| 200 L/day, 50 kg | 0 / 0 / 0 | 0 / 0 / 0 |
| 200 L/day, 74.7 kg | 0 / 0 / 0 | 40 / 42 / 40 |
| 100 L/day, 50 kg, A = 2 m² | 2 / 11 / 2 | — |
| 200 L/day, 74.7 kg, A = 5 m² | 0 / 0 / 0 | — |

The screening outcome changes materially with the demand basis; a 50 kg bed cannot meet a 200 L/day duty at share 0.5 in any cluster.

### 7.3 PCM share 0.5 [INFER]
The 0.5 comes from integrated tank-plus-PCM literature (second-hand, F-19). In Objective 2's architecture a 500 L tank holds 500·4.186·25 K = 52.3 MJ of sensible heat between 75 °C and 50 °C, against 10.0 MJ latent capacity for 50 kg at 200 kJ/kg, and a daily 200 L demand of 27.5 MJ (C0). Share 0.5 would require the bed to deliver 13.7 MJ/day, more than its own capacity. The 0.5 is not transferable to the Objective 2 architecture.

### 7.4 Recommendation [REC]
Treat Objective 1 as the material screen that feeds Objective 2: (i) screen on material-intrinsic necessary conditions (Tm window, L, k, ρ, cycling, supercooling, safety, corrosion) and capacity per unit mass; (ii) report the duty screen at both 100 and 200 L/day as diagnostics; (iii) use 200 L/day (Objective 2 baseline) for any list handed to Objective 2; (iv) let Objective 2's design optimisation decide area, mass and capsule size. Decision A5 is HOLD because it depends on what you intend Objective 1 to represent.

---

## 8. Safety and corrosion audit

### 8.1 Evidence actually available [FACT]
Flammability: manufacturer-reported for 7 rows (5 "Yes", 2 "No"), imputed for 48, missing for 3. Flash point (source CSV): reported for 16 of 55 rows (155–225 °C; several are bounds such as ">180"), imputed for 39; the 3 added literature rows have none. Maximum operating temperature: reported for 21 rows (65–100 °C). No SDS text, no corrosion column, no containment specification, no leakage or enclosure assessment exists in the repository. Searching the repository's md/txt/py files found no corrosion evidence for any candidate; PDFs were not text-searched.

### 8.2 Is the four-way classification supported?
- **Confirmed acceptable:** supported as an empty class (nothing meets the evidence needed).
- **Conditional on controls/containment:** supported; but the rule "flash point ≥ 93 °C → conditional" uses a GHS cut-off [LIT] that is not in the repository and must be cited; the "T_hot = 75 °C" comes from the Objective 2 tank limit and has no margin defined (`obj1_method_classify.py` line 14, 20).
- **Confirmed unacceptable:** flash point ≤ 75 °C is a minimal rule; the margin above the maximum PCM temperature is undefined; none of the 16 reported flash points is anywhere near it.
- **Unknown:** correct as default for imputed or missing evidence.

### 8.3 Is a flash-point threshold alone sufficient for a safety PASS? **No.** [INFER/LIT]
A flash point says when vapour can ignite; it does not address leak or rupture of molten PCM onto hot surfaces, auto-ignition, thermal decomposition or additive hazards in composites, enclosure and ventilation, fire protection, toxicity, spill and disposal, or overheating beyond the manufacturer's maximum operating temperature. The flash point must also be a sourced closed-cup value, not an imputed one (the Objective 2 evidence register already says OM46 and OM49 need a sourced value "before any safety use").

### 8.4 Missing corrosion evidence: what status?
UNKNOWN, capping the status at CONDITIONAL, never PASS and never an automatic rejection. A candidate with a *demonstrated* incompatibility with the specified containment is REJECTED. The fatty-acid statement ("Cu/Al attack") in the methodology is an unreferenced literature claim [LIT]; it must be cited or removed.

### 8.5 Minimum evidence for a defensible PASS [REC]
- **Safety PASS:** (1) current SDS for the exact product/grade with GHS classification, closed-cup flash point (method named), auto-ignition temperature, decomposition products; (2) flash point above the maximum credible PCM temperature by a margin set by the project's safety reviewer (no number is invented here); (3) manufacturer maximum operating temperature ≥ the maximum credible PCM temperature including overheating scenarios; (4) leak/rupture/enclosure/fire-protection assessment for the specified enclosure; (5) toxicity and environmental data from the SDS; (6) for composites (HDPE, NBR, expanded graphite) the additive SDS and thermal stability.
- **Corrosion PASS:** (1) a specified list of wetted materials (capsule wall, coating, heat exchanger, pipes, fittings, seals); (2) for every wetted metal, a manufacturer compatibility statement for that grade or an immersion/coupon test at or above the maximum operating temperature for a duration tied to design life [LIT: ASTM G31-type practice, to be confirmed], reported as a corrosion rate against an acceptance limit set by the designer; (3) for polymers, compatibility, swelling, permeation and creep at temperature; (4) galvanic couples; (5) tests on the exact grade, not the chemical family. No organic family is assumed benign.
- **Containment specification still needed:** capsule material, wall thickness and any coating; sealing and pressure class; fill fraction; heat-exchanger and pipe materials; maximum design temperature; leak containment and venting.

### 8.6 Rajasthan comparison [FACT]
Rajasthan flags flammability and toxicity as "needs a human review" and never excludes on an unqualified Yes/No flag (`07_feasibility_filter.py` docstring point C); its corrosion veto only targets bare salt hydrates in humid clusters and was inert on its database. Assam's current script (reject on Reported "Yes"; PASS by name pattern) is stricter on safety and looser on corrosion than Rajasthan; the methodology's rule is closer to Rajasthan's safety treatment and stricter than Rajasthan on corrosion evidence, which is justified.

---

## 9. Pathway boundary audit

- **56 °C boundary:** follows arithmetically from 50 + 6 K (F-08/F-09 note it is a threshold condition, not a delivery proof). Pathway assignment on Tm rather than the freezing onset T_f = Tm − SC: no Path A candidate is affected today [CALC]; the rule should still use T_f.
- **Upper bound:** missing (F-11). The system limit and the charging fluid temperature Tm + approach must be applied; three candidates (RT69HC at 75.0, n-Dotriacontane at 75.5, RT70HC at 76.0 °C) are at or above 75 °C.
- **Can the collector charge at the required temperature?** Yes in the model (§5.4), conditional on unmeasured parameters; efficiency falls to about 0.42 at 64 °C.
- **Is 6 K defensible, is 4–8 K adequate?** 6 K is the existing design constant (Rajasthan uses 7 K, the midpoint of a 5–8 K range in the plan document); defensible as a nominal. For a capsule bed the effective approach is set by conduction through the frozen layer (F-08) and may exceed 6 K at the end of a draw, so a 4–10 K range, with the charging ΔT separated from it (F-09), is more adequate. Pathway counts versus approach [CALC]: 2 K → 39, 4 K → 32, 6 K → 26, 8 K → 24 Path A candidates.
- **Path B output temperature and boost:** T_pre = Tm − approach is the correct maximum outlet in the same limiting sense; preheat fractions of the 50 °C load 0.61–0.97 [CALC, previous round]. The auxiliary energy per 50 kg draw is 0.06–0.70 kWh. Consistent, but entirely dependent on an auxiliary that does not exist in the project.
- **Can Path B exceed CONDITIONAL without a defined auxiliary system?** **No.** [REC]
- Pathways are not merged, and MCDM cannot override feasibility status in the specification (V13).

---

## 10. Candidate classification and count audit

- **Database:** 58 candidates (31 manufacturer, 27 literature). **Path A 26, Path B 32** (reproduced from Tm ≥ 56). [CALC]
- **Counts:** FEASIBLE 0, CONDITIONAL 51, UNVERIFIED 5, REJECTED 2 in each cluster. Independently reproduced with fresh code and 0 mismatches. [CALC]
- **Rejection reasons:** RT42 (Tm 40.5 °C < 42 °C); Myristic acid/NBR-1.0 (R_wm 0.03/0.33/0.16 at the favourable corner; capacity ratio 0.93). UNVERIFIED: two eutectics (ρ and k missing), three Paraffin/HDPE composites (L imputed).
- **Identical counts across clusters:** genuinely produced by the rules (F-14); not a copy-through error. The cluster-dependent information is in the nominal-pass flags (7 candidates differ by cluster) and in the R_wm values.
- **MCDM eligibility:** the methodology specifies FEASIBLE/CONDITIONAL only; no MCDM code was changed, so "follows the methodology" can only be judged on the specification, which it does.
- **Imputed-property promotion:** none is promoted to FEASIBLE (it requires Reported k, cycles, corrosion and safety evidence). But nominal-pass flags rest on imputed k (15/19/14) and imputed ρ (3 each) and the status rules are inconsistent (F-12).
- **Unsupported claims in the methodology document** (should be revised before any approval): "δ = 10–30 mm spans the Objective 2 range" (F-03); "median day of the worst month" (F-06); "Rajasthan-consistent threshold" (F-06); unqualified nominal-pass counts (F-07); the 74.7 kg figure without its provisional constraints (F-15); fatty-acid corrosion statement (F-13).
- **Missing columns:** source type and data-source detail (F-16). `corrosion_detail`, `safety_detail`, `flash_point_C`, `max_op_temp_C` are present.
- **Stale results:** the CSVs match the methodology document (appendix status counts equal CSV counts). They are, however, results of a model with the F-01/F-02 defects and must be regenerated after correction.
- **The 49-survivor defect:** prevented only by specification (F-20).

---

## 11. Capacity and latent-heat calculations [CALC]

Formulas: per-draw heat E_d = M_w c_p (50 − T_mains); daily PCM duty D = 2 · s · E_d (two draws, PCM share s); required specific latent heat at usable fraction u and mass m: L_min = D/(m u); required mass for a given L: m_req = D/(L u). Units: M_w kg, c_p = 4.186 kJ/kgK, T °C, E in kJ, L in kJ/kg.

**Worked example, 100 L/day (50 + 50 kg), C0 mean mains 17.2 °C, s = 0.5, u = 1, m = 50 kg:**
E_d = 50 · 4.186 · (50 − 17.2) = 6 865 kJ; D = 2 · 0.5 · 6 865 = 6 865 kJ = 6.87 MJ; L_min = 6 865/50 = **137.3 kJ/kg**; p95 (cold) day (T_mains 8.7 °C): D = 8 640 kJ, L_min = 172.8 kJ/kg. Specification check: 0.5 · 100 · 4.186 · (50 − 19.96)/50 = 125.75 kJ/kg, which reproduces the Phase 4 value 125.74 (regional-mean mains).

**Worked example, 200 L/day (100 + 100 kg), same conditions:** E_d = 13 730 kJ; D = 13 730 kJ = 13.73 MJ; L_min = **274.6 kJ/kg** at 50 kg, 183.8 kJ/kg at 74.7 kg; p95 day: 345.6 and 231.3 kJ/kg. Required mass at L = 200 kJ/kg: 68.7 kg (mean day) and 86.4 kg (p95 day).

| Basis | s | m (kg) | L_min mean / p95 (kJ/kg) | Candidates below L_min mean / p95 (of 58) |
|---|---|---|---|---|
| 100 L/day | 0.4 | 50 | 109.8 / 138.2 | 0 / 1 |
| 100 L/day | 0.5 | 50 | 137.3 / 172.8 | 1 / 14 |
| 100 L/day | 0.5 | 74.7 | 91.9 / 115.7 | 0 / 0 |
| 200 L/day | 0.4 | 50 | 219.7 / 276.5 | 35 / 58 |
| 200 L/day | 0.4 | 74.7 | 147.1 / 185.1 | 2 / 19 |
| 200 L/day | 0.5 | 50 | 274.6 / 345.6 | 58 / 58 |
| 200 L/day | 0.5 | 74.7 | 183.8 / 231.3 | 18 / 42 |

Geometric mass bound: reproduced independently, **74.74 kg inline** (t = 51.2 mm, N = 108) and **68.76 kg staggered** (t = 43.9 mm, N = 137) for ρ = 910 kg/m³; a typical Objective 2 design (t = 40 mm, N = 100) is 41.3 kg. Capacity-based rejection as written in methodology §4.3 (reject only if the largest constructible mass at share 0.4 cannot cover the p95 day) was **not implemented in the classification script**; applied at the 200 L/day basis with 74.7 kg it would reject the 19 candidates below 185.1 kJ/kg. The status therefore depends on the demand basis and on this unimplemented rule. Specific latent heat and total usable storage are separate: the same L passes or fails depending on mass, demand and share, so a fixed kJ/kg threshold cannot be derived without the system definition (A5).

---

## 12. Rajasthan comparison (consolidated)

| Issue | Rajasthan source | Same issue in Rajasthan? | Validated practice or inherited assumption? |
|---|---|---|---|
| Stefan film / carry-over | no Stefan; constant H_p = 800 W/m²K lumped model (`physics_lib.py`) | n/a (different model); constant H_p ignores front growth, optimistic | inherited Barqawi constant from another rig |
| Collector | stagnation-style T_c = T_a + 0.7 G/U_L, U_L calibrated to hit a benchmark; A_c = 4 m²; no tilt | yes (unmeasured; calibrated to targets) | inherited, calibrated |
| Reliability metric | none; solar-fraction benchmark calibration | no R_wm | — |
| Demand | 300 kg/day, delivery 60 °C (re-based 2026-09-13), T_mains = T_a − 2 (TODO) | system-specific | inherited |
| PCM share | SHARE 0.5 (0.4–0.78), "Option B: ranking criterion not gate" | same second-hand citations | inherited literature |
| κ | fixed-κ run κ = 0.7 (4/7/5 survivors); **κ-calibrated companion** κ = 0.5/0.6/0.5, 8–20 band (9/14/16 and 9/15/17) | count-targeting in the calibrated run | **not validated; not to be copied** |
| Charging ceiling | worst-month clearness, area-agnostic | yes | inherited heuristic |
| Safety/corrosion | flag-only | no strict exclusion on Yes/No | closer to the methodology's rule |
| Provenance guards | `provenance_lib` fingerprints, `09` cross-phase precondition | no | adoptable practice |

Rajasthan counts are historical outputs of runs whose inputs this audit cannot reproduce; they are not used as targets, and no parity with Rajasthan is claimed. Parity for Assam: the methodology's replay of the Rajasthan rules reproduces the 2026-10-04 frozen Assam run (identical inputs by SHA-256) and nothing more.

---

## 13. A1–A10 decision matrix (independent recommendations)

| ID | Topic | Recommendation | Justification | Evidence needed to resolve |
|---|---|---|---|---|
| A1 | Separate Path A and Path B | **APPROVE WITH CONDITIONS** | Different jobs; mixing labels would be wrong. Conditions: Path A upper bound (F-11) and freezing-temperature boundary; define the flow arrangement and outlet constraint (F-08); Path B never above CONDITIONAL. | system arrangement; T_max,system |
| A2 | 6 K approach | **APPROVE WITH CONDITIONS** | Defensible as the existing nominal; separate it from the charging ΔT (F-09); use 4–10 K sensitivity. | heat-exchanger/UA basis |
| A3 | R_wm ≥ 0.5 | **HOLD** | Convention only, mislabelled, January-dominated, 50 % failure fraction (F-06). | a duty/reliability requirement from Objective 2 (for example solar-fraction target) or a decision to keep R_wm as a diagnostic |
| A4 | Nominal and favourable design | **HOLD** | A = 4 m² is Rajasthan's (F-04); δ = 20 mm is outside Objective 2's capsules (F-03); the favourable corner confounds six factors (F-14). | chosen reference system; capsule-diameter-to-δ mapping; corrected model |
| A5 | Demand basis | **HOLD** | Outcome changes materially; Objectives 1 and 2 differ (§7). Recommended: screen on material properties, report both demands, hand 200 L/day results to Objective 2. | your decision on what Objective 1 represents |
| A6 | Safety classification | **APPROVE WITH CONDITIONS** | Flash-point-based conditional class is reasonable; "Yes" alone is not an exclusion. Conditions: cite the GHS line, define the margin, require SDS/enclosure assessment for any PASS (§8.5). | SDS for each shortlisted product; safety reviewer margin |
| A7 | Corrosion and containment | **APPROVE WITH CONDITIONS** | UNKNOWN-capped is right; remove or cite the fatty-acid claim; specify the containment before any PASS. | containment specification; coupon/compatibility evidence |
| A8 | Four-status classification | **APPROVE WITH CONDITIONS** | Structure is sound; fix the imputed/missing rule (F-12) and regenerate after model correction. | corrected model |
| A9 | Database property additions | **APPROVE** | Flash point, maximum operating temperature and volume expansion exist in the source and were dropped; add them, plus source type, SDS reference and closed-cup method. | none |
| A10 | Auxiliary boost for Path B | **APPROVE WITH CONDITIONS** | Cap at CONDITIONAL until an auxiliary arrangement is defined (agree with the methodology). | auxiliary device definition |

---

## 14. Required methodology corrections (all require your approval before implementation)

1. Replace the scalar-plus-snapshot Stefan scheme by the explicit interface stack or a 1-D enthalpy solution; remove the carry-over reset (F-01).
2. Converge the time step to ≤ 0.01 R_wm; report Δt (F-02).
3. Map capsule diameter to the equivalent slab δ (R/2) or use the cylindrical geometry; add R_ext (F-03).
4. Use the system's own collector area as nominal; keep other areas as sensitivity (F-04).
5. Fix the demand basis and PCM-share treatment (A5, F-05); do not inherit 0.5 unqualified.
6. Correct the R_wm wording and either justify or demote the threshold; report month-year variability (F-06).
7. Report nominal passes as "with reported k" and "with imputed k" (F-07).
8. Add the outlet-temperature (ε–NTU) condition and a flow-arrangement definition for Path A (F-08).
9. Separate the approach from the charging driving difference (F-09).
10. Collector: sensitivity ranges per §5.6; tilt and a₂ both reported (F-10).
11. Apply the Path A upper bound and freezing-temperature boundary in classification (F-11).
12. One imputed/missing-property rule (F-12).
13. Cite or remove the unreferenced safety/corrosion statements; define the safety margin (F-13).
14. State that equal cluster counts are a property of the gates; report cluster-specific values (F-14).
15. Qualify the 74.7 kg bound (F-15); add provenance columns (F-16); remove dead code and report the floor (F-17); test spin-up (F-18).
16. Implement the capacity-based rejection rule or delete it from the methodology (§11).

## 15. Unresolved questions and unavailable evidence

Which system Objective 1 represents; the flow arrangement and location of the PCM heat exchanger; the auxiliary heater definition; measured collector parameters; measured PCM–water conductance; SDS and coupon-test evidence; the containment specification; the Objective 2 reliability or solar-fraction requirement; Rajasthan's original per-run outputs and the 62- versus 55-row database discrepancy; the text of the PDF literature in `Sources/pdfs/`.

## 16. Recommended next steps

1. You decide A5 and the reference system (A4), and whether R_wm stays as a diagnostic (A3).
2. You approve specifying and testing the corrected Stefan formulation of §4.4 in an isolated scratch/new-directory study (no production change), including the time-step and geometry tests.
3. Regenerate the sensitivity and status tables from the corrected model and re-issue the methodology as a new revision for review.
4. Only then request implementation approval, with the test plan V1–V18 plus the added tests: carry-over (two consecutive draws without charging), outlet constraint, geometry mapping, and rule-consistency tests for imputed properties.

## 17. Files and processes left untouched

- **Modified:** none. **Created:** this report only (scratch calculations are in the session scratchpad, outside the project).
- **Not modified:** the methodology document and the audit addendum, all assets, all production scripts and configs, the candidate database, historical results, Assam ERA5 preprocessing and GMM clustering outputs, Rajasthan, Tamil Nadu and Uttarakhand.
- **Phase B:** not run. **Stage E:** I did not start, stop, restart or alter any Stage E process or file. By the time of this report the five Δt = 7.5 s runs (E1–E5) had finished on their own at 12:24 and their output files exist; I did not analyse them, because that would create new project outputs and this task is read-only.
