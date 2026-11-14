# Assam Objective 1 — Corrected Multi-Front Stefan Formulation: Isolated Study

**ISOLATED RESEARCH STUDY. No production script, dataset, configuration, historical output or existing methodology file was modified. Phase B was not run. Rajasthan, Tamil Nadu and Uttarakhand were not touched. Nothing here is approval to implement anything.**

Written for: the project owner, who authorised this study (item D of the decision message) and must decide what, if anything, is adopted.

Evidence labels: **[FACT]** read from source/data; **[CALC]** independent calculation by this study; **[MODEL]** result of a screening model (not experimentally validated); **[ASSUME]** assumption; **[OPEN]** unresolved.

Scratch code and outputs (outside the project): `C:\Users\hp\AppData\Local\Temp\claude\0189d894ab42dcbb\de02d978-3298-4895-a1f7-5f4bfe890c04\scratchpad\` — `stefan_study.py` (corrected model), `run_stefan.py`, `analyze_stefan.py`, `tol_study.py`, `enthalpy_bench.py`, `audit_core.py` (current model + explicit stack reference); results in `obj1_audit\stefan_study_*.csv/json`, `enthalpy_bench.json`.

---

## 1. Executive summary

1. **Defect confirmed and isolated [CALC].** Making the frozen shell persist between the evening and morning draws (the only change between the `reset_slab` and `persist_slab` runs) lowers the mean daily full-duty reliability R_all by **0.20–0.24** and the mean morning coverage by 0.02–0.03 (max 0.16) in every cluster and configuration. It flips candidates across R_wm = 0.5 (C0, audit nominal: PlusICE A52 0.52 → 0.00, Myristic acid C14 0.53 → 0.00).
2. **The new analytic implementation reproduces the old structure when told to reset [CALC].** The `reset_slab` option agrees with the current vectorised model at Δt = 112.5 s to a mean |ΔR_wm| of 0.0004–0.0064 (max 0.0066–0.0395), and at 900 s to a max of 0.04–0.12; so the corrected code is not a different model by accident, and the old model's remaining distance from it is step error.
3. **Time step.** The corrected formulation is integrated **exactly in time** (event-based closed forms, §3.4), so it has no Δt. The current model's step error is real: R_wm differs by up to 0.115 between 900 s and 112.5 s. The explicit multi-front reference does **not** converge to the analytic result in R_wm (§5): the 0.999 full-coverage test is a hard threshold applied to a quantity with step error ≈ 10⁻³, so R_all/R_wm flip. Reliability metrics built on a hard 0.999 test are not numerically robust (§6).
4. **Conduction resistance dominates [MODEL].** Replacing the Stefan front by the lumped constant-coefficient treatments used by Assam's own physics validation (h = 150 W/m²K) and Rajasthan's `physics_lib` (800 W/m²K scaled by k) gives identical, far higher reliability: those coefficients are never binding at these fluxes, so those treatments are energy/collector-limited. Pass counts differ by a factor 2–3 between the treatments; the heat-transfer model is the largest model-form uncertainty.
5. **Geometry and external resistance matter.** Cylindrical geometry (equal-energy equivalent) lowers mean R_all from 0.475 to 0.290 (audit nominal, C0); adding the Objective 1 coefficient h = 150 W/m²K as an external resistance lowers it to 0.419.
6. **Objective 1 reference configuration (A = 2 m², 50 kg, A_pcm = 2.5 m²) [MODEL].** With the corrected Stefan model **no candidate reaches R_wm ≥ 0.25 in C0 and C2 and one Path A candidate does in C1** (0 at 0.5, 0.75, 0.9). Under the lumped treatments 19–28 candidates pass 0.5. Which behaviour is right depends on an unmeasured PCM–water conductance.
7. **Quasi-steady validity [CALC].** A 1-D enthalpy finite-volume benchmark agrees with the quasi-steady front to 0.5 % (Ste = 0.07, start at T_m) and 3.4 % (Ste = 0.2), but to 14 % and 10 % when the PCM starts 10 K sub-cooled/super-heated: sensible heat is neglected and the screen is optimistic by that order in those states.
8. **Nothing is ready to adopt.** The formulation is validated against closed forms, an explicit reference and an enthalpy benchmark only. It remains unvalidated against experiment.

---

## 2. Verified facts about the current implementation [FACT]

(Line numbers: `ASSAM_OBJECTIVE1_CORRECTED_FEASIBILITY_METHODOLOGY_assets/obj1_method_analysis.py`.)

- State: liquid fraction x per element (line 168, initial 0.5); no layer arrangement.
- Charging: film φ_c = max((x − x_ref)δ, s_min), x_ref reset at 08:00 (lines 189, 194); rate A_face k_l ΔT_app/φ_c (195).
- Discharging: shell φ_d = max((x_r − x)δ, s_min), x_r reset at the start of **every** draw (lines 177, 180); rate A_face k_s (T_m − SC − T̄_w)/φ_d.
- s_min = 0.5 mm, Δt = 900 s, explicit (165); success `cov ≥ 0.999` on both draws (200); spin-up 5 days (201).
- Rajasthan (read-only): `physics_lib.py` line 479–480 (`SURFACE_TO_VOLUME_RATIO_PER_M = 100`, `H_P_WM2K_BASE = 800`), 683–684 (`Ap = Vp·S/V`, `Hp = 800·k_s/0.2`), 765 (`dQ = Hp·Ap·max(0, Tw − Tm)·dt`): constant-coefficient lumped melting, no moving-front resistance, no layer memory.
- Assam Objective 1 physics (`10_physics_validation.py`): (UA)_pcm = 375 W/K (h = 150 W/m²K, A = 2.5 m²), A_c = 2.0 m², PCM 50 kg, tank water 100 kg.

---

## 3. Corrected formulation (specification)

### 3.1 Variables and units
| Symbol | Meaning | Unit |
|---|---|---|
| v | PCM volume per unit capsule-wall area (layer "thickness" in energy-equivalent measure) | m³/m² = m |
| v_tot = m/(ρ_s A_face) | total of all layers (δ for a slab; R_pcm/2 for a cylinder) | m |
| layers | ordered list (phase, v) from the wall inward; alternating phases; Σv = v_tot | — |
| ρL | volumetric latent heat | J/m³ |
| a | imposed flux cap: collector Q_col/A_face (charge) or duty/(A_face·3600 s) (discharge) | W/m² |
| ΔT | driving difference: ΔT_app (charge) or T_m − SC − T̄_w (discharge) | K |
| R_ext | wall + convective resistance per wall area | m²K/W |
| k | k_l (charge film) or k_s (discharge shell) | W/mK |

### 3.2 Resistance of the wall-adjacent layer (thickness v)
- Slab: R(v) = v/k.
- Cylinder, melting/freezing inward from the outer wall, total v_tot: R(v) = −(v_tot/k) ln(1 − v/v_tot) (limit v/k for v ≪ v_tot; derived from R_cyl = ln(R/r)·R/(2k) with v = R·s − s²/2 per unit wall area).

### 3.3 Dynamics of the wall layer
Flux q = min(a, ΔT/(R(v) + R_ext)); advance dv/dt = q/(ρL), consuming the neighbouring layer of the other phase.
- **Flux-limited regime** (ΔT/(R(v)+R_ext) ≥ a): dv/dt = a/ρL, valid until R(v*) = ΔT/a − R_ext; slab v* = k(ΔT/a − R_ext).
- **Resistance-limited regime:** Φ(v) + R_ext v = Φ(v₀) + R_ext v₀ + ΔT τ/(ρL), with Φ(v) = ∫₀ᵛ R dv: slab v²/(2k); cylinder (v_tot²/k)[(1−w)ln(1−w) + w], w = v/v_tot. Slab closed form v = k[−R_ext + √(R_ext² + 2C/k)]; cylinder by bracketed bisection.
- **Events:** when the neighbouring layer is exhausted, the wall layer absorbs it and the next same-phase layer (thickness added), the front disappears, and a new neighbour appears; zero-thickness layers are removed.
- **Phase change at the wall:** if the wall-adjacent phase is the wrong one, a zero-thickness layer of the needed phase is created there; its resistance starts at R_ext, so **no numerical floor is needed**.
- **Persistence:** the layer list is never reset; the shell frozen by a draw stays until a charge melts it; the morning draw starts with the evening shell. Phase fronts therefore persist across hours and days.

### 3.4 Time integration
Within each hour G, T_a and the duty rate are constant, so each hour is integrated exactly by the closed forms above; no time step exists. Hours: draws at 07:00 and 19:00 rows; charging rows 08–18 with G ≥ 10 W/m² (unchanged from the current model so differences are attributable to the formulation).

### 3.5 Energy ledger
Liquid volume V_L = Σ v_liquid. Over a run ΔV_L must equal Σ(v converted by charging) − Σ(v converted by discharging). Delivered heat = ρL A_face Δv.

---

## 4. Validation checks

### 4.1 Closed-form and explicit checks of the integrator [CALC]
ρL = 1.8×10⁸ J/m³, k = 0.2 W/mK, ΔT = 6 K, 3600 s, from zero film:

| Test | Analytic code | Independent value | Relative difference |
|---|---|---|---|
| Resistance-limited slab, √(2kΔTt/ρL) | 6.92820 mm | 6.92820 mm | 1.6×10⁻¹⁴ |
| Flux-limited slab, a = 100 W/m² | 2.000 mm | 2.000 mm | 0 |
| Regime switch, a = 500 W/m² | 6.49923 mm | 6.49923 mm | 0 |
| Slab + R_ext = 0.0067 m²K/W vs explicit 0.5 s steps | 5.19605 mm | 5.19618 mm | 2.4×10⁻⁵ |
| Cylinder, v_tot = 10 mm vs explicit 0.5 s steps | 5.22694 mm | 5.22708 mm | 2.6×10⁻⁵ |
| Layer exhaustion and merge | single liquid layer, converted 1.000 mm | expected | exact |

### 4.2 Energy conservation [CALC]
Maximum relative residual |ΔV_L − (Σ charge − Σ discharge)|/Σ charge over all 2 016 evaluable (configuration × model × cluster × candidate) 10-year runs: **1.7×10⁻¹⁴**. (The current model's residual was 0.0075 J in absolute terms.)

### 4.3 New code versus current model with reset enabled [CALC]
`reset_slab` (shell measured from the start of each draw, no floor) versus the current vectorised model, all 56 evaluable candidates per cluster:

| Configuration | Cluster | Δt = 900 s: mean / max \|ΔR_wm\| | Δt = 112.5 s: mean / max \|ΔR_wm\| |
|---|---|---|---|
| Audit nominal (A = 4 m², δ = 20 mm) | C0 / C1 / C2 | 0.022 / 0.102; 0.020 / 0.118; 0.020 / 0.076 | 0.0049 / 0.040; 0.0064 / 0.032; 0.0043 / 0.026 |
| Objective 1 reference (A = 2 m², A_pcm = 2.5 m²) | C0 / C1 / C2 | 0.0033 / 0.069; 0.011 / 0.105; 0.0033 / 0.040 | 0.0004 / 0.0066; 0.0025 / 0.032; 0.0004 / 0.0099 |

The remaining distance at fine step reflects the 0.5 mm floor and residual step error of the old model.

### 4.4 Quasi-steady assumption versus a 1-D enthalpy solution [CALC]
Finite-volume enthalpy solution, 100 cells, half-thickness 20 mm, ρ = 900, L = 200 kJ/kg, c_s = 2.0, c_l = 2.2 kJ/kgK, isothermal phase change, 3600 s, wall temperature fixed:

| Case | Ste | Front (enthalpy) | Front (quasi-steady) | Ratio |
|---|---|---|---|---|
| Melt, ΔT = 6 K, start at T_m (k_l = 0.1) | 0.066 | 4.876 mm | 4.899 mm | 0.995 |
| Melt, ΔT = 6 K, start 10 K below T_m | 0.066 | 4.212 mm | 4.899 mm | 0.860 |
| Freeze, ΔT = 20 K, start at T_m (k_s = 0.2) | 0.20 | 12.217 mm | 12.649 mm | 0.966 |
| Freeze, ΔT = 20 K, start 10 K above T_m | 0.20 | 11.381 mm | 12.649 mm | 0.900 |

Quasi-steady is accurate when the PCM sits at T_m and optimistic by 10–14 % when sensible heat must be supplied or removed. The screen does not track PCM sensible temperature, so it is optimistic in those states. [OPEN: the distribution of such states in the real cycle.]

---

## 5. Time-step comparison (900, 225, 112.5 s)

### 5.1 Current vectorised model [CALC]
Maximum |R_wm(900 s) − R_wm(112.5 s)| over all 56 evaluable candidates: audit nominal C0/C1/C2 0.079 / 0.115 / 0.066; Objective 1 reference 0.063 / 0.082 / 0.033. (From the earlier audit, C0 audit nominal: 450 s 0.039, 225 s 0.016 versus 112.5 s.) The step error is not negligible against any threshold.

### 5.2 Explicit multi-front stack versus the exact analytic solution [CALC]
Eight sentinel candidates × 3 clusters, audit nominal, persistent shell, floor 0.5 mm unless noted. Differences are explicit minus analytic.

| Δt (s) | floor | max \|ΔR_wm\| | mean ΔR_wm | max \|ΔR_all\| | mean ΔR_all |
|---|---|---|---|---|---|
| 900 | 0.5 mm | 0.349 | −0.012 | 0.088 | −0.040 |
| 225 | 0.5 mm | 0.122 | −0.048 | 0.092 | −0.049 |
| 225 | 0.05 mm | 0.122 | −0.048 | 0.092 | −0.049 |
| 112.5 | 0.5 mm | 0.122 | −0.053 | 0.091 | −0.050 |

The result does not converge towards the analytic value as Δt falls from 225 to 112.5 s, and the floor (0.5 vs 0.05 mm) changes nothing. A day-by-day comparison (savE OM49, C0, Δt = 225 s) shows the **mean** coverage agreeing to 0.0012 (morning) and 0.0004 (evening), but R_all 0.927 (explicit) vs 0.948 (analytic): the difference arises on days whose coverage lies within about 10⁻³ of the 0.999 acceptance level. The persistent difference is therefore a **metric-threshold sensitivity**, not an error in the dynamics. [INFER, supported by §6.]

### 5.3 Corrected analytic model
Independent of Δt by construction; there is nothing to converge, and its validity rests on §4.1–§4.4.

---

## 6. Sensitivity of the reliability metric to the coverage tolerance [CALC]

Corrected analytic model, mean R_all (and R_wm ≥ 0.5 counts Path A/Path B) as the success tolerance on both draws varies:

| Configuration | tol 0.95 | 0.99 | 0.999 (current) | 0.9999 |
|---|---|---|---|---|
| Audit nominal, C0 mean R_all | 0.583 | 0.510 | 0.475 | 0.466 |
| Audit nominal, C0 counts at R_wm ≥ 0.5 | 9 / 7 | 7 / 7 | 7 / 7 | 7 / 7 |
| Audit nominal, C1 mean R_all; counts | 0.583; 13 / 7 | 0.519; 13 / 7 | 0.486; 12 / 7 | 0.476; 12 / 7 |
| Audit nominal, C2 mean R_all; counts | 0.595; 8 / 7 | 0.527; 6 / 7 | 0.492; 6 / 7 | 0.481; 6 / 7 |
| Objective 1 reference, all clusters mean R_all | 0.37–0.39 | 0.32–0.33 | 0.29–0.30 | 0.28–0.30 |
| Objective 1 reference counts at 0.5 | 0 | 0 | 0 | 0 |

R_all moves by 0.1 across the range of tolerances, and the explicit schemes' step error is of the size of the 0.999 margin. [REC] report mean coverage and energy-weighted shortfall alongside the binary success fraction, and state the tolerance.

---

## 7. Effect of each formulation change

### 7.1 Persistent versus reset shell (analytic, same everything else) [CALC]
| Configuration | Cluster | mean ΔR_all (reset − persist) | mean / max ΔR_wm | mean / max Δ morning coverage |
|---|---|---|---|---|
| Objective 1 reference | C0 | 0.200 | 0.007 / 0.227 | 0.030 / 0.161 |
| | C1 | 0.193 | 0.027 / 0.548 | 0.026 / 0.130 |
| | C2 | 0.206 | 0.010 / 0.293 | 0.030 / 0.145 |
| Audit nominal | C0 | 0.236 | 0.040 / 0.533 | 0.024 / 0.162 |
| | C1 | 0.209 | 0.053 / 0.647 | 0.020 / 0.148 |
| | C2 | 0.227 | 0.042 / 0.549 | 0.022 / 0.164 |

### 7.2 Geometry and external resistance [CALC] (mean R_all; C0 / C1 / C2)
| Model | Audit nominal | Objective 1 reference |
|---|---|---|
| Persistent slab | 0.475 / 0.486 / 0.492 | 0.291 / 0.293 / 0.304 |
| + R_ext = 1/150 | 0.419 / 0.427 / 0.433 | 0.231 / 0.233 / 0.244 |
| Persistent cylinder (equal-energy) | 0.290 / 0.302 / 0.310 | 0.137 / 0.158 / 0.158 |
| Reset slab (current structure) | 0.711 / 0.694 / 0.718 | 0.491 / 0.486 / 0.510 |
| Lumped h = 150 W/m²K, A_pcm 2.5 m² (Assam O1 physics) | 0.879 / 0.839 / 0.874 | 0.770 / 0.739 / 0.757 |
| Lumped 800·k/0.2 W/m²K, A = 5.88 m² (Rajasthan treatment) | identical to the line above | identical |

The two lumped treatments coincide because neither limits the flux below the collector or duty rate (flux ≥ 900 W/m² at ΔT = 6 K); they behave as infinite conductance. The slab and cylinder rows are the only ones where conduction limits the result. [MODEL]

### 7.3 Numerical lower bound
The analytic scheme has no floor and none is needed (§3.3). In the explicit scheme, floors of 0.5 and 0.05 mm give identical results at Δt = 225 s (§5.2), while 5 mm changed R_wm by up to 0.18 in the earlier audit; a floor of ≤ 2 mm is immaterial, as the audit found. The omitted R_ext is the more important lower-resistance term (a 1/150 m²K/W resistance equals a 1.3 mm film at k = 0.2).

---

## 8. Candidate-count changes [MODEL]

Number of candidates with R_wm ≥ threshold, Path A (Tm ≥ 56 °C; 26 candidates) / Path B (Tm < 56 °C; 30 evaluable), clusters C0 / C1 / C2. **The thresholds are shown as a sensitivity only; none is an engineering requirement (decision C).**

### 8.1 Audit nominal (A = 4 m², m = 50 kg, δ = 20 mm)
| Model | ≥ 0.25 | ≥ 0.50 | ≥ 0.75 | ≥ 0.90 |
|---|---|---|---|---|
| Current model, 900 s | A 11/16/12; B 10/12/11 | A 9/13/8; B 9/11/9 | A 2/0/0; B 6/5/6 | A 0/0/0; B 2/0/2 |
| Current model, 112.5 s | A 10/15/11; B 10/11/10 | A 7/13/7; B 9/11/9 | A 0/0/0; B 6/4/6 | A 0/0/0; B 2/0/2 |
| Reset slab (analytic) | A 10/15/11; B 10/11/10 | A 7/13/7; B 9/11/9 | A 0/0/0; B 6/3/6 | A 0/0/0; B 1/0/1 |
| **Persistent slab (corrected)** | A 9/15/10; B 7/7/7 | A 7/12/6; B 7/7/7 | A 0/0/0; B 3/1/3 | A 0/0/0; B 0/0/1 |
| Persistent + R_ext = 1/150 | A 7/10/8; B 3/7/7 | A 3/8/3; B 2/7/2 | A 0/0/0; B 0/0/1 | A 0/0/0; B 0/0/0 |
| Persistent cylinder | A 2/5/2; B 0/0/0 | A 0/4/0; B 0/0/0 | A 0/0/0; B 0/0/0 | A 0/0/0; B 0/0/0 |
| Lumped h = 150 (≡ Rajasthan treatment) | A 23/24/23; B 23/28/25 | A 22/23/22; B 21/25/23 | A 15/2/8; B 20/12/21 | A 0/0/0; B 8/0/8 |


### 8.2 Objective 1 reference (A = 2 m², PCM 50 kg, A_pcm = 2.5 m², R_ext = 0 unless stated)
| Model | ≥ 0.25 | ≥ 0.50 | ≥ 0.75 | ≥ 0.90 |
|---|---|---|---|---|
| Current model, 900 s | A 0/2/0; B 1/3/1 | A 0/0/0; B 0/2/0 | 0 | 0 |
| Current model, 112.5 s | A 0/2/0; B 0/2/1 | A 0/0/0; B 0/2/0 | 0 | 0 |
| Reset slab | A 0/1/0; B 0/2/1 | A 0/0/0; B 0/1/0 | 0 | 0 |
| **Persistent slab** | A 0/1/0; B 0/0/0 | 0 | 0 | 0 |
| Persistent + R_ext, cylinder | 0 | 0 | 0 | 0 |
| Lumped h = 150 / Rajasthan-style | A 8/21/6; B 23/28/25 | A 0/9/0; B 19/25/16 | A 0/0/0; B 4/0/1 | 0 |

### 8.3 Reading the changes
With the Stefan resistance and the Objective 1 reference collector, practically no candidate delivers its full duty on the median-month basis: the collector area (2 m² for 50 kg, 0.04 m²/kg) is the limit in January, not the material. The corrected persistent-shell model lowers the audit-nominal pass counts at 0.5 from A 9/13/8, B 9/11/9 (900 s model) to A 7/12/6, B 7/7/7; the reset model at fine step is already A 7/13/7, B 9/11/9, so about half of the audit's earlier 18 → 13 shift came from time-step error and half from the structural change.

---

## 9. Monthly success fractions and the worst month [MODEL]

Persistent slab. The worst calendar month is January for 55 (C0), 38 (C1) and 56 (C2) of 56 evaluable candidates in the audit nominal configuration (C1: February for 18; C0: December for 1). Objective 1 reference: January 55/50/56, December 1/6/0.

Monthly fractions Jan…Dec (R_wm = minimum):

| Config | Candidate | Cluster | Monthly success fractions | R_all | R_wm (month) |
|---|---|---|---|---|---|
| Audit nominal | savE OM49 | C0 | .87 .92 .95 .94 .97 .95 .97 .98 .97 .94 .98 .93 | 0.948 | 0.868 (Jan) |
| | savE OM49 | C1 | .85 .75 .85 .85 .87 .90 .93 .98 .98 .93 .92 .94 | 0.897 | 0.749 (Feb) |
| | n-Hexacosane | C0 | .74 .85 .96 .95 .98 .94 .95 .98 .98 .94 .98 .88 | 0.929 | 0.74 (Jan) |
| | RT64HC | C1 | .78 .66 .83 .80 .82 .85 .85 .94 .95 .89 .90 .88 | 0.847 | 0.657 (Feb) |
| Obj 1 reference | savE OM49 | C0 | 0 0 0 .21 .55 .80 .85 .91 .78 .18 0 0 | 0.358 | 0 (Jan) |
| | n-Hexacosane | C0 | 0 .06 .75 .84 .83 .80 .86 .93 .86 .76 .39 0 | 0.594 | 0 (Jan) |
| | RT64HC | C1 | .29 .46 .69 .72 .70 .77 .75 .86 .87 .79 .82 .43 | 0.680 | 0.293 (Jan) |

At the Objective 1 reference, the winter months (Nov–Feb) fail completely for most candidates; the annual fraction hides it. These tables, not a threshold, are the reportable result until a reliability requirement exists.

---

## 10. Rajasthan formulation (read-only comparison)

| Aspect | Rajasthan `physics_lib.py` | Corrected Assam formulation |
|---|---|---|
| Melting law | Q̇ = H_p A_p max(0, T_w − T_m) with H_p = 800·k_s/0.2 W/m²K, A_p = 100 m⁻¹·V (lines 479–480, 683–684, 765) | moving-front conduction through the wall-adjacent layer (+ R_ext) |
| Front memory | none (state is stored energy Q_p and T_p) | persistent layer list |
| Geometry | none (area ratio only) | slab or cylinder |
| Parameter origin | Barqawi 2025 rig values, scaled by k | candidate k, ρ, L; geometry from design |
| Status | validated only against a solar-fraction benchmark by calibration of the collector, not of H_p | validated against closed forms, explicit reference, enthalpy benchmark only |

Same vulnerabilities: unmeasured PCM–water coupling; no sensible-heat/Ste-number test; no experimental validation. Rajasthan avoids the carry-over defect only by assuming a constant coefficient that is never binding at these fluxes (§7.2), which is the optimistic limit. It must not be copied.

---

## 11. Assumptions, limits and unresolved questions

- [ASSUME] Slab/cylinder with wall at the PCM temperature plus ΔT_app (charge) and mean water temperature (discharge); a single wall-adjacent resistance; no natural convection; no density change; supercooling only through the discharge ΔT.
- [ASSUME] The same duty definition (share 0.5, two 50 kg draws, mains from the medoid weather) and charging window (08:00–18:00) as the audited model, so only the formulation differs.
- [OPEN] Hot-end outlet-temperature condition for Path A (audit F-08) is not addressed here.
- [OPEN] Sensible heat and sub-cooled/superheated states (§4.4: up to 14 % front error).
- [OPEN] Cylinder implementation is the equal-energy equivalent with inward melting from the outer wall; real capsules have wall conduction, headspace and a fill fraction.
- [OPEN] PCM–water coupling coefficient, collector parameters and the capsule-to-δ mapping remain unmeasured.
- [OPEN] A reliability requirement (decision A3) and the success tolerance (§6).

## 12. Files and processes left untouched
No project file modified; this report is the only file created in the project (together with the Stage E review). No Phase B run, no Stage F, no Stage E rerun. Rajasthan, Tamil Nadu and Uttarakhand untouched.
