# Assam M3 Stage E — Read-Only Results Review

**READ-ONLY REVIEW of existing outputs in `Objective---2/objective2-assam/objective2-assam-fpc/results/m3_stageE/`. Stage E was not rerun, Stage F was not started, no Stage E file was modified, created or regenerated (the project's `run_m3_stageE.py --analyze` was deliberately not used because it writes an analysis file into that folder). No Phase B, no Rajasthan/Tamil Nadu/Uttarakhand access.**

Written for: the project owner, to decide whether to authorise Stage F.

Evidence labels: **[FACT]** read from the output files/configs; **[CALC]** independent recalculation by this review (scratch, outside the project: `scratchpad\stageE_review.py`, results `obj1_audit\stageE_review.json`); **[MODEL]** a simulation result (provisional, not validated); **[ASSUME]**; **[OPEN]**.

---

## 1. Verdict

**All five expected runs are present, complete, uncorrupted and internally consistent, and the declared Stage E acceptance checks are met. Nothing in the outputs blocks Stage F.** Whether to start Stage F is your decision. The limits that Stage F must carry as stated limitations are in §9: the time step is not demonstrated converged, the pending closure-defect and controller-reference decisions are unresolved, the drift metric is confounded by the weather trend, and the controller cycles about 9.5 times per day.

---

## 2. Inventory and integrity [FACT/CALC]

All 18 expected files exist (5 reference-resolution runs × {json, daily csv, samples csv}, plus the supplementary Δt = 30 s run of E1 × 3). No `*.FAILED.txt`, no analysis file, no orphan files; the only other files are the two run logs. The Δt = 7.5 s log lists all five runs `done` (elapsed 16 884–17 413 s, about 4.7–4.8 h each).

| File | Bytes | SHA-256 (first 16) |
|---|---|---|
| E1_M10_Nr20_dt7.5.json / _daily.csv / _samples.csv | 10 362 / 11 648 / 1 764 570 | 70e1e0e22282f509 / 8a2ca8604af087d5 / 776dc87d657e87de |
| E2 | 10 508 / 11 982 / 1 867 878 | 426f84ce26e919fa / 7850a76c74515dfc / 984451c203414bf6 |
| E3 | 10 243 / 11 523 / 1 745 148 | cd144154e6aa4b21 / 2e8c87d47860d68f / aec614f5f8d5fa29 |
| E4 | 10 365 / 11 640 / 1 770 094 | 2d78fc7510cecc99 / b0b0ce28bd57d9da / c592acb54a6fe467 |
| E5 | 10 370 / 11 636 / 1 768 654 | 4fce1a314200e4e2 / b1d49e04eefd3d40 / 079c4f9df5506472 |
| E1 (Δt = 30 s) | 10 374 / 11 636 / 1 765 619 | 554d337354c9f785 / 5d13cc463131c831 / 5c566046232b26b1 |

Integrity checks on every run (including E1 Δt = 30 s): JSON and CSV parse; 30 daily rows; 8 640 samples (30 d × 288 at 300 s); no NaN or non-finite value; time strictly increasing with a constant 300 s step; `n_steps_eval` 345 600 = 30 d × 86 400 s / 7.5 s (86 400 for Δt = 30 s); dates cover exactly the declared window; liquid fraction within [0, 1]; tank temperature within physical bounds. **No missing or corrupted output.**

Provenance: the SHA-256 of all six configuration files recorded in each run (bed system, system, thermal, evidence, tolerances, regimes) **equals the current file hashes**, so the runs used the configuration now on disk [CALC]. Each run records `solver: fast` (the optimised solver that Stage D showed bit-identical to the original), M = 10, N_r = 20, coupling A, status `PROVISIONAL_NOT_VALIDATED`.

Windows were re-derived independently from the weather with the declared rule: representative start **2020-10-17** (mean GHI 179.29 W/m², 23.25 °C) and stress start **2016-01-06** (123.70 W/m²), matching the run records [CALC].

---

## 3. Completion and spin-up [FACT]

| Run | PCM / window / initial state | Spin-up blocks | Converged | Verified periodic | Last block ΔT_tank (K) | Last block ΔE_bed (fraction of latent capacity) | End-of-spin-up tank (°C) |
|---|---|---|---|---|---|---|---|
| E1 | OM46, representative, A (45 °C) | 3 | yes | yes | −3.5×10⁻⁴ | 4.5×10⁻⁵ | 55.5023 |
| E2 | OM49, representative, A | 3 | yes | yes | 5.9×10⁻⁸ | 9.9×10⁻¹⁰ | 55.4411 |
| E3 | OM46, stress, A | 3 | yes | yes | 6.0×10⁻⁶ | 1.4×10⁻⁷ | 37.4564 |
| E4 | OM46, representative, B (30 °C) | 3 | yes | yes | −3.1×10⁻⁷ | — | 55.5023 |
| E5 | OM46, representative, C (65 °C liquid) | 3 | yes | yes | −2×10⁻⁸ | — | 55.5023 |

Declared limits: |ΔT| ≤ 0.05 K, |ΔE| ≤ 0.5 % of latent capacity, mode unchanged; maximum 6 blocks. All runs satisfy them with three blocks (the minimum: converge, then verify). The controller mode at the end of spin-up is `off` with the tank latch set for E1, E2, E4, E5. [Block-1 end states differ between E1, E4, E5 (55.5066, 55.5067, 55.5027 °C), so the different initial conditions were applied; see §7.]

---

## 4. Solver convergence and energy conservation [FACT/CALC]

Declared tolerances: per-step closure ≤ 1×10⁻⁹, cumulative ≤ 1×10⁻⁸.

| Run | Max per-step closure (tank / bed / pipe / loop / system) | Cumulative closure (max) | Pass |
|---|---|---|---|
| E1 | 4.0e-11 / 4.7e-12 / 4.3e-12 / 2.4e-10 / 7.1e-11 | 8.8e-13 | yes |
| E2 | 3.9e-11 / 5.0e-12 / 4.4e-12 / 2.9e-10 / 9.3e-11 | 1.8e-12 | yes |
| E3 | 3.7e-11 / 3.6e-12 / 2.6e-12 / 3.0e-11 / 2.5e-11 | 7.5e-13 | yes |
| E4 | 3.8e-11 / 4.5e-12 / 4.4e-12 / 2.4e-10 / 6.9e-11 | 9.0e-13 | yes |
| E5 | 3.8e-11 / 4.5e-12 / 4.4e-12 / 2.4e-10 / 7.0e-11 | 9.0e-13 | yes |
| E1 Δt = 30 s | 7.1e-12 / 9.6e-13 / 4.2e-12 / 5.0e-11 / 1.6e-11 | 6.1e-14 | yes |

**None of the Stage E runs contains a step with the closure defect found in Stage D** (7 of 72 Stage D runs exceeded the per-step limit; the largest Stage E per-step value is 2.9×10⁻¹⁰). The slice-commit defect remains an unresolved, unapproved item for Stage F's limitations, but it is not visible here.

Independent recomputation from the cumulative ledgers in each JSON (not from the stored closure numbers) [CALC]:
- Tank balance E_u − E_load − E_loss,T − E_sink − ΔU_T: relative residual 2.9–3.3×10⁻¹² of E_u (0.0024–0.0038 J on 10⁹ J).
- System balance (E_u + E_pump,el) − (E_load + E_loss,T + E_loss,bed + E_loss,pipe + E_pump,amb) − (ΔU_T + ΔU_bw + ΔU_pipe + ΔE_el): 2.5–2.9×10⁻¹² (3.7×10⁻¹⁴ for Δt = 30 s).
- Loop balance: 1×10⁻¹²–1×10⁻¹¹ relative to E_sink.
- Daily ledger rows sum to the cumulative ledger within 10⁻¹⁵ for E_u, E_load, E_sink, ΔU_T.
- Tank capacity implied by ΔU_T/ΔT_tank = **2.0930×10⁶ J/K in every run = 500 kg × 4 186 J/kgK** (independent of the code's own closure).
- Mean collector efficiency E_u/(5 m² · ΣGHI) = 0.491 (E1, E2, E4, E5), 0.529 (E3): plausible for a flat plate and not suspicious.

The stored closure values and the independent identities agree. **Energy conservation is verified to rounding level.**

---

## 5. Time-step consistency: Δt = 30 s versus 7.5 s for E1 [CALC]

Compared at the 8 640 common 300 s marks against the **declared Stage D resolution tolerances**:

| Metric | Δt 30 vs 7.5 s | Declared tolerance | Pass |
|---|---|---|---|
| T_tank max \|Δ\| | 0.548 K | 0.10 K | no |
| T_tank rms | 0.133 K | 0.03 K | no |
| T_return rms | 0.579 K | 0.20 K | no |
| SF | 1.2×10⁻⁴ (0.92626 vs 0.92613) | 0.001 | yes |
| E_over fraction | 3.3×10⁻⁵ | 0.001 | yes |
| Event-mean outlet max \|Δ\| | 0.340 K | 0.10 K | no |
| DS flips | 1 of 60 events | ≤ 3 | yes |
| Pump-on time | 2.5 % | 2 % | no |
| Mode switches | 284 vs 278 (6) | ≤ 2 | no |
| Bed heat exchange (E_sink) | 0.29 % | 2 % | yes |
| Liquid fraction max \|Δ\| | 0.049 | 0.02 | no |

This reproduces the Stage D conclusion [FACT: `STAGE_D_REPORT.md`]: no coarser step meets the declared tolerances. It also means the finest tested step (7.5 s) is **not shown to be converged**: a 0.55 K tank difference between 30 s and 7.5 s does not tell how far 7.5 s is from the limit. Integral metrics (SF, E_over) are insensitive; event-level and controller metrics are not. The Stage E results are therefore reference-resolution results, not converged results. [OPEN]

---

## 6. Controller consistency [FACT/CALC]

Modes found: `off`, `charge`, `discharge` only. Checked on the 300 s samples:
- Direct charge ↔ discharge transitions: **0** in all runs (off always intervenes).
- Samples in `discharge` with tank above 50 °C: **0** (stop threshold respected); tank never above 75 °C (maximum 68.54 °C in E1/E4/E5, 68.52 °C in E2, 49.20 °C in E3).
- Sample-level mode changes equal the reported `mode_switches` in every run (284, 268, 164, 284, 284).
- Pump-on time 0.90 h/day (E1, E2, E4, E5), 0.45 h/day (E3).
- Median dwell: charge 600 s, discharge 900 s, off 4 200–6 600 s; minimum dwell 300 s (the sampling interval; the control step is 7.5 s so shorter dwell is not visible).

**Concern [FACT]:** about **9.5 mode changes per day** (E1) with charge dwells of about 10 minutes is high-frequency cycling of a pump; it follows from the approved 5 K start / 2 K stop thresholds (provisional policy) and the collector signal, and may be unrealistic for a real pump. This is a controller-policy observation, not a numerical defect, and no threshold was changed.

---

## 7. Initial-condition independence and periodicity [CALC]

| Quantity | E4 (30 °C) vs E1 | E5 (65 °C liquid) vs E1 | Declared limit |
|---|---|---|---|
| ΔT_tank at end of spin-up | 3.4×10⁻⁶ K | 3.4×10⁻⁶ K | 0.05 K |
| ΔE_bed / latent capacity | 7.8×10⁻⁸ | 7.8×10⁻⁸ | 0.005 |
| ΔSF | 3.7×10⁻¹⁰ | 3.7×10⁻¹⁰ | 0.001 |
| ΔE_over fraction | 4.2×10⁻⁹ | 4.2×10⁻⁹ | 0.001 |
| Event-mean outlet max \|Δ\| | 2.9×10⁻⁶ K | 2.9×10⁻⁶ K | 0.10 K |
| DS flips | 0 | 0 | ≤ 3 |
| Mode switches difference | 0 | 0 | — |
| Max \|ΔT_tank\| over the whole evaluation series | 3.4×10⁻⁶ K | 3.4×10⁻⁶ K | — |

**The declared independence criterion is met.** Caution on its strength [INFER]: the representative window is sunny enough that the 500 L tank returns to about 55.5 °C within the first 14-day block from any of the three states, so the three runs converge almost completely by block 1 (their block-1 end temperatures differ by only 0.004 K). The test confirms forgetting of the initial state in this regime but does not probe a regime where memory is long (for example a window starting with the PCM bed in an intermediate state while the tank is limited by weather). It was not run for the stress window (E3 has one initial state). The 14-day spin-up block is recycled from the first 14 days of the evaluation window by design, so the state at the start of the evaluation corresponds to the end of day 14 of the same weather: a declared design choice (Phase 5B workflow).

Day-to-day drift over the evaluation (declared "reported" metric) [CALC]: end-of-day tank temperature slope −0.093 K/day (E1, E2, E4, E5) and −0.175 K/day (E3); bed stored energy slope +0.0065 (E1), +0.0083 (E2), −0.0037 (E3) of latent capacity per day; first-5-day versus last-5-day mean end-of-day tank 57.7 → 51.6 °C (E1). This trend follows the weather (mid-October to mid-November irradiance declines) and is **not** evidence that the numerical state is unsettled; the declared drift metric cannot separate the two because the forcing is not stationary. [OPEN: a stationary-forcing check was not part of the declared design.]

---

## 8. Results summary [MODEL]

| Run | SF | DS_event (07:00 / 19:00) | E_over fraction of ideal | Min / mean event outlet (°C) | Tank max (°C) | PCM end-of-day liquid fraction < 1 |
|---|---|---|---|---|---|---|
| E1 OM46 representative | 0.9263 | 0.550 (0.233 / 0.867) | 0.0940 | 36.1 / 50.66 | 68.5 | 5 of 30 days |
| E2 OM49 representative | 0.9265 | 0.550 (0.233 / 0.867) | 0.0944 | 36.2 / 50.69 | 68.5 | 6 of 30 |
| E3 OM46 stress | 0.6252 | **0.000** (0 / 0) | 0.0000 | 27.0 / 34.78 | 49.2 | 30 of 30 (liquid fraction ≤ 0.0037) |
| E4 / E5 | identical to E1 to 10⁻⁹ | identical | identical | identical | identical | 5 of 30 |

Observations: (i) OM46 and OM49 give nearly identical results (SF differs by 2.6×10⁻⁴): the PCM barely cycles because the tank sits near 55 °C and the bed stays liquid; (ii) in the stress window the bed **never melts** and no draw event reaches 50 °C (DS = 0), so this design and collector cannot meet the zero-tolerance 50 °C target in low-insolation winter conditions; (iii) the 07:00 draws succeed far less often than the 19:00 draws (0.23 vs 0.87). These are provisional numerical results for one design (D1) and cluster 0 with placeholder hydraulics and no experimental validation, and they are not findings about real hardware.

---

## 9. Does the output support proceeding to Stage F?

**Supported (declared acceptance checks):** complete and uncorrupted outputs; spin-up converged and verified in every run; per-step and cumulative closure within tolerance with an independent recomputation; configuration unchanged since the runs; initial-condition independence met; controller invariants respected; windows reproduce the declared rule.

**Must be stated as limitations in Stage F:**
1. The Δt = 30 s check fails the declared tolerances and 7.5 s is not shown converged (§5; Stage D temporal order not reached).
2. The closure-defect correction and the controller reference definition (volume-weighted versus unweighted) remain unapproved; Stage E steps happen to be free of the defect.
3. The drift metric is confounded by the weather trend; the independence test is weak outside sunny regimes.
4. High controller cycling (≈ 9.5 changes/day).
5. Stress-window result DS = 0 and PCM never melting; E1 and E2 nearly identical (PCM contribution small in this design).
6. Only cluster 0, one design (D1), two PCM entries; Δt = 30 s run only for E1; no experimental validation; hydraulics are placeholders.
7. The M2 all-points temporal-slope failure and the single xfail remain open (unchanged by Stage E).

I did **not** start Stage F. Awaiting your explicit approval.

## 10. Files and processes left untouched
No file in `results/m3_stageE/` or elsewhere in the project was modified; no process was started or stopped for Stage E; this report is the only Stage E-related file created (in `era5-assam/`). Rajasthan, Tamil Nadu and Uttarakhand untouched; no Phase B.
