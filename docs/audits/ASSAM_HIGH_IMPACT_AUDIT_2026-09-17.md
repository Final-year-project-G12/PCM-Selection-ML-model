# Assam — High-Impact Audit (2026-09-17)

Scope: `era5-assam/*.py`, `docs/assam/*.md` (00 through 24, plus
`20_IMPLEMENTATION_ISSUES.md` and `DOCUMENTATION_CONSISTENCY_REPORT.md`). Read-only
audit; nothing in this pass changed any file.

**Scope note:** Assam has no `pcm_shared_config.py`-style shared constants file —
its constants are genuinely hardcoded per-script, as the project's root CLAUDE.md
claims. Several of those hardcoded values (Findings 1 and 2 below) diverge from
Rajasthan's corrected methodology with no cited justification for the divergence.

**Not flagged as new (already correctly documented and still open):** the K=4-vs-K=3
dual-chain design, Monte Carlo skip, and corrosion-veto inertness are already
correctly documented as known/resolved and match the code.
`n_confirmed=[0,0,0]` with 1 conditional survivor (C24) was verified in
`07_feasibility_filter_final.py`; Phase 7/8 NOT PERFORMED/SKIPPED was confirmed;
`DOCUMENTATION_CONSISTENCY_REPORT.md`'s own 2026-09-14 correction note (that
`04_preprocess` never reads `bias_decision_assam.txt`) was verified still true in
current code.

---

## 1. Sign-flipped PCM melting-temperature target formula — CONFIRMED, likely root cause of the "NOT PHYSICALLY SUPPORTED" Phase 10 finding

**Files:** `era5-assam/04b_climate_signature.py:57` —
`TM_TARGET = T_DELIVERY - DT_APPROACH` (= 50 − 6 = **44°C**) — confirmed by direct
read. Repeated via a hardcoded `Tm_target_C=44.0` in `05_cluster_assam.py:311-312`,
`05b_swh_design_specification.py`, `07_feasibility_filter*.py`, and
`08_mcdm_ranking*.py`.

Assam sets the PCM's target melting point *below* the delivery temperature (44°C
target for a 50°C delivery). The project's own corrected Rajasthan/Tamil Nadu
formula (root CLAUDE.md §3.2, confirmed directly in `pcm_shared_config.py:52`) is
`TM_TARGET_C = T_DELIVERY_C + DT_APPROACH_C` — melting point *above* delivery,
which is the physically correct direction: a PCM must melt above the temperature
threshold it needs to keep water above, so its discharge stays hot enough after
heat-exchanger losses.

Assam's own `docs/assam/23_PHASE_10_AUDIT.md` independently rediscovers this exact
physics via brute-force simulation — finding that OM48 (Tm=51°C, above delivery)
hugely outperforms RT44HC (Tm=43°C, below delivery, the MCDM's Gaussian-fitness #1
pick) — but frames this as a "profound thermodynamic finding about
decision-criteria design," explicitly stating "not a code defect."

**Why high-impact:** given the target formula's sign is backwards relative to the
correctly-implemented Rajasthan/Tamil Nadu version, and directly contradicts the
physics the same audit doc describes, this reads far more like a sign bug than a
genuine methodological insight — no citation or derivation is given for Assam's
sign choice anywhere. If it is a bug, it invalidates every downstream Tm-based
screening/ranking step (Phase 6 melting window, Phase 7 MCDM Gaussian fitness) for
the entire K=4 chain.

**Fix direction:** recompute with `TM_TARGET = T_DELIVERY + DT_APPROACH` (≈56°C)
to match the rest of the project, or explicitly justify (with a cited
heat-exchanger derivation) why Assam's approach-temperature sign should differ —
and if kept as-is, correct `23_PHASE_10_AUDIT.md`'s "not a code defect" framing to
reflect that it is asserted, not demonstrated.

---

## 2. Assam's `L_required` omits the SHARE_PCM fractional-share correction applied project-wide

**Files:** `04b_climate_signature.py:205`, `05_cluster_assam.py:291-294` —
`L_required = M_draw * Cp * (T_delivery - T_mains) / M_PCM`, i.e. 100% of the full
night-discharge sensible deficit is assigned to PCM latent heat alone.

Per the project's root CLAUDE.md §3.1, this exact formulation (PCM latent heat
alone supplying 100% of night load) was identified as a root-cause bug in
Rajasthan and corrected to `L_required = SHARE_PCM * Q_night / m_PCM` with
`SHARE_PCM=0.5` (cited: Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021).
Assam's `docs/assam/06_PHASE_4_AUDIT.md` never mentions SHARE_PCM or these
citations, and the code shows no such factor — Assam appears to have inherited
the pre-fix Rajasthan formula.

**Why high-impact:** this produces artificially high `L_required` floors
(252–280 kJ/kg) and is a plausible contributor — alongside the data-completeness
gating the docs already cite — to Phase 6's `n_confirmed=[0,0,0]` result across
all three regimes.

**Fix direction:** apply the same literature-anchored SHARE_PCM fraction used in
Rajasthan for cross-state methodological consistency, or explicitly document/cite
why Assam's basis should differ.

---

## 3. Doc/code mismatch on collector heat-loss coefficient in the "Authoritative Final" Phase 9 audit

`docs/assam/10_PHASE_9_AUDIT.md:21` states the collector loss coefficient
a1 = 3.8 W/m²K. The actual code (`10_physics_validation.py:13,63,105`, both the
docstring and the live constant `FR_UL_WM2K = 4.5`) uses 4.5 W/m²K consistently.
The doc number does not match the code that supposedly produced it — in a document
marked "Status: COMPLETE (Authoritative Final)."

**Fix direction:** correct the doc to 4.5 W/m²K (or confirm which value the 24
reported simulation runs actually used and fix whichever side is wrong).

---

## 4. Assam's physics constants are uncited/uncalibrated, unlike Rajasthan's literature-grounded calibration pass

`10_physics_validation.py:60-67` labels `FR_TAU_ALPHA=0.72`, `FR_UL_WM2K=4.5`,
`UA_TANK_WK=1.0`, `UA_PCM_WK=375.0` all as "assumed" with only generic "typical
FPC" ranges, no citations, and no calibration against a literature solar-fraction
benchmark (contrast Rajasthan's joint-calibration-to-54–84%-band exercise,
CLAUDE.md §3.3/§3.4). This is honestly disclosed in code comments, but
`10_PHASE_9_AUDIT.md` presents the 24 simulation runs as "rigorous" and
"Authoritative Final" without surfacing that caveat prominently — a reader of the
docs alone would not know these are uncalibrated assumptions.

**Fix direction:** either run the same citation-anchored calibration Rajasthan
did, or add an explicit uncertainty caveat to `10_PHASE_9_AUDIT.md`.
