# Tamil Nadu — High-Impact Audit (2026-09-17)

Scope: `era5-tamilnadu/*.py`, `docs/tamilnadu/*.md`, `era5-tamilnadu/outputs/*`,
`pcm_shared_config.py`. Read-only audit; nothing in this pass changed any file.

**Not flagged (already correctly handled):** `LATENT_HEAT_FRACTION` /
`LATENT_HEAT_ABSOLUTE_MIN_KJ_KG` being defined locally in `config.py` (not shared)
is intentional and documented as Phase-5-only, not part of the cross-state basis.
`SHARE_PCM`, `ASSUMED_PCM_MASS_KG`, `TM_TARGET_C`, `PCA_N_COMPONENTS` are all
correctly imported from `pcm_shared_config.py` everywhere except the one hardcode
in Finding 1 below.

---

## 1. CONFIRMED — `T_DELIVERY_C` mismatch between Phase 3 and Phase 7/10 (50°C vs 60°C)

**Files:** `pcm_shared_config.py:50` (`T_DELIVERY_C = 60.0`) vs.
`era5-tamilnadu/10_physics_validation.py:164` (`T_DELIVERY_C = 50.0`, a local
hardcode) — directly confirmed by reading both files. Its own docstring
(~line 56) — "Target delivery temperature 50°C … same T_delivery used throughout
this pipeline's Tm_target rule" — is itself now false. By contrast,
`04b_climate_signature.py` and `11_seasonal_pcm_sensitivity.py` correctly
`from config import T_DELIVERY_C` (→ 60.0, `TM_TARGET_C` → 67.0).

**Why high-impact:** Phase 3 now sizes the PCM melting-point target and
`L_required` around a 67°C delivery target (`Q_night` computed at a 60°C rise),
while Phase 7/10's grey-box physics simulator still validates delivery success
(`if Tw >= T_DELIVERY_C`, line 333) and demand energy (line 337) against 50°C — a
17°C-lower bar. This makes the simulated "solar fraction" and "delivery hours met"
trivially easier to hit than what the material was actually selected for, silently
inflating Phase 7's apparent success and invalidating any comparison between
MCDM-predicted suitability and simulated performance (the Spearman ρ reported in
`docs/tamilnadu/09_PHASE_7_AUDIT.md` and `recommendation_cards.md`). This is
exactly the mismatch flagged as a known TODO in the project's root CLAUDE.md
(§3.2) — confirmed still unfixed here.

**Fix direction:** replace the local `T_DELIVERY_C = 50.0` in
`10_physics_validation.py` with the shared import (`from config import
T_DELIVERY_C`), re-derive `DRAW_MASS_KG`/demand accordingly, and re-run
Phase 7 → 09 → 11.

---

## 2. On-disk Tamil Nadu outputs are stale relative to current code/config — predate even the pre-T_DELIVERY_C pipeline unification

**Files:** `era5-tamilnadu/data/processed/signatures/climate_signature_tamilnadu.csv`
(last modified 2026-09-08, `Tm_target_C=57.0` — the pre-fix value),
`data/processed/pcm/recommendation_cards.md` (shows **5 clusters**, Tm_target=57.0°C,
L_required≈299 kJ/kg — but current docs describe Phase 4 as yielding k=3 clusters
post-unification, so this file predates even that step).

**Why high-impact:** anyone reading `recommendation_cards.md` or citing
survivor/ρ numbers from `docs/tamilnadu/*` today is citing numbers computed under
an already-superseded climate-signature regime (k=5, Tm_target=57°C), not the
current 60/67°C, k=3, unified-Phase-5 pipeline. The docs (`00_PHASE_5_AUDIT.md`,
`12_FINAL_READINESS_REPORT.md`) flag "re-run pending" for the unification, but
nobody has flagged that the *subsequent* T_DELIVERY_C fix (2026-09-13) is a
second, additional reason the current artifacts are invalid —
`docs/tamilnadu/05_PHASE_3_AUDIT.md:52` still states "Tm,target = 50.0+7.0 =
57.0°C" as the current documented value, un-updated.

**Fix direction:** full clean re-run of the CORE chain per
`12_FINAL_READINESS_REPORT.md`'s recommended order, after fixing #1 and #3;
update `05_PHASE_3_AUDIT.md`'s stated Tm_target/T_delivery numbers.

---

## 3. `11_seasonal_pcm_sensitivity.py` still uses a fixed `LATENT_HEAT_FRACTION=0.7` floor instead of each cluster's calibrated kappa — the exact bug already found and fixed in Rajasthan

**File:** `era5-tamilnadu/11_seasonal_pcm_sensitivity.py:78,111` —
`LATENT_HEAT_FRACTION = 0.7` hardcoded, applied via
`latent_heat_floor_kj_kg(l_required, LATENT_HEAT_FRACTION)` inside
`rank_seasonal()`, independent of Phase 5's per-cluster `calibrated_kappa` values
that the survivor pool it re-ranks was actually filtered with.

**Why high-impact:** the project's root CLAUDE.md (§3.4) documents this precise
bug for Rajasthan ("re-filtering each cluster's already-calibrated survivor pool
with a hardcoded LATENT_HEAT_FRACTION=0.7, stricter than every cluster's own
Phase 5 calibrated kappa … fixed to use each cluster's own calibrated kappa") and
the fix was never ported to Tamil Nadu. TN's current L_required (~300 kJ/kg) is
low enough this may not yet zero out survivors, but once Finding 1 propagates
(T_DELIVERY_C=60 → higher L_required, mirroring Rajasthan's jump from ~300 to
~440 kJ/kg), this will very likely reproduce Rajasthan's exact "degenerate — <2
survivors per cluster-season" failure and silently erase the seasonal-sensitivity
finding.

**Fix direction:** mirror Rajasthan's fix — read each cluster's
`calibrated_kappa` from the Phase 5 kappa-calibrated survivors file and use it in
place of the fixed 0.7 constant.

---

## 4. Phase 7 tank/collector calibration diverges from Rajasthan's and misses the benchmark band entirely — self-disclosed but still unresolved

**Files:** `10_physics_validation.py:119-127` (`M_W_KG=150`,
`DRAW_MASS_KG=75×2=150 kg/day`, `A_C_M2=2.5`, `COLLECTOR_EFF=0.70`) vs.
Rajasthan's calibrated `physics_lib.py` (`M_W_KG=200`,
`COLLECTOR_UL_WM2K=2.0` after joint retuning). `docs/tamilnadu/09_PHASE_7_AUDIT.md`
self-reports **0 of 42 simulations land in the 54–84% published benchmark band**
(SF range 30–53%), explicitly flagged "TANK/COLLECTOR CALIBRATION DIVERGES FROM
RAJASTHAN."

**Why high-impact:** already self-disclosed as open (restated here, not newly
found), but worth keeping in the high-impact list because it's a standalone,
uncalibrated model — unlike Rajasthan's, which went through a documented joint
retune to land all medoids in-band. TN's Phase 7 ρ numbers (mean ≈0.62–0.65 in
the docs) are not trustworthy evidence of MCDM-physics agreement until
recalibrated, and this compounds with Finding 1 since both live in the same
script.

**Fix direction:** port Rajasthan's shared `physics_lib.py` tank/collector model
to Tamil Nadu (already flagged in the docs as "the next unification step") rather
than maintaining a second, divergent standalone implementation in
`10_physics_validation.py`.
