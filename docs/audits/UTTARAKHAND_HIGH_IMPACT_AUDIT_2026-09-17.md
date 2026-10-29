# Uttarakhand — High-Impact Audit (2026-09-17)

Scope: `era5-uttarakhand/*.py`, `docs/uttarakhand/*.md` (00 through 13, plus
`11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md` and
`CONSOLIDATION_SUMMARY.md`). Read-only audit; nothing in this pass changed any
file.

**Not flagged as new (already correctly documented and verified still fixed):**
the Tamil Nadu map-centre bug, `passes_all` filtering, the `deaccumulate()` bug,
TOPSIS/VIKOR bugs, the 07b regime-cap bug, and the elevation proxy were all
checked directly against current code and confirmed genuinely resolved. Also
already correctly flagged and still open in the existing docs (omitted here per
"don't repeat"): the `T_mains_est_C` −2.0°C offset being uncited, K=5 exceeding the
"realistically 2–4" recommendation, GMM soft-membership collapse, PCM database
imputation footprint, the monsoon_index JJAS/JJA mismatch, and the plot-layer
defects already catalogued in `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`.

---

## 1. Orchestrator (`run_all_uttarakhand.py`) skips the Phase-1 combine step it documents as required — pipeline fails on a fresh run

**File:** `era5-uttarakhand/run_all_uttarakhand.py` (~lines 181–194,
`CORE_SCRIPTS`). The module docstring lists `02_combine_uttarakhand.py` as step 1,
"the single input every later script ultimately traces back to." But in the actual
`CORE_SCRIPTS` list it is commented out:

```python
CORE_SCRIPTS = [
    # ("02_combine_uttarakhand.py", True),
    ("02b_build_daily_aggregates.py", True),
    ("04_preprocess_uttarakhand.py", True),
    ...
```

`04_preprocess_uttarakhand.py` reads `COMBINED_POINTS_FILE`
(= `climate_uttarakhand_points.csv`, `config.py:41`), which only
`02_combine_uttarakhand.py` produces.

**Why high-impact:** running `python run_all_uttarakhand.py` on a clean checkout
will crash at `04_preprocess_uttarakhand.py` with a `FileNotFoundError` — or, worse,
silently reuse a stale copy of that file if one happens to exist from a prior
manual run, producing results from an outdated combine step with no warning.
`12_FINAL_READINESS_REPORT.md` item 12 still frames "add a run_all script" as
future work, which is now out of date (one exists) and masks this bug.

**Fix direction:** uncomment the `02_combine_uttarakhand.py` line in
`CORE_SCRIPTS`.

---

## 2. Phase 7's physics validation uses half the domestic draw volume that Phase 3 uses to derive `L_required` — the two "independent" checks aren't testing the same system

- `04b_climate_signature.py:81`: `DRAW_VOLUME_L = 300.0` (300 L/day) — sizes
  `L_required_kJ_per_kg`, the value that gates PCM feasibility in Phase 5.
- `10_physics_validation.py:190` + docstring table: `DRAW_MASS_KG = 75.0`, drawn
  twice a day (07:00/19:00) = **150 L/day total** — exactly half.

`T_DELIVERY_C=50°C` is at least consistent between the two scripts, but the demand
magnitude is not.

**Why high-impact:** Phase 7 is meant to be an independent physical check on
whether the Phase 5/6-selected PCMs actually work ("this phase makes the claim
falsifiable" per the script's own docstring) — but it simulates a materially
smaller household than the one used to size the feasibility gate. This is not
flagged anywhere in `05_PHASE_3_AUDIT.md`, `09_PHASE_7_AUDIT.md`, or
`12_FINAL_READINESS_REPORT.md`. Given Phase 7 already reports solar fraction an
order of magnitude below benchmark (Finding 3), doubling the simulated draw to
match Phase 3's own assumption would make that shortfall worse, not better.

**Fix direction:** reconcile the two draw assumptions — pick one number and cite
it once in `config.py`, the way `SHARE_PCM` already is.

---

## 3. Uttarakhand never adjusts warm-climate assumptions (T_DELIVERY_C=50°C, no backup heater) for its own colder climate — and its own physics results are the smoking gun

Confirmed directly: `T_DELIVERY_C=50.0` appears unmodified in
`04b_climate_signature.py`, `10_physics_validation.py`, and
`11_level_b_seasonal_analysis.py` — all three hardcode their own local copy
rather than importing a shared constant, and none differ from Rajasthan's
*pre-correction* value (Rajasthan's own history raised this to 60°C for a
reason — Uttarakhand never revisited it, let alone in the colder-climate
direction that the project's root CLAUDE.md explicitly flags as needed). No
backup-heater term exists anywhere in `era5-uttarakhand/` (confirmed by grep —
zero matches for backup/auxiliary/immersion heating).

The pipeline's own Phase 7 result is direct evidence this matters:
`10_physics_validation.py`'s corrected run finds **0% of simulated PCM-cluster
pairs land in the 54–84% published solar-fraction benchmark band — actual
~12–19%, an order-of-magnitude shortfall** (`09_PHASE_7_AUDIT.md` §1,
`12_FINAL_READINESS_REPORT.md`). This is honestly reported as an open
"methodology question," but the code's own inline comment
(`10_physics_validation.py:182-186`) already half-answers it: `UA_TANK_W_K=2.0`
(tank-to-ambient loss) is "a plains/moderate-climate estimate… arguably too low,
not too high" for high-Himalaya clusters — meaning a climate-appropriate
correction would push solar fraction even further below benchmark, not fix it.

**Why high-impact:** nowhere does the documentation connect these two facts
(chronically low solar fraction + admittedly-too-low cold-climate loss term) to
the obvious implication: a 50°C-delivery, no-backup-heater domestic SWH system
may simply not be viable as modeled for Uttarakhand's colder clusters. The fix
path likely isn't recalibrating tank constants the way Rajasthan did, but adding
a backup-heater term or lowering the delivery-temperature target for cold
clusters. This is the single highest-impact open item for O1/O4's applicability
to Uttarakhand, and it currently only surfaces in the docs as a numeric benchmark
miss, not as a design problem.

**Fix direction:** explicitly decide and document either (a) a backup-heater term
for cold clusters, or (b) a lower, climate-appropriate delivery-temperature target
— before treating Uttarakhand's Phase 7 solar-fraction numbers as comparable to
the other three states'.

---

## 4. Stale/contradictory documentation: `12_FINAL_READINESS_REPORT.md` still lists SHARE_PCM as missing after `05_PHASE_3_AUDIT.md` confirms it's fixed

`12_FINAL_READINESS_REPORT.md` item 15 (under "Open, medium priority"): *"`L_required`
has no PCM fractional-contribution (`SHARE_PCM`) factor — the PCM is implicitly
assumed to supply the whole night load."* This is stale — `config.py` defines
`SHARE_PCM=0.5` and `04b_climate_signature.py:54` imports and uses it
(`sig["L_required_kJ_per_kg"] = (q_total_kJ * SHARE_PCM) / ASSUMED_PCM_MASS_KG`),
and `05_PHASE_3_AUDIT.md`'s own Stage 3 section explicitly marks this RESOLVED.

**Why high-impact:** low-to-medium impact by itself, but two files in the same
2026-09-dated audit set now disagree with each other, risking a future reader
"re-fixing" an already-fixed item or citing the wrong (unfixed) formula in the
paper.

**Fix direction:** update item 15 in `12_FINAL_READINESS_REPORT.md` to match `05`'s
resolved status.
