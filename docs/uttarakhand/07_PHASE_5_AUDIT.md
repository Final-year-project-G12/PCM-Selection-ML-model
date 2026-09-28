# 07 — Phase 5 Audit: PCM Database & Feasibility Filtering

**Scripts**: `PCM_data/PCM_data/01_preprocess.py`, `06_build_pcm_database.py`,
`07b_charging_feasibility.py` (optional), `07_feasibility_filter.py`

**Status**: **COMPLETE.** The PCM source CSVs are among the very few data files actually committed
in `era5-uttarakhand/`, so this phase is the most directly verifiable in the whole pipeline.

---

## PCM property cleaning — `PCM_data/PCM_data/01_preprocess.py`

### Method

MICE (chained-equation) imputation with a **Random Forest per column**, refined by **Predictive
Mean Matching**, so "every filled value is a REAL, previously-measured value donated from the most
physically-similar PCM — never a synthetic average."

```python
IN_PATH      = data/PCM_Properties_55records_42_70C_dense.csv
OUT_LEAN     = data/PCM_Properties_cleaned_mice_pmm.csv
OUT_DETAILED = data/PCM_Properties_cleaned_mice_pmm_detailed.csv
N_ITER       = 8      # MICE refinement rounds
N_DONORS     = 3      # PMM donor pool size per missing cell
RANDOM_STATE = 42
```

The design rationale targets a specific failure mode: several properties are missing across an
entire product line, so a naive nearest-neighbour fill has no donor. MICE+RF+PMM avoids it because
"the Random Forest for a given column trains ONLY on rows where that column is actually observed —
regardless of which product line they belong to", and PMM then ranks donors by closeness of *model
predictions*, not raw feature distance.

Every imputed numeric cell is donor-logged, "so cross-series borrowing can be verified directly in
the output rather than taken on faith."

> **Documentation lag:** the docstring's worked example describes a dataset of "10/10" Rubitherm RT
> rows and "8/8" Pluss savE/OM rows — an 18-row database. `IN_PATH` points at the 55-record file.
> The narrative is stale; the code paths are current.

### Diagnostics produced (committed)

`PCM_data/PCM_data/data/`:
- `01_missingness_before_after.png`
- `02_cross_series_donor_audit.png`
- `03_imputed_vs_reported_sanity.png`
- `04_correlation_heatmap.png`
- `05_imputation_provenance.csv` (168 KB — the per-cell donor log)

---

## `06_build_pcm_database.py` — candidate database

### Input path resolution

```python
INPUT_CSV = PROCESSED_DIR.parent.parent / "PCM_data" / "data" /
            "PCM_Properties_cleaned_mice_pmm_detailed.csv"
```

`PROCESSED_DIR` = `era5-uttarakhand/data/processed`, so `.parent.parent` = `era5-uttarakhand/`.
The resolved path is `era5-uttarakhand/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv`
— which **exists** (34,909 bytes).

> `README.md` describes this as expecting "`PCM_data/` as a sibling folder **of this pipeline**",
> but the code resolves to a **child** folder of `era5-uttarakhand/`. The code is what runs; the
> README sentence is imprecise. Note also that the repository contains the file at **two** paths —
> `PCM_data/data/…` (the one `06` reads) and `PCM_data/PCM_data/data/…` (the cleaner's own output
> directory) — with identical size.

### Composition (verified directly against the committed CSV)

**55 rows × 59 columns.**

| Manufacturer | Rows |
|---|---|
| Literature | **24** |
| Rubitherm Technologies | 14 |
| Pluss Advanced Technologies | 7 |
| PureTemp | 5 |
| PCM Products Ltd. | 4 |
| CrodaTherm | 1 |
| **Total** | **55** |

31 manufacturer rows + 24 literature rows — exactly as `06`'s docstring claims.

| `pcm_type` | Rows |
|---|---|
| Organic (RT-line) | 14 |
| Organic n-alkane | 11 |
| Organic | 7 |
| Organic PCM | 5 |
| Organic fatty acid | 4 |
| Organic bio-based PCM | 4 |
| Organic/composite blend | 3 |
| Organic blend | 3 |
| Organic/polymer blend | 2 |
| Organic commercial PCM | 1 |
| Organic/eutectic composite | 1 |

**Every row is organic.** There are no salt hydrates, no eutectic salts, and no inorganic PCMs of
any kind in the Uttarakhand database. This has a direct consequence in `07`: `corrosion_class` is
derived as `"check_manually" if "Inorganic" in pcm_type else "low_organic"`, so it evaluates to
`low_organic` for all 55 rows and carries zero discriminating information.

### Property ranges (verified)

| Property | Range across the 55 rows |
|---|---|
| `Tm_melting` | **40.5 – 70.0 °C** |
| `latent_heat_melting` | **128 – 260 kJ/kg** |
| Rows inside the 42–70 °C absolute band | **54 of 55** (one row at 40.5 °C falls below) |
| Rows inside the [52, 65] °C melting window at `Tm_target = 57` | **29** |
| `cycles_tested_status` = "Reported by manufacturer" | **7** |
| `cycles_tested_status` = "Estimated via MICE-RF-PMM" | **48** |
| `flammability` = "Yes" | 45 |
| `flammability` = "No" | 10 |

### Imputation footprint (verified from the `*_imputed` flag columns)

**618 of 1,045** flagged property cells (55 rows × 19 flagged properties) were imputed — **59.1 %**.
**All 55 rows** carry at least one imputed property, so `any_property_imputed` is `True` for every
candidate.

| Property | Rows imputed | Property | Rows imputed |
|---|---|---|---|
| `Tm_melting` | **0** | `TC_liquid` | 34 |
| `latent_heat_melting` | **3** | `TC_solid` | 39 |
| `density_solid` | 14 | `TC_both` | 36 |
| `density_liquid` | 14 | `cycles_tested` | 48 |
| `Cp_liquid` | 22 | `flammability` | 48 |
| `Cp_solid` | 24 | `appearance` | 48 |
| `Tm_freezing` | **29** | `volume_expansion` | 48 |
| `Tm_nucleation` | 54 | `max_op_temp` | 34 |
| `latent_heat_freezing` | 43 | `flash_point` | 39 |
| `heat_storage_Wh_kg` | 41 | | |

This is the single most important caveat for Phases 5 and 6:

- **`Tm_melting` is never imputed** (0/55) and **`latent_heat_melting` is imputed for only 3/55** —
  the two properties that drive the melting-window filter and the latent-heat floor are almost
  entirely measured.
- **`TC_W_mK` — an MCDM ranking criterion — is derived as `(TC_liquid + TC_solid)/2`, and those two
  columns are imputed for 34 and 39 of 55 rows respectively.**
- **`cycles_confidence` — another MCDM ranking criterion — derives from `cycles_tested`, imputed
  for 48 of 55 rows.**
- **`supercooling_K = Tm_C − Tm_freezing_C` — a feasibility filter — depends on `Tm_freezing`,
  imputed for 29 of 55 rows.**
- **`rho_H_MJ_m3` — an MCDM criterion — depends on `density_solid`/`density_liquid`, imputed for
  14 of 55 rows.**

### Column mapping and derived properties

```python
out["Tm_C"]                = df["Tm_melting"]
out["latent_heat_kJ_kg"]   = df["latent_heat_melting"]
out["TC_W_mK"]             = (df["TC_liquid"] + df["TC_solid"]) / 2.0   # prefers per-phase
                                                                        # average over TC_both
out["supercooling_K"]      = out["Tm_C"] - out["Tm_freezing_C"]
out["n_properties_imputed"]= df[[c + "_imputed" for c in IMPUTABLE_PROPS]].sum(axis=1)
out["any_property_imputed"]= out["n_properties_imputed"] > 0
out["source"]              = "literature_MICE_RF_PMM_completed"      if manufacturer=="Literature"
                             else "manufacturer_datasheet_MICE_RF_PMM_completed"

rho_H_MJ_m3      = density_solid.fillna(density_liquid) * latent_heat_kJ_kg / 1000
Cp_avg_kJ_kgK    = mean of Cp_liquid/Cp_solid with mutual fillna
cycles_confidence= log1p(cycles_tested) / log1p(max_cycles)          # NaN where cycles unknown
in_absolute_band = Tm_C.between(42.0, 70.0)
corrosion_class  = "check_manually" if "Inorganic" in pcm_type else "low_organic"
```

Family labels are assigned from `manufacturer` (Rubitherm RT, PLUSS savE, PCM Products, PureTemp,
CrodaTherm) or, for literature rows, from `pcm_type` (n-Alkane, Fatty acid, Composite, Blend,
Polymer blend, Eutectic composite, Organic PCM, Bio-based PCM, Commercial PCM, Organic, and
"Organic (RT-line)" -> Rubitherm RT).

### Output

`data/processed/pcm/pcm_database_uttarakhand.csv`, sorted by `Tm_C`. Git-ignored — not committed.

---

## `07b_charging_feasibility.py` — optional regime-dependent Tm cap

### What it is for

The docstring is unusually candid, and the honesty note belongs in any write-up verbatim:

> This is a **HEURISTIC PROXY, not a real collector thermal model.** A rigorous version needs the
> cluster's 5th-percentile daily insolation fed through an actual collector efficiency curve
> (`eta_th = F_R[S − U·(T_in − T_amb)/G]` …) — that's Phase 7 territory, not something to
> improvise here under deadline pressure.

Its purpose is to break the constant-`Tm_target` degeneracy: "without it, every cluster shares the
same constant Tm_target and the same feasibility window, so every cluster gets an identical
survivor list (see 07's output — you'll have seen this if you ran it before this script)."

### Method — CONFIRMED BUG, RESOLVED (2026-09)

```python
REFERENCE_GOOD_DAY_TEMP_C = 70.0     # stated assumption, not measured
MIN_ACHIEVABLE_TEMP_C     = 42.0
POOR_DAY_Z                = 1.28     # ~5th percentile under a normal approximation

poor_day_kt        = (kt_mean - 1.28 * kt_std).clip(lower=0.05)
reliability_ratio  = (poor_day_kt / kt_mean).clip(0, 1)      # <-- THE BUG (see below)
achievable_temp    = 42 + reliability_ratio * (70 - 42)
Tm_target_C_regime_capped = min(Tm_target_C, achievable_temp)
```

**This formula was, in fact, run — this section's own inference below ("evidence points to not
run") was the closest possible diagnosis without seeing the actual numbers, but the real cause was
different.** `reliability_ratio = poor_day_kt / kt_mean` algebraically collapses to
`1 - Z*(kt_std/kt_mean)` — a coefficient-of-variation measure depending only on each cluster's
*relative* day-to-day scatter, not its *absolute* clearness level. `kt_mean`'s absolute value
cancels out of the ratio almost entirely, so `achievable_temp` landed within ~1C of the same value
for every cluster regardless of how sunny or cloudy it actually was — which is exactly why the
script always printed "0/5 clusters where the regime cap actually lowers Tm_target," every time it
was run, not because it wasn't run. **Fixed** by using `poor_day_kt` directly instead of the ratio:
`achievable_temp = 42 + poor_day_kt * (70 - 42)`. Post-fix, 2/5 clusters get a real, differentiated
cap (Cluster 1: 55.16C, Cluster 2: 56.51C).

The 70 °C ceiling is described as "a generic collector-physics ceiling, not a Uttarakhand-specific
number", cited as "roughly consistent with Al-Mamun2023's cited FPC 25-100C operating band". This
is the **only substantive external literature citation anywhere in `era5-uttarakhand/`'s pipeline
code**.

The script also records an explicit Uttarakhand-specific reasoning step: "if anything, this state's
higher-altitude clusters see LESS reliable clear-sky access than the plains (more cloud/fog
persistence, not less), which is exactly what kt_mean/kt_std already capture per cluster."

### Side effect

It **overwrites `cluster_profiles_uttarakhand.csv` in place**, adding `poor_day_kt_estimate` and
`Tm_target_C_regime_capped`. `07_feasibility_filter.py` then silently prefers the capped column if
present:

```python
tm_target = (prof["Tm_target_C_regime_capped"]
             if "Tm_target_C_regime_capped" in prof.index else prof["Tm_target_C"])
```

**RESOLVED (2026-09) — `07b` WAS being run; the inference above was the best possible read of the
symptom, but the actual cause was the normalization bug described above, not a skipped step.** Now
that the bug is fixed, `Tm_target_C_regime_capped` genuinely differs from `Tm_target_C` for
Clusters 1 and 2, and `07`'s survivor sets are no longer identical across all five clusters (see
"Survival rate" below).

---

## `07_feasibility_filter.py` — the feasibility filter

### Why filtering precedes ranking

> This matters because MCDM is compensatory — a PCM with an unreachable melting point but great
> latent heat can still score well in TOPSIS and be physically useless. Filtering first prevents
> that.

### Constants

```python
ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX  = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7   # RESOLVED (2026-09): now imported from config.py rather than
                             # hardcoded locally (was numerically identical either way; removes
                             # future drift risk if config.py's value ever changes)
CYCLES_FLOOR         = 300
SUPERCOOLING_MAX_K   = 8.0
MIN_SURVIVORS, MAX_RELAX_STEPS, RELAX_STEP_K = 5, 4, 2.0
```

### Filters actually applied

| # | Filter | Rule | Missing-data policy |
|---|---|---|---|
| 1 | Melting window | `Tm` in `[Tm_target − 5, Tm_target + 8]` -> **[52, 65] °C** at `Tm_target = 57` | — |
| 2 | Absolute band | `Tm` in `[42, 70] °C` | — |
| 3 | Latent-heat floor | `L >= 0.7 × L_required` for that cluster | — |
| 4 | Cycling stability | `cycles_tested >= 300` where reported | **retained and flagged** where unknown — "absence of data is not evidence of failure" |
| 5 | Supercooling veto | `abs(supercooling_K) <= 8 K` where known | NaN passes through flagged, not excluded |

### Filters explicitly NOT applied (from the script's own docstring)

> NOT applied (need data this project doesn't have yet — flagged as future work, not silently
> skipped):
> - Charging feasibility at the cluster's 5th-percentile insolation day (needs a full daily GHI
>   percentile per cluster, not just the mean in `cluster_profiles_uttarakhand.csv`)
> - Corrosion veto against cluster HSI 75th percentile (needs a real `corrosion_class` per PCM;
>   the database currently only distinguishes "low_organic" vs "check_manually" for the one
>   inorganic PCM)
> - Safety exclusion (no toxicity data in the current database)

Note the parenthetical about "the one inorganic PCM": the 55-row database this pipeline actually
consumes contains **zero** inorganic rows, so even that residual distinction is inert.

### Auto-relaxation

If a cluster keeps fewer than `MIN_SURVIVORS = 5`, the melting window is widened by
`RELAX_STEP_K = 2 K` and retried, up to `MAX_RELAX_STEPS = 4` (i.e. up to +8 K). If a cluster keeps
more than 25, that is reported but **not** narrowed — "Phase 6's ranking is what should separate
them."

Status reported per cluster: `OK` for 5–25 survivors, `LOW` for < 5, `HIGH` for > 25.

### Output shape — an important detail

`07` writes **every** PCM × cluster row, not just survivors:

```python
result = filter_cluster(pcm_db, tm_target, l_required, window_relax=relax)   # all 55 rows
result.insert(0, "cluster_id", cid); …; all_rows.append(result)
full = pd.concat(all_rows, ignore_index=True); full.to_csv(OUT_FILE)
```

`feasibility_survivors_by_cluster.csv` therefore contains **55 × 5 = 275 rows**, each carrying
per-filter booleans (`pass_melting_window`, `pass_absolute_band`, `pass_latent_heat`,
`pass_cycling`, `pass_supercooling`), the aggregate `passes_all`, and the window bounds
(`window_lo`, `window_hi`, `window_relax_applied`, `latent_heat_floor_used`).

The docstring calls this "the per-filter pass/fail detail kept alongside for your methodology
section's survivor-count table" — a deliberate design choice. **Consumers must filter on
`passes_all`.** `08_mcdm_ranking.py` and `09_recommendation_cards.py` do. Four Objective 1 plots
and `verify_03_feasibility.py` do not (see
`11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`).

---

## Observed Phase 5 results

### Confirmed from committed artefacts

`data/plots/verify_feasibility/06_summary.png`:

```
Total Survivors: 275
Number of Clusters: 5
Avg Survivors per Cluster: 55.0
  Cluster 0: 55 PCMs   Cluster 1: 55 PCMs   Cluster 2: 55 PCMs
  Cluster 3: 55 PCMs   Cluster 4: 55 PCMs
```

These are **row counts, not survivor counts** — the verification script counted every row of the
survivors file (which carries every candidate with pass/fail flag columns, not just the passers).
This section originally reported that `05_pcm_survivors_per_cluster_interactive.html`'s bars
encoded a flat 55-per-cluster because `generate_objective1_plots.py`'s `p05()` used
`df.groupby("cluster_id").size()` without filtering `passes_all` first. **RESOLVED (before this
2026-09 session — the current `p05()` already has `if "passes_all" in df.columns: df=df[df["passes_all"]]`
before the groupby.** The regenerated plot now correctly shows the true per-cluster survivor counts
(29/30/29/27/29), not a flat 55.

### Reproduced survivor count

The four filters that do not depend on the un-committed `L_required` can be reproduced exactly
against the committed PCM CSV. Applying `Tm` in [52, 65] **and** `Tm` in [42, 70] **and**
`abs(Tm_melting − Tm_freezing) <= 8 K` **and** `cycles_tested >= 300`:

**29 candidates survive** for the Tm_target=57C clusters (0, 2, 4). No candidate in the [52, 65] °C
window fails on either supercooling or cycling — every one of the 29 window-passers passes all
four. Clusters 1 and 2 differ post-regime-cap-fix (30 and 29 respectively, per a shifted window) —
see the regime-cap resolution above.

The fifth filter, `L >= 0.7 × L_required`, is non-binding: the observed cluster `Ta_mean` medians
**RESOLVED — the ~63-82 kJ/kg L_required estimate below described the pre-2026-09-fix formula.**
`04b`'s current formula (see `05_PHASE_3_AUDIT.md` Stage 3) gives `L_required` in the
**~113-190 kJ/kg** range across the 45 points, and the 0.7x floor is **~79-133 kJ/kg**, while the
*minimum* latent heat in the whole 55-row database is 128 kJ/kg — the floor is largely, not
absolutely, non-binding now (a handful of low-latent-heat candidates near 128 kJ/kg could fail it
for the higher-`L_required` clusters). `08_mcdm_ranking.py`'s own diagnostic text still confirms
most candidates clear it: "every candidate's latent heat comfortably clearing L_required in every
cluster."

**29 survivors for Clusters 0/2/4 (Tm_target=57C); 30 for Cluster 1, 27 for Cluster 3** — no
longer identical across all five clusters. Since 27-30 > 25 in every case, `07` prints status
`HIGH` for all five clusters and the auto-relaxation never triggers (`window_relax_applied = 0.0`
throughout) — that part of the original observation still holds.

### The 29 surviving candidates

| Name | Manufacturer | Tm (°C) | L (kJ/kg) | Supercool (K) | Cycles |
|---|---|---|---|---|---|
| n-Tetracosane (C24) | Literature | 52.0 | 255 | 2.7 | 1368 |
| PlusICE A52 | PCM Products Ltd. | 52.0 | 220 | 2.8 | 1447 |
| Paraffin/Expanded graphite (92 % paraffin) | Literature | 52.2 | 170 | 3.0 | 2000 |
| PureTemp 53 | PureTemp | 53.0 | 225 | 1.9 | 1686 |
| Myristic acid (C14) | Literature | 53.0 | 199 | 1.8 | 1686 |
| RT54HC | Rubitherm | 53.5 | 200 | 0.0 | 1474 |
| n-Pentacosane (C25) | Literature | 54.0 | 238 | 3.4 | 1404 |
| RT55 | Rubitherm | 54.0 | 170 | −2.5 | 2000 |
| Myristic acid/NBR-1.0 | Literature | 54.1 | **128** | 4.9 | 2000 |
| Myristic acid/NBR-0.5 | Literature | 54.6 | 142 | 4.1 | 2000 |
| **savE® OM55** | Pluss | 55.0 | 188 | 1.0 | 2000 |
| **Palmitic-stearic acid/Expanded graphite** | Literature | 55.2 | 176 | 0.3 | 2000 |
| **n-Hexacosane (C26)** | Literature | 56.5 | **256** | 0.3 | 1404 |
| RT57HC | Rubitherm | 56.5 | 240 | 0.0 | 1404 |
| **RT60** | Rubitherm | 58.0 | 160 | 0.0 | 2000 |
| **PureTemp 58** | PureTemp | 58.0 | 225 | −0.1 | 1620 |
| PlusICE A58 | PCM Products Ltd. | 58.0 | 215 | −0.2 | 1581 |
| n-Heptacosane (C27) | Literature | 59.0 | 236 | −0.7 | 1404 |
| CrodaTherm 60 | CrodaTherm | 59.8 | 217 | −1.7 | 1533 |
| Palmitic acid/Expanded graphite (80/20) | Literature | 60.9 | 148 | 0.1 | 2000 |
| PureTemp 60 | PureTemp | 61.0 | 220 | −0.5 | 1695 |
| RT65 | Rubitherm | 61.5 | 150 | 0.0 | 2000 |
| n-Octacosane (C28) | Literature | 61.6 | 253 | −0.7 | 1581 |
| PlusICE A62 | PCM Products Ltd. | 62.0 | 205 | 0.0 | 1581 |
| RT62HC | Rubitherm | 62.5 | 230 | 0.5 | 1404 |
| Palmitic acid (C16) | Literature | 62.6 | 198 | 0.5 | 1695 |
| PureTemp 63 | PureTemp | 63.0 | 206 | 1.0 | 1510 |
| n-Nonacosane (C29) | Literature | 64.0 | 240 | 1.7 | 1404 |
| RT64HC | Rubitherm | 64.0 | 250 | 1.5 | 1404 |

Bold rows are the five that appear in a Top-3 in Phase 6.

Survival rate: **29/55 = 52.7 %** of the database, in Clusters 0, 2, and 4 (Tm_target=57C,
unchanged). Against `VERIFICATION_METHODOLOGY.md`'s own success criterion of "10–50 % of candidates
survive (not too strict or loose)", this sits marginally above the upper bound.

**RESOLVED (2026-09) — survivor sets are no longer identical across all five clusters.**
`07b_charging_feasibility.py`'s regime-dependent Tm cap had a normalization bug (dividing
`poor_day_kt` by `kt_mean`, which erased the absolute-clearness signal it needed) that made it a
mathematical no-op — this is why every cluster shared exactly the same 29-candidate survivor set.
Fixed to use `poor_day_kt` directly: Cluster 1 now has `Tm_target=55.16C` (30 survivors — gains
Paraffin/HDPE PCM6, Paraffin/HDPE PCM2, and savE OM48, none of which pass the higher 57C-based
window) and Cluster 2 has `Tm_target=56.51C` (29 survivors, a slightly different set from Clusters
0/4's 29). Cluster 3 (27 survivors) was already slightly different due to its own `L_required`
value. Only Clusters 0 and 4 remain identical to each other, which is a legitimate finding (they
have very similar climate profiles) rather than a bug.

---

## Literature support

`07b_charging_feasibility.py` cites **Al-Mamun 2023** for the flat-plate-collector 25–100 °C
operating band — the only substantive external citation in the pipeline code.
`06_build_pcm_database.py` mentions **Singh 2025** historically, describing a superseded code path
that appended 7 hardcoded literature rows. The 42–70 °C band and the filter set are cited to plan
v3.0 Table 12. The MICE + RF + PMM method is described at length with **no** citation. See
`13_LITERATURE_MAPPING.md`.

## Validation

| Check | Result |
|---|---|
| Filter precedes ranking | **Confirmed** — and justified in the docstring |
| Unimplemented filters declared | **Confirmed** — three named explicitly in the docstring |
| Missing data does not cause exclusion | **Confirmed** — cycling and supercooling retain-and-flag on NaN |
| Per-filter detail retained for audit | **Confirmed** — all 275 rows written with five pass/fail booleans |
| Auto-relaxation on low survivors | **Implemented**; never triggered (27-30 >= 5 in every cluster) |
| Survivor count inside the 5–25 `OK` band | **FAIL** — 27-30 per cluster, status reads `HIGH` |
| Survival rate inside the 10–50 % criterion | **MARGINAL FAIL** — ~49-55% depending on cluster |
| Per-cluster differentiation | **RESOLVED (2026-09), now PASS** — was **FAIL** (identical survivor set in all five clusters); survivor counts are now 29/30/29/27/29, genuinely differentiated for Clusters 1-3 |

## Problems / risks

1. **~~All five clusters have identical survivor sets.~~ RESOLVED (2026-09).** This was never an
   unavoidable consequence of a constant `Tm_target` — it was a real bug in
   `07b_charging_feasibility.py`'s regime cap (a normalization step divided away its own signal,
   making the cap a no-op regardless of how the script was run). Fixed; Clusters 1 (Tm_target=55.16C)
   and 2 (56.51C) now get real, differentiated survivor sets (30 and 29 respectively, vs. 29/27/29
   for Clusters 0/3/4).
2. **~~Three of the five plan Table-12 filters are not implemented~~ — now two.** Corrosion veto
   is implemented (2026-09, see item 3); 5th-percentile-day charging feasibility and safety
   exclusion remain unimplemented, and the script says so in its own docstring rather than hiding it.
3. **The corrosion veto is implemented but cannot activate with the current database** — every one
   of the 55 candidates is organic, so `corrosion_class` is `low_organic` for all of them (verified
   directly against the database), leaving nothing for the veto to reject. `NEXT_STEPS.md`'s
   expectation that the veto would "bite for high-monsoon-humidity Uttarakhand clusters" still
   cannot be realised with this database — but the veto will fire automatically the moment an
   inorganic candidate (e.g. a salt hydrate) is added, without any further code change.
4. **`07`'s low-survivor warning string is stale**: it prints "your database (25 rows) is thin for
   this" while the database is 55 rows. It would not have fired in this run anyway (29 > 5).
5. **Auto-relaxation never triggered** (29 >= 5 in every cluster), so `window_relax_applied` is 0
   throughout and the relaxation policy question is moot for this run.
6. **59.1 % of the PCM database's flagged property cells are MICE-RF-PMM estimates**, and three of
   the five MCDM criteria (`TC_W_mK`, `cycles_confidence`, `rho_H_MJ_m3`) rest substantially on
   them. The pipeline carries `any_property_imputed` and `n_properties_imputed` forward precisely
   so this can be reported — `09_recommendation_cards.py`'s caveat text mentions it, but only for
   "the literature-added candidates", which understates the scope: **all 55 rows** carry at least
   one imputed property.
7. **The survivor count exceeds the pipeline's own upper comfort bound** (29 vs the `OK` range of
   5–25, and 52.7 % vs the 10–50 % criterion). The 13 K-wide melting window at `Tm_target = 57 °C`
   admits over half the database.
8. **`07b`'s constants are stated assumptions, not measured values** — the script says so itself.
   If it is ever run, `REFERENCE_GOOD_DAY_TEMP_C = 70`, `MIN_ACHIEVABLE_TEMP_C = 42` and
   `POOR_DAY_Z = 1.28` must be declared as assumptions in the write-up.

## Status

**COMPLETE, and RESOLVED (2026-09) — no longer degenerate.** The database build is thorough and
fully auditable — the imputation footprint is recoverable cell-by-cell from the committed CSV,
which is more transparency than the climate data offers. The filter is correctly ordered before
ranking, declares its own gaps, and handles missing data conservatively. Survivor sets are now
genuinely differentiated across regimes (29/30/29/27/29 for Clusters 0-4) thanks to the
`07b_charging_feasibility.py` regime-cap fix — the previous "does not discriminate between regimes"
finding was traced to that script's normalization bug, not an inevitable consequence of the
constant Phase-3 `Tm_target=57C`.
