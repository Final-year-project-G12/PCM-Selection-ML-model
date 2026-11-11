# 07 — Phase 5 Audit: Curated PCM Database and Feasibility Filtering (Assam)

**Database builder:** `06_build_pcm_database_final.py`  
**Final database:** `pcm_database_final.csv`  
**Feasibility stage:** `07_feasibility_filter_final.py` (verify the exact active filename in the current Assam repository before execution)

**Status:** Database facts updated from the supplied Assam audit. Feasibility summary updated to the latest audited Assam Objective 1 results. Some implementation details remain to be verified against the active code and output files.

---

## Purpose

Phase 5 maintains the curated PCM property database and applies physical, thermal, and safety feasibility checks before candidates are passed to the MCDM ranking stage.

The database and feasibility results must correspond to the same current Assam K=3 cluster-profile run. Historical prototype results and the Rajasthan pipeline's thresholds, candidate counts, and methodology must not be substituted for Assam results.

---

## Final Curated PCM Database

The current Assam audit identifies `pcm_database_final.csv` as the authoritative production database.

| Database attribute | Current audited value |
|---|---:|
| Total PCM records | **58 deduplicated records** |
| Property columns | **41** |
| Detailed source composition | 55 detailed-source records + 3 unique literature additions |
| Inorganic PCM records | **0** |
| Historical prototype | 25 rows, superseded |

### Material families represented

- Organic paraffins: commercial Rubitherm RT products and pure alkanes (\(C_{20}\)–\(C_{30}\)).
- Bio-based organic materials: PLUSS savE OM products and fatty acids, including myristic, palmitic, and stearic acid.
- Organic eutectic mixtures, including binary blends such as myristic–palmitic 58/42.

**Important limitation:** all 58 records are classified as organic or organic composites. The database contains no inorganic salt hydrates. Therefore, the corrosion-veto logic cannot currently demonstrate screening of salt-hydrate candidates. Do not describe corrosion screening of inorganic PCMs as an active differentiator in the current Assam results.

---

## Provenance and Data-Quality Fields

The final database records provenance and value status at the property level.

### `source_type`

| Value | Meaning |
|---|---|
| `Manufacturer_Datasheet` | Value transcribed from a manufacturer technical datasheet |
| `Literature_Primary` | Value obtained from a primary research publication |
| `Imputed_Model` | Value estimated through the database's statistical imputation process |

### `value_status`

| Value | Meaning |
|---|---|
| `Reported` | Reported/measured value attributed to its source |
| `Imputed` | Model-estimated value |
| `Missing` | Left unpopulated because no defensible value is available |

Do not describe imputed values as manufacturer-measured values. Preserve the provenance and status fields when preparing downstream model inputs.

---

## Specific Heat Capacity (\(C_p\)) Policy

The final database follows a strict dual-phase policy:

\[
C_{p,\text{avg}} =
\frac{C_{p,\text{solid}}+C_{p,\text{liquid}}}{2}
\]

If either phase-specific heat capacity is missing, \(C_{p,\text{avg}}\) remains uncomputed (`NaN`). There is no single-phase fallback.

This avoids silently treating solid-phase heat capacity as an average when liquid-phase data are unavailable. Downstream calculations must handle missing \(C_{p,\text{avg}}\) explicitly.

---

## Current Assam Objective 1 Feasibility Context

The feasibility stage must consume the latest audited Assam K=3 cluster profiles, not historical K=4 or 129-point outputs.

| Upstream input | Latest audited Assam value |
|---|---|
| Spatial points | 80 (`ASP_0001`–`ASP_0080`) |
| Climate regimes | 3 |
| Cluster sizes | 27 / 28 / 25 |
| Medoids | `ASP_0003`, `ASP_0036`, `ASP_0080` |
| GMM covariance | Diagonal |
| BIC at K=3 | 1030.97 |
| Nominal PCM melting target | 56.0°C |
| Cluster 0 melting-temperature cap | 49.659°C |
| Cluster 1 melting-temperature cap | 49.395°C |
| Cluster 2 melting-temperature cap | 49.437°C |

The three regime caps are lower than the nominal target. The active feasibility implementation must use the current cluster-specific capped values where its charging-feasibility rule requires them.

### Latest audited feasibility summary

The latest Assam Objective 1 audit records:

| Feasibility output | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---:|---:|---:|
| Confirmed candidates | 2 | 2 | 2 |
| Deployable survivors | 5 | 5 | 5 |

These are separate reported categories in the audit and must not be added together or treated as interchangeable. The supplied summary does not define the exact distinction between “confirmed candidates” and “deployable survivors”; verify the relevant output columns and code before assigning a more specific interpretation.

The current summary is **5 deployable survivors per cluster**. Do not replace it with the Rajasthan 4/8/11, 9/14/16, or 0/0/0 results, which belong to a different state, demand basis, and pipeline run.

---

## Feasibility Constraints

The latest Assam Objective 1 pipeline uses seven feasibility filters:

| # | Constraint | Audited rule / interpretation |
|---|---|---|
| 1 | Melting window | Candidate melting temperature must satisfy the configured target window |
| 2 | Absolute melting-temperature band | \(42^\circ C \leq T_m \leq 70^\circ C\) |
| 3 | Latent heat | \(L \geq 0.7\,L_{\text{required}}\) |
| 4 | Cycling stability | At least 300 cycles |
| 5 | Supercooling | No more than 8 K |
| 6 | Corrosion veto | Excludes candidates when the corrosion-related condition is triggered |
| 7 | Safety keyword veto | Excludes candidates matching the configured safety keywords |

**Implementation note:** the exact current melting-window bounds, treatment of missing values, corrosion trigger, and safety keyword list must be read from the active Assam feasibility script. Do not import the Rajasthan audit's eight-constraint design, its charging-feasibility constraint, its flag-only behavior, or its κ-calibration procedure without verifying that the Assam code implements them.

Because the current database has zero inorganic records, the corrosion filter has no inorganic salt-hydrate candidates on which to demonstrate its intended behavior. This is a database-coverage limitation, not evidence that inorganic corrosion risk has been validated.

---

## Relationship to MCDM Ranking

Feasibility filtering precedes MCDM. The ranking stage should receive the feasible/deployable candidate pool for each Assam climate regime and should not use infeasible candidates as if they had passed the screening stage.

The latest audited Assam consensus result identifies **savE® OM49 as the MCDM consensus Top-1 PCM in all three clusters**. This is a downstream ranking result; it should not be confused with the count of feasible or deployable candidates.

The current audited PCM database contains 58 records, while the feasibility summary reports 5 deployable survivors per cluster. The exact candidate names in each cluster should be taken from the current feasibility output, not reconstructed from the database or inferred from the Top-1 ranking.

---

## Historical Results and Values Not to Reuse

The following belong to older or different pipeline states and must not be presented as current Assam Phase 5 results:

- The 25-row `pcm_database_assam.csv` prototype.
- Any historical K=4 feasibility results.
- Rajasthan's 55-row detailed source database plus seven appended literature rows (62 evaluated candidates).
- Rajasthan's 60°C delivery-temperature correction, `SHARE_PCM=0.5` derivation, κ calibration values, and candidate-pool counts.
- Rajasthan's 4/8/11, 9/14/16, or 0/0/0 survivor counts.
- Rajasthan's eight-constraint implementation details unless independently confirmed in the Assam code.
- Any claim that the Assam database includes inorganic salt hydrates.

---

## Open Verification Items

1. Confirm that `pcm_database_final.csv` in the active Assam working tree is the 58-row, 41-column file described here; record its modification time and checksum or other provenance identifier.
2. Confirm the active feasibility script filename and the exact database path it reads.
3. Verify that the feasibility run consumes the latest 80-point, K=3 Assam cluster profiles and current regime-specific caps.
4. Record the exact candidate names and output columns underlying “2 confirmed candidates” and “5 deployable survivors” per cluster.
5. Verify the precise melting-window equation, missing-data handling, corrosion-veto implementation, and safety keyword list from the active Assam code.
6. Confirm that the downstream MCDM input files were generated from the same feasibility run and current cluster-profile version.
7. Preserve the strict dual-phase \(C_p\) policy and confirm that no downstream script reintroduces a single-phase fallback.

**Status recommendation:** the 58-record database architecture, zero-inorganic finding, provenance policy, and strict \(C_p\) policy are documented as current Assam audit facts. The feasibility counts are recorded from the latest audit summary, but Phase 5 should only be marked fully reproducible after the active script, current output files, candidate identities, and upstream provenance are checked.
