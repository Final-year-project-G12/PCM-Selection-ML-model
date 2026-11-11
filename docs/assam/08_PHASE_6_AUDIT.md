# 08 — Phase 6 Audit: Feasibility Filtering Engine (Assam)

**Final-governance script:** `07_feasibility_filter_final.py`  
**Legacy script:** `07_feasibility_filter.py`  
**Candidate database:** `pcm_database_final.csv` (58 records × 41 properties)

**Status:** The supplied audit documents a locked six-criterion governance implementation and a separate historical K=4 survivor set. Its original cluster medoids, 44°C target context, and regime-specific latent-heat requirements do not match the latest audited Assam Objective 1 dataset. The method description is retained, but current-run reproducibility is **NOT VERIFIED** until those inputs are reconciled and the final script is rerun.

---

## Objective and Pipeline-Version Distinction

Phase 6 screens PCM candidates using evidence-based physical, operational, and safety criteria. Two implementations must be kept separate:

1. **Final locked K=3 governance:** `07_feasibility_filter_final.py`, using the 58-row production database, six criteria, PASS/FAIL/UNKNOWN evidence states, and no automatic relaxation.
2. **Historical pre-audit K=4 screening:** `07_feasibility_filter.py`, using the 25-row prototype database, a seven-constraint design, a \(0.7\,L_{\text{required}}\) floor, and automatic melting-window relaxation. This older run produced the historical eight-candidate set listed in Section 3.

Do not combine the survivor counts or criteria from these two implementations.

---

## 1. Final Locked K=3 Governance — Method Recorded in the Supplied Audit

### Screening setup recorded by the supplied Phase 6 audit

| Input | Value in supplied Phase 6 audit | Current-data assessment |
|---|---|---|
| Candidate pool | 58-row `pcm_database_final.csv` | Consistent with latest audited database |
| Climate regimes | K=3 | Consistent with latest Assam clustering |
| Medoids | `ASP_0012`, `ASP_0092`, `ASP_0028` | **Obsolete**; latest medoids are `ASP_0003`, `ASP_0036`, `ASP_0080` |
| Target melting temperature | 44.0°C | Conflicts with latest Objective 1 nominal target of 56.0°C; governing relationship unresolved |
| \(L_{\text{required}}\) | 252.09 / 258.69 / 279.70 kJ/kg | **Not verified** for the latest 80-point clusters |

The latest audited Assam clustering basis is:

- 80 spatial points, `ASP_0001`–`ASP_0080`, with 87.8% population coverage.
- Three clusters of sizes 27 / 28 / 25.
- Diagonal-covariance GMM, BIC@K=3 = 1030.97.
- Bootstrap mean ARI = 0.6939; median ARI = 0.7255.
- Current medoids: `ASP_0003`, `ASP_0036`, `ASP_0080`.

The 44°C value is the formula-derived value \(50-6\) in the supplied Phase 4/Phase 6 material. The latest Objective 1 audit separately records a nominal PCM melting target of 56.0°C and regime caps of 49.659°C, 49.395°C, and 49.437°C. The files do not establish whether these represent different design quantities or whether one configuration is stale. **Do not silently replace one with the other.** The final-governance script's active target and cluster-specific inputs must be checked before claiming a current K=3 run.

### Six-criterion architecture

The supplied final-governance audit describes the following evidence-based criteria:

1. **Combined melting-temperature window:**  
   \[
   T_m \in [\max(T_m^{\text{target}}-6,\ 42.0),\ T_m^{\text{target}}+8]
   \]
   The supplied audit evaluates this as 42–52°C when \(T_m^{\text{target}}=44^\circ C\). This numeric interval is tied to the unresolved 44°C target and must not be assumed to be the current interval.

2. **Latent-heat capacity floor:**  
   \[
   L \geq 1.0\,L_{\text{required}}
   \]
   The floor is strict and unrelaxed. The audit states that missing reported latent heat produces `UNKNOWN`.

3. **Thermal cycling durability:** `PASS` only when cycle life is explicitly reported (not imputed) and is at least 300 cycles; otherwise `UNKNOWN`.

4. **Supercooling limit:** `PASS` only when supercooling is explicitly reported, or both \(T_m\) and \(T_{m,\text{freezing}}\) are reported, and the value is no more than 8.0 K; otherwise `UNKNOWN`.

5. **Corrosion veto:** `FAIL` only when `is_inorganic` is true and the cluster humidity proxy (`RH_mean_mean`, used as an HSI stand-in) exceeds the global 75th percentile. `PASS` requires explicit documented compatibility evidence; otherwise `UNKNOWN`. The 58-row database currently has `is_inorganic=False` for all records, so this veto cannot currently fail a candidate based on inorganic classification. It is structurally inert for the present database.

6. **Chemical and fire safety:** `PASS` or `FAIL` only on explicit reported flammability evidence; otherwise `UNKNOWN`.

### Feasibility status logic

The supplied audit defines:

- `CONFIRMED_FEASIBLE`: all six criteria are `PASS`.
- `CONDITIONALLY_FEASIBLE`: no criterion is `FAIL`, but at least one is `UNKNOWN`.
- `INFEASIBLE`: at least one criterion is `FAIL`.

This is an evidence-based governance approach: imputed values do not automatically count as reported evidence for criteria that explicitly require reported measurements.

### Governance findings in the supplied audit

The supplied Phase 6 audit reports:

\[
n_{\text{confirmed}}=[0,0,0]
\]

It also reports exactly one conditional candidate:

- `n-Tetracosane (C24)`, \(T_m=52.0^\circ C\), \(L=255.0\text{ kJ/kg}\), in Cluster 0.
- The supplied audit states that it meets the recorded Cluster 0 latent-heat requirement of approximately 252.0 kJ/kg and lies at the upper boundary of the recorded 42–52°C melting window.
- It is classified as **conditional**, not confirmed and not MCDM-ranked.

**Current-run qualification:** these candidate findings depend on the old medoids, 44°C target/window, and old \(L_{\text{required}}\) values in the supplied audit. The exact `n_confirmed=[0,0,0]` and the single conditional-candidate result should therefore be cited as findings of the supplied final-governance run, not yet as independently verified results for the latest 80-point Assam dataset.

### Reconciliation with the Phase 5 audit summary

The Phase 5 update records 2 “confirmed candidates” and 5 “deployable survivors” per cluster. The supplied Phase 6 final-governance audit instead reports zero `CONFIRMED_FEASIBLE` candidates and one `CONDITIONALLY_FEASIBLE` candidate overall.

These counts must **not** be merged or treated as equivalent. The Phase 5 summary does not define the exact output columns behind “confirmed candidates” and “deployable survivors”; the Phase 6 audit defines a stricter six-criterion evidence status. The likely distinction is between different scripts, status definitions, or pipeline runs, but the supplied files alone do not prove which. Resolve this by inspecting both scripts and their output files, including timestamps and upstream fingerprints. Until then, report the counts with their respective script/version labels.

---

## 2. No Automatic Relaxation

The final-governance script is described as having zero automatic relaxation:

- No automatic widening of the melting-temperature interval.
- No automatic reduction of the latent-heat threshold.
- No relaxation merely to increase the candidate count.

This is a methodological distinction from the older seven-constraint implementation. Do not describe the final-governance script as using the older \(0.7\,L_{\text{required}}\) floor or automatic melting-window relaxation.

---

## 3. Historical Pre-Audit K=4 Survivor Set

The supplied audit records an exploratory run using the 25-row prototype `pcm_database_assam.csv`, K=4 climate forcing, and a relaxed latent-heat threshold \(\kappa=0.7\). It reports the following eight historical survivors:

| # | Candidate material | Material family | \(T_m\) (°C) | Latent heat (kJ/kg) |
|---:|---|---|---:|---:|
| 1 | Myristic–Palmitic eutectic (58/42) | Binary organic eutectic | 42.6 | 169.7 |
| 2 | RT44HC | Rubitherm paraffin | 43.0 | 250.0 |
| 3 | savE® OM42 | PLUSS bio-based organic | 44.0 | 199.0 |
| 4 | C22H46 (docosane-class paraffin) | Pure alkane paraffin | 44.5 | 249.0 |
| 5 | savE® OM46 | PLUSS bio-based organic | 47.0 | 177.0 |
| 6 | RT45HC | Rubitherm paraffin | 47.0 | 230.0 |
| 7 | savE® OM50 | PLUSS bio-based organic | 50.0 | 189.0 |
| 8 | savE® OM48 | PLUSS bio-based organic | 51.0 | 165.0 |

These eight candidates belong to the historical 25-row/K=4 screening run. They are not the output of the final 58-row, six-criterion governance run.

The supplied audit says these eight candidates were subsequently evaluated by the dynamic physics simulation. When referenced downstream, preserve the historical qualification and do not imply that they passed the final locked K=3 governance.

### Negative disclosures

- **RT47 is not a historical survivor.** The supplied audit states that RT47 (\(T_m=46.0^\circ C\), \(L=160\text{ kJ/kg}\)) failed the historical latent-heat floor.
- **n-Tetracosane (C24) is not in the historical eight-candidate set.** It was introduced in the 58-row database and appears only in the supplied final-governance conditional-candidate finding.

---

## 4. Database and Corrosion-Screening Limitation

The current audited PCM database contains 58 records and 41 columns, all organic or organic-composite classifications; it contains zero inorganic entries.

Consequently:

- The `is_inorganic`-based corrosion veto cannot identify an inorganic salt-hydrate candidate in the current pool.
- Do not claim that the current Assam results demonstrate comparative corrosion screening across organic and inorganic PCM families.
- The use of `RH_mean_mean` as an HSI stand-in is a proxy choice, not a direct measurement of PCM corrosion compatibility.
- Explicit compatibility evidence remains necessary for a `PASS` under the supplied final-governance rule.

---

## 5. Open Verification Items Before Calling the Current K=3 Run Authoritative

1. Confirm the exact active version and modification time of `07_feasibility_filter_final.py`.
2. Confirm that its cluster-profile input corresponds to the latest 80-point K=3 run and medoids `ASP_0003`, `ASP_0036`, `ASP_0080`.
3. Reconcile the 44°C formula-derived value with the latest 56°C nominal Objective 1 target and the regime caps 49.659 / 49.395 / 49.437°C.
4. Recalculate \(L_{\text{required}}\) from current cluster profiles and verify the demand/mains-water assumptions used by the final-governance script.
5. Re-run the final-governance script against the current 58-row database and current cluster-profile artifacts.
6. Record current per-cluster counts for `CONFIRMED_FEASIBLE`, `CONDITIONALLY_FEASIBLE`, and `INFEASIBLE`, plus the candidate names and criterion-level PASS/FAIL/UNKNOWN evidence.
7. Reconcile those outputs with the Phase 5 “2 confirmed / 5 deployable” summary by tracing the exact script, status columns, timestamps, and provenance fingerprints.
8. Verify that downstream MCDM inputs are generated from the same current feasibility output. Do not pass conditional candidates into MCDM unless the pipeline explicitly defines and documents that policy.

**Status recommendation:** retain the six-criterion governance method and the historical eight-candidate table as documented audit evidence. Mark the supplied final K=3 counts and conditional-candidate result as **run-specific / pending current-input reconciliation**. Do not label the latest 80-point Assam Phase 6 execution “Authoritative Final” until the current cluster profiles, target-temperature definition, latent-heat requirements, and output provenance have been verified.
