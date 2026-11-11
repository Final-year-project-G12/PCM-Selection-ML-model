# 06 — Phase 4 Audit: Solar Water Heating (SWH) Design Specification (Assam)

**Script:** `05_cluster_assam.py`

**Status:** Partially updated to the latest audited Assam Objective 1 dataset. The system demand assumptions are retained from the supplied Assam audit; historical climate profiles and unresolved thermal-design assumptions are explicitly identified rather than silently carried forward.

---

## SWH System Sizing & Operational Requirements

Phase 4 defines the engineering design parameters and operational boundary conditions for the climate-adaptive solar water heating (SWH) system with latent thermal energy storage (LTES).

The supplied Assam audit records the following design assumptions:

| System parameter | Value in supplied Assam audit | Audit treatment |
|---|---:|---|
| Target delivery temperature, \(T_{\text{delivery}}\) | 50.0 °C | Retained as the documented design assumption |
| Heat-exchanger approach temperature, \(\Delta T_{\text{approach}}\) | 6.0 K | Retained as the documented design assumption |
| PCM target melting temperature, \(T_m^{\text{target}}\) | 44.0 °C | Historical derived value; conflicts with the latest Objective 1 target basis (see below) |
| Daily domestic demand | 100 L/day | Retained |
| Morning draw | 50 L at 07:00 local | Retained |
| Evening draw | 50 L at 19:00 local | Retained |
| Tank water mass, \(M_w\) | 100 kg | Retained |
| PCM mass, \(M_p\) | 50 kg | Retained |
| Collector surface area | 2.0 m² | Retained; design justification not independently revalidated here |

The 100 L/day demand profile and two 50 L draws are the documented Assam design basis in this audit. Do not interpret the 07:00 and 19:00 times as UTC; the source specifies local time.

### PCM melting-temperature reconciliation — important

The supplied Phase 4 file derives:

\[
T_m^{\text{target}} = T_{\text{delivery}}-\Delta T_{\text{approach}}
=50-6=44^\circ\text{C}
\]

However, the latest audited Assam Objective 1 results record a **nominal PCM melting target of 56.0°C**, with regime-specific caps:

- Cluster 0: 49.659°C
- Cluster 1: 49.395°C
- Cluster 2: 49.437°C

These are different design quantities and cannot both be described as the same current \(T_m^{\text{target}}\). The 44.0°C value is the formula-derived value in the supplied Phase 4 document; 56.0°C and the caps are the latest audited Objective 1 values. The governing design rule connecting these quantities has **not been established by the supplied files**.

**Required action:** inspect the active `05_cluster_assam.py`, the current cluster-profile output, and the PCM target/capping logic in the feasibility stage. Then document whether 56.0°C is a nominal search target, whether the caps are operational selection limits, and whether 44.0°C remains a separate heat-exchanger design calculation. Until reconciled, do not label 44.0°C as the current Objective 1 PCM target.

---

## Final Climate Forcing — Current Assam K=3 GMM Result

The climate forcing must use the latest audited Assam Phase 3 clustering result, not the historical profile table in the original Phase 4 document.

### Current audited clustering summary

| Metric | Current Assam result |
|---|---:|
| Number of climate regimes | 3 |
| GMM covariance type | Diagonal (`diag`) |
| BIC at K=3 | 1030.97 |
| Bootstrap mean ARI | 0.6939 |
| Bootstrap median ARI | 0.7255 |
| Cluster sizes | 27 / 28 / 25 |
| Medoid station IDs | `ASP_0003`, `ASP_0036`, `ASP_0080` |
| Active spatial points | 80 (`ASP_0001`–`ASP_0080`) |
| Population coverage | 87.8% |

The current clustering uses the five core physical features recorded in the Assam audit: `GHI_mean`, `Ta_mean`, `DTR`, `RH_mean`, and `wind_mean`. The current audited covariance type is **diagonal**, not the **full covariance** stated in the original Assam Phase 4 file.

### Regime profile table

The original Assam Phase 4 file contains a historical profile table with 33/61/35 points, medoids `ASP_0012`/`ASP_0092`/`ASP_0028`, and population counts tied to the earlier 129-point dataset. Those values do not describe the current 80-point run and are retired from the current-state description.

The current cluster sizes and medoid IDs are known, but the latest cluster-profile values for covered population, annual mean GHI, estimated daily GHI, mean ambient temperature, DTR, and RH were not provided in the supplied current audit evidence. **Do not copy the historical profile values into the current table.** Populate these fields from the current `table04_cluster_profiles_k3.csv` (or its current Assam equivalent) after verifying its timestamp and point IDs.

These regimes represent climate-feature similarity groupings. They should not be described as contiguous geographic territories unless a separate geographic analysis supports that claim.

---

## Regime-Specific Thermal Requirements

The supplied Phase 4 audit defines the night-discharge thermal deficit as:

\[
Q_{\text{deficit}} =
M_w \cdot C_{p,w} \cdot
(T_{\text{delivery}}-T_{\text{mains}})
\]

and the required latent heat per unit PCM mass as:

\[
L_{\text{required}}=\frac{Q_{\text{deficit}}}{M_p}
\]

It records the mains-water estimate as:

\[
T_{\text{mains}}=\max(5.0,\ T_{a,\text{mean}}-6.0)
\]

The supplied file reports the following per-cluster \(L_{\text{required}}\) values: 252.09, 258.69, and 279.70 kJ/kg for clusters 0, 1, and 2 respectively. These values are retained here only as **values reported by the supplied Phase 4 audit**. The source profile medoids and ambient-temperature values are from the historical 129-point run, while the latest audited clustering is the 80-point run. Therefore, the three \(L_{\text{required}}\) values have **not been verified for the current 80-point clusters** and must not be presented as current results until recalculated from the current cluster profiles and current demand assumptions.

The historical K=4 range of 232.3–248.9 kJ/kg is also retained only as historical context; it is not a current K=3 result.

### Current-run calculation checklist

Before publishing current regime-specific thermal requirements, verify:

1. Current cluster-profile mean ambient temperature and medoid IDs for all three clusters.
2. The exact current \(T_{\text{mains}}\) equation and whether it is consistent with the physics-validation implementation.
3. \(M_w\), \(M_p\), \(C_{p,w}\), and \(T_{\text{delivery}}\) used in the active calculation.
4. Whether the calculation uses the documented 100 kg tank water mass and 50 kg PCM mass.
5. The current per-cluster output values from the active feasibility/cluster-profile artifacts.

---

## Current Dataset Reconciliation

| Item | Current audited Assam basis |
|---|---:|
| Active population-weighted points | 80 |
| Point ID range | `ASP_0001`–`ASP_0080` |
| Population coverage | 87.8% |
| Climate regimes | 3 |
| GMM covariance type | Diagonal |
| Cluster sizes | 27 / 28 / 25 |
| GMM BIC at K=3 | 1030.97 |
| Bootstrap mean ARI | 0.6939 |
| Bootstrap median ARI | 0.7255 |
| Medoids | `ASP_0003`, `ASP_0036`, `ASP_0080` |
| Daily demand assumption in supplied Phase 4 audit | 100 L/day |
| Draw schedule in supplied Phase 4 audit | 50 L at 07:00 and 50 L at 19:00 local |
| Latest Objective 1 nominal PCM melting target | 56.0°C |
| Latest Objective 1 regime-specific caps | 49.659 / 49.395 / 49.437°C |

### Historical values to retire from current-state descriptions

- 129 spatial points and the historical 33/61/35 cluster counts.
- Historical medoids `ASP_0012`, `ASP_0092`, and `ASP_0028`.
- Historical covered-population counts and climate-profile means in the original table.
- Full-covariance GMM; the current audited Assam model is diagonal covariance.
- The 44.0°C value as the latest Objective 1 PCM target.
- The reported 252.09 / 258.69 / 279.70 kJ/kg values as current-run thermal requirements, until recalculated for the current clusters.

## Open Verification Items

1. Reconcile the 44.0°C formula-derived melting temperature with the latest Objective 1 nominal target of 56.0°C and the three regime caps.
2. Regenerate or verify the current cluster-profile table for the 80-point K=3 run; replace the historical profile table only with verified current values.
3. Recalculate \(T_{\text{mains}}\), \(Q_{\text{deficit}}\), and \(L_{\text{required}}\) for the current three clusters.
4. Confirm that the current cluster-profile and physics-validation scripts use the same mains-water temperature definition.
5. Confirm the provenance/timestamp of `05_cluster_assam.py` outputs before marking this phase fully authoritative.

**Status recommendation:** keep the SWH sizing assumptions as documented inputs, but mark the climate-profile and regime-specific thermal-requirement sections as pending current-run reconciliation. Do not label the entire Phase 4 audit “Authoritative Final” until the open items above are resolved.
