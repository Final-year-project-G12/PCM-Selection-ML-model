# 07 — Phase 5 Audit: Curated PCM Property Database

**Script**: `06_build_pcm_database.py`

**Status**: COMPLETE (Authoritative Final)

---

## Final Curated PCM Database (`pcm_database_final.csv`)

Phase 5 establishes the authoritative material property repository for latent thermal energy storage (LTES). While initial exploratory runs utilized an early 25-row prototype (`pcm_database_assam.csv`), the production repository was expanded, deduplicated, and audited into **`pcm_database_final.csv`**.

### Database Architecture & Verification
- **Total Records**: **58 deduplicated PCM records**
- **Property Columns**: **41 properties** capturing thermodynamic, chemical, kinetic, safety, and physical behavior
- **Material Families**:
  - *Organic Paraffins*: Commercial Rubitherm RT series, pure alkanes ($C_{20}–C_{30}$)
  - *Bio-based Organics*: PLUSS savE OM series, fatty acids (myristic, palmitic, stearic)
  - *Eutectic Mixtures*: Binary organic eutectics (e.g. Myristic-Palmitic 58/42)
  - *Inorganic Salt Hydrates & Eutectics*: High-density hydrated salts (subject to corrosion screening)

---

## Strict Provenance & Uncertainty Framework

To eliminate scientific ambiguity, every numerical entry in `pcm_database_final.csv` carries strict origin tracking across two dedicated provenance dimensions:

1. **Source Attribution (`source_type`)**:
   - `Manufacturer_Datasheet`: Directly transcribed from certified technical datasheets (Rubitherm GmbH, PLUSS Advanced Technologies).
   - `Literature_Primary`: Peer-reviewed experimental studies (e.g., Singh et al. 2025).
   - `Imputed_Model`: Statistically imputed thermodynamic values.

2. **Data Integrity Status (`value_status`)**:
   - **`Reported`**: Experimentally measured and manufacturer-verified parameter.
   - **`Imputed`**: Derived via validated MICE-RF-PMM predictive imputation (only for non-critical secondary properties).
   - **`Missing`**: Explicitly unpopulated when neither measured nor imputable without violating physical laws.

---

## Specific Heat Capacity ($C_p$) Policy

A critical audit finding across early PCM databases was the "silent fallback" bug, where missing liquid-phase heat capacity ($C_{p,\text{liquid}}$) caused code to silently substitute solid-phase capacity ($C_{p,\text{solid}}$) as the average.

Under `pcm_database_final.csv`:
- **Strict Dual-Phase Requirement**:
  $$C_{p,\text{avg}} = \frac{C_{p,\text{solid}} + C_{p,\text{liquid}}}{2}$$
- **Zero Fallback Policy**: If either phase-specific capacity is missing, $C_{p,\text{avg}}$ remains **uncomputed (`NaN`)** rather than falling back to a single phase. Downstream models handle this missingness explicitly without biasing energy calculations.

---

## Comparison: Final Database vs. Historical Prototype

| Feature | Historical Prototype (`pcm_database_assam.csv`) | Final Production Database (`pcm_database_final.csv`) |
|---|---|---|
| **Row Count** | 25 rows (preliminary exploration) | **58 deduplicated rows** (authoritative repository) |
| **Column Count** | 24 columns | **41 columns** |
| **Provenance Tracking** | Ad-hoc source string | Cell-level `source_type` and `value_status` tracking |
| **$C_p$ Averaging** | Occasional silent single-phase fallback | **Strict dual-phase requirement (zero silent fallback)** |
| **Pipeline Role** | **Historical artifact**; fed preliminary $K=4$ run | **Final locked database**; evaluated in final $K=3$ governance |
