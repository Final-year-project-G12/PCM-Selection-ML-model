# 08 — Phase 6 Audit: Feasibility Filtering Engine

**Script**: `07_feasibility_filter.py`

**Status**: GOVERNED (Authoritative Final)

---

## Objective & Methodological Distinction

Phase 6 screens candidate phase-change materials against 7 multi-physics, safety, and operational constraints. To preserve scientific integrity, this audit explicitly distinguishes between two separate pipeline versions:
1. **Final Locked $K=3$ Feasibility Governance** (Audited 58-PCM database under final $K=3$ climate forcing).
2. **Historical Pre-Audit $K=4$ Feasibility Survivors** (Preliminary 25-PCM screening under historical $K=4$ forcing, which provided the 8 candidates evaluated in Phase 9 physics validation).

---

## 1. Final Locked $K=3$ Feasibility Governance

### Screening Setup
- **Candidate Pool**: Audited 58-row production database (`pcm_database_final.csv`).
- **Climate Forcing**: Final locked $K=3$ climate regimes (Medoids: `ASP_0012`, `ASP_0092`, `ASP_0028`).
- **Target Melting Temperature**: $T_m^{\text{target}} = 44.0^\circ\text{C}$ ($T_{\text{del}} = 50.0^\circ\text{C}, \Delta T_{\text{approach}} = 6.0\text{ K}$).
- **Target Latent Heat Demand**: Regime-specific $L_{\text{required}}$ based on ambient temperature baselines ($232–252\text{ kJ/kg}$).

### Constraint Architecture (7 Vetoes)
1. **Melting Window**: $T_m \in [T_m^{\text{target}} - 6, T_m^{\text{target}} + 8] = [38.0, 52.0]^\circ\text{C}$.
2. **Absolute SWH Band**: $T_m \in [42.0, 70.0]^\circ\text{C}$.
3. **Latent Heat Capacity Floor**: $L \ge 1.0 \times L_{\text{required}}$ (strict unrelaxed floor).
4. **Thermal Cycling Durability**: Tested $\ge 300$ cycles without degradation.
5. **Corrosion Veto**: Prohibition of inorganic salt hydrates in high-humidity regimes ($HSI > \text{global } p_{75}$).
6. **Supercooling Limit**: Supercooling $\Delta T_{\text{sub}} \le 8.0\text{ K}$.
7. **Chemical & Fire Safety**: Elimination of toxic, explosive, or highly flammable materials.

### Governance Findings
- **Zero Automatic Relaxation**: Unlike early prototypes that relaxed temperature bands or latent heat thresholds when candidate counts dropped, final governance strictly prohibits automatic relaxation.
- **Confirmed Feasible Candidates**:
  $$n_{\text{confirmed}} = [0, 0, 0]$$
  Zero PCMs in the 58-row database met all 7 strict criteria simultaneously without relaxation across all three regimes.
- **Conditional Candidate**: Exactly **one candidate** qualified under conditional status:
  - **`n-Tetracosane (C24)`** ($T_m = 52.0^\circ\text{C}$, $L = 255.0\text{ kJ/kg}$) in **Cluster 0**.
  - In Cluster 0 ($L_{\text{required}} = 252.0\text{ kJ/kg}$), `n-Tetracosane` satisfies $L \ge L_{\text{required}}$ and touches the upper boundary of the melting window ($52.0^\circ\text{C}$). It is classified strictly as a **Conditional candidate**, not a confirmed or MCDM-ranked survivor.
  *(Crucial Note: `n-Tetracosane` belongs exclusively to the final 58-row database screening; it was NOT part of the historical 8-PCM survivor set).*

---

## 2. Historical Pre-Audit $K=4$ Feasibility Survivors

During preliminary research, feasibility filtering was applied to the initial 25-row prototype database (`pcm_database_assam.csv`) under exploratory 4-cluster forcing. With latent-heat threshold relaxation ($\kappa = 0.7$), an 8-PCM candidate set survived.

### Authoritative Historical 8-PCM Survivor Set
These exact 8 materials formed the candidate universe evaluated independently by the Phase 9 dynamic physics simulation:

| # | Candidate Material | Material Family | $T_m$ (°C) | Latent Heat ($L$, kJ/kg) | Historical Screening Role |
|---|---|---|---|---|---|
| 1 | **Myristic-Palmitic eutectic (58/42)** | Binary Organic Eutectic | 42.6 | 169.7 | Historical survivor |
| 2 | **RT44HC** | Rubitherm Paraffin | 43.0 | 250.0 | Historical survivor (#1 MCDM rank) |
| 3 | **savE® OM42** | PLUSS Bio-based Organic | 44.0 | 199.0 | Historical survivor |
| 4 | **C22H46 (docosane-class paraffin)** | Pure Alkane Paraffin | 44.5 | 249.0 | Historical survivor |
| 5 | **savE® OM46** | PLUSS Bio-based Organic | 47.0 | 177.0 | Historical survivor |
| 6 | **RT45HC** | Rubitherm Paraffin | 47.0 | 230.0 | Historical survivor |
| 7 | **savE® OM50** | PLUSS Bio-based Organic | 50.0 | 189.0 | Historical survivor |
| 8 | **savE® OM48** | PLUSS Bio-based Organic | 51.0 | 165.0 | Historical survivor |

### Negative Disclosures (Integrity Mandates)
- **NO "RT47"**: `RT47` ($T_m=46.0^\circ\text{C}$, $L=160\text{ kJ/kg}$) failed the latent heat floor and was eliminated; it is **not** a historical survivor.
- **NO n-Tetracosane**: `n-Tetracosane (C24)` was introduced in the 58-row database and was **not** part of this historical 8-PCM set.
- **Terminology**: When referencing these materials in downstream validation, they must be designated:
  *"Phase 6-screened historical candidate evaluated independently under final Phase 3 $K=3$ climate forcing."*
