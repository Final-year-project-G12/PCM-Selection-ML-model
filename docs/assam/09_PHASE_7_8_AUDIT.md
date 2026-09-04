# 09 — Phase 7 & 8 Audit: MCDM Ranking Engine & Monte Carlo Uncertainty

**Scripts**: `08_mcdm_ranking.py`, `09_recommendation_cards.py`

**Status**: GOVERNED (Authoritative Final)

---

## Methodological Distinction: Final Governance vs. Historical Benchmark

Phases 7 and 8 constitute the multi-criteria decision making (MCDM) ranking and stochastic uncertainty propagation layer. 

In strict adherence to project governance, this audit distinguishes:
1. **Final Locked $K=3$ Governance** (Audited pipeline status under final $K=3$ climate forcing and 58-row database).
2. **Historical Pre-Audit $K=4$ Benchmark** (Historical exploratory ranking and 5,000-draw Monte Carlo simulation).

---

## 1. Final Locked $K=3$ Governance

### Phase 7: MCDM Status — NOT PERFORMED
Under the authoritative Phase 6 feasibility screening of the 58-row PCM database without arbitrary relaxation:
$$n_{\text{confirmed}} = [0, 0, 0]$$
- Because no PCM met all 7 strict physical, durability, and corrosion constraints simultaneously across the three regimes, the formal multi-criteria ranking engine was **`NOT PERFORMED`**.
- The solitary conditional candidate, `n-Tetracosane (C24)` in Cluster 0, is reported transparently as a conditional candidate and was **not** processed through MCDM scoring.
- **Critical Policy**: No material in the final $K=3$ pipeline is designated as an "MCDM Winner."

### Phase 8: Monte Carlo Status — SKIPPED
In accordance with `data/processed/pcm/monte_carlo_stability_assam.csv` and `final_outputs/tables/table08_monte_carlo_stability_k3.csv`:
- **Execution Parameter**: $n_{\text{draws}} = 0$
- **Governance Status**: **`SKIPPED`** across all three regimes (Cluster 0, 1, 2).
- **Authoritative Skip Reason**: *"Monte Carlo stability analysis skipped due to insufficient eligible candidates ($n < 2$)."*

---

## 2. Historical Pre-Audit $K=4$ MCDM & Monte Carlo Benchmark

During preliminary research, the multi-method MCDM framework and 5,000-draw Monte Carlo uncertainty engine were executed against the historical 8-PCM survivor set under exploratory 4-cluster forcing. These outputs are preserved in `final_outputs/tables/table07_historical_mcdm_rankings_k4.csv` as a historical reference benchmark.

### Method Stack & Consensus Layer
The historical engine implemented four independent multi-criteria methods:
- **TOPSIS**: Relative closeness coefficient $C_i \in [0, 1]$ (benefit).
- **Grey Relational Analysis (GRA)**: Grey relational grade $\gamma_i \in [0, 1]$ (benefit).
- **PROMETHEE II**: Net outranking flow $\Phi \in [-1, +1]$ (benefit).
- **VIKOR**: Compromise ranking index $Q_i \in [0, 1]$ (cost metric; lower is better).
- **Consensus**: Borda count rank sum (primary), Copeland pairwise dominance (secondary), and Kendall's coefficient of concordance ($W$).

### MCDM Criteria Stack
1. `f_Tm`: Gaussian temperature fitness $\exp\left(-\frac{(T_m - 44.0)^2}{2 \times 4.0^2}\right)$, scoring proximity to target.
2. `latent_heat_margin_ratio`: Climate-relative latent heat ($L / L_{\text{required}}$).
3. `rho_H_MJ_m3`: Volumetric latent storage density (MJ/m³).
4. `TC_W_mK`: Mean thermal conductivity ($W/\text{m}\cdot\text{K}$).
5. `cycles_confidence`: Durability confidence score based on thermal cycling data.

### Weighting & Historical 5,000-Draw Monte Carlo
- Blended weighting: 50% objective Shannon entropy + 50% subjective AHP priors.
- **Executed Monte Carlo Simulation**: Exactly **5,000 Dirichlet-perturbed stochastic draws** were executed on this historical $K=4$ set, perturbing criteria weights (Dirichlet $\alpha=30$) and material properties ($T_m \pm 1\text{ K}$, $L \pm 5\%$, $k \pm 10\%$, $\rho H \pm 8\%$).

### Historical $K=4$ Results Summary
- **Unanimous #1 Rank**: `RT44HC` ($T_m = 43.0^\circ\text{C}$, $L = 250\text{ kJ/kg}$) achieved the top Borda rank across all 4 historical clusters, driven by its high Gaussian fitness at $44.0^\circ\text{C}$ and superior latent heat.
- **Concordance**: High inter-method concordance (Kendall's $W = 0.807$ to $0.845$).
- **Monte Carlo Retention**: `RT44HC` exhibited a 95.2% to 96.2% Top-1 retention probability.

*Thesis Context*: As established in Phase 10, this historical MCDM prioritization of `RT44HC` was subsequently refuted by independent dynamic physics validation, establishing that MCDM proximity scoring does not reflect operational solar fraction.
