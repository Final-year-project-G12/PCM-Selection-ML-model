# Data Fusion Methodology — Literature Traceability

Climate-adaptive PCM selection for Tamil Nadu SWH systems using ERA5 reanalysis and Grey Relational Analysis.

## Equations

### Step 1 — Climate–PCM compatibility (T_peak)

**Project spec** ([`datafusion.txt`](../../../datafusion.txt) Step 1):

```
T_proxy(h) = T_amb(h) + 0.02 × GHI(h)
T_peak_day = max_h T_proxy  (per location, per day)
T_peak_mean(district, month) = mean of district-mean daily T_peak
```

**Paper support (form, not 0.02 coefficient):**
- Barqawi (2025) Eq. (3): T_c = T_amb + (η × I_solar) / 20
- Assareh (2023) Eq. (1): Q_u = A_c·F_R·[S − U(T_c − T_a)]
- Kou (2025) Eq. (18): RRTD_HS = Q_sol,ave / (T_set − T_out,ave)

### Step 2 — Climate-adaptive PCM filter

**Project spec** ([`datafusion.txt`](../../../datafusion.txt) Step 2):

```
T_melt ∈ [T_peak_min − 5, T_peak_max + 5]
```

**Paper support:**
- Yan et al. (2025): PCM T_m should be 5–10 °C below HTF temperature
- Kou et al. (2025): climate-dependent T_m; ΔT = 2 °C phase band
- Singh et al. (2025): global SWH band 40–70 °C (pre-filter 35–75 °C)

### Step 3 — Grey Relational Grade (GRG)

**Chen et al. (2025)** Eqs. (15)–(17):

```
x*_j(k) = (x_j(k) − min x_j(k)) / (max x_j(k) − min x_j(k))     [Eq. 15]
ξ_j(k) = (Δ_min + 0.5·Δ_max) / (Δ_j(k) + 0.5·Δ_max)              [Eq. 16]
GRG_j = Σ w(k) · ξ_j(k)                                           [Eq. 17]
```

**Criteria and weights** (Singh 2025 priority + datafusion.txt):

| Criterion | Weight | Source |
|-----------|--------|--------|
| Latent heat | 0.35 | Singh §5a priority #1 |
| Thermal conductivity | 0.25 | Singh §5a priority #2 |
| T_melt_match = 100 − \|T_melt − T_peak_mean\| | 0.25 | Singh #3; Kou T_m alignment; Yan gap |
| Specific heat | 0.15 | Singh §5a priority #4 |
| Density | excluded | Singh §5a priority #5 (lowest) |

## IEEE references

1. G.-R. Chen et al., "Using the Taguchi method and grey relational analysis to optimize the parameter design of flat-plate collectors with nanofluids, and phase change materials in an integrated solar water heating system," *Energy Convers. Manag.: X*, vol. 26, p. 100910, 2025.
2. B. Singh et al., "Application of phase change materials in solar water heating systems — A comprehensive review," *Sol. Energy Mater. Sol. Cells*, vol. 293, p. 113888, 2025.
3. F. Kou et al., "A novel solar heating building integrated heat pipes and PCMs: Optimizing thermophysical properties and reducing energy consumption," *Build. Environ.*, vol. 285, p. 113674, 2025.
4. P. Yan et al., "The potential of machine learning to predict melting response time of phase change materials in triplex-tube latent thermal energy storage systems," *Appl. Energy*, vol. 390, p. 125863, 2025.
5. F. A. Barqawi, "Dynamic simulation of phase change material-integrated solar water heating systems: A machine learning approach to energy conversion optimization," *Muthanna J. Eng. Technol.*, vol. 13, no. 3, pp. 1–14, 2025.

## Engineering assumptions

| Assumption | Rationale |
|------------|-----------|
| ERA5 only (no NASA POWER) | Available in pipeline; NASA POWER planned for cross-validation |
| District = mean of grid-city daily T_peak | 222 ERA5 points per district; avoids single-cell bias |
| RT38 used where PROJECT SUMMARY lists RT38HC | Catalog limitation |
| Annual top-3 by mean GRG across months | Presentation summary; extends Chen single-winner GRG |

## Code mapping

| Code variable | Paper / spec |
|---------------|--------------|
| `GHI_COEFF = 0.02` | datafusion.txt |
| `FILTER_TOLERANCE_C = 5.0` | datafusion.txt; Yan (2025) |
| `ZETA = 0.5` | Chen (2025) Eq. (16) |
| `GRG_WEIGHTS` | datafusion.txt; Singh (2025) ordering |
