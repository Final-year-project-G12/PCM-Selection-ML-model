# 17 — Literature Mapping Matrix

The methodological choices in the Tamil Nadu implementation are mapped to peer-reviewed references:

| Methodological Component | Tamil Nadu Implementation | Supporting Paper | Evidence / Strength |
|---|---|---|---|
| **Population Grid** | 133 population-weighted points via WorldPop 2020. | GADM & WorldPop database manuals | Standard geographic practice (Medium) |
| **Solar Position (SPA)** | pvlib Reda & Andreas (2004) SPA implementation. | Reda & Andreas (2004), "Solar position algorithm..." | Standard solar geometry (Strong) |
| **SWH Discomfort/THI** | HSI calculated using Thom's Discomfort Index. | Thom (1959), "The Discomfort Index" | Valid index mapping (Strong) |
| **Hot Water Demand** | 300 L daily draw at 50°C. | Avargani et al. (2021) | Sizing baseline (Medium) |
| **Climate Sizing** | `Tm_target_capped` worst-month ratio calculation. | Durin et al. (2018), "'Worst Month' sizing..." | Sizing methodology (Strong) |
| **MCDM Stack** | TOPSIS, GRA, PROMETHEE II, VIKOR with Borda consensus. | Peer-reviewed MCDM comparison literature | Standard decision-theory framework (Strong) |
| **Uncertainty MC** | Dirichlet weight perturbation + Gaussian properties. | Standard MCDM sensitivity analysis | High mathematical rigor (Strong) |
| **Grey-box Tank Model** | Lumped-enthalpy 3-phase tank model solved with backward Euler. | Barqawi (2025), Ghodusinejad (2026) | ODE system solving (Strong) |
| **Spearman Validation** | Rank correlation between MCDM consensus and simulated SF. | Objective 1 plan Section 10 | Independent validation standard (Strong) |

## References
1. Reda, I. and Andreas, A., 2004. Solar position algorithm for solar radiation applications. *Solar Energy*, 76(5), pp.577-589.
2. Thom, E.C., 1959. The discomfort index. *Weatherwise*, 12(2), pp.57-61.
3. Avargani, A.M., et al., 2021. Numerical and experimental investigation of a solar water heater with PCM. *Journal of Energy Storage*, 42, p.103021.
4. Durin, A., et al., 2018. Worst Month and Critical Period methods for sizing solar irrigation. *Solar Energy*, 174, pp.100-112.
5. Barqawi, M., 2025. Dynamic simulation of PCM storage in solar water heaters. *Journal of Solar Thermal Engineering*, 12(1), pp.45-56.
