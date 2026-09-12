# 17 — Literature Mapping Matrix

The methodological choices in the Tamil Nadu implementation are mapped to peer-reviewed references from `sources/` (literature summaries in `sources.zip`):

| Methodological Component | Tamil Nadu Implementation | Supporting Paper | Source File | Evidence |
|---|---|---|---|---|
| **Population Grid** | 133 population-weighted points via WorldPop 2020 | GADM & WorldPop manuals | — | Standard geographic practice |
| **Solar Position (SPA)** | pvlib Reda & Andreas (2004) | Reda & Andreas (2004) | — | Strong |
| **Cross-Source GHI Validation** | MBE/RMSE/r + quantile mapping | Ghodusinejad et al. (2026) | `Ghodusinejad2026SolarIrradianceForecasting_summary.md` | Strong |
| **Bias Correction** | Per-season empirical quantile mapping | Mansouri et al. (2025) | `Mansouri2025MultimodalRenewableForecasting_summary.md` | Medium |
| **SWH Discomfort/HSI** | Thom's Discomfort Index | Thom (1959) | — | Strong |
| **Hot Water Demand** | 300 L daily draw at 50°C | Avargani et al. (2021) | — | Medium |
| **PCM Database** | 62 candidates: 55 manufacturer-derived MICE+RF+PMM-completed records plus 7 literature records | Singh et al. (2025); Martinez (2025) | `Singh2025PCM_SWH_ComprehensiveReview_summary.md`, `Martinez2025PCM_Industrial_TES_summary.md` | Strong |
| **Climate Sizing** | Tm_target worst-month ratio | Durin et al. (2018) | — | Strong |
| **GMM Regime Discovery** | K=5, diagonal covariance | Liu et al. (2025) | `Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` | Medium |
| **MCDM Stack** | TOPSIS, GRA, PROMETHEE II, VIKOR + Borda | Chen et al. (2025) | `Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md` | Strong |
| **Uncertainty MC** | Dirichlet weights + Gaussian properties | Chopra et al. (2023) | `Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md` | Strong |
| **Grey-box Tank Model** | Lumped-enthalpy 3-phase, backward Euler | Barqawi (2025) | `Barqawi2025DynamicSimulationPCM_SWH_summary.md` | Strong |
| **PCM Modeling Review** | Melting band, cycling, supercooling | Abdellatif (2025) | `Abdellatif2025PCM_Modeling_Review_summary.md` | Strong |
| **AI/DRL Future Control** | Phase 9+ adaptive controller | Emami (2026); Terfai (2025) | `Emami2026DRL_Solar_ORC_TES_summary.md`, `Terfai2025SSP_ANN_MPC_Experimental_summary.md` | Medium |
| **Techno-Economic Sizing** | DSTS system optimization | Duraivel (2025) | `Duraivel2025DSTS_TechnoEconomic_summary.md` | Medium |
| **SWH State of Art** | System-level context | Al-Mamun (2023); Odoi & Yorke (2025) | `AlMamun2023SWH_StateOfArt_summary.md`, `OdoiYorke2025AI_SWH_Review_summary.md` | Strong |
| **Spearman Validation** | MCDM vs simulated solar fraction | Framework doc §10 | — | Strong |

## Full Reference List
1. Reda, I. and Andreas, A., 2004. Solar position algorithm for solar radiation applications. *Solar Energy*, 76(5), pp.577-589.
2. Thom, E.C., 1959. The discomfort index. *Weatherwise*, 12(2), pp.57-61.
3. Avargani, A.M., et al., 2021. Numerical and experimental investigation of a solar water heater with PCM. *Journal of Energy Storage*, 42, p.103021.
4. Durin, A., et al., 2018. Worst Month and Critical Period methods for sizing solar irrigation. *Solar Energy*, 174, pp.100-112.
5. Barqawi, M., 2025. Dynamic simulation of PCM storage in solar water heaters. *Journal of Solar Thermal Engineering*.
6. Ghodusinejad, M.H., et al., 2026. Systematic review of solar irradiance forecasting. *Solar Compass*, 17, 100154.
7. Singh, R., et al., 2025. PCM in solar water heating: comprehensive review. *Renewable and Sustainable Energy Reviews*.
8. Chen, et al., 2025. Taguchi + GRA for PCM-nanofluid SWH optimization. *Applied Thermal Engineering*.
9. Liu, et al., 2025. AI for PCM TES prediction and optimization. *Energy*.
