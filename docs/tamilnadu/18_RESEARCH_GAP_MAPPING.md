# 18 — Research Gap Mapping

Objective 1 addresses the broader final-year project goals. The table below maps how each phase contributes to resolving the key research gaps:

| Phase | Research Gap | How It Contributes |
|---|---|---|
| **Phase 1 — Data Collection** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Establishes the population-weighted grid and temporal solar windows to represent real meteorological stress. |
| **Phase 2 — Preprocessing & QA** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Filters, cleans, and imputes historical weather, providing a high-fidelity dataset for ML/DRL. |
| **Phase 3 — Climate Signature** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Distills raw weather into PCM-facing thermal targets. |
| **Phase 4 — GMM Clustering** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Discovers 5 spatial climate regimes, replacing arbitrary administrative boundaries with GMM profiles. |
| **Phase 5 — Feasibility Filter** | **RG5**: Lack of predictive optimization under climatic uncertainty. | Implements 8 physical screening constraints to prevent compensatory MCDM errors. |
| **Phase 6 — MCDM Ranking** | **RG5**: Lack of predictive optimization under climatic uncertainty. | propagation-weighted ranking identifies Top-3 candidates robust to parameter uncertainty. |
| **Phase 7 — Physics Validation** | **RG4**: Limited real-world experimental validation. | Provides a grey-box simulation model to verify that MCDM rankings correlate with physical solar fraction. |
| **Phase 8 — Rec Cards** | **RG3**: Poor alignment with household demand. | Distills the final recommendations into actionable cards aligned to domestic hot water profiles. |
| **Phase 9+ — DRL Controller** | **RG1**: Lack of real-time adaptive control. | Future work: DRL uses discovered regimes to optimize PCM charging online (Emami 2026; Terfai 2025). |

## Literature Support
| Research Gap | Key Reference | Source |
|---|---|---|
| RG3 — household demand alignment | Odoi & Yorke (2025) | `sources/OdoiYorke2025AI_SWH_Review_summary.md` |
| RG4 — experimental validation | Barqawi (2025); Yan et al. (2025) | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md`, `sources/Yan2025ML_MeltingTime_TriplexTube_PCM_summary.md` |
| RG5 — climatic uncertainty | Ghodusinejad (2026); Mansouri (2025) | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| RG1 — adaptive control (future) | Emami (2026) DRL; Terfai (2025) MPC | `sources/Emami2026DRL_Solar_ORC_TES_summary.md`, `sources/Terfai2025SSP_ANN_MPC_Experimental_summary.md` |
| Techno-economic gap | Duraivel (2025); Chopra (2023) | `sources/Duraivel2025DSTS_TechnoEconomic_summary.md`, `sources/Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md` |
