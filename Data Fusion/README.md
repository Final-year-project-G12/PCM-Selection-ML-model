# Climate–PCM Data Fusion (GRG)

District-level PCM selection for Tamil Nadu solar water heating using **Grey Relational Analysis (GRA)** on ERA5 climate data and Rubitherm/PLUSS PCM properties.

## Quick start

```bash
cd "Data Fusion"
pip install pandas numpy
python 05_grg_climate_fusion.py
```

## Inputs

| File | Source |
|------|--------|
| `../era5-tamilnadu-pipeline/data/processed/climate_tamilnadu_all.csv` | ERA5 hourly climate (38 districts) |
| `../PCM_data/pcm_cleaned.csv` | 18 SWH-suitable PCMs (Singh 2025 band) |

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `district_pcm_monthly.csv` | Best PCM per district × month |
| `district_pcm_grg_rankings.csv` | Full GRG ranking audit trail |
| `district_pcm_top3_annual.csv` | Top-3 PCMs per district (mean GRG) |

## Pipeline steps

1. **T_peak** = daily max(T_amb + 0.02 × GHI), aggregated to district-month mean/min/max
2. **Filter** PCMs where T_melt ∈ [T_peak_min − 5, T_peak_max + 5]
3. **GRG** rank survivors (Chen 2025 Eqs. 15–17, ζ = 0.5)
4. **Merge** into unified monthly table + annual top-3 summary

## Primary references

- **Chen et al. (2025)** — GRA/GRG methodology
- **Singh et al. (2025)** — PCM selection criteria priority
- **Kou et al. (2025)** — Climate-dependent T_m alignment
- **Yan et al. (2025)** — ±5 °C T_m tolerance precedent

See [`docs/data_fusion_methodology.md`](docs/data_fusion_methodology.md) for full literature traceability.

## Files

```
Data Fusion/
├── grg_utils.py                  # T_peak, filter, GRG helpers
├── 05_grg_climate_fusion.py      # Main pipeline
├── docs/
│   └── data_fusion_methodology.md
├── data/processed/               # Generated CSVs
└── README.md
```
