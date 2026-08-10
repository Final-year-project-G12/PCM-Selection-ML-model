# 02 — Data Sources and Variables

## Variable Mapping Table
The following variables are collected, processed, and validated across the pipeline:

| Variable | ERA5 Variable Name | Original Unit | Stored Unit | Transformation | Reason / Physical Meaning |
|---|---|---|---|---|---|
| **GHI** | `ssrd` | J/m² (accumulated) | W/m² | `(deaccumulate(ssrd) / 3600)` | Global Horizontal Irradiance (horizontal solar input) |
| **DNI** | `msdwswrf` | W/m² (mean rate) | W/m² | `.clip(0)` (no division) | Direct Normal Irradiance (direct solar beam) |
| **DHI** | Derived | W/m² | W/m² | `(GHI - DNI * cos(SZA))` | Diffuse Horizontal Irradiance (scattered solar input) |
| **T_amb** | `t2m` | K | °C | `kelvin - 273.15` | Ambient air temperature at 2 m |
| **T_dew** | `d2m` | K | °C | `kelvin - 273.15` | Dewpoint temperature at 2 m |
| **RHum** | Derived | % | % | Magnus-Tetens formula on T/Td | Relative humidity |
| **W_spd** | `u10`, `v10` | m/s | m/s | `sqrt(u10² + v10²)` | Wind speed at 10 m |
| **W_dir** | `u10`, `v10` | m/s | Degrees | `(atan2(u,v) + 360) % 360` | Wind direction in degrees |
| **P_atm** | `sp` | Pa | hPa | `sp / 100.0` | Surface atmospheric pressure |
| **cloud_cover** | `tcc` | Fraction (0–1) | Fraction (0–1) | None | Total cloud fraction |
| **precipitation** | `tp` | m (accumulated) | mm | `deaccumulate(tp) * 1000` | Hourly precipitation |
| **SZA** | Derived (pvlib) | Degrees | Degrees | Solar position algorithm (SPA) | Solar Zenith Angle |
| **solar_azimuth**| Derived (pvlib) | Degrees | Degrees | Solar position algorithm (SPA) | Solar azimuth angle |
| **GHI_clearsky** | Derived (pvlib) | W/m² | W/m² | Ineichen clear-sky model | Ideal/maximum horizontal solar input |
| **CSI** | Derived | Fraction | Fraction | `GHI / GHI_clearsky` | Clearness Index |

## NASA POWER Variables (Tier 2 cross-source validation)
- `ALLSKY_SFC_SW_DWN` (GHI, W/m²)
- `CLRSKY_SFC_SW_DWN` (Clear-sky GHI, W/m²)
- `T2M` (Ambient temperature at 2 m, °C)
- `RH2M` (Relative humidity at 2 m, %)
- `WS10M` (Wind speed at 10 m, m/s)
