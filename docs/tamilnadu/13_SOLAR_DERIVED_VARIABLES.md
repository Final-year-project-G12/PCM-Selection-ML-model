# 13 — Solar Derived Variables

## Solar Splits
When direct solar radiation (`DNI`) is not directly available, it is derived. In `02_combine_tamilnadu.py`:
- **GHI (Global Horizontal Irradiance)**: Read from `ssrd` (after `accum_to_flux()`, v3.1).
- **DNI (Direct Normal Irradiance)**:
  - If a direct radiation variable like `avg_sdirswrf` (mean direct solar flux) is present, DNI is set to it.
  - If missing, it falls back to:
    `DNI = GHI / cos(SZA)`, clipped to `[0, 1400]` W/m².
- **DHI (Diffuse Horizontal Irradiance)**:
  - Calculated by subtracting the direct component from the total horizontal:
    `DHI = GHI - DNI * cos(SZA)`, clipped to non-negative.

## Physical Meaning
- GHI represents the total solar radiation on a horizontal surface (direct + diffuse).
- DNI represents the direct beam of the sun perpendicular to the rays (critical for concentrating collectors).
- DHI represents the scattered light from the sky dome.
- These variables determine the charging power of flat-plate and evacuated-tube solar collectors.

## Literature Support
| Variable | Reference | Source |
|---|---|---|
| GHI/DNI/DHI decomposition | Ghodusinejad et al. (2026) — irradiance components | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| Collector charging power | Singh et al. (2025) — SWH collector sizing | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Clearness index CSI | Standard solar meteorology | `12_SOLAR_GEOMETRY.md` |
| Assareh (2023) PCM enhancement | Assareh et al. (2023) | `sources/Assareh2023EnhancingSolarThermalPCM_summary.md` |
