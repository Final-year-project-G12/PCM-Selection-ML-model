# 12 — Solar Geometry Calculations

## Calculations in `02_combine_tamilnadu.py`
For every timestamp, the solar position is calculated using `pvlib.location.Location.get_solarposition()`:
- **Solar Zenith Angle (SZA)**: Angle between the sun and the vertical.
- **Solar Azimuth Angle**: Angle of the sun along the horizon.
- **Extraterrestrial Radiation (ETR)**: Incident solar radiation at the top of the atmosphere.
- **Clear-sky GHI**: Maximum horizontal solar radiation under clear skies, calculated using the **Ineichen clear-sky model** (`pvlib.location.Location.get_clearsky(model="ineichen")`).

## Application
- Night-masking: When `SZA >= 90.0` (sun below horizon), solar variables (`GHI`, `DNI`, `DHI`, `GHI_clearsky`, `CSI`) are forced to 0.0.
- Clearness Index: `CSI = GHI / GHI_clearsky` (capped at 1.5).
