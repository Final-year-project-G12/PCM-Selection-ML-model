# 16 — Climate Signature Construction: Climatological Mapping

## Mapping Climate Features to PCM Requirements
The climate signature variables represent specific physical stresses on a solar water heating (SWH) system with PCM storage. The table below outlines the physical justification:

| Climate Feature | Thermal Behavior of SWH | PCM Target/Constraint | PCM Property Impacted |
|---|---|---|---|
| **Ta_mean** (mean air temp) | Governs baseline hot water heat loss to ambient environment. | Derived target Tm_target and L_required. | Selection of optimal melting point. |
| **DTR** (Diurnal Temp Range) | Larger range implies lower night temperatures and higher cooling loads. | Determines thermal demand swing. | Volumetric latent heat capacity. |
| **GHI_daily_kWh** | Controls total solar thermal energy available for collector charging. | Governs sizing and autonomy requirements. | Latent heat storage capacity. |
| **cloudy_frac** | High fraction implies frequent consecutive low-radiation days. | Restricts charging window and capacity. | Latent heat capacity and conductivity. |
| **RH_mean** | High humidity increases convective and condensation losses. | Corrosion veto trigger. | Material compatibility (encapsulation). |
| **wind_mean** | Strong winds cause convective losses from collector glass cover. | Autonomy margin. | Latent heat storage margin. |
| **monsoon_index** | Concentrated monsoon rainfall reduces solar fractions for specific months. | Seasonal PCM suitability. | Melting point window and target width. |

## Interaction Terms
- `int_GHI_x_ktstd`: Flags erratic solar resources where daily integrals are large but variable.
- `int_DTR_x_cloudyfrac`: Captures thermal cycling stress under high weather intermittency.
- `int_RH_x_TaMinusTm`: Measures ambient condensation risk on cold storage boundaries.
- `int_wind_x_TaMinusTsoil`: Estimates evening heat loss from tank to surroundings.
- `int_CCI_x_1minusSAI`: Quantifies combined cloudy autonomy requirement.
