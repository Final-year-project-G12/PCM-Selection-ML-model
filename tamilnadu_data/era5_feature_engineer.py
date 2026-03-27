"""
era5_feature_engineer.py
========================
Builds the full feature-engineered dataset from the base CSV + extra NC files.

Sources:
  Base CSV   : era5_climate_tamilnadu_2024.csv   (GHI, T_amb, Wind_speed, RH, Cloud, Precip)
  Instant NC : era5_2024_MM__*instant*.nc         (d2m → T_dew, u10/v10 → W_dir)
  Extra NC   : era5_2024_MM_extra*.nc             (fdir → DNI/DHI, strd → LW_down, sp → P_atm)
  pvlib      : SZA, azimuth, ETR, GHI_clearsky, CSI, sunrise, sunset
  Static     : city, altitude_m, climate_zone, T_set, high_solar_resource

Output: era5_climate_tamilnadu_2024_features.csv
"""

import os, re, glob
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON SOLAR GEOMETRY (fallback if pvlib fails)
# ─────────────────────────────────────────────────────────────────────────────
def solar_geometry_no_pvlib(df, lat_col='latitude', lon_col='longitude', time_col='timestamp'):
    """
    Compute solar zenith angle and clear-sky GHI without pvlib.
    Uses Spencer (1971) solar declination model — accurate to ±0.5°.
    Returns DataFrame with added columns: SZA, solar_azimuth, ETR, GHI_clearsky.
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    dt = df[time_col]
    lat_rad = np.radians(df[lat_col].values)
    lon_deg = df[lon_col].values
    
    # --- Day of year ---
    doy = dt.dt.dayofyear.values.astype(float)
    B   = 2 * np.pi * (doy - 1) / 365.0   # radians
    
    # --- Solar declination (Spencer 1971) ---
    decl = (0.006918
            - 0.399912 * np.cos(B)
            + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B)
            + 0.000907 * np.sin(2*B)
            - 0.002697 * np.cos(3*B)
            + 0.00148  * np.sin(3*B))   # radians
    
    # --- Equation of time (minutes) ---
    eot = (0.000075
           + 0.001868 * np.cos(B)
           - 0.032077 * np.sin(B)
           - 0.014615 * np.cos(2*B)
           - 0.04089  * np.sin(2*B)) * 229.18
    
    # --- Solar time correction ---
    utc_hour = dt.dt.hour.values + dt.dt.minute.values / 60.0
    solar_time = utc_hour + lon_deg / 15.0 + eot / 60.0   # decimal hours
    
    # --- Hour angle ---
    hour_angle = np.radians((solar_time - 12.0) * 15.0)   # radians
    
    # --- Solar zenith angle ---
    cos_zenith = (np.sin(lat_rad) * np.sin(decl)
                  + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    
    zenith_deg = np.degrees(np.arccos(cos_zenith))
    
    df['SZA'] = zenith_deg
    
    # --- Solar azimuth (0=N, 90=E, 180=S, 270=W) ---
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_decl = np.sin(decl)
    cos_decl = np.cos(decl)
    sin_ha = np.sin(hour_angle)
    cos_ha = np.cos(hour_angle)
    
    # Azimuth angle using standard formula
    num = sin_ha
    denom = (sin_lat * cos_ha - cos_lat * sin_decl * sin_ha)
    denom = np.clip(denom, -1.0, 1.0)
    
    azimuth_rad = np.arctan2(num, denom)  # radians from -π to π
    azimuth_deg = np.degrees(azimuth_rad)  # -180 to 180
    azimuth_deg = (azimuth_deg + 180) % 360  # convert to 0-360, where 0=N
    
    df['solar_azimuth'] = azimuth_deg
    
    # --- Extraterrestrial radiation (Bourges 1985) ---
    # Seasonal variation: ~±3.3% from mean ~1360.8 W/m²
    Gsc = 1360.8  # solar constant (W/m²)
    Rn = doy / 365.25 * 2 * np.pi
    I0 = Gsc * (1.00011 + 0.034221*np.cos(Rn) + 0.00128*np.sin(Rn) 
                          + 0.000719*np.cos(2*Rn) + 0.000077*np.sin(2*Rn))
    
    df['ETR'] = I0
    
    # --- Clear-sky GHI (Haurwitz 1945) ---
    # Simple and accurate model: GHI_cs = 910*cos(z)*exp(-0.059/cos(z)) for cos(z) > 0
    cos_z_safe = np.where(cos_zenith > 0.0, cos_zenith, 0.0)
    clear_sky_ghi = 910.0 * cos_z_safe * np.exp(-0.059 / (cos_z_safe + 1e-6))
    clear_sky_ghi = np.where(cos_zenith > 0.0, clear_sky_ghi, 0.0)
    
    df['GHI_clearsky'] = clear_sky_ghi
    
    # --- Clear-sky index ---
    csi = np.where(
        clear_sky_ghi > 10.0,
        df['GHI'].values / clear_sky_ghi,
        0.0
    )
    df['CSI'] = np.clip(csi, 0.0, 1.5)
    
    return df

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_CSV     = os.path.join(SCRIPT_DIR, 'data', 'era5_climate_tamilnadu_2024.csv')
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, 'data', 'era5_climate_tamilnadu_2024_features.csv')

# Tamil Nadu city label map (closest city to each 0.25° grid cell centre)
# Key = (lat_rounded, lon_rounded), Value = city name
# Pre-built for the 13 major Tamil Nadu 0.25° cells covering main cities
CITY_MAP = {
    (13.0, 80.25): 'Chennai',   (13.0, 80.0): 'Chennai',
    (13.25, 80.25): 'Chennai',  (12.75, 80.0): 'Chennai',
    (11.0, 77.0):  'Coimbatore',(11.0, 76.75): 'Coimbatore',
    (10.75,76.75): 'Coimbatore',(11.25,76.75): 'Coimbatore',
    (9.75, 78.0):  'Madurai',   (9.75, 77.75): 'Madurai',
    (10.0, 78.0):  'Madurai',   (10.75,78.5):  'Tiruchirappalli',
    (10.5, 78.25): 'Tiruchirappalli', (12.5, 79.25): 'Vellore',
    (8.5,  76.75): 'Tirunelveli',(10.0, 79.0): 'Thanjavur',
    (11.5,  79.5): 'Cuddalore', (13.0,  80.25):'Chennai',
}

# Climate zone classification (simplified Koppen for Tamil Nadu latitudes)
def get_climate_zone(lat, lon):
    if lat >= 12.5:  return 'Aw'    # Tropical savanna (north TN / Chennai)
    if lat >= 10.0:  return 'Aw'    # Tropical savanna (central TN)
    return 'Am'                     # Tropical monsoon (south TN)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: open + clean an NC file → xarray Dataset
# ─────────────────────────────────────────────────────────────────────────────
def open_nc(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4)
    engine = 'netcdf4' if header[:4] == b'\x89HDF' else 'scipy'
    try:
        ds = xr.open_dataset(filepath, engine=engine)
    except Exception:
        for eng in ['netcdf4', 'scipy', 'h5netcdf']:
            try: ds = xr.open_dataset(filepath, engine=eng); break
            except: continue
        else: return None
    for dim in ['number', 'surface', 'step']:
        if dim in ds.dims and ds.sizes.get(dim, 0) == 1:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim, errors='ignore')
    if 'expver' in ds.dims:
        ds = ds.sel(expver=ds.expver[0], drop=True)
    for tname in ['time', 'valid_time']:
        if tname in ds.coords and tname != 'valid_time':
            ds = ds.rename({tname: 'valid_time'}); break
    return ds

def deaccum_per_grid(df, col, new_col):
    """Deaccumulate an accumulated field per grid point, then ÷3600 → W/m²."""
    df = df.sort_values(['latitude', 'longitude', 'timestamp'])
    df[col] = df.groupby(['latitude', 'longitude'])[col].diff().clip(lower=0)
    df[new_col] = (df[col] / 3600.0).clip(lower=0)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load base CSV
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ERA5 FULL FEATURE ENGINEERING")
print("=" * 65)

if not os.path.exists(BASE_CSV):
    print(f"❌ Base CSV not found: {BASE_CSV}"); exit()

print(f"\n📂 Loading base CSV ...")
df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
df['latitude']  = df['latitude'].round(4)
df['longitude'] = df['longitude'].round(4)
print(f"   Shape: {df.shape}  |  Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: T_dew and W_dir from existing instant .nc files
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂 Extracting T_dew and W_dir from instant .nc files ...")

instant_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'data', 'raw', '*instant*.nc')))
print(f"   Found {len(instant_files)} instant files")
if instant_files:
    raw_dfs = []
    for f in instant_files:
        try:
            ds = open_nc(f)
            if ds is None:
                print(f"   ⚠️  Failed to open {os.path.basename(f)} (open_nc returned None)")
                continue
            raw = ds.to_dataframe().reset_index()
            ds.close()
            raw = raw.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
            for tcol in ['valid_time', 'time']:
                if tcol in raw.columns:
                    raw['timestamp'] = pd.to_datetime(raw[tcol]); break
            keep = ['timestamp', 'latitude', 'longitude']
            for c in ['d2m', 'u10', 'v10']:
                if c in raw.columns: keep.append(c)
            if len(keep) > 3:  # only add if we have more than just timestamp/lat/lon
                raw_dfs.append(raw[[k for k in keep if k in raw.columns]])
                print(f"   ✓ {os.path.basename(f)}: {list(set(keep) - {'timestamp', 'latitude', 'longitude'})}")
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(f)}: {str(e)}")

    if raw_dfs:
        raw_combined = pd.concat(raw_dfs, ignore_index=True)
        raw_combined['latitude']  = raw_combined['latitude'].round(4)
        raw_combined['longitude'] = raw_combined['longitude'].round(4)
        raw_combined['timestamp'] = pd.to_datetime(raw_combined['timestamp'])

        # Remove duplicate rows
        raw_combined = raw_combined.drop_duplicates(
            subset=['timestamp', 'latitude', 'longitude'], keep='first'
        )

        # T_dew: d2m K → °C
        if 'd2m' in raw_combined.columns:
            raw_combined['T_dew'] = raw_combined['d2m'] - 273.15
            print(f"   ℹ️  T_dew extracted from d2m: {raw_combined['T_dew'].notna().sum()} values")

        # W_dir: meteorological wind direction (0=N, 90=E, 180=S, 270=W)
        if 'u10' in raw_combined.columns and 'v10' in raw_combined.columns:
            raw_combined['W_dir'] = (
                (270 - np.degrees(np.arctan2(raw_combined['v10'],
                                             raw_combined['u10']))) % 360
            ).round(1)
            print(f"   ℹ️  W_dir extracted from u10/v10: {raw_combined['W_dir'].notna().sum()} values")

        keep_cols = ['timestamp', 'latitude', 'longitude']
        for c in ['T_dew', 'W_dir']:
            if c in raw_combined.columns: keep_cols.append(c)

        df = df.merge(raw_combined[keep_cols], on=['timestamp','latitude','longitude'], how='left', suffixes=('', '_instant'))
        print(f"   ✅ T_dew and W_dir merged")
    else:
        print("  ⚠️  No data extracted from instant .nc files — T_dew/W_dir will be NaN")
        df['T_dew'] = np.nan
        df['W_dir'] = np.nan
else:
    print("  ⚠️  No *instant*.nc files found — T_dew/W_dir will be NaN")
    df['T_dew'] = np.nan
    df['W_dir'] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: fdir (DNI), strd (LW_down), sp (P_atm) from extra .nc files
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂 Extracting DNI/LW_down/P_atm from _extra .nc files ...")

# Process accum and instant files separately!
accum_files   = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'data', 'raw', '*_extra*accum*.nc')))
instant_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'data', 'raw', '*_extra*instant*.nc')))
print(f"   Found {len(accum_files)} accum + {len(instant_files)} instant extra files")

extra_combined = None

# Process accum files (fdir, strd)
if accum_files:
    accum_dfs = []
    for f in accum_files:
        try:
            ds = open_nc(f)
            if ds is None:
                print(f"   ⚠️  Failed to open {os.path.basename(f)}")
                continue
            raw = ds.to_dataframe().reset_index()
            ds.close()
            raw = raw.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
            for tcol in ['valid_time', 'time']:
                if tcol in raw.columns:
                    raw['timestamp'] = pd.to_datetime(raw[tcol]); break
            accum_dfs.append(raw)
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(f)}: {str(e)}")
    
    if accum_dfs:
        extra_combined = pd.concat(accum_dfs, ignore_index=True)
        extra_combined['latitude']  = extra_combined['latitude'].round(4)
        extra_combined['longitude'] = extra_combined['longitude'].round(4)
        extra_combined['timestamp'] = pd.to_datetime(extra_combined['timestamp'])
        extra_combined = extra_combined.sort_values(['latitude', 'longitude', 'timestamp'])
        
        # fdir → avg_sdirswrf (W/m²), DNI proxy
        if 'fdir' in extra_combined.columns:
            extra_combined = deaccum_per_grid(extra_combined, 'fdir', 'avg_sdirswrf')
            print("  ℹ️  avg_sdirswrf ← fdir deaccumulated ÷ 3600")
        
        # strd → LW_down (W/m²)
        if 'strd' in extra_combined.columns:
            extra_combined = deaccum_per_grid(extra_combined, 'strd', 'LW_down')
            print("  ℹ️  LW_down ← strd deaccumulated ÷ 3600")

# Process instant files (sp → P_atm)
if instant_files:
    instant_dfs = []
    for f in instant_files:
        try:
            ds = open_nc(f)
            if ds is None:
                print(f"   ⚠️  Failed to open {os.path.basename(f)}")
                continue
            raw = ds.to_dataframe().reset_index()
            ds.close()
            raw = raw.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
            for tcol in ['valid_time', 'time']:
                if tcol in raw.columns:
                    raw['timestamp'] = pd.to_datetime(raw[tcol]); break
            instant_dfs.append(raw)
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(f)}: {str(e)}")
    
    if instant_dfs:
        instant_combined = pd.concat(instant_dfs, ignore_index=True)
        instant_combined['latitude']  = instant_combined['latitude'].round(4)
        instant_combined['longitude'] = instant_combined['longitude'].round(4)
        instant_combined['timestamp'] = pd.to_datetime(instant_combined['timestamp'])
        
        # sp → P_atm (instantaneous, no deaccum)
        if 'sp' in instant_combined.columns:
            instant_combined['P_atm'] = instant_combined['sp']
            # Also derive altitude from pressure: z = 44330×(1-(P/P0)^0.1903)
            instant_combined['altitude_m'] = (
                44330 * (1 - (instant_combined['P_atm'] / 101325) ** 0.1903)
            ).round(0)
            print("  ℹ️  P_atm ← sp,  altitude_m derived from P_atm. Non-null P_atm: {0}".format(
                instant_combined['P_atm'].notna().sum()))
        
        # Merge instant data into accum data if accum exists
        if extra_combined is not None:
            instant_cols = ['timestamp', 'latitude', 'longitude', 'P_atm', 'altitude_m']
            instant_keep = instant_combined[[c for c in instant_cols if c in instant_combined.columns]].drop_duplicates(
                subset=['timestamp', 'latitude', 'longitude'], keep='first')
            extra_combined = extra_combined.merge(instant_keep, on=['timestamp', 'latitude', 'longitude'], how='left')
        else:
            extra_combined = instant_combined

if extra_combined is not None:
    keep = ['timestamp', 'latitude', 'longitude']
    for c in ['avg_sdirswrf', 'LW_down', 'P_atm', 'altitude_m']:
        if c in extra_combined.columns: keep.append(c)
    
    extra_keep = extra_combined[[c for c in keep if c in extra_combined.columns]].drop_duplicates(
        subset=['timestamp', 'latitude', 'longitude'], keep='first')
    
    # Fill zeros for irradiance fields (nighttime values)
    for col in ['avg_sdirswrf', 'LW_down']:
        if col in extra_keep.columns:
            extra_keep[col] = extra_keep[col].fillna(0)
    
    df = df.merge(extra_keep, on=['timestamp', 'latitude', 'longitude'], how='left')
    print("  ✅ Extra variables merged")
else:
    print("  ⚠️  No data extracted from _extra*.nc files")
    for c in ['avg_sdirswrf', 'LW_down', 'P_atm', 'altitude_m']:
        df[c] = np.nan

# Derive DNI and DHI from fdir and existing GHI
if 'avg_sdirswrf' in df.columns:
    df['DNI'] = df['avg_sdirswrf']
    df['DHI'] = (df['GHI_Wm2'] - df['avg_sdirswrf']).clip(lower=0)
else:
    if 'DNI' not in df.columns: df['DNI'] = np.nan
    if 'DHI' not in df.columns: df['DHI'] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Time features
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏰ Computing time features ...")
df['hour']  = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['DOY']   = df['timestamp'].dt.day_of_year
df['year']  = df['timestamp'].dt.year

season_map  = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
season_name = {0:'Winter', 1:'Spring', 2:'Summer', 3:'Autumn'}
df['season_code'] = df['month'].map(season_map)
df['season']      = df['season_code'].map(season_name)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Solar geometry via pvlib
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Solar geometry (use fallback to avoid pvlib DLL issues)
# ─────────────────────────────────────────────────────────────────────────────
print("\n☀️  Computing solar geometry (pure Python, no pvlib) ...")
try:
    # Note: GHI column is still called 'GHI_Wm2' at this point, passed as a parameter
    df_temp = df.copy()
    df_temp['GHI'] = df_temp['GHI_Wm2']  # Create alias for the function to use
    df_temp = solar_geometry_no_pvlib(df_temp, lat_col='latitude', lon_col='longitude', time_col='timestamp')
    df[['SZA', 'solar_azimuth', 'ETR', 'GHI_clearsky', 'CSI']] = df_temp[['SZA', 'solar_azimuth', 'ETR', 'GHI_clearsky', 'CSI']]
    print(f"   ✓ Solar zenith angle (SZA) calculated: {df['SZA'].notna().sum()} values")
    print(f"   ✓ Solar azimuth calculated: {df['solar_azimuth'].notna().sum()} values")
    print(f"   ✓ Extraterrestrial radiation (ETR) calculated: {df['ETR'].notna().sum()} values")
    print(f"   ✓ Clear-sky GHI calculated: {df['GHI_clearsky'].notna().sum()} values")
    print(f"   ✓ Clear-sky index (CSI) calculated: {df['CSI'].notna().sum()} values")
    print("  ✅ Solar geometry done (Haurwitz 1945 model, Spencer 1971 declination)")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")
    for c in ['SZA', 'solar_azimuth', 'ETR', 'GHI_clearsky', 'CSI']:
        df[c] = np.nan

# Sunrise/Sunset computation (commented out - use SZA > 90 from results if needed)
# Skipped in final output for performance with 3.4M rows
# Users can compute locally: sunrise_hour = first SZA(lat,lon,date) where SZA < 90
# sunset_hour = last SZA(lat,lon,date) where SZA < 90
print("  ⏭️  Sunrise/sunset computation skipped (SZA available for local computation)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: RRTDHS (Relative Required Thermal Discharge per Hour per Setpoint)
# RRTDHS = GHI_avg / (T_set - T_amb_avg)  per Kou 2025
# Computed as instantaneous row-wise approximation
# ─────────────────────────────────────────────────────────────────────────────
T_SET = 26.0   # standard cooling setpoint (°C) — change if needed
df['T_set'] = T_SET
delta_T = (T_SET - df['T_ambient_C']).abs().clip(lower=0.1)  # avoid / 0
df['RRTDHS'] = (df['GHI_Wm2'] / delta_T).round(4)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Static metadata per grid point
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏙️  Adding static metadata ...")

# City lookup
def get_city(lat, lon):
    key = (round(lat, 2), round(lon, 2))
    for (k_lat, k_lon), city in CITY_MAP.items():
        if abs(k_lat - lat) <= 0.13 and abs(k_lon - lon) <= 0.13:
            return city
    return f"TN_{round(lat,2)},{round(lon,2)}"

df['city']         = df.apply(lambda r: get_city(r['latitude'], r['longitude']), axis=1)
df['lat']          = df['latitude']
df['lon']          = df['longitude']
df['climate_zone'] = df.apply(lambda r: get_climate_zone(r['latitude'], r['longitude']), axis=1)

# high_solar_resource: annual mean GHI > 200 W/m²
ghi_mean = df.groupby(['latitude', 'longitude'])['GHI_Wm2'].mean().rename('ghi_mean')
df = df.merge(ghi_mean.reset_index(), on=['latitude', 'longitude'], how='left')
df['high_solar_resource'] = (df['ghi_mean'] >= 200).astype(int)
df.drop(columns='ghi_mean', inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Rename to final schema and select columns
# ─────────────────────────────────────────────────────────────────────────────
df = df.rename(columns={
    'T_ambient_C':          'T_amb',
    'Wind_speed_ms':        'Wind_speed',
    'RH_percent':           'RHum',
    'Precip_mm':            'precipitation',
    'Cloud_cover_fraction': 'cloud_cover',
    'GHI_Wm2':              'GHI',
    'DNI_Wm2':              'DNI',
    'DHI_Wm2':              'DHI',
    'surface_pressure_Pa':  'P_atm_from_nc',   # keep original if available
})

# If P_atm was merged from extra, use that; else use from nc
if 'P_atm' not in df.columns and 'P_atm_from_nc' in df.columns:
    df['P_atm'] = df['P_atm_from_nc']

FINAL_COLS = [
    'timestamp',
    'avg_sdirswrf',      # direct horizontal irradiance (W/m²)
    'T_amb',             # 2m temperature (°C)
    'T_dew',             # dewpoint temperature (°C)
    'RHum',              # relative humidity (%)
    'W_dir',             # wind direction (deg meteorological)
    'Wind_speed',        # wind speed (m/s)
    'GHI',               # global horizontal irradiance (W/m²)
    'DNI',               # direct normal irradiance proxy (W/m²)
    'DHI',               # diffuse horizontal irradiance (W/m²)
    'LW_down',           # longwave downwelling radiation (W/m²)
    'cloud_cover',       # cloud cover fraction (0–1)
    'precipitation',     # precipitation (mm)
    'P_atm',             # surface pressure (Pa)
    'hour',              # hour of day (0–23)
    'month',             # month (1–12)
    'DOY',               # day of year (1–366)
    'year',              # year
    'season',            # season name
    'season_code',       # season number (0=winter … 3=autumn)
    'SZA',               # solar zenith angle (deg)
    'solar_azimuth',     # solar azimuth (deg)
    'ETR',               # extraterrestrial radiation (W/m²)
    'GHI_clearsky',      # clear-sky GHI from pvlib Ineichen (W/m²)
    'CSI',               # clear-sky index = GHI / GHI_clearsky
    'RRTDHS',            # Relative Required Thermal Discharge per Hour per Setpoint
    'city',              # nearest city name
    'lat',               # latitude (°N)
    'lon',               # longitude (°E)
    'altitude_m',        # elevation (m) derived from surface pressure
    'climate_zone',      # Koppen climate zone
    'T_set',             # PCM cooling setpoint temperature (°C)
    'high_solar_resource', # 1 if annual mean GHI ≥ 200 W/m²
]

out_cols  = [c for c in FINAL_COLS if c in df.columns]
missing   = [c for c in FINAL_COLS if c not in df.columns]
final_df  = df[out_cols]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Save
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n💾 Writing {OUTPUT_CSV} ...")
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 Done!")
print(f"   Rows      : {len(final_df):,}")
print(f"   Columns   : {len(out_cols)}  →  {out_cols}")
if missing:
    print(f"   ⚠️  Missing (run download scripts first): {missing}")
print(f"   File size : ~{os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB")
print(f"\nNaN audit (top issues):")
nan_s = final_df.isna().sum()
print(nan_s[nan_s > 0].to_string() if nan_s.any() else "   No NaNs!")
print(f"\nSample (3 rows):\n{final_df.head(3).to_string(index=False)}")
