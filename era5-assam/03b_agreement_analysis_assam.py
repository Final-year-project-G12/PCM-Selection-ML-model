import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import pearsonr
import config

def compute_metrics(y_true, y_pred):
    """Compute MBE, RMSE, and Pearson correlation."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
        
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mbe = np.mean(y_pred - y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r, _ = pearsonr(y_true, y_pred)
    
    return mbe, rmse, r

def build_quantile_map(source_series, target_series, n_quantiles=100):
    """
    Builds an empirical quantile mapping function from source (ERA5) to target (POWER).
    Returns a dictionary of quantiles that can be used for interpolation.
    """
    source_clean = source_series.dropna()
    target_clean = target_series.dropna()
    
    if len(source_clean) == 0 or len(target_clean) == 0:
        return None
        
    q_vals = np.linspace(0, 1, n_quantiles)
    source_quantiles = np.quantile(source_clean, q_vals)
    target_quantiles = np.quantile(target_clean, q_vals)
    
    return {
        'q_vals': q_vals,
        'source_q': source_quantiles,
        'target_q': target_quantiles
    }

def main():
    print("Loading climate_assam_points.csv...")
    input_file = os.path.join(config.PROCESSED_DIR, "climate_assam_points.csv")
    df = pd.read_csv(input_file)

    
    print("Computing ERA5 vs NASA POWER agreement metrics...")
    
    variables = {
        'GHI': ('power_ALLSKY_SFC_SW_DWN', 'era5_GHI'),
        'T_amb': ('power_T2M', 'era5_T_amb'),
        'RHum': ('power_RH2M', 'era5_RHum'),
        'W_spd': ('power_WS10M', 'era5_W_spd')
    }
    
    results = []
    
    # We stratify by season. If season is not available, default to 'All'
    if 'season' not in df.columns:
        df['season'] = 'All'
        
    # We want to focus on daytime for GHI
    day_mask = df['era5_SZA'] < 90
    
    quantile_maps = {}
    
    for season in df['season'].unique():
        df_season = df[df['season'] == season]
        
        for var_name, (power_col, era5_col) in variables.items():
            if power_col not in df.columns or era5_col not in df.columns:
                continue
                
            # For GHI, only compute metrics when the sun is up
            if var_name == 'GHI':
                mask = day_mask & (df['season'] == season)
            else:
                mask = df['season'] == season
                
            df_subset = df[mask]
            
            y_true = df_subset[power_col]
            y_pred = df_subset[era5_col]
            
            mbe, rmse, r = compute_metrics(y_true, y_pred)
            
            results.append({
                'season': season,
                'variable': var_name,
                'MBE': mbe,
                'RMSE': rmse,
                'Pearson_r': r,
                'n_samples': mask.sum()
            })
            
            # Build quantile map
            qmap = build_quantile_map(y_pred, y_true)
            if qmap:
                quantile_maps[f"{season}_{var_name}"] = qmap

    results_df = pd.DataFrame(results)
    output_metrics = os.path.join(config.PROCESSED_DIR, "era5_power_agreement_assam.csv")
    results_df.to_csv(output_metrics, index=False)
    print(f"Metrics saved to {output_metrics}")
    
    # Determine Bias Decision based on GHI MBE across all seasons
    ghi_mbe_mean = results_df[results_df['variable'] == 'GHI']['MBE'].mean()
    ghi_true_mean = df.loc[day_mask, 'power_ALLSKY_SFC_SW_DWN'].mean()
    
    # If mean bias is > 10% of the mean value, we trigger QUANTILE_MAP
    bias_percentage = abs(ghi_mbe_mean / ghi_true_mean) * 100 if ghi_true_mean > 0 else 0
    
    decision = "BACKBONE"
    if bias_percentage > 10:
        decision = "QUANTILE_MAP"
        
    decision_file = os.path.join(config.PROCESSED_DIR, "bias_decision_assam.txt")
    with open(decision_file, 'w') as f:
        f.write(decision)
    print(f"Bias decision: {decision} (Bias: {bias_percentage:.1f}%) saved to {decision_file}")
    
    if decision == "QUANTILE_MAP":
        qmap_file = os.path.join(config.PROCESSED_DIR, "quantile_maps_assam.joblib")
        joblib.dump(quantile_maps, qmap_file)
        print(f"Quantile maps saved to {qmap_file}")
        
if __name__ == "__main__":
    main()
