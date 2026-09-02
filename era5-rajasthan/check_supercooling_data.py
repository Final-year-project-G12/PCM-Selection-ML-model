import pandas as pd
from pathlib import Path

# Load survivors and manufacturer data
base = Path('d:/Final Year Project/PCM-Selection-ML-model/era5-rajasthan')
survivors_file = base / 'data' / 'processed' / 'feasibility_survivors_rajasthan_kappa_calibrated.csv'
manuf_file = Path('d:/Final Year Project/PCM-Selection-ML-model/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv')

survivors = pd.read_csv(survivors_file)
manuf = pd.read_csv(manuf_file)

# Get unique survivors
survivors_unique = survivors[survivors['survives_all']].drop_duplicates(subset='pcm_id')[['pcm_id', 'Tm_C']].copy()

# Merge with manufacturer data to get nucleation temps
survivors_unique = survivors_unique.merge(
    manuf[['product', 'Tm_freezing', 'Tm_nucleation']],
    left_on='pcm_id',
    right_on='product',
    how='left'
)

# Calculate supercooling offset
survivors_unique['delta_T_subcool'] = survivors_unique['Tm_freezing'] - survivors_unique['Tm_nucleation']
survivors_unique = survivors_unique[['pcm_id', 'Tm_freezing', 'Tm_nucleation', 'delta_T_subcool']].sort_values('delta_T_subcool', ascending=False)

print('SUPERCOOLING DATA FOR SURVIVOR PCMS:')
print(survivors_unique.to_string(index=False))
print()
print(f'Mean supercooling: {survivors_unique["delta_T_subcool"].mean():.2f} K')
print(f'Std supercooling: {survivors_unique["delta_T_subcool"].std():.2f} K')
print(f'Min supercooling: {survivors_unique["delta_T_subcool"].min():.2f} K')
print(f'Max supercooling: {survivors_unique["delta_T_subcool"].max():.2f} K')
print(f'Unique values: {sorted(survivors_unique["delta_T_subcool"].unique())}')
