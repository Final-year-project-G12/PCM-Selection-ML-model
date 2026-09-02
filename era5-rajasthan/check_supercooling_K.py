import pandas as pd
from pathlib import Path

survivors_file = Path('d:/Final Year Project/PCM-Selection-ML-model/era5-rajasthan/data/processed/feasibility_survivors_rajasthan_kappa_calibrated.csv')
survivors = pd.read_csv(survivors_file)
print('Columns in survivors CSV:')
print(survivors.columns.tolist())
print()
survivors = survivors[survivors['survives_all']].drop_duplicates(subset='pcm_id')
print('Supercooling data:')
print(survivors[['pcm_id', 'Tm_C', 'supercooling_K']].sort_values('supercooling_K', ascending=False).to_string(index=False))
print()
print(f'Mean: {survivors["supercooling_K"].mean():.2f} K')
print(f'Std: {survivors["supercooling_K"].std():.2f} K')
print(f'Min: {survivors["supercooling_K"].min():.2f} K')
print(f'Max: {survivors["supercooling_K"].max():.2f} K')
