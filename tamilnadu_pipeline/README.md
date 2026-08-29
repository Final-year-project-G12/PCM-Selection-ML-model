# ERA5 Tamil Nadu Pipeline (population-weighted points, sun-event-aligned, 10-year)

Same method as the Rajasthan pipeline your friend built, applied to Tamil
Nadu, so both regions' outputs are directly comparable for the cross-region
clustering step (Objective 1).

## Pipeline overview

```
00a_build_population_grid.py   →  data/processed/population_grid_points.csv
00b_build_suntimes.py          →  data/processed/suntimes.csv
01_download_era5_tamilnadu.py  →  data/raw/era5/points/*.nc
01b_download_nasapower.py      →  data/raw/nasapower/*.json
00_unzip_accum.py              →  (fixes zip-disguised-as-.nc files in place)
02_combine_tamilnadu.py        →  data/processed/climate_tamilnadu_points.csv
```

## Run order

```
python 00a_build_population_grid.py
python 00b_build_suntimes.py
python 01_download_era5_tamilnadu.py
python 01b_download_nasapower.py
python 00_unzip_accum.py
python 02_combine_tamilnadu.py
```

Each script is resumable — safe to Ctrl-C and re-run.

## What's different from the old (222-city, fixed-hour, 2-year) TN pipeline

| | Old pipeline | This pipeline |
|---|---|---|
| Sampling locations | 260+ hand-picked named cities | population-weighted 0.25° grid cells covering ~87.5% of state population |
| Time sampling | fixed hourly, all 24h/day | 3 events/day: sunrise, solar noon, sunset (astronomically computed per point/date via pvlib) |
| Study period | 2024–2025 (2 years) | 2016–2025 (10 years) |
| Cross-check source | ERA5 only | ERA5 **and** NASA POWER, side by side per row |
| Output | `climate_tamilnadu_all.csv` | `climate_tamilnadu_points.csv` |

## Requirements

```
pip install geopandas rasterio requests pandas numpy xarray netCDF4 pvlib scipy cdsapi
```

`geopandas`/`rasterio` are only needed for `00a`.

## Notes

- **Puducherry**: GADM's "Tamil Nadu" polygon does not include the
  Puducherry union territory enclaves. Set `INCLUDE_PUDUCHERRY = True` in
  `00a_build_population_grid.py` if you want them folded in.
- **Elevation**: population points don't carry elevation, so
  `02_combine_tamilnadu.py` uses a flat 150m approximation for solar-geometry
  calculations (Tamil Nadu's population sits mostly on the coastal plain
  and interior plateau — lower than Rajasthan's 300m approximation).
- **First day of the dataset**: 2016-01-01 has no prior-day predecessor
  hour for deaccumulation if a sun event's window touches hour 0 UTC —
  affected `era5_GHI`/related columns for that one day come out `NaN`
  rather than a wrong value. Every other month boundary is bridged
  automatically.
- If you already ran the *old* 222-point/2-year pipeline in this same
  project folder, its files live under `data/raw/era5/grid/` and are
  untouched by any script here — the new pipeline uses entirely separate
  paths (`data/raw/era5/points/`).

## Next: cross-region clustering (Objective 1)

Once all four regions (Tamil Nadu, Rajasthan, + the other two) have a
`climate_{region}_points.csv`, run `03_cluster_regions.py` to combine them
into one climate-feature table, cluster it (k-means over standardized
GHI/T_amb/RH/cloud-cover/seasonal features, weighted by population), and
rank Top-2/Top-3 PCM candidates per cluster using the same GRA methodology
already in `data_fusion_methodology.md`.
