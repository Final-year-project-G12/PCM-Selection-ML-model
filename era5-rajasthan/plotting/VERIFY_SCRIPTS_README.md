# Moved → `../PLOTSV2/PLOTS_GUIDE.md` §4

The four verification scripts now live in `../PLOTSV2/` and are documented in
**[`../PLOTSV2/PLOTS_GUIDE.md`](../PLOTSV2/PLOTS_GUIDE.md) §4**.

```
cd ../PLOTSV2 && python run_all_plots_v2.py verify
```

| Script | Output |
| :--- | :--- |
| `verify_01_preprocessing_rajasthan.py` | `verify_preprocessing/` (7 plots) |
| `verify_02_clustering_rajasthan.py` | `verify_clustering/` (6 plots) |
| `verify_03_feasibility_rajasthan.py` | `verify_feasibility/` (6 plots) |
| `verify_04_ranking_rajasthan.py` | `verify_ranking/` (6 plots) |

**Do not use the numbers this document used to quote.** The versions it described
had bugs the PLOTSV2 ones fix — most consequentially, the feasibility script
counted all 186 evaluation rows as survivors and reported "62 survivors" in every
cluster. The correct counts are **9 / 14 / 16**. `PLOTS_GUIDE.md` §4 lists each
correction and the current results (silhouette 0.313, bootstrap ARI 0.827,
constraint breakdown).

See [`README.md`](README.md) for why this folder was retired.
