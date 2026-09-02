# PCM Properties Dataset — PCM List, Preprocessing Methods & Implementation

**Dataset:** `PCM_Properties_55records_42_70C_dense.csv` → `PCM_Properties_cleaned_mice_pmm.csv`
**Scope:** 55 phase change materials (PCMs), melting range 40.5 °C – 70 °C
**Script:** `01_preprocess.py`
**Part of:** Objective 1 — Tamil Nadu climate–PCM ranking pipeline

---

## 1. The PCM List (55 records)

All 55 PCMs, sorted by melting temperature. Melting temperature, latent heat of melting, solid density, and solid thermal conductivity are shown after cleaning/imputation (see Section 2 for how gaps were filled).

| Product | Manufacturer | Type | Tm melting (°C) | Latent heat – melting (kJ/kg) | Density – solid (kg/m³) | TC – solid (W/mK) |
|---|---|---|---:|---:|---:|---:|
| RT42 | Rubitherm Technologies | Organic (RT-line) | 40.5 | 165 | 880 | 0.207 |
| RT44HC | Rubitherm Technologies | Organic (RT-line) | 42.5 | 250 | 800 | 0.306 |
| savE® OM42 | Pluss Advanced Technologies | Organic | 44.0 | 199 | 903 | 0.190 |
| RT47 | Rubitherm Technologies | Organic (RT-line) | 44.5 | 160 | 880 | 0.206 |
| n-Docosane (C22) | Literature | Organic n-alkane | 44.5 | 249 | 880 | 0.307 |
| Lauric acid (C12) | Literature | Organic fatty acid | 44.8 | 184 | 960 | 0.181 |
| RT45HC | Rubitherm Technologies | Organic (RT-line) | 45.0 | 240 | 880 | 0.306 |
| savE® OM46 | Pluss Advanced Technologies | Organic | 47.0 | 177 | 917 | 0.200 |
| n-Tricosane (C23) | Literature | Organic n-alkane | 47.5 | 232 | 797 | 0.307 |
| RT50 | Rubitherm Technologies | Organic (RT-line) | 48.0 | 160 | 880 | 0.208 |
| Paraffin/Expanded graphite (85.56% paraffin) | Literature | Organic/composite blend | 48.8 | 161 | 937 | 0.193 |
| savE® OM49 | Pluss Advanced Technologies | Organic | 49.0 | 224 | 816 | 0.330 |
| Paraffin/HDPE PCM3 | Literature | Organic blend | 49.9 | 187 | 952 | 0.205 |
| PlusICE A50 | PCM Products Ltd. | Organic PCM | 50.0 | 190 | 810 | 0.180 |
| savE® OM50 | Pluss Advanced Technologies | Organic | 50.0 | 189 | 961 | 0.210 |
| Paraffin/HDPE PCM6 | Literature | Organic blend | 50.3 | 187 | 953 | 0.206 |
| Paraffin/HDPE PCM2 | Literature | Organic blend | 50.6 | 187 | 954 | 0.208 |
| savE® OM48 | Pluss Advanced Technologies | Organic | 51.0 | 165 | 960 | 0.200 |
| PlusICE A52 | PCM Products Ltd. | Organic PCM | 52.0 | 220 | 810 | 0.180 |
| n-Tetracosane (C24) | Literature | Organic n-alkane | 52.0 | 255 | 799 | 0.306 |
| Paraffin/Expanded graphite (92% paraffin) | Literature | Organic/composite blend | 52.2 | 170 | 871 | 0.185 |
| Myristic acid (C14) | Literature | Organic fatty acid | 53.0 | 199 | 990 | 0.216 |
| PureTemp 53 | PureTemp | Organic bio-based PCM | 53.0 | 225 | 920 | 0.250 |
| RT54HC | Rubitherm Technologies | Organic (RT-line) | 53.5 | 200 | 850 | 0.199 |
| RT55 | Rubitherm Technologies | Organic (RT-line) | 54.0 | 170 | 880 | 0.197 |
| n-Pentacosane (C25) | Literature | Organic n-alkane | 54.0 | 238 | 801 | 0.305 |
| Myristic acid/NBR-1.0 | Literature | Organic/polymer blend | 54.1 | 128 | 860 | 0.171 |
| Myristic acid/NBR-0.5 | Literature | Organic/polymer blend | 54.6 | 142 | 865 | 0.176 |
| savE® OM55 | Pluss Advanced Technologies | Organic | 55.0 | 188 | 935 | 0.160 |
| Palmitic-stearic acid/Expanded graphite | Literature | Organic/eutectic composite | 55.2 | 176 | 856 | 0.181 |
| RT57HC | Rubitherm Technologies | Organic (RT-line) | 56.5 | 240 | 900 | 0.304 |
| n-Hexacosane (C26) | Literature | Organic n-alkane | 56.5 | 256 | 770 | 0.299 |
| PlusICE A58 | PCM Products Ltd. | Organic PCM | 58.0 | 215 | 910 | 0.220 |
| PureTemp 58 | PureTemp | Organic bio-based PCM | 58.0 | 225 | 890 | 0.250 |
| RT60 | Rubitherm Technologies | Organic (RT-line) | 58.0 | 160 | 880 | 0.192 |
| n-Heptacosane (C27) | Literature | Organic n-alkane | 59.0 | 236 | 773 | 0.303 |
| CrodaTherm 60 | CrodaTherm | Organic PCM | 59.8 | 217 | 922 | 0.290 |
| Palmitic acid/Expanded graphite (80/20) | Literature | Organic/composite blend | 60.9 | 148 | 860 | 0.191 |
| PureTemp 60 | PureTemp | Organic bio-based PCM | 61.0 | 220 | 960 | 0.250 |
| RT65 | Rubitherm Technologies | Organic (RT-line) | 61.5 | 150 | 880 | 0.183 |
| n-Octacosane (C28) | Literature | Organic n-alkane | 61.6 | 253 | 910 | 0.307 |
| PlusICE A62 | PCM Products Ltd. | Organic PCM | 62.0 | 205 | 910 | 0.220 |
| RT62HC | Rubitherm Technologies | Organic (RT-line) | 62.5 | 230 | 850 | 0.304 |
| Palmitic acid (C16) | Literature | Organic fatty acid | 62.6 | 198 | 989 | 0.214 |
| PureTemp 63 | PureTemp | Organic bio-based PCM | 63.0 | 206 | 920 | 0.250 |
| RT64HC | Rubitherm Technologies | Organic (RT-line) | 64.0 | 250 | 880 | 0.303 |
| n-Nonacosane (C29) | Literature | Organic n-alkane | 64.0 | 240 | 790 | 0.303 |
| n-Triacontane (C30) | Literature | Organic n-alkane | 65.4 | 251 | 910 | 0.308 |
| savE® OM65 | Pluss Advanced Technologies | Organic | 67.0 | 188 | 924 | 0.190 |
| Stearic acid (C18) | Literature | Organic fatty acid | 67.9 | 259 | 965 | 0.250 |
| PureTemp 68 | PureTemp | Organic commercial PCM | 68.0 | 213 | 811 | 0.300 |
| n-Hentriacontane (C31) | Literature | Organic n-alkane | 68.0 | 242 | 930 | 0.250 |
| RT69HC | Rubitherm Technologies | Organic (RT-line) | 69.0 | 230 | 940 | 0.304 |
| n-Dotriacontane (C32) | Literature | Organic n-alkane | 69.5 | 170 | 848 | 0.191 |
| RT70HC | Rubitherm Technologies | Organic (RT-line) | 70.0 | 260 | 880 | 0.303 |

**Manufacturer / source breakdown (55 total):**

| Source | Count | Notes |
|---|---:|---|
| Literature | 24 | Pure n-alkanes (C22–C32), fatty acids, and composite/blend PCMs from published studies |
| Rubitherm Technologies (RT-line) | 14 | Scraped from Rubitherm datasheets |
| Pluss Advanced Technologies (savE® OM-line) | 7 | Scraped from Pluss datasheets |
| PureTemp | 5 | Bio-based organic PCMs |
| PCM Products Ltd. (PlusICE A-line) | 4 | Organic PCMs |
| CrodaTherm | 1 | Organic PCM |

All 55 records are **organic PCMs** (n-alkanes, fatty acids, RT-line paraffins, OM-line organic mixtures, bio-based, and organic composite/blend systems) — the dataset was deliberately scoped to the 42–70 °C organic PCM band relevant to Tamil Nadu's climate-load range, and expanded by scraping additional Rubitherm and Pluss datasheets to fill gaps in that range.

---

## 2. Preprocessing Methods Used

The raw scraped/compiled dataset (`PCM_Properties_55records_42_70C_dense.csv`, 22 columns) had **618 missing cells** across its 18 numeric + 2 categorical properties before cleaning. The preprocessing pipeline (`01_preprocess.py`) addressed this in five stages:

### 2.1 Text parsing and standardization
Manufacturer datasheets report values inconsistently — ranges (`"43-37"`, `"48 and 43"`), annotated peaks (`"peak: 45.2"`), and plain numbers all appear in the same column. A custom parser (`parse_messy_numeric`):
- Extracts the `peak:` value when present.
- Converts `"X and Y"` and dash-range formats (`"X-Y"`) to their midpoint, using an anchored regex so a descending range like `"43-37"` isn't misread as `43` and `-37` (a naive `findall` on a bare minus-sign pattern silently corrupts ranges this way).
- Falls back to the first number found for any unrecognized format.
- Preserves the original manufacturer text in a `*_original_text` audit column alongside the parsed numeric value.

### 2.2 Feature engineering
- **Nucleation temperature reparametrization:** nucleation is only meaningful as a *degree of supercooling* relative to a PCM's own freezing point, not as an absolute temperature. It is imputed internally as `Tm_freezing − Tm_nucleation` (the subcooling offset) and converted back to an absolute temperature afterward.
- **Categorical predictors:** the `pcm_type` field (11 distinct organic subtypes — n-alkane, fatty acid, RT-line, composite blend, polymer blend, eutectic composite, bio-based, etc.) and `manufacturer` are one-hot encoded and used as predictors for every imputation model, preserving chemical-family signal.

### 2.3 Numeric imputation — MICE + Random Forest + Predictive Mean Matching (PMM)
This is the core method, chosen specifically to solve a cross-manufacturer missingness problem: several properties (thermal conductivity, specific heat–solid, flammability) are missing across an *entire* product line (e.g. all 14 RT-line rows) and only reported for another line (e.g. Pluss OM). A distance-based nearest-neighbor fill would fail here, since an RT PCM's nearest neighbors are other RT PCMs — which are also missing the value.

**Algorithm** (multiple imputation by chained equations, 8 iterations, 3 donors per cell):
1. **Initialize** every missing cell with the column mean (bootstrap fill).
2. **Order columns** by ascending missingness (standard MICE heuristic — easy columns stabilize first).
3. For each column, in each of 8 iterations:
   - Fit a **Random Forest Regressor** (300 trees, max depth 4, min 2 samples/leaf) using every *other* property plus the type/manufacturer one-hot features as predictors, trained **only on rows where that column is genuinely observed** — regardless of manufacturer.
   - Predict a value for every row (including rows whose entire product line lacks the property) from that row's own characteristics.
   - **PMM donor matching:** rather than using the raw regression prediction, rank the real observed values by how close their *predicted* values are to the target row's predicted value, take the 3 nearest, and fill with an inverse-distance-weighted blend of those real donor values. This guarantees every imputed number is a blend of values that were actually measured somewhere in the dataset — never a synthetic average.
4. **Low-confidence fallback:** for columns with fewer than 3 real observations in the entire dataset (e.g. nucleation temperature, with only 1 known value out of 55), a full regression model can't be fit; these are filled from the mean of whatever real values exist and explicitly flagged `LOW (n<3 donors)` in the provenance log rather than presented as equal-quality to the MICE+RF+PMM result.

### 2.4 Categorical imputation — Random Forest Classifier
`flammability` and `appearance` (48/55 missing for flammability) are filled the same cross-series way: a **Random Forest Classifier** (300 trees, max depth 4) trains on whichever rows have a known label, predicts for every row from its own physical-property profile, and logs the 3 nearest known-label PCMs (Euclidean distance in the same predictor space) as supporting evidence.

### 2.5 Provenance tracking
Every one of the 618 imputed cells is logged with:
- The predicted value and confidence level (Standard / Low).
- The prediction method used.
- Its top-3 donor PCMs, their manufacturers, their actual reported values, and their blend weights.

This produces `05_imputation_provenance.csv` — a fully inspectable, cell-by-cell audit trail answering "which real PCMs supplied this not-reported value, and how much did each contribute?"

---

## 3. Implementation Summary

**Pipeline:** `01_preprocess.py` (single script, run end-to-end)

| Step | Function | Output |
|---|---|---|
| Load & parse | `load_raw()` | Standardized column names, parsed numerics, subcooling feature |
| Impute | `mice_rf_pmm_impute()` | Fully imputed numeric + categorical columns, donor logs |
| Audit print | `print_cross_series_audit()` | Console summary of cross-manufacturer donor usage |
| Provenance table | `build_provenance_table()` | `05_imputation_provenance.csv` |
| Diagnostics | `make_plots()` | 4 PNG figures |
| Final outputs | `main()` | Lean + detailed CSVs |

**Configuration:** 8 MICE iterations, 3 PMM donors/cell, `random_state=42` for reproducibility, scikit-learn `RandomForestRegressor`/`RandomForestClassifier`.

### Results (verified by re-running the script on the uploaded data)

| Property | Missing before | Missing after |
|---|---:|---:|
| Tm_freezing | 29 | 0 |
| Tm_nucleation | 54 | 0 |
| Latent heat – melting | 3 | 0 |
| Latent heat – freezing | 43 | 0 |
| Heat storage capacity | 41 | 0 |
| Density – liquid / solid | 14 / 14 | 0 / 0 |
| Specific heat – liquid / solid | 22 / 24 | 0 / 0 |
| Thermal conductivity – liquid / solid / both | 34 / 39 / 36 | 0 / 0 / 0 |
| Volume expansion | 48 | 0 |
| Max operating temp | 34 | 0 |
| Flash point | 39 | 0 |
| Cycles tested | 48 | 0 |
| Flammability / Appearance | 48 / 48 | 0 / 0 |
| **Total** | **618** | **0** |

**Cross-manufacturer donor audit** (proves the method reaches across product lines when a whole line lacks a property): thermal conductivity–solid, heat-storage capacity, max operating temperature, and cycles-tested were imputed almost entirely (≥89–100%) using at least one donor from a *different* manufacturer than the row's own — exactly the scenario a naive same-line nearest-neighbor fill would have failed on.

### Deliverables produced
- **`PCM_Properties_cleaned_mice_pmm.csv`** — the lean, analysis-ready file: one clean numeric/categorical column per property, 55 rows × 22 columns, no missing values.
- **`PCM_Properties_cleaned_mice_pmm_detailed.csv`** — audit version with per-property imputed-flags and original manufacturer text preserved alongside each value.
- **`05_imputation_provenance.csv`** — 618-row cell-level provenance table (predicted value, confidence, donor PCMs, donor values, donor weights).
- **Diagnostic plots:**
  - `01_missingness_before_after.png` — heatmap of missingness before vs. after.
  - `02_cross_series_donor_audit.png` — % of imputed rows per column using a cross-manufacturer donor.
  - `03_imputed_vs_reported_sanity.png` — strip plot comparing imputed vs. reported values by manufacturer for key properties.
  - `04_correlation_heatmap.png` — correlation matrix of the final imputed numeric properties.

---

## 4. References

**Imputation methodology**
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67. https://doi.org/10.18637/jss.v045.i03
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman and Hall/CRC. https://stefvanbuuren.name/fimd/
- Stekhoven, D. J., & Bühlmann, P. (2012). MissForest — non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118. https://doi.org/10.1093/bioinformatics/btr597
- Little, R. J. A. (1988). Missing-Data Adjustments in Large Surveys. *Journal of Business & Economic Statistics*, 6(3), 287–296. https://doi.org/10.2307/1391878 (predictive mean matching)
- Rubin, D. B. (1986). Statistical Matching Using File Concatenation with Adjusted Weights and Multiple Imputations. *Journal of Business & Economic Statistics*, 4(1), 87–94. https://doi.org/10.2307/1391390
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

**PCM selection / MCDM (relevant to downstream ranking in Objective 1)**
- Sharma, P., Banerjee, R., & Modi, A. A multi-criteria decision making framework for optimal phase change material–insulation combinations in building envelopes (Mumbai, India). *ScienceDirect*. https://www.sciencedirect.com/science/article/abs/pii/S2352152X25041428
- Comparative Framework for Climate-Responsive Selection of Phase Change Materials in Energy-Efficient Buildings. *Energies*, 18(22), 5982. https://doi.org/10.3390/en18225982
- Selection and thermophysical assessment of PCMs for space cooling applications in hot subtropical climates (AHP + TOPSIS + VIKOR). *Numerical Heat Transfer, Part A*, 86(8). https://doi.org/10.1080/10407782.2023.2292183
- Selection of phase change material suitable for building heating applications based on a qualitative decision matrix (TOPSIS). https://www.academia.edu/62989662

**Manufacturer datasheet sources (scraped for this dataset)**
- Rubitherm® RT-line PCM datasheets — https://www.rubitherm.eu/en/productcategory/organische-pcm-rt
- Pluss savE® PCM (OM-series) technical data sheets — https://www.pluss.co.in/knowledge-center/data-sheets/
