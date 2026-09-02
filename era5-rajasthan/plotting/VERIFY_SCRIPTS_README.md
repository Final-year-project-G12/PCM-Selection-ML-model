# Rajasthan Verification Scripts — Complete Documentation

**Generated:** 2026-09-02  
**Status:** ✅ All 4 scripts functional and tested  
**Output Location:** `outputs/objective1_plots_rajasthan/verify_*/`

This document describes all 4 verification scripts adapted from the Tamil Nadu pipeline for Rajasthan.

---

## Overview

The verification scripts provide **pipeline quality assurance** at each phase:
- **verify_01:** Phase 2 data preprocessing validation
- **verify_02:** Phase 4 clustering validation
- **verify_03:** Phase 5 PCM feasibility screening validation
- **verify_04:** Phase 6 MCDM ranking validation

Each script generates 6 PNG visualization plots + 1 summary figure.

---

## Quick Stats

| Script | Phase | Plots | Size | Status |
|--------|-------|-------|------|--------|
| `verify_01_preprocessing_rajasthan.py` | 2 | 6 | 0.5 MB | ✅ |
| `verify_02_clustering_rajasthan.py` | 4 | 6 | 0.8 MB | ✅ |
| `verify_03_feasibility_rajasthan.py` | 5 | 6 | 0.26 MB | ✅ |
| `verify_04_ranking_rajasthan.py` | 6 | 6 | 0.45 MB | ✅ |
| **TOTAL** | **2-6** | **24 plots** | **2.01 MB** | **✅** |

---

## Script 1: Verify Preprocessing (`verify_01_preprocessing_rajasthan.py`)

**Phase:** 2 (Data Cleaning)  
**Run After:** Phase 2 preprocessing complete  
**Output Directory:** `verify_preprocessing/`

### What It Checks
- ✅ Raw vs. cleaned data dimensions
- ✅ Climate variable distributions (6 core variables)
- ✅ Data completeness/coverage per variable
- ✅ Statistical summary (mean, std)
- ✅ Engineered features (lag, rolling, delta)
- ✅ Variable correlations
- ✅ Data quality metrics

### Plots Generated

1. **01_climate_distributions.png** (Histograms)
   - GHI, T_amb, RHum, W_spd, P_atm, Precipitation
   - Shows before/after preprocessing

2. **02_data_completeness.png** (Bar chart)
   - Coverage % per variable (>95% green, >90% orange, <90% red)
   - Identifies variables with missing data

3. **03_statistical_summary.png** (Bar chart)
   - Mean and Std Dev per variable
   - Quick statistical overview

4. **04_feature_engineering.png** (Histogram)
   - Engineered feature distributions (lag/rolling/delta)
   - Only if engineered features exist

5. **05_correlation_analysis.png** (Heatmap)
   - Pearson correlation between all climate variables
   - Identifies multicollinearity

6. **06_data_quality_metrics.png** (Horizontal bar)
   - Total records, complete cases, variables, engineered features
   - High-level dataset metrics

7. **07_preprocessing_summary.png** (Text report)
   - Input/output record counts
   - Data retention %
   - Per-variable coverage summary

### How to Run
```bash
cd era5-rajasthan/plotting/
python verify_01_preprocessing_rajasthan.py
```

### What to Look For
- ✅ Data retention should be 80%+ after cleaning
- ✅ All core variables >95% coverage
- ✅ No extreme outliers in distributions
- ✅ Reasonable correlations (r² < 0.95 to avoid multicollinearity)

---

## Script 2: Verify Clustering (`verify_02_clustering_rajasthan.py`)

**Phase:** 4 (Clustering)  
**Run After:** Phase 4 clustering complete  
**Output Directory:** `verify_clustering/`

### What It Checks
- ✅ Cluster quality (Silhouette, DB, Calinski-Harabasz)
- ✅ Optimal k selection via elbow curves
- ✅ PCA projection (2D visualization)
- ✅ Geographic distribution (if coordinates available)
- ✅ Cluster profiles (feature distributions by cluster)
- ✅ Cluster balance (size per cluster)

### Plots Generated

1. **01_elbow_curves.png** (4 subplots)
   - Silhouette score (higher is better)
   - BIC (lower is better)
   - Davies-Bouldin (lower is better)
   - Calinski-Harabasz (higher is better)
   - Red line marks actual k value

2. **02_silhouette_plot.png** (Horizontal bar chart)
   - Silhouette coefficient per cluster
   - Average silhouette score marked
   - 0.4 threshold line (good separation)

3. **03_pca_projection.png** (Scatter plot)
   - First 2 PCA components
   - Points colored by cluster
   - Shows low-dimensional separation

4. **04_geographic_map.png** (Scatter plot)
   - Latitude vs. Longitude
   - Points colored by cluster
   - Shows geographic stratification

5. **05_cluster_profiles.png** (6 boxplots)
   - Feature distributions by cluster
   - Top 6 features shown
   - Validates cluster separability

6. **06_cluster_sizes.png** (Bar chart)
   - Number of samples per cluster
   - Identifies imbalanced clusters

### How to Run
```bash
cd era5-rajasthan/plotting/
python verify_02_clustering_rajasthan.py
```

### What to Look For
- ✅ Silhouette score 0.4+ indicates good clustering
- ✅ Davies-Bouldin < 1.5 indicates good separation
- ✅ Clusters should be geographically coherent
- ✅ Balanced cluster sizes (not one huge cluster)
- ✅ Clear cluster separation in PCA projection

---

## Script 3: Verify Feasibility (`verify_03_feasibility_rajasthan.py`)

**Phase:** 5 (Feasibility Screening)  
**Run After:** Phase 5 feasibility filtering complete  
**Output Directory:** `verify_feasibility/`

### What It Checks
- ✅ PCM survival rate per cluster
- ✅ Feasible property space (Tm vs. latent heat)
- ✅ Top candidates per cluster (highest latent heat)
- ✅ Constraint satisfaction (pass/fail counts)
- ✅ Property distributions by cluster
- ✅ Overall survival statistics

### Plots Generated

1. **01_survival_rate_by_cluster.png** (Bar chart)
   - Survivor count or % per cluster
   - Overall survival rate 10-50% is healthy

2. **02_feasible_property_space.png** (Scatter plot)
   - Melting point (Tm) vs. latent heat
   - Points colored by cluster
   - Shows PCM candidates in viable region

3. **03_top_candidates_per_cluster.png** (Horizontal bar)
   - Most frequent top-3 candidates by cluster
   - Shows which PCMs are most versatile

4. **04_constraint_analysis.png** (Stacked horizontal bar)
   - Pass/fail counts for each constraint
   - Shows which constraints are most restrictive

5. **05_property_distributions.png** (Boxplots)
   - PCM properties by cluster (Tm, latent heat, etc.)
   - Identifies property ranges per cluster

6. **06_summary.png** (Text report)
   - Total survivors
   - Survivors per cluster
   - Overall survival rate with interpretation

### How to Run
```bash
cd era5-rajasthan/plotting/
python verify_03_feasibility_rajasthan.py
```

### What to Look For
- ✅ Total survivors 10-50% of all candidates (not too strict/loose)
- ✅ Balanced survivors per cluster
- ✅ Feasible region has clear thermal bounds
- ✅ No single constraint eliminates all candidates
- ✅ Property distributions are realistic

---

## Script 4: Verify Ranking (`verify_04_ranking_rajasthan.py`)

**Phase:** 6 (MCDM Ranking)  
**Run After:** Phase 6 MCDM ranking complete  
**Output Directory:** `verify_ranking/`

### What It Checks
- ✅ Method agreement (Spearman correlation heatmap)
- ✅ Top-3 inclusion probability (robustness)
- ✅ Rank distribution across methods
- ✅ Rank reversal frequency (instability)
- ✅ Consensus vs. individual method agreement
- ✅ Top-3 recommendations summary

### Plots Generated

1. **01_method_correlation.png** (Heatmap)
   - Spearman correlation between ranking methods
   - Should be 0.5+ (good agreement)
   - Red = low correlation, Green = high correlation

2. **02_top3_inclusion_probability.png** (Horizontal bar)
   - % of rankings where each PCM appears in top-3
   - 80%+ = robust recommendation
   - 50-80% = acceptable
   - <50% = fragile

3. **03_rank_distributions.png** (Boxplot)
   - Rank distribution per method
   - Shows whether methods rank similarly
   - Similar boxes = good agreement

4. **04_rank_reversal_frequency.png** (Horizontal bar)
   - Rank instability (spread between methods)
   - Low values = stable rankings
   - High values = method-dependent rankings

5. **05_method_agreement.png** (Scatter plot)
   - Consensus vs. TOPSIS rank
   - Points on diagonal = perfect agreement
   - Deviation from line = disagreement

6. **06_summary.png** (Text report)
   - Method count and ranked candidates
   - Spearman rho between key method pairs
   - Top-3 consensus rankings

### How to Run
```bash
cd era5-rajasthan/plotting/
python verify_04_ranking_rajasthan.py
```

### What to Look For
- ✅ Method correlation 0.5+ indicates good agreement
- ✅ Top-1 PCM should have 70-90% top-3 inclusion
- ✅ Top-ranked PCMs should have low rank reversal
- ✅ Consensus and TOPSIS should correlate well
- ✅ No extreme method outliers

---

## Running All Verification Scripts

### Sequential Execution
```bash
cd era5-rajasthan/plotting/

echo "Running all verify scripts..."
python verify_01_preprocessing_rajasthan.py
python verify_02_clustering_rajasthan.py
python verify_03_feasibility_rajasthan.py
python verify_04_ranking_rajasthan.py

echo "All verification complete!"
```

### Expected Runtime
- verify_01: ~10 seconds
- verify_02: ~30 seconds  
- verify_03: ~5 seconds
- verify_04: ~5 seconds
- **Total: ~50 seconds**

---

## Interpreting Results

### Preprocessing (Script 1) — "Is data clean?"
| Finding | Status | Action |
|---------|--------|--------|
| All variables >95% coverage | ✅ Good | Continue |
| Some variables 90-95% coverage | ⚠️ Fair | Monitor closely |
| Any variable <90% coverage | ❌ Problem | Investigate outlier filtering |
| Extreme outliers in distributions | ❌ Problem | Check data source |
| High correlations (r² > 0.95) | ⚠️ Caution | May cause MCDM weighting issues |

### Clustering (Script 2) — "Are clusters well-separated?"
| Finding | Status | Action |
|---------|--------|--------|
| Silhouette > 0.4 | ✅ Good | Clusters well-separated |
| Silhouette 0.2-0.4 | ⚠️ Fair | Acceptable separation |
| Silhouette < 0.2 | ❌ Problem | Re-cluster with different k |
| Davies-Bouldin < 1.5 | ✅ Good | Good cluster compactness |
| Unbalanced clusters (1 huge) | ❌ Problem | Re-evaluate k or features |
| Geographic coherence | ✅ Good | Clustering makes physical sense |

### Feasibility (Script 3) — "Are candidates appropriately filtered?"
| Finding | Status | Action |
|---------|--------|--------|
| 10-50% survival rate | ✅ Good | Balanced screening |
| <10% survival rate | ⚠️ Strict | May be over-filtering |
| >50% survival rate | ⚠️ Loose | May be under-filtering |
| 0 survivors in any cluster | ❌ Problem | Constraints too strict |
| All constraints satisfied | ✅ Good | No single blocker |
| One constraint eliminates all | ❌ Problem | Constraint too aggressive |

### Ranking (Script 4) — "Is consensus robust?"
| Finding | Status | Action |
|---------|--------|--------|
| Method correlation > 0.5 | ✅ Good | Methods agree well |
| Method correlation 0.3-0.5 | ⚠️ Fair | Acceptable agreement |
| Method correlation < 0.3 | ❌ Problem | Methods disagree; check weights |
| Top-1 inclusion > 70% | ✅ Good | Stable recommendation |
| Top-1 inclusion < 50% | ⚠️ Fragile | May need multiple recommendations |
| Low rank reversal | ✅ Good | Stable across methods |
| High rank reversal | ⚠️ Caution | Weighting may be sensitive |

---

## Common Issues & Troubleshooting

### Issue: "FileNotFoundError: climate_rajasthan_points.csv"
**Solution:** Ensure Phase 2 preprocessing has been run  
**Check:** `data/processed/climate_rajasthan_points.csv` and `climate_rajasthan_points_clean.csv` exist

### Issue: "No engineered features found" (verify_01)
**Status:** Not a problem—just means lag/rolling/delta features weren't created  
**Fix:** Add these in Phase 2 preprocessing if needed

### Issue: Silhouette score < 0.2 (verify_02)
**Cause:** Clusters not well-separated  
**Fix:** Re-run clustering with different k value or features

### Issue: "0 survivors after filtering" (verify_03)
**Cause:** Constraints too strict (L_required too high)  
**Solution:** See CLAUDE.md for L_required correction (SHARE_PCM=0.5)

### Issue: Method correlation < 0.3 (verify_04)
**Cause:** MCDM weight matrix may be unbalanced  
**Fix:** Review criterion weights in Phase 6 ranking

---

## Integration with Main Pipeline

```
Phase 2 → verify_01 ✓
Phase 4 → verify_02 ✓
Phase 5 → verify_03 ✓
Phase 6 → verify_04 ✓
         ↓
   All verification plots in outputs/
```

Run verification scripts after each phase to catch issues early.

---

## Output Files Reference

### Verify 01 Outputs
```
verify_preprocessing/
├── 01_climate_distributions.png
├── 02_data_completeness.png
├── 03_statistical_summary.png
├── 05_correlation_analysis.png
├── 06_data_quality_metrics.png
└── 07_preprocessing_summary.png
```

### Verify 02 Outputs
```
verify_clustering/
├── 01_elbow_curves.png
├── 02_silhouette_plot.png
├── 03_pca_projection.png
├── 04_geographic_map.png
├── 05_cluster_profiles.png
└── 06_cluster_sizes.png
```

### Verify 03 Outputs
```
verify_feasibility/
├── 01_survival_rate_by_cluster.png
├── 02_feasible_property_space.png
├── 03_top_candidates_per_cluster.png
├── 04_constraint_analysis.png
├── 05_property_distributions.png
└── 06_summary.png
```

### Verify 04 Outputs
```
verify_ranking/
├── 01_method_correlation.png
├── 02_top3_inclusion_probability.png
├── 03_rank_distributions.png
├── 04_rank_reversal_frequency.png
├── 05_method_agreement.png
└── 06_summary.png
```

---

## Data Files Required

| Script | Required Files | Status |
|--------|---|---|
| verify_01 | `climate_rajasthan_points.csv`, `climate_rajasthan_points_clean.csv` | ✅ |
| verify_02 | `climate_signature_rajasthan.csv`, `cluster_assignments_rajasthan_levelB.csv` | ✅ |
| verify_03 | `feasibility_survivors_rajasthan_kappa_calibrated.csv` | ✅ |
| verify_04 | `mcdm_rankings_rajasthan.csv` | ✅ |

---

## Adaptation Notes (Tamil Nadu → Rajasthan)

Key changes made:
1. File paths updated to Rajasthan directory structure
2. Column names adapted to Rajasthan CSV formats
3. Fallback column names added for compatibility
4. All "Tamil Nadu" references changed to "Rajasthan"
5. Output paths changed to `outputs/objective1_plots_rajasthan/verify_*/`

**Robustness features:**
- Column name auto-detection with fallbacks
- Graceful handling of missing coordinates
- Optional database file handling
- Smart feature selection (skip metadata columns)

---

## Quality Assurance Checklist

After running all verify scripts:

- [ ] Verify 01: Data retention >80%, all vars >90% coverage
- [ ] Verify 02: Silhouette >0.4, geographic coherence visible
- [ ] Verify 03: 10-50% survival rate, balanced per cluster
- [ ] Verify 04: Method correlation >0.5, top-1 inclusion >70%

**If all checked:** Pipeline quality is good ✅  
**If any unchecked:** Investigate corresponding phase 🔧

---

## Citation & Reference

**Original:** Tamil Nadu verification scripts (`tamilnadu_pipeline/plots/verify_*.py`)

**Rajasthan Adaptation:** `era5-rajasthan/plotting/verify_0[1-4]_*_rajasthan.py`

**Total Output:** 24 PNG files (2.01 MB) across 4 directories  
**Generation Time:** ~50 seconds for all 4 scripts

---

## Questions & Support

For issues:
1. Check console output for specific error messages
2. Verify required data files exist in `data/processed/`
3. Check troubleshooting section above
4. Refer to CLAUDE.md for project context
5. Review COMPARISON_PLOTS_README.md for validation approach

**All verify scripts tested and working ✅**

Generated: 2026-09-02
