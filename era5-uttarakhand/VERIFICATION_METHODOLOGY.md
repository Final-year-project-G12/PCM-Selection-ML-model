# Verification & Validation Strategy for Objective 1 Pipeline

## Overview

This document explains how to verify the correctness of each stage in the climate-to-PCM-selection pipeline. Each stage has comparison plots to validate assumptions, detect issues, and ensure data quality.

---

## Pipeline Stages & Verification

### Stage 1: **Data Download & Preparation**
**What happens:** ERA5 and NASA POWER climate data are downloaded and combined by spatial grid.

**Verification focus:**
- Data completeness (no extreme gaps in time series)
- Geographic coverage (all population grid cells represented)
- Data range sanity (temperature within physical bounds)

**Plots:**
- Temporal coverage heatmap (days available per grid cell per variable)
- Spatial extent map (grid cells with data vs. missing)
- Climate variable distributions (raw ERA5 vs. NASA POWER agreement)

---

### Stage 2: **Preprocessing & QC**
**What happens:** Raw climate time series are cleaned, imputed, and feature-engineered.

**Verification focus:**
- Outlier detection effectiveness (are bad values removed?)
- Missing data imputation quality (is interpolation sensible?)
- Scaling correctness (are features normalized for clustering?)
- Feature engineering validity (do lag/rolling/delta features capture regime changes?)

**Plots:**
- **Before/After histograms** per variable (raw vs. cleaned)
- **Outlier detection** (flagged points shown on time series)
- **Missing data pattern** (heatmap of NaN locations before imputation)
- **Scaling impact** (raw vs. scaled distributions)
- **Feature correlation** (do engineered features correlate with raw?)
- **Hampel filter effectiveness** (outliers detected and removed)

**Success criteria:**
- Distributions are smoother after cleaning
- Outliers are isolated, not entire clusters removed
- No NaN values remain after imputation
- Scaled data mean ≈ 0, std ≈ 1 for first 70% (training set)

---

### Stage 3: **Climate Signature & Clustering**
**What happens:** Multivariate climate profiles are computed and clustered into regimes.

**Verification focus:**
- Are clusters distinct (good separation)?
- Do clusters make geographic/climate sense?
- Is the cluster count optimal (k)?
- Are within-cluster samples similar and between-cluster samples different?

**Plots:**
- **Elbow curve** (BIC, silhouette, Davies-Bouldin for k=2..8)
- **Silhouette plot** (per sample, per cluster)
- **PCA scatter** (2D projection of clusters)
- **Geographic cluster map** (clusters colored by location)
- **Climate variable profiles** (box plots of key vars per cluster)
- **Cluster size distribution** (how many samples per cluster)
- **Within-cluster vs. between-cluster distances**

**Success criteria:**
- Elbow appears around k=3–5
- Silhouette scores > 0.4 for most samples
- Geographic clusters are contiguous (not scattered)
- Profiles make climate sense (e.g., high altitude cluster is cooler)

---

### Stage 4: **Feasibility Filtering**
**What happens:** PCM candidates are tested against thermal performance constraints per cluster.

**Verification focus:**
- How many candidates survive per constraint?
- Are constraints too strict (very few survivors) or too loose (most pass)?
- Do survivors make thermophysical sense?

**Plots:**
- **Constraint violation distribution** (how many fail each test?)
- **Survivors per cluster** (count and percentage)
- **Melting point vs. latent heat scatter** (color by pass/fail)
- **Constraint feasibility region** (visualize acceptable property space)
- **Candidate elimination flow** (Sankey: input → fail constraint 1 → fail constraint 2 → survivors)

**Success criteria:**
- 10–50% of candidates survive (not too strict or loose)
- Survivors cluster in physically plausible region
- Geographic regions with different climates have different survivor sets

---

### Stage 5: **Multi-Criteria Decision Making (MCDM) Ranking**
**What happens:** Surviving PCM candidates are ranked by TOPSIS, GRA, PROMETHEE II, and VIKOR; consensus ranking via Borda or Kendall.

**Verification focus:**
- Do different MCDM methods agree?
- Are the top recommendations robust (appear high in all methods)?
- Is there rank reversal (same candidate ranked very differently across methods)?

**Plots:**
- **Method rank correlation heatmap** (Spearman between methods)
- **Top-10 inclusion probability** (how often is each candidate in top-10 across methods)
- **Rank distributions** (box plot: method A ranks vs. method B)
- **Bump chart** (rank trends for top 5 PCMs across methods)
- **Consensus vs. single method** (scatter: consensus rank vs. TOPSIS rank)
- **Rank reversal frequency** (how often does a PCM rank high in one method but low in another?)
- **Sensitivity plot** (how sensitive are top ranks to small weight changes?)

**Success criteria:**
- Spearman correlation > 0.7 between methods (good agreement)
- Top 3 candidates appear in top-10 for ≥80% of methods
- Rank reversal is rare (< 20% of candidates)
- Final consensus ranking is stable (small weight perturbations don't flip top 3)

---

### Stage 6: **Recommendation Cards**
**What happens:** Top PCM recommendations are synthesized per climate regime with thermal performance profiles.

**Verification focus:**
- Are recommendations aligned with cluster properties?
- Do top PCMs cover different use cases (melt range, cost, availability)?
- Are recommendations actionable (real PCM products)?

**Plots:**
- **Recommended PCM properties by cluster** (melting point, latent heat, cost)
- **Thermal performance profile** (temperature range, phase-change performance)
- **Recommendation diversity** (different top-picks for different clusters)
- **Cost vs. performance Pareto** (efficiency frontier per cluster)

**Success criteria:**
- Different clusters have different top recommendations (not identical for all)
- Recommendations span different melting point ranges
- Top 3 are real materials (e.g., paraffin wax, salt hydrates, fatty acids)

---

## How to Use This Verification Suite

### Quick Start

Each stage has a dedicated verification script:

```bash
# Stage 2: Preprocessing validation
python verify_01_preprocessing.py

# Stage 3: Clustering validation
python verify_02_clustering.py

# Stage 4: Feasibility filtering
python verify_03_feasibility.py

# Stage 5: MCDM ranking verification
python verify_04_ranking.py
```

### Output Folders

```
data/
├── plots/
│   ├── objective1/           # Main presentation plots (generated earlier)
│   ├── verify_preprocessing/ # Raw vs. cleaned, outlier detection, scaling
│   ├── verify_clustering/    # Elbow, silhouette, PCA, geographic map
│   ├── verify_feasibility/   # Constraint violations, survivors, Sankey
│   └── verify_ranking/       # Method agreement, rank reversal, sensitivity
```

### Interpretation Guide

**Green light (results correct):**
- Preprocessing: outliers isolated, no data loss, smooth distributions
- Clustering: silhouette > 0.4, elbow visible, clusters are geographic
- Feasibility: 10–50% survival, survivors in physical space
- Ranking: method correlation > 0.7, top-3 stable, low rank reversal

**Red flag (investigate):**
- Preprocessing: entire sections missing or scaled incorrectly
- Clustering: silhouette < 0.3, k-choice ambiguous, scattered clusters
- Feasibility: < 5% or > 90% survival (constraints misconfigured)
- Ranking: low method agreement, high rank reversal, unstable top-3

---

## Why Each Plot Type?

| Plot | Detects | Metric |
|------|---------|--------|
| **Before/After histogram** | Data quality improvement | KL divergence, outliers removed |
| **Silhouette plot** | Cluster validity | Silhouette score per sample |
| **Elbow curve** | Optimal cluster count | BIC, silhouette, Davies-Bouldin |
| **PCA scatter** | Visual cluster separation | Within vs. between distance |
| **Geographic map** | Spatial coherence | Contiguity, region size |
| **Constraint violation** | Feasibility stringency | % passing per constraint |
| **Rank correlation** | MCDM agreement | Spearman/Kendall ρ |
| **Bump chart** | Rank stability | Top-3 consistency |
| **Sensitivity plot** | Recommendation robustness | Rank shift under perturbation |

---

## Validation Checklist

- [ ] Preprocessing: no NaN remaining, distributions smooth, no data loss > 5%
- [ ] Clustering: silhouette ≥ 0.4 (avg), k ∈ {3,4,5}, geographic coherence ✓
- [ ] Feasibility: survivors 10–50%, distinct property cluster, per-cluster variation ✓
- [ ] Ranking: method correlation > 0.7, top-3 consistent, rank reversal < 20%
- [ ] Recommendations: per-cluster variation ✓, real materials ✓, actionable ✓

---

## Next Steps

1. Run each verification script to generate comparison plots
2. Review the plots in their respective output folders
3. Check green light criteria above
4. If any red flags, see debugging guide in each script
5. Use plots in thesis/presentation to justify methodology choices

