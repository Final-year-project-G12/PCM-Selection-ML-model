# Objective 1: Methods Summary Table
## Quick Lookup Guide for Each Technique

---

## **TABLE 1: ALL METHODS & ALGORITHMS USED**

| Stage | Process | Method/Algorithm | Why This? | Input | Output | Success Metric |
|-------|---------|------------------|-----------|-------|--------|-----------------|
| **1** | Data Download | ERA5 Reanalysis + NASA POWER API | Global, hourly coverage | Coordinates + date range | 493K records × 36 vars | 0 download errors |
| **2a** | Outlier Detection | Hampel's MAD Filter (Median Absolute Deviation) | Robust (not affected by extremes), fast O(n log n) | Raw time series | Flagged outliers, imputed | Remove extremes >3×MAD |
| **2b** | Missing Data | Forward-fill + Linear interpolation | Preserves trends better than mean imputation | Flagged time series | Continuous time series | 0 NaN values |
| **2c** | Temporal Lags | Autoregressive features (t-24h, t-7d, t-30d) | Captures thermal inertia, autoregressive patterns | Time series | 3 lag features per variable | Correlation with target |
| **2d** | Rolling Windows | Moving average & std (7-day, 30-day) | Captures medium-term variability, weather patterns | Time series | 4 rolling features per variable | Smoothing of noise |
| **2e** | Delta Features | Rate of change (ΔT per 24h) | Detects rapid transitions, cycling frequency | Time series | 1 delta feature per variable | Correlation with PCM cycling |
| **2f** | Solar Features | Clear Sky Index, azimuth, hour angle | Radiation intensity & direction for charging timing | GHI, DNI, location | 3 solar features | CSI ∈ [0,1], azimuth ∈ [0,360°] |
| **2g** | Wind Transformation | sin(θ), cos(θ) for direction | Avoids circular discontinuity (360°≈0°) | Wind speed + direction | 2 wind features | sin/cos ∈ [-1,1] |
| **2h** | Scaling | Min-Max normalization (70% chronological) | [0,1] range, interpretable, preserves relationships | Clean features | Scaled features [0,1] | mean≈0.5, std≈0.3 on full data |
| **3a** | Aggregation | Hourly → location summaries (mean, p95, p05, std) | Summarize 6 years into climate profile | Time series | 30 climate metrics | Statistical consistency |
| **3b** | Dimensionality | PCA (Principal Component Analysis) | Reduce 30→3 features, 85% variance retained | Climate metrics | PC1, PC2, PC3 | Explained variance ratio |
| **3c** | Standardization | Z-score (mean=0, std=1) | Prepare for Euclidean-distance clustering | Metrics | Standardized signatures | mean≈0, std=1 per feature |
| **4a** | Clustering | Gaussian Mixture Model (EM algorithm) | Soft assignment (probabilistic), continuous data | Standardized signatures | Cluster probabilities | Convergence in <100 iterations |
| **4b** | Model Selection | BIC (Bayesian Information Criterion) | Balance fit vs. complexity | GMM models k=2..8 | Best k score | BIC elbow visible |
| **4c** | Validation 1 | Silhouette Score | Within vs. between-cluster distance | Cluster assignments | Silhouette per sample | Score ∈ [-1,1], ideally >0.4 |
| **4d** | Validation 2 | Davies-Bouldin Index | Cluster compactness & separation | Cluster assignments | DB index | Lower is better |
| **4e** | Validation 3 | Calinski-Harabasz Index | Variance ratio (between/within) | Cluster assignments | CH index | Higher is better |
| **5a** | Filtering | Rule-based constraints (5 feasibility criteria) | Domain knowledge (thermophysics + PCM properties) | PCM database + Tm_target | Survivor list | Pass rate 10-50% |
| **5b** | Constraint 1 | Melting point window (Tm ± margin) | Physics: PCM must melt/refreeze in daily cycle | Tm_PCM, Tm_target | Boolean: pass/fail | Tm ∈ [target-5, target+5] |
| **5c** | Constraint 2 | Latent heat floor (L ≥ 120 kJ/kg) | Efficiency: minimum useful heat storage | L_PCM | Boolean: pass/fail | L ≥ 120 kJ/kg |
| **5d** | Constraint 3 | Cycling stability (cycles ≥ 1000, status=stable) | Reliability: avoid sudden degradation | cycles_tested, cycles_status | Boolean: pass/fail | cycles ≥ 1000 tested |
| **5e** | Constraint 4 | Supercooling tolerance (≤ 5K) | Functionality: supercooling = no solidification | supercooling_K | Boolean: pass/fail | supercooling ≤ 5K |
| **5f** | Constraint 5 | Absolute temperature band (0-60°C) | Safety: avoid hazardous extremes | Tm_PCM | Boolean: pass/fail | 0°C ≤ Tm ≤ 60°C |
| **6a** | Criteria Prep | Min-Max normalization (criterion-level) | Equalize scales across different units | Criteria matrix M | M_norm ∈ [0,1] | max-min = 1 per criterion |
| **6b** | Weighting A | Entropy weighting (Shannon entropy) | Data-driven: criteria that discriminate get higher weight | Normalized criteria | w_entropy vector | Σw = 1 |
| **6c** | Weighting B | AHP prior (Analytic Hierarchy Process) | Expert weights from domain knowledge | Expert judgment | w_AHP vector | Σw = 1, f_Tm=0.3 (highest) |
| **6d** | Weighting C | Hybrid weighting (50% entropy + 50% AHP) | Balance data + expertise | w_entropy, w_AHP | w_final vector | Σw = 1 |
| **6e** | Ranking Method 1 | TOPSIS (distance to ideal solution) | Favors excellence (good in key criteria) | M_norm, w_final | TOPSIS_score ∈ [0,1] | Score ∈ [0,1], 1=ideal |
| **6f** | TOPSIS Detail | S⁺ = √(Σ wⱼ(Mᵢⱼ-ideal)²), S⁻ = √(Σ wⱼ(Mᵢⱼ-negative)²) | Euclidean distance (normalized) | Normalized criteria | S⁺, S⁻ per candidate | Positive distance to ideal |
| **6g** | TOPSIS Score | TOPSIS = S⁻/(S⁺+S⁻) | Relative closeness to ideal solution | S⁺, S⁻ | TOPSIS_score | Score ∈ [0,1] |
| **6h** | Ranking Method 2 | GRA (Grey Relational Analysis) | Favors consistency (balanced performance) | M_norm, w_final | GRA_grade ∈ [0,1] | Grade ∈ [0,1], 1=ideal |
| **6i** | GRA Detail | ρᵢⱼ = (min_Δ + ζ×max_Δ)/(Δᵢⱼ + ζ×max_Δ), ζ=0.5 | Correlation coefficient (grey relational) | Deviations Δ | Correlation per criterion | ρ ∈ [0,1] |
| **6j** | GRA Grade | GRA_grade = Σ wⱼ × ρᵢⱼ | Weighted correlation (consistency metric) | ρᵢⱼ, weights | GRA_grade | Grade ∈ [0,1] |
| **6k** | Ranking Method 3 | Borda Voting (rank aggregation) | Democratic consensus (all methods equal voice) | TOPSIS_rank, GRA_rank | borda_score | Score = sum of (n-rank+1) |
| **6l** | Borda Count | borda_score = Σ(n - rank + 1) per method | Points system: rank #1=n pts, rank #n=1 pt | Ranks from TOPSIS & GRA | Borda score | score = (n-rank+1) per method |
| **6m** | Consensus Rank | consensus_rank = rank(borda_score) | Convert Borda scores back to rank positions | borda_score | consensus_rank [1..n] | Rank 1 = highest borda score |
| **6n** | Agreement Metric | Kendall's W (coefficient of concordance) | Measure agreement between TOPSIS & GRA | TOPSIS_rank, GRA_rank | Kendall's W ∈ [-1,1] | W > 0.6 = high agreement |
| **7a** | Synthesis | Recommendation cards per cluster | Generate actionable output with justification | Top-3 PCMs per cluster | Markdown + HTML | Readability + completeness |
| **7b** | Documentation | Thermal profile generation | Show melt fraction vs. temperature curve | PCM properties | Profile plot | S-curve shape |

---

## **TABLE 2: CRITERIA FOR PCM RANKING (Stage 6)**

| Criterion | Symbol | Definition | Range | Unit | Why Important? | Normalization |
|-----------|--------|------------|-------|------|-----------------|----------------|
| Melting Point Fit | f_Tm | exp(-(Tm_PCM - Tm_target)²/(2σ²)) where σ=10°C | [0, 1] | Dimensionless | PCM melts during charge, refreezes during discharge | Gaussian |
| Latent Heat | L | Enthalpy of fusion per unit mass | [140, 200] | kJ/kg | Energy stored per kg (higher = more compact system) | Min-Max [0,1] |
| Storage Density | ρH | ρ_solid × L (volumetric energy capacity) | [19, 27] | MJ/m³ | Tank size efficiency (higher = smaller, cheaper) | Min-Max [0,1] |
| Specific Heat | Cp | (Cp_solid + Cp_liquid) / 2 (sensible heat) | [1.8, 3.5] | kJ/(kg·K) | Quicker response before melting | Min-Max [0,1] |
| Cycling Durability | cycles_conf | Confidence based on testing history | [0, 1] | Dimensionless | Avoid sudden failure; proven robustness | 5000+→1.0, 2000→0.8, 1000→0.5, <1000→0.2 |

---

## **TABLE 3: VALIDATION METRICS & THRESHOLDS**

| Metric | Stage | Formula | Interpretation | ✅ Good | ⚠️ Marginal | ❌ Poor |
|--------|-------|---------|-----------------|----------|------------|---------|
| **Data Retention** | Preprocessing | % survivors after cleaning | How much data lost? | >98% | 95-98% | <95% |
| **NaN Coverage** | Preprocessing | % missing values per feature | Data completeness? | 0% NaN | <1% NaN | >5% NaN |
| **Silhouette Score** | Clustering | (b-a)/max(a,b) per sample | Cluster tightness? | >0.5 | 0.3-0.5 | <0.3 |
| **BIC Score** | Clustering | -2ln(L) + k×ln(n) | Model fit vs. k? | Elbow visible | Flat | No clear optimum |
| **Davies-Bouldin** | Clustering | Avg(R_ij) across pairs | Separation quality? | <1.0 | 1.0-1.5 | >1.5 |
| **Calinski-Harabasz** | Clustering | (B/W)×(n-k)/(k-1) | Variance ratio? | >25 | 15-25 | <15 |
| **Cluster Balance** | Clustering | max_size / min_size | Size imbalance? | <2.0 | 2.0-5.0 | >5.0 |
| **Feasibility Rate** | Filtering | survivors / total_candidates | Constraint stringency? | 10-50% | <10% or >50% | None survive |
| **Kendall's W** | Ranking | 12S/(m²(n³-n)) | Method agreement? | >0.6 | 0.3-0.6 | <0.3 |
| **Rank Stability** | Ranking | % candidates with rank_range<3 | Rank consistency? | >80% | 60-80% | <60% |

---

## **TABLE 4: WHERE EACH METHOD IS USED (Pipeline Mapping)**

```
STAGE 1: Data Collection
├─ ERA5 API
└─ NASA POWER API

STAGE 2: Preprocessing
├─ Hampel MAD Filter
├─ Interpolation
├─ Lag Engineering (t-24h, t-7d, t-30d)
├─ Rolling Windows (mean, std on 7d, 30d)
├─ Delta Features (rate of change)
├─ Solar Features (CSI, azimuth, angle)
├─ Wind Transformation (sin/cos)
└─ Min-Max Scaling (70% chronological)

STAGE 3: Climate Signatures
├─ Aggregation (hourly → location summaries)
├─ PCA (30 → 3 components)
└─ Z-score Standardization

STAGE 4: Clustering
├─ Gaussian Mixture Model (EM fit)
├─ BIC Evaluation (k=2..8)
├─ Silhouette Score
├─ Davies-Bouldin Index
└─ Calinski-Harabasz Index

STAGE 5: Feasibility Filtering
├─ Constraint 1: Melting Point Window
├─ Constraint 2: Latent Heat Floor
├─ Constraint 3: Cycling Stability
├─ Constraint 4: Supercooling
└─ Constraint 5: Absolute Temp Band

STAGE 6: MCDM Ranking
├─ Min-Max Normalization (per cluster)
├─ Entropy Weighting (Shannon entropy)
├─ AHP Prior (expert weights)
├─ Hybrid Weighting (50-50 blend)
├─ TOPSIS (Euclidean distance to ideal)
├─ GRA (correlation with ideal pattern)
├─ Borda Voting (rank aggregation)
└─ Kendall's W (agreement metric)

STAGE 7: Recommendations
├─ Top-3 Selection per cluster
├─ Thermal Profile Synthesis
└─ Documentation Generation
```

---

## **TABLE 5: ERROR HANDLING & EDGE CASES**

| Situation | How It's Handled | Why This Approach? |
|-----------|-----------------|------------------|
| **Missing values in time series** | Forward-fill + linear interpolation | Preserves temporal continuity, avoids artificial breaks |
| **Extreme outliers** (e.g., T=150°C) | Hampel MAD filter (>3×MAD flagged) | Robust to extremes; won't remove entire sections |
| **Circular wind direction** (359° → 1°) | sin(θ), cos(θ) transformation | Treats angles as periodic; avoids discontinuity |
| **Constant criterion in cluster** | Shannon entropy → 0 → low weight | Criteria that don't discriminate get less influence |
| **All PCMs fail a constraint** | No survivors for that cluster | Data-driven; indicates constraint too strict |
| **Borda tie (same score)** | method="min" in rank() | Ties get same rank; next rank skipped |
| **Kendall's W undefined** (n=1 candidate) | W = NaN | Can't measure agreement with 1 item |
| **TOPSIS score all 0.5** (indifference) | All candidates equally far from ideal | Indicates data not discriminative enough |
| **Cluster size imbalance** (1 vs 20 items) | Report & visualize (not removed) | Imbalance is real; may indicate climate boundary |

---

## **TABLE 6: HYPERPARAMETERS & CONFIGURATION**

| Parameter | Value | Stage | Why? |
|-----------|-------|-------|------|
| **Hampel threshold** | 3×MAD | Preprocessing | Standard robust statistics threshold |
| **Lag windows** | [24h, 168h, 720h] | Preprocessing | Daily, weekly, monthly patterns |
| **Rolling windows** | [7d, 30d] | Preprocessing | Short & medium-term variability |
| **Scaling split** | 70% chronological | Preprocessing | Avoid temporal leakage |
| **PCA components** | 3 (reduce 30→3) | Signatures | Retain ~85% variance |
| **GMM k range** | 2 to 8 | Clustering | Reasonable for 45 locations |
| **Melting point margin** | ±5°C | Feasibility | Balance precision vs. flexibility |
| **Latent heat floor** | 120 kJ/kg | Feasibility | Minimum practical threshold |
| **Cycling threshold** | 1000+ cycles | Feasibility | Industry standard for stability proof |
| **Supercooling limit** | ≤5K | Feasibility | Acceptable subcooling range |
| **Gaussian σ (Tm fit)** | 10°C | Ranking | Bell curve width for temperature match |
| **Entropy-AHP lambda** | 0.5 | Ranking | Equal weighting of data + expertise |
| **AHP prior weights** | f_Tm=0.3, L=0.3, ρH=0.2, Cp=0.1, cycles=0.1 | Ranking | Expert consensus from PCM literature |
| **Borda n** | # candidates in cluster | Ranking | Dynamic based on survivors |

---

## **QUICK DECISION TREE: WHICH METHOD TO USE WHEN?**

```
Q: Need to CLEAN raw time series data?
├─ YES → Use Hampel MAD Filter (robust to outliers)
└─ NO → Skip to next

Q: Want to PREDICT future climate?
├─ YES → Use Lag Features (t-24h, t-7d, t-30d)
└─ NO → Skip to next

Q: Want to REDUCE dimensions (30 → 3)?
├─ YES → Use PCA
└─ NO → Use all 30 metrics

Q: Need to CLUSTER locations into regimes?
├─ YES → Use Gaussian Mixture Model
│   └─ Then evaluate k via BIC, Silhouette, DB, CH
└─ NO → Skip to next

Q: Need to FILTER PCM candidates?
├─ YES → Apply 5 feasibility constraints (rules-based)
└─ NO → Skip to next

Q: Need to RANK feasible PCMs?
├─ YES → Use TOPSIS + GRA + Borda (3 methods)
│   ├─ TOPSIS if you favor EXCELLENCE (few great criteria)
│   ├─ GRA if you favor BALANCE (good at everything)
│   └─ Borda if you want CONSENSUS (democratic vote)
└─ NO → Done!

Q: Worried about METHOD DISAGREEMENT?
├─ YES → Compute Kendall's W (measure agreement)
└─ NO → Trust Borda consensus rank

Q: Need to EXPLAIN results to non-experts?
├─ YES → Use recommendations_cards.md (plain language)
└─ NO → Use raw TOPSIS/GRA/Borda scores
```

---

## **KEY EQUATIONS REFERENCE**

### **Hampel's Filter**
```
median = p50(X)
mad = median(|X - median|)
outlier = |X - median| > 3 × mad
```

### **Gaussian Melting Point Fit**
```
f_Tm = exp(-(Tm_PCM - Tm_target)² / (2 × σ²))
where σ = 10°C
```

### **Min-Max Normalization**
```
X_norm = (X - X_min) / (X_max - X_min)
Result: X_norm ∈ [0, 1]
```

### **TOPSIS Score**
```
S⁺ = √(Σ wⱼ × (Mᵢⱼ - ideal_j)²)      # Distance to ideal
S⁻ = √(Σ wⱼ × (Mᵢⱼ - negative_j)²)   # Distance to negative-ideal
TOPSIS_score = S⁻ / (S⁺ + S⁻)
Result: TOPSIS_score ∈ [0, 1], 1=ideal
```

### **GRA Relational Coefficient**
```
Δᵢⱼ = |Mᵢⱼ - ideal_j|
ρᵢⱼ = (min_Δ + ζ × max_Δ) / (Δᵢⱼ + ζ × max_Δ)
where ζ = 0.5 (distinguishing coefficient)
```

### **Borda Count**
```
borda_score_i = Σ_m (n - rank_m(i) + 1)
where n = total candidates, rank ∈ [1, n]
```

### **Kendall's W (Coefficient of Concordance)**
```
R_i = Σ_m rank_m(i)           # Sum of ranks for candidate i
R̄ = mean(R_i)
S = Σ (R_i - R̄)²
W = 12 × S / (m² × (n³ - n))  # m=# methods, n=# candidates
Result: W ∈ [-1, 1], 1=perfect agreement
```

---

This is your complete **methods reference** for defending Objective 1!
