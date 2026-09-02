# Objective 1: Complete Pipeline Explanation
## Climate Data → PCM Selection through Methods & Analysis

---

## **OVERVIEW: What is Objective 1?**

**Goal:** Select the best Phase Change Materials (PCMs) for thermal energy storage in different climate regions of Uttarakhand, India.

**Process:**
```
Climate Data (ERA5 + NASA POWER)
    ↓
Clean & Preprocess
    ↓
Extract Climate Signatures
    ↓
Cluster into Climate Regimes
    ↓
Filter PCM Candidates by Feasibility
    ↓
Rank PCMs using Multi-Criteria Methods
    ↓
Generate Recommendations per Climate Region
```

---

# **DETAILED STAGE-BY-STAGE BREAKDOWN**

## **STAGE 1: DATA DOWNLOAD & COMBINATION**

### **What Happens**
Collect climate data for population-weighted grid points across Uttarakhand.

### **Methods Used**

#### **1.1 ERA5 Reanalysis Download**
- **Tool:** `01_download_era5_uttarakhand.py`
- **Data Source:** Copernicus Climate Data Store (ERA5)
- **Variables Downloaded:**
  - Temperature (T2M)
  - Dew point (T_dew)
  - Relative humidity (RH2M)
  - Wind speed (W_spd)
  - Wind direction (W_dir)
  - Global Horizontal Irradiance (GHI)
  - Direct Normal Irradiance (DNI)
  - Diffuse Horizontal Irradiance (DHI)
  - Longwave radiation (LW_down)
  - Cloud cover
  - Precipitation
  - Pressure (P_atm)
  - Solar zenith angle (SZA)

- **Frequency:** Hourly from 2015-01-01 to 2020-12-31 (6 years)
- **Spatial Resolution:** 0.25° × 0.25° grid cells

#### **1.2 NASA POWER Download**
- **Tool:** `01b_download_nasapower.py`
- **Data Source:** NASA POWER (meteorology + solar)
- **Variables:** Same as ERA5 for comparison/backup
- **Purpose:** Cross-validation and data gap filling

#### **1.3 Population Grid Builder**
- **Tool:** `00a_build_population_grid.py`
- **Method:** Spatial population weighting
  - Read population raster (CIESIN data)
  - Create 0.25° grid aligned with ERA5
  - Weight each cell by population density
  - Flag grid cells with zero population (exclude)
- **Output:** 45 population-weighted grid points for Uttarakhand

#### **1.4 Data Combination**
- **Tool:** `02_combine_rajasthan.py` (template; adapted for Uttarakhand)
- **Method:** Time-series merge
  - Match ERA5 and NASA POWER by date/time
  - Interpolate missing values (linear)
  - Take ensemble average where both available
  - Create hourly climate time series per grid point
- **Output:** 493,155 records × 36 variables (6 years × 365 days × 24 hours, ~45 locations)

---

## **STAGE 2: PREPROCESSING & QUALITY CONTROL**

### **What Happens**
Clean raw climate data, handle missing values, engineer features for clustering.

### **Methods Used**

#### **2.1 Physical Bounds Check**
- **Method:** Min-max validation
- **Purpose:** Detect impossible values (e.g., T > 60°C, RH > 120%)
- **Action:** Flag and remove records outside physical bounds
- **Thresholds Checked:**
  ```
  -50°C < Temperature < 60°C
  0 < Relative Humidity < 100%
  0 < Wind Speed < 30 m/s
  700 hPa < Pressure < 1100 hPa
  0 < GHI < 1400 W/m²
  0 < Precipitation < 100 mm/hr
  ```

#### **2.2 Missing Data Imputation (Hampel MAD Filter)**
- **Method:** Hampel's robust outlier detection + Kalman smoothing
- **Algorithm:** Median Absolute Deviation (MAD)
  ```
  median = np.median(X)
  mad = median_abs_deviation(X)
  outlier_mask = |X - median| > 3 * mad
  ```
- **Action:**
  1. Detect outliers (beyond 3×MAD)
  2. Mark as missing
  3. Impute using forward-fill + interpolation
- **Purpose:** Remove noise without losing trend data
- **Output:** Clean climate time series, 489,105 valid records

#### **2.3 Feature Engineering**

**A. Temporal Features:**
```python
month = extract_month(date)
day_of_year (DOY) = julian_day(date)
year = extract_year(date)
season = classify(month) → ['DRY', 'MON_ONSET', 'MONSOON', 'RETREAT']
hour_decimal = hour + minute/60
is_daytime = (SZA < 90°)  # Solar zenith angle
```

**B. Lag Features (Past Climate):**
```python
for var in ['T2M', 'GHI', 'RH2M', 'W_spd', 'cloud_cover']:
    var_lag1d = var[t-24]   # 1 day ago
    var_lag7d = var[t-168]  # 7 days ago
    var_lag30d = var[t-720] # 30 days ago
```
- **Purpose:** Capture autoregressive patterns, thermal inertia

**C. Rolling Statistics (7-day & 30-day windows):**
```python
for var in ['T2M', 'GHI', 'RH2M', 'W_spd', 'cloud_cover']:
    var_roll7d_mean = rolling_mean(var, window=168)
    var_roll7d_std = rolling_std(var, window=168)
    var_roll30d_mean = rolling_mean(var, window=720)
    var_roll30d_std = rolling_std(var, window=720)
```
- **Purpose:** Capture medium-term variability (weather patterns, monsoon intensity)

**D. Delta Features (Rate of Change):**
```python
var_delta1d = var[t] - var[t-24]  # Change in past 24 hours
```
- **Purpose:** Detect rapid transitions (PCM cycling frequency)

**E. Solar & Radiation Features:**
```python
clear_sky_index (CSI) = GHI / GHI_clearsky
direct_horizontal_irradiance = DNI * sin(elevation_angle)
solar_azimuth = compute_solar_azimuth(lat, lon, time)
solar_hour_angle = (t - 12) * 15°
```

**F. Wind Features (Direction transformation):**
```python
wind_dir_sin = sin(wind_direction)  # Avoid circular discontinuity
wind_dir_cos = cos(wind_direction)
```

**G. Thermophysical Proxy Features:**
```python
cloud_opacity = 1 - (GHI / GHI_clearsky)
T_depression = T_ambient - T_dewpoint  # Dryness indicator
```

#### **2.4 Data Scaling (Chronological Split)**
- **Method:** Min-Max scaling (preserves interpretability)
- **Split:** First 70% (chronological) = training set
- **Algorithm:**
  ```
  X_scaled = (X - X_min[70%]) / (X_max[70%] - X_min[70%])
  ```
- **Why chronological?** Climate data has trends; random split pollutes with future knowledge
- **Purpose:** Normalize features to [0,1] for clustering
- **Output:** 89 features ready for signature computation

#### **2.5 Validation Gate**
- **Check:** No NaN values remain
- **Output:** 489,105 records × 89 variables

---

## **STAGE 3: CLIMATE SIGNATURE EXTRACTION**

### **What Happens**
Compute multivariate climate metrics that characterize each location's thermal environment.

### **Methods Used**

#### **3.1 Temperature Metrics**
```python
Ta_mean_proxy = mean(T_ambient)           # Average temperature
Ta_p95_proxy = percentile(T_ambient, 95)  # Hot extreme
Ta_p05_proxy = percentile(T_ambient, 5)   # Cold extreme
DTR_proxy = Ta_p95 - Ta_p05               # Diurnal temp range

Tm_target_C = weighted_mean(T_ambient)    # Target melting point
                                          # (closer to mean = less switching)
```
- **Purpose:** Characterize thermal environment for PCM selection

#### **3.2 Solar Radiation Metrics**
```python
GHI_mean = mean(Global_Horizontal_Irradiance)
GHI_daily_kWh_proxy = sum_daily(GHI)      # Daily insolation
kt_mean_proxy = mean(clearness_index)     # Clear-sky clarity
kt_std_proxy = std(clearness_index)       # Variability

SAI_proxy = sum(hourly_direct)            # Solar aggregation index
                                          # (high = potential for direct)
```
- **Purpose:** Quantify solar resource and variability

#### **3.3 Cloud & Atmospheric Metrics**
```python
cloudy_frac_proxy = fraction(cloud_cover > 50%)
CCI_proxy = cloud_cover_index             # Cloud cover intensity
```
- **Purpose:** Assess cloud-induced temperature swings

#### **3.4 Humidity & Moisture Metrics**
```python
RH_mean = mean(Relative_Humidity)
T_depression = T_ambient - T_dewpoint    # Dryness proxy
```

#### **3.5 Heating Degree Days (HDD) & Cooling Degree Days (CDD)**
```python
HDD18_proxy = sum(max(0, 18 - T_ambient)) # Heating requirement (base 18°C)
CDD24_proxy = sum(max(0, T_ambient - 24)) # Cooling requirement (base 24°C)
```
- **Purpose:** Quantify heating/cooling load intensity

#### **3.6 Wind Metrics**
```python
wind_mean = mean(wind_speed)
wind_std = std(wind_speed)                # Wind variability
```

#### **3.7 Seasonality Metrics**
```python
monsoon_index = sum_june_to_sept(precipitation) / total_annual
seasonality_proxy = std_by_month(temperature)  # Month-to-month variation
```
- **Purpose:** Capture seasonal regime strength

#### **3.8 PCA Reduction (Dimensionality Reduction)**
- **Method:** Principal Component Analysis
- **Input:** ~30 computed signatures per location
- **Algorithm:**
  ```
  Covariance = cov(signature_matrix)
  Eigenvectors = eig(Covariance)
  PC1, PC2, PC3 = top 3 eigenvectors
  explained_variance = [λ1/Σλ, λ2/Σλ, λ3/Σλ]
  ```
- **Output:** PC1, PC2, PC3 (captures ~85% variance)
- **Purpose:** Compress into 3 components for visualization

#### **3.9 Standardization (Z-Score)**
- **Method:** Standardize all signatures to mean=0, std=1
- **Algorithm:**
  ```
  X_z = (X - mean(X)) / std(X)
  ```
- **Output:** 90 features per location (signatures + PCs, all standardized)

---

## **STAGE 4: CLUSTERING INTO CLIMATE REGIMES**

### **What Happens**
Group 45 locations into k climate regimes based on similarity.

### **Methods Used**

#### **4.1 Optimal Cluster Selection (k=2..8)**

**A. Gaussian Mixture Model (GMM)**
- **Algorithm:** Expectation-Maximization
- **What it does:**
  ```
  Assume: Each climate regime follows a multivariate Gaussian distribution
  Fit: p(X | k) = Σ π_i * N(X | μ_i, Σ_i)
         where π_i = mixing coefficient (probability of regime i)
               μ_i = mean profile of regime i
               Σ_i = covariance (spread) of regime i
  ```
- **Why GMM not K-Means?**
  - GMM gives **probability** of belonging to each cluster (soft assignment)
  - K-Means gives hard assignment (50% in cluster 1, 50% in cluster 2 is impossible)
  - Climate data is **continuous**, not discrete → GMM better fit

**B. BIC (Bayesian Information Criterion)**
- **Formula:**
  ```
  BIC = -2 * log(likelihood) + k * log(n)
                                ↑
                                penalty for complexity
  ```
- **Interpretation:** Lower BIC = better k
- **What it does:** Balances model fit vs. overfitting

**C. Silhouette Score**
- **Formula:**
  ```
  silhouette = (b - a) / max(a, b)
  where a = avg distance to points in same cluster
        b = avg distance to points in nearest other cluster
  ```
- **Range:** [-1, 1]
  - Silhouette = 0.7 → Very tight clusters (well-separated)
  - Silhouette = 0.4 → Moderate clusters
  - Silhouette < 0.3 → Overlapping clusters
- **Your data:** 0.253 (overlapping, but acceptable for continuous climate data)

**D. Davies-Bouldin Index**
- **Formula:**
  ```
  DB = (1/k) * Σ max(R_ij)
       where R_ij = (σ_i + σ_j) / d_ij
                    ↑ within-cluster spread
                              ↑ between-cluster distance
  ```
- **Interpretation:** Lower DB = better clusters

**E. Calinski-Harabasz Index**
- **Formula:**
  ```
  CH = (B / W) * (n - k) / (k - 1)
       where B = between-cluster variance
             W = within-cluster variance
  ```
- **Interpretation:** Higher CH = better clusters

#### **4.2 Chosen k = 5 Clusters**
- **Decision Criteria:**
  - BIC shows elbow around k=5
  - Silhouette 0.253 acceptable (climate is continuous)
  - Davies-Bouldin 1.194 (moderate)
  - Calinski-Harabasz 21.7 (fair)
  - **Result: 5 climate regimes identified**

#### **4.3 Cluster Interpretation**

| Cluster | Size | Characteristics |
|---------|------|-----------------|
| **0** | 9 locations | High altitude, cold winters, moderate solar |
| **1** | 10 locations | Mid-elevation, moderate temps, variable monsoon |
| **2** | 3 locations | Tropical zone, hot year-round, low seasonality |
| **3** | 12 locations | High monsoon influence, strong seasonality |
| **4** | 11 locations | Moderate altitude, balanced temps, strong solar |

---

## **STAGE 5: PCM FEASIBILITY FILTERING**

### **What Happens**
Filter PCM candidates against thermal performance constraints for each climate regime.

### **Methods Used**

#### **5.1 PCM Database**
- **Source:** Literature review + commercial databases
- **Database:** 25 PCM candidates (paraffins, fatty acids, salt hydrates)
- **Properties per PCM:**
  - Melting temperature (Tm_C)
  - Latent heat (L)
  - Density (solid & liquid)
  - Specific heat (Cp)
  - Thermal conductivity (k)
  - Cycling stability (cycles tested)
  - Supercooling tendency
  - Corrosion class
  - Cost indicator

#### **5.2 Feasibility Constraints**

**Constraint 1: Melting Temperature Window**
```python
# PCM melting point should be within target range:
lower_bound = Tm_target - temperature_margin  # e.g., -5°C
upper_bound = Tm_target + temperature_margin  # e.g., +5°C

pass_melting_window = (Tm_PCM >= lower_bound) & (Tm_PCM <= upper_bound)
```
- **Why:** PCM must melt during discharge (useful) & refreeze during charge
- **Target Tm calculation:** 
  ```
  Tm_target = weighted_average(T_ambient_yearly)
  ```

**Constraint 2: Latent Heat Floor**
```python
latent_heat_floor_kJ_kg = 120  # Minimum useful heat storage

pass_latent_heat = (L_PCM >= latent_heat_floor_kJ_kg)
```
- **Why:** Low latent heat = inefficient thermal buffer

**Constraint 3: Cycling Stability**
```python
pass_cycling = (cycles_tested >= 1000) & (cycles_tested_status == 'stable')
```
- **Why:** PCM must survive thermal cycling (charge/discharge cycles)

**Constraint 4: Supercooling Tolerance**
```python
pass_supercooling = (supercooling_K <= 5)  # Acceptable subcooling
```
- **Why:** Supercooling = PCM doesn't solidify → useless

**Constraint 5: Absolute Temperature Band**
```python
# For safety in storage systems:
pass_absolute_band = (Tm_PCM >= 0°C) & (Tm_PCM <= 60°C)
```
- **Why:** Avoid hazardous temperatures (too cold = ice expansion; too hot = volatilization)

#### **5.3 Filtering Logic**
```python
passes_all = (pass_melting_window & 
              pass_latent_heat & 
              pass_cycling & 
              pass_supercooling & 
              pass_absolute_band)

survivors = PCM_database[survivors[passes_all == True]]
```

#### **5.4 Results**
- **Total survivors:** 125 PCM entries (25 PCMs × 5 clusters)
- **Per cluster:** 25 survivors (identical across all clusters)
- **Note:** Constraints are **cluster-agnostic** (same thresholds apply everywhere)
  - *Alternative:* Could use climate-specific thresholds (temperature range ± 2°C for hot cluster vs. ± 5°C for cold cluster)

---

## **STAGE 6: MULTI-CRITERIA DECISION MAKING (MCDM) RANKING**

### **What Happens**
Rank PCM candidates using 3 methods, aggregate via consensus (Borda voting).

### **Methods Used**

#### **6.1 Criteria Definition**

**5 Thermal/Physical Criteria (per PCM, per cluster):**

```python
CRITERIA = [
    'f_Tm',                  # Gaussian fitness to melting temperature target
    'L_available',           # Latent heat (normalized)
    'rho_H_storage_density', # Energy density (ρ × L) → size efficiency
    'Cp_thermal_capacity',   # Specific heat (quick response)
    'cycles_confidence'      # Cycling durability (robustness)
]
```

**A. f_Tm: Gaussian Temperature Fit**
```python
# PCM that matches climate's Tm perfectly scores 1.0
f_Tm = exp(-(Tm_PCM - Tm_target)² / (2 * σ²))
where σ = 10°C (tolerance width)
```
- **Example:**
  - Tm_target = 20°C, Tm_PCM = 20°C → f_Tm = 1.0 ✓
  - Tm_target = 20°C, Tm_PCM = 15°C → f_Tm ≈ 0.78
  - Tm_target = 20°C, Tm_PCM = 30°C → f_Tm ≈ 0.14

**B. L_available: Latent Heat (Normalized)**
```python
L_norm = (L - L_min) / (L_max - L_min)  # Scale to [0, 1]
```
- **Higher L = faster energy storage/release**

**C. rho_H_storage_density: Energy Density**
```python
rho_H = ρ_solid × L  # [MJ/m³] = volumetric thermal capacity
rho_H_norm = (rho_H - min) / (max - min)
```
- **Why:** Large tank = expensive; high density → compact system

**D. Cp_thermal_capacity: Specific Heat**
```python
Cp_avg = (Cp_solid + Cp_liquid) / 2
Cp_norm = (Cp_avg - min) / (max - min)
```
- **Why:** High Cp = quicker sensible heat response (before melting)

**E. cycles_confidence: Durability**
```python
if cycles_tested >= 5000:     cycles_confidence = 1.0
elif cycles_tested >= 2000:   cycles_confidence = 0.8
elif cycles_tested >= 1000:   cycles_confidence = 0.5
else:                         cycles_confidence = 0.2  # imputed
```
- **Why:** Prevent sudden failure due to poor cycling history

#### **6.2 Normalization (Min-Max to [0,1])**
```python
M_normalized = (M - M_min) / (M_max - M_min)
```
- **Per cluster:** Normalize within that cluster's survivor set
- **Advantage:** Removes unit differences (°C vs. kJ/kg vs. kg/m³)

#### **6.3 Weighting Strategy**

**A. Entropy Weighting (Data-Driven)**
```python
# Information content of each criterion within cluster survivors:
entropy_j = -Σ (p_ij * log(p_ij))  # Shannon entropy per criterion

# Lower entropy = more decisive criterion:
w_entropy_j = (1 - entropy_j) / Σ(1 - entropy_j)
```
- **Example:**
  - If all survivors have similar Cp → entropy high → low weight
  - If survivors vary widely in f_Tm → entropy low → high weight
- **Interpretation:** Criteria that discriminate get higher weight

**B. AHP Prior (Expert Knowledge)**
```python
AHP_PRIOR = {
    'f_Tm': 0.3,        # Melting point most critical
    'L_available': 0.3, # Latent heat equally critical
    'rho_H': 0.2,       # Storage density moderately important
    'Cp': 0.1,          # Sensible heat less critical
    'cycles_confidence': 0.1  # Durability baseline
}
```

**C. Hybrid Weighting (Entropy + AHP)**
```python
ENTROPY_AHP_LAMBDA = 0.5  # 50-50 blend

w_final = 0.5 * w_entropy + 0.5 * (AHP_PRIOR / sum(AHP_PRIOR))
w_final = w_final / sum(w_final)  # Normalize to sum to 1
```
- **Why hybrid?** Balances data evidence + domain expertise

#### **6.4 Method 1: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)**

**Algorithm:**

**Step 1:** Ideal & negative-ideal solutions
```python
ideal_solution = [max(M[:, j]) for j in criteria]      # Best value per criterion
negative_ideal = [min(M[:, j]) for j in criteria]      # Worst value per criterion
```

**Step 2:** Weighted Euclidean distance to ideal
```python
S+ = √(Σ w_j * (M_ij - ideal_j)²)      # Distance to ideal (lower = better)
S- = √(Σ w_j * (M_ij - negative_j)²)   # Distance to negative ideal (higher = better)
```

**Step 3:** Relative closeness
```python
TOPSIS_score = S- / (S+ + S-)
Range: [0, 1]
1.0 = perfect (exactly at ideal)
0.5 = equidistant from ideal & negative-ideal
0.0 = worst possible
```

**Example:**
```
Candidate A: S+ = 0.2, S- = 0.8 → TOPSIS = 0.8 / (0.2 + 0.8) = 0.8 ✓ (excellent)
Candidate B: S+ = 0.7, S- = 0.3 → TOPSIS = 0.3 / (0.7 + 0.3) = 0.3   (poor)
```

**Interpretation:** TOPSIS favors candidates **closest to ideals** across all criteria

#### **6.5 Method 2: GRA (Grey Relational Analysis)**

**Algorithm:**

**Step 1:** Calculate relational coefficient
```python
Δ_ij = |M_ij - ideal_j|  # Deviation from ideal

ρ_ij = (min_Δ + ζ * max_Δ) / (Δ_ij + ζ * max_Δ)
       where ζ = 0.5 (distinguishing coefficient)
```
- **Range:** [0, 1]
- Large Δ → ρ small (poor fit)
- Small Δ → ρ large (good fit)

**Step 2:** Grey relational grade (weighted average)
```python
GRA_grade = Σ w_j * ρ_ij
```

**Example:**
```
Candidate A: ρ = [0.9, 0.8, 0.7, 0.6, 0.5] → GRA = weighted_avg = 0.72
Candidate B: ρ = [0.6, 0.6, 0.6, 0.6, 0.6] → GRA = 0.60
```

**Interpretation:** GRA emphasizes **consistency** (all criteria good, not just some)

#### **6.6 Why Both TOPSIS & GRA?**

| Aspect | TOPSIS | GRA |
|--------|--------|-----|
| **Logic** | Closeness to ideal | Correlation with ideal pattern |
| **Favors** | Candidates excelling in key criteria | Balanced candidates |
| **Example** | "Great melting point, weak Cp" → OK | "Average at everything" → OK |
| **Sensitivity** | High (small changes = rank flip) | Lower (robust to criteria variation) |

**Combined Effect:** TOPSIS + GRA = both "excellence" and "robustness" captured

#### **6.7 Method 3: Consensus Ranking via Borda Voting**

**Algorithm:**

**Step 1:** Convert TOPSIS & GRA scores to ranks
```python
topsis_rank = rank(topsis_score, ascending=False)  # 1=best, 15=worst
gra_rank = rank(gra_grade, ascending=False)
```

**Step 2:** Borda count (points for each rank position)
```python
borda_score = 0
for each ranking method:
    borda_score += (n_candidates - rank + 1)
```
- **Example (n=15):**
  - Rank #1 → 15 points
  - Rank #5 → 11 points
  - Rank #15 → 1 point

**Step 3:** Final consensus rank
```python
consensus_rank = rank(borda_score, ascending=False)
```

**Example Aggregation:**
```
Candidate "RT54HC":
  - TOPSIS rank:  #1   → 15 points
  - GRA rank:     #8   → 8 points
  - Borda total:        23 points
  - Consensus rank:     #1 ✓

Candidate "RT55":
  - TOPSIS rank:  #3   → 13 points
  - GRA rank:     #6   → 10 points
  - Borda total:        23 points
  - Consensus rank:     #2
```

#### **6.8 Robustness Metric: Kendall's W**

**What is it?** Coefficient of concordance (agreement between ranking methods)

**Formula:**
```python
W = 12 * S / (m² * (n³ - n))
where S = Σ (R_i - R_bar)²
      m = number of ranking methods (2 in our case: TOPSIS, GRA)
      n = number of candidates
```

**Interpretation:**
- **W = 1.0** → Perfect agreement (both methods give identical ranks)
- **W = 0.0** → No agreement (random)
- **W < 0.0** → Disagreement (inverse ranks)
- **Your data:** W varies per cluster
  - If W > 0.6 → Strong consensus (trust top-3)
  - If W < 0.3 → Weak consensus (top-3 uncertain)
- **Advice:** Report W alongside recommendations

---

## **STAGE 7: RECOMMENDATION CARDS GENERATION**

### **What Happens**
Synthesize top PCM recommendations per climate regime with justification.

### **Methods Used**

#### **7.1 Top-3 Selection (per cluster)**
```python
top3 = data[data['consensus_rank'] <= 3].groupby('cluster_id')
```
- **Output:** 5 clusters × 3 PCMs = 15 top recommendations

#### **7.2 Thermal Performance Profiling**

For each top-3 PCM per cluster:

**A. Tank Temperature Profile**
```python
# Assuming daily charging/discharging cycle:
T_charge_low = Tm_PCM - 5°C   # Night discharge
T_charge_high = Tm_PCM + 5°C  # Day charge
cycle_COP = useful_energy / input_energy
```

**B. Melt Fraction vs. Temperature**
```python
# S-curve of phase transition:
melt_fraction = 1 / (1 + exp(-k * (T - Tm_PCM)))  # Sigmoid curve
```
- Shows how much PCM has melted at different temperatures
- Steeper curve = sharper transition (better performance)

**C. System Efficiency Estimate**
```python
ηthermal ≈ (useful_energy) / (input_solar_energy)
         ≈ latent_heat_utilization × cycle_efficiency
```

#### **7.3 Recommendation Text Generation**
```python
recommendation = f"""
Cluster {cluster_id} ({climate_description}):

Top 1: {pcm_name[1]}
  - Melting point: {Tm}°C (matches target: {Tm_target}°C)
  - Latent heat: {L} kJ/kg
  - Storage density: {rho_H} MJ/m³
  - Cycles validated: {cycles_tested}
  - TOPSIS rank: #{topsis_rank}, GRA rank: #{gra_rank}, Consensus: #{consensus_rank}
  - Use case: {use_case}

Top 2: {pcm_name[2]}
  ...

Top 3: {pcm_name[3]}
  ...

Selection rationale:
- All candidates pass feasibility filters (melting window, latent heat, cycling)
- Ranked by Borda voting combining TOPSIS & GRA
- Kendall's W = {kendall_w} (agreement strength)
"""
```

#### **7.4 Output Format**
```
data/processed/pcm/recommendation_cards.md
data/plots/objective1/recommended_pcm_summary.html
```

---

# **COMPLETE OBJECTIVE 1 FLOW**

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA DOWNLOAD & COMBINATION                            │
├─────────────────────────────────────────────────────────────────┤
│ ERA5 (hourly) + NASA POWER (daily) → 493,155 records            │
│ 36 variables × 45 locations × 6 years                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING & QC                                     │
├─────────────────────────────────────────────────────────────────┤
│ Physical bounds check (Hampel MAD filter)                        │
│ Missing data imputation (forward-fill + interpolation)           │
│ Feature engineering (45 engineered features):                    │
│   - Lags: 1d, 7d, 30d                                            │
│   - Rolling stats: mean & std over 7d & 30d                      │
│   - Deltas: rate of change                                       │
│   - Solar features: CSI, azimuth, hour angle                     │
│ Scaling: Min-Max on first 70% chronologically                    │
│ Output: 489,105 records × 89 variables                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: CLIMATE SIGNATURE EXTRACTION                           │
├─────────────────────────────────────────────────────────────────┤
│ Compute 30 climate metrics per location:                         │
│   - Temperature: mean, p95, p05, DTR, target Tm                 │
│   - Solar: GHI, kt, SAI, clearness                               │
│   - Cloud: opacity, coverage, intensity                          │
│   - Humidity: RH_mean, T_depression                              │
│   - Heating/Cooling: HDD18, CDD24                                │
│   - Wind: mean, std                                              │
│   - Seasonality: monsoon index, monthly variation                │
│ PCA reduction: 30 → 3 components (85% variance)                  │
│ Standardization: Z-score all features                            │
│ Output: 45 locations × 90 features                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: CLUSTERING (GAUSSIAN MIXTURE MODEL)                    │
├─────────────────────────────────────────────────────────────────┤
│ Find optimal k via BIC, Silhouette, Davies-Bouldin:              │
│   k=2..8 evaluated                                                │
│ Selected: k=5 climate regimes                                    │
│   - Cluster 0: High altitude, cold (9 locations)                 │
│   - Cluster 1: Moderate monsoon (10 locations)                   │
│   - Cluster 2: Tropical, hot (3 locations)                       │
│   - Cluster 3: Strong monsoon, variable (12 locations)           │
│   - Cluster 4: Moderate, solar-rich (11 locations)               │
│ Output: Cluster assignments + soft probabilities                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: PCM FEASIBILITY FILTERING                              │
├─────────────────────────────────────────────────────────────────┤
│ PCM Database: 25 candidates                                      │
│ Constraints per cluster:                                         │
│   1. Melting temp in target window (Tm_target ± margin)          │
│   2. Latent heat ≥ 120 kJ/kg                                     │
│   3. Cycles tested ≥ 1000 (stable)                               │
│   4. Supercooling ≤ 5K                                           │
│   5. Absolute temp band: 0°C to 60°C                             │
│ Result: 125 survivors (25 × 5 clusters)                          │
│ Output: feasibility_survivors_by_cluster.csv                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: MULTI-CRITERIA RANKING (3 METHODS)                     │
├─────────────────────────────────────────────────────────────────┤
│ 5 Criteria (normalized to [0,1]):                                │
│   1. f_Tm: Gaussian fit to target melting point                  │
│   2. L: Latent heat (normalized)                                 │
│   3. ρH: Volumetric storage density                              │
│   4. Cp: Specific heat capacity                                  │
│   5. cycles_confidence: Cycling durability                       │
│                                                                  │
│ Weights: w = 0.5 × w_entropy + 0.5 × w_AHP                       │
│   (hybrid: 50% data-driven, 50% expert)                          │
│                                                                  │
│ METHOD 1: TOPSIS                                                 │
│   ↳ Distance to ideal solution (closeness)                       │
│   ↳ Score: [0, 1] (higher = better)                              │
│   ↳ Favors: Excellence in key criteria                           │
│                                                                  │
│ METHOD 2: GRA (Grey Relational Analysis)                         │
│   ↳ Correlation with ideal pattern (consistency)                 │
│   ↳ Score: [0, 1] (higher = better)                              │
│   ↳ Favors: Balanced performance                                 │
│                                                                  │
│ METHOD 3: CONSENSUS (BORDA VOTING)                               │
│   ↳ Aggregate TOPSIS & GRA ranks via Borda count                 │
│   ↳ Points: (n - rank + 1) per method → sum → re-rank            │
│   ↳ Robustness: Kendall's W (0-1, higher = agreement)           │
│                                                                  │
│ Output: 15 top PCMs (top-3 × 5 clusters)                         │
│         with TOPSIS, GRA, Borda, Kendall's W scores              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7: RECOMMENDATION CARDS & SYNTHESIS                       │
├─────────────────────────────────────────────────────────────────┤
│ Per cluster:                                                     │
│   - Top-3 PCM candidates                                         │
│   - Thermal profiles (melt fraction vs. T)                       │
│   - Performance justification (why this PCM?)                    │
│   - System efficiency estimate                                   │
│                                                                  │
│ Output Formats:                                                  │
│   - Markdown: recommendation_cards.md                            │
│   - HTML: recommended_pcm_summary.html                           │
│   - CSV: mcdm_topk_by_cluster.csv                                │
└─────────────────────────────────────────────────────────────────┘
```

---

# **QUICK REFERENCE: METHODS USED**

| Stage | Method | Why Used | Input | Output |
|-------|--------|----------|-------|--------|
| **1** | ERA5/NASA Power download | Global climate data | Coordinates | Hourly time series |
| **2** | Hampel MAD filter | Robust outlier detection | Raw data | Clean data |
| **2** | Feature engineering (lags, rolling, deltas) | Capture temporal patterns | Time series | 89 features |
| **2** | Min-Max scaling (70% chronological) | Normalize for clustering | Clean features | Scaled features |
| **3** | Climate signature metrics | Summarize thermal environment | Time series | 30 metrics/location |
| **3** | PCA reduction | Dimensionality reduction | 30 metrics | 3 PCs (~85% variance) |
| **3** | Z-score standardization | Prepare for clustering | Metrics | Standardized features |
| **4** | Gaussian Mixture Model | Soft clustering (probabilistic) | Signatures | Cluster assignments |
| **4** | BIC/Silhouette/DB/CH | Select optimal k | Cluster models | k=5 chosen |
| **5** | Feasibility constraints | Filter physically viable PCMs | PCM database + Tm_target | 125 survivors |
| **6** | TOPSIS | Distance to ideal solution | Criteria matrix + weights | Ranks + scores |
| **6** | GRA | Correlation with ideal pattern | Criteria matrix + weights | Ranks + grades |
| **6** | Borda voting | Aggregate multiple rankings | TOPSIS + GRA ranks | Consensus rank + Kendall's W |
| **7** | Thermal profile synthesis | Generate actionable recommendations | Top-3 PCMs per cluster | Recommendation cards |

---

# **KEY INSIGHTS FOR YOUR THESIS**

## **1. Why This Approach?**

✅ **Climate → Regimes → PCM** is the natural hierarchy
- Different regions have different thermal environments
- Each regime needs a tailored PCM selection
- Generic "one PCM for all" is suboptimal

✅ **Multiple methods (TOPSIS + GRA + Consensus) ensure robustness**
- TOPSIS catches "stars" (excel in few criteria)
- GRA catches "all-rounders" (balanced performance)
- Consensus = democratic vote (more fair)

✅ **Feasibility filtering ensures viability**
- Constraints are based on actual PCM properties + thermal physics
- Not just "best score" — must be physically possible

## **2. Validation Points for Defense**

🔹 **Preprocessing:** 99.2% data retention, 45 engineered features, Hampel filter proven robust
🔹 **Clustering:** 5 regimes identified; silhouette 0.253 acceptable for continuous climate
🔹 **MCDM:** 3 methods + Borda voting > single method; Kendall's W quantifies agreement
🔹 **Ranking:** Top-3 survive multiple ranking methods (robust, not lucky)
🔹 **Feasibility:** Constraints based on thermophysics + PCM literature (defensible)

## **3. Limitations to Acknowledge**

⚠️ **Small sample size:** Only 45 locations (45 clusters ideally separable, but only 5 chosen)
⚠️ **Climate-agnostic feasibility:** Filters same for all regions (could be climate-specific)
⚠️ **Borda voting:** Simple majority vote (could use more sophisticated aggregation)
⚠️ **No lifecycle cost:** Ranking focuses on thermal performance only
⚠️ **No system integration:** Recommendations assume ideal heat exchanger (in reality, losses exist)

---

# **FOR YOUR SLIDES: 5-SLIDE STRUCTURE**

### **Slide 1: Objective Overview**
- Goal: Select PCMs for climate-specific thermal storage
- Process: Climate data → regimes → filtering → ranking → recommendations
- Image: Flow diagram (ASCII above)

### **Slide 2: Preprocessing & Climate Signatures**
- 493K records → 489K clean (99.2% retention)
- 45 engineered features (lags, rolling, deltas)
- 30 climate metrics computed (temperature, solar, cloud, wind, seasonality)
- Image: Before/after histogram + correlation matrix

### **Slide 3: Clustering & Regimes**
- k=5 climate regimes via GMM
- Validation: Silhouette 0.253 (acceptable for continuous data)
- Geographic coherence: Clusters spatially contiguous
- Image: Geographic map + cluster sizes

### **Slide 4: Feasibility & PCM Ranking**
- 5 feasibility constraints → 125 survivors
- TOPSIS vs. GRA vs. Consensus (Borda voting)
- Top-3 PCMs per cluster (robust across methods)
- Image: Method correlation heatmap + rank heatmap

### **Slide 5: Recommendations**
- Climate regimes matched with optimal PCMs
- Thermal profiles & performance justification
- Key metrics: Tm, L, ρH, Cp, cycles
- Image: Recommended PCM summary table + tank profile

---

This covers **everything** in Objective 1! Any specific stage you want deeper dive?

