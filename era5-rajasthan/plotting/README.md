# Objective 1 Plotting Audit — Rajasthan Pipeline

**Status:** Implementation in progress (2026-09-02)  
**Context:** Comprehensive audit of Phases 1–9 via structured plotting and verification

This directory contains automated plotting scripts that generate all 13 main plots and 8 comparison plots for the Rajasthan Objective-1 PCM selection and optimization pipeline, per the detailed audit specification in `../docs/rajasthan/11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`.

## Quick Start

```bash
cd era5-rajasthan
python plotting/run_all_plots.py
```

Outputs are written to `outputs/objective1_plots_rajasthan/` with subdirectories by plot/phase.

---

## Table of Contents
- [Quick Overview of All Plots](#quick-overview-of-all-plots)
- [Part A: Main Plots (13 Required)](#part-a--main-plots-13-required)
- [Part B: Comparison Plots (8 Additional)](#part-b--comparison-plots-8-additional)
- [Architecture & Verification Pattern](#architecture--verification-pattern)
- [Output Structure](#output-structure)
- [Running Individual Plots](#running-individual-plots)
- [Known Issues & TODOs](#known-issues--todos)

## ⚠️ Staleness Notice

As of **2026-08-31**, Phase 3's `L_required` methodology was corrected (SHARE_PCM=0.5), cascading through Phases 5–9.

**Current state:**
- Data on disk is **post-correction** (L_required ~285–344 kJ/kg, 39 total calibrated survivors)
- All plotting scripts include fingerprint checks via `provenance_lib.py`
- If Phases 5–9 are re-run before plotting, plots will be re-generated against new data automatically
- If not, plots will be stamped with a staleness watermark

---

# Quick Overview of All Plots

## Part A: Main Plots (13 Required)

| # | Plot Name | Purpose | Key Output |
|---|-----------|---------|------------|
| 1 | **Raw vs. Preprocessed Radiation** | Verify data cleaning didn't over-filter | KS statistics showing data drift |
| 2 | **Climate Regime Map** | Visualize 3 climate clusters geographically | Folium map showing cluster regions |
| 3 | **PCM Feasibility Scatter** | Show which PCMs survive the screening | Tm vs. Latent heat with feasible region highlighted |
| 4 | **Survivors per Cluster** | Compare primary vs. calibrated runs | Bar chart of survivor counts (39 post-correction) |
| 5 | **Bump Chart** | Track rank agreement across 4 MCDM methods | Connected lines showing rank stability per cluster |
| 6 | **Method Correlation Heatmap** | Quantify MCDM method agreement | Spearman correlation matrices (1 per cluster) |
| 7 | **MC Inclusion Probability** | Show stability of top-3 recommendations | Histogram of Monte Carlo inclusion rates |
| 8 | **Rank-Reversal Frequency** | Measure how often candidates swap order | Frequency distribution per cluster (higher in Cluster 0) |
| 9 | **MCDM vs. Physics Agreement** | Validate if MCDM ranks match simulated performance | Scatter + trend line (Spearman rho per cluster) |
| 10 | **Tank Profile (Day-Night)** | Show instantaneous thermal dynamics | Hourly Tw, Tp, melt fraction arrays (awaiting instrumentation) |
| 11 | **Summary Cards** | One-page recommendation summary | 3-panel figure with Top-1 PCM per cluster |
| — | **MC Consolidation (Plot 7)** | Combine inclusion & imputation analysis | Copy of existing + added imputation watermarks |

## Part B: Comparison Plots (8 Additional)

| Phase | Comparison | Purpose | Data Source |
|-------|-----------|---------|------------|
| 2.5 | **Raw vs. Clean (5 variables)** | Generalize Plot 1 to all climate variables | T_amb, RHum, W_spd, GHI, CSI distributions |
| 3 | **Tier 1 vs. Tier 2** | Show impact of climate stratification method | diurnal_gradient vs. DTR_true comparison |
| 3 | **Tm_target_capped (old vs. new)** | Document 2026-08-11 methodology fix | Pre: 40.8-49.5°C vs. Post: 51.1-55.2°C |
| 4 | **Level A vs. Level B** | Show seasonal cluster-assignment drift | Cluster label changes across seasons |
| 5 | **L_required Before/After** | Demonstrate impact of SHARE_PCM correction | Pre: 0 survivors vs. Post: 39 survivors |
| 6 | **VIKOR Sign-Inversion Bugfix** | Show rank order reversal fix (if backup exists) | Requires pre/post-bugfix data backups |
| 7 | **PCM vs. Plain Tank** | Show added value of PCM integration | Plain tank (latent_heat=0) vs. PCM performance |
| 8 | **Penalty k=0.0 vs. k=0.3** | Evaluate supercooling penalty impact | Survivor counts under two penalty scenarios |

---

## PART A: Main Plots — Detailed Documentation

### Plot 1: Raw vs. Preprocessed Radiation

**File:** `01_raw_vs_preprocessed.py`

**What it does:**
Overlays histograms of raw vs. cleaned climate data, specifically verifying that GHI (global horizontal irradiance) remains nearly unchanged while T_amb (ambient temperature) shows visible tail-trimming after Hampel filtering.

**Key Code:**
```python
# Load both datasets
raw_df = pd.read_csv("climate_rajasthan_points.csv")
clean_df = pd.read_csv("climate_rajasthan_points_clean.csv")

# Compute KS (Kolmogorov-Smirnov) statistics
ks_ghi, ks_ghi_pval = ks_2samp(ghi_raw, ghi_clean)
ks_tamb, ks_tamb_pval = ks_2samp(tamb_raw, tamb_clean)

# Create subplots: GHI (left), T_amb (right)
fig = make_subplots(rows=1, cols=2, ...)
fig.add_trace(go.Histogram(x=ghi_raw, name="GHI Raw", ...))
fig.add_trace(go.Histogram(x=ghi_clean, name="GHI Clean", ...))
# ... similar for T_amb
```

**Verification Block:**
- **KS stat for GHI:** Should be small (<0.05) because GHI is deliberately excluded from filtering
- **KS stat for T_amb:** Should be large (>0.1) because T_amb IS Hampel-filtered
- **Output:** `01_raw_vs_preprocessed/ghi_tamb_distributions.html`

**What you'll see:**
- Two side-by-side histograms
- Left panel: GHI distributions nearly identical (raw ≈ clean)
- Right panel: T_amb showing visible tail-trimming (clean narrower than raw)
- KS statistics printed to console during execution

**Why this matters:**
Confirms that data cleaning targeted the right variables without over-filtering irradiance.

---

### Plot 2: Climate Regime Map

**File:** `02_climate_regime_map_copy.py`

**What it does:**
Copies and consolidates the existing folium-based geographic map from Phase 4 output, showing the 3 climate clusters across Rajasthan. Does NOT regenerate; reuses existing Phase 4 output to avoid redundant computation.

**Key Code:**
```python
# Load cluster profiles to verify metadata
profiles_df = pd.read_csv("cluster_profiles_rajasthan.csv")

# Verify k=3 and canonical relabeling
print(f"Number of clusters (k): {len(profiles_df)}")  # Should be 3
print(f"Cluster IDs: {sorted(profiles_df['cluster_id'].unique())}")  # Should be [0, 1, 2]

# Copy existing map
shutil.copy2("../outputs/qc_cluster_map_rajasthan.html", 
             "../outputs/objective1_plots_rajasthan/02_climate_regime_map/climate_regime_map_rajasthan.html")
```

**Verification Block:**
- **k value:** Must be exactly 3 clusters
- **Cluster IDs:** Must be in canonical order [0, 1, 2] (lowest latitude → highest)
- **Cluster sizes:** Printed for sanity check
- **Output:** `02_climate_regime_map/climate_regime_map_rajasthan.html` (copy from Phase 4)

**What you'll see:**
- Interactive folium map with Rajasthan data points colored by cluster
- Cluster 0 (blue), Cluster 1 (orange), Cluster 2 (green)
- Hover tooltips showing lat/lon and cluster assignment
- Legend and zoom/pan controls

**Why this matters:**
Confirms that climate stratification produced meaningful geographic separation (not random clustering).

---

### Plot 3: PCM Feasibility Scatter

**File:** `03_pcm_feasibility_scatter.py`

**What it does:**
Scatter plot of all PCM candidates (melting point Tm on x-axis vs. latent heat on y-axis). Survivors are colored by cluster; non-survivors are light gray. Also shows target Tm range (42–70°C band) and L_required threshold line.

**Key Code:**
```python
# Load all three datasets
profiles_df = pd.read_csv("cluster_profiles_rajasthan.csv")
survivors_df = pd.read_csv("feasibility_survivors_rajasthan_kappa_calibrated.csv")
pcm_db_df = pd.read_csv("PCM_Properties_cleaned_mice_pmm_detailed.csv")

# Check fingerprint staleness (provenance_lib)
current_fp = file_fingerprint(CLUSTER_PROFILE_FILE)
if survivors_df["upstream_cluster_profile_fingerprint"].iloc[0] != fingerprint_id(current_fp):
    print("⚠ WARNING: Data is STALE")  # But continue with watermark

# Count survivors per cluster
for cluster_id in profiles_df["cluster_id"].unique():
    count = len(survivors_df[survivors_df["cluster_id"] == cluster_id])
    print(f"Cluster {cluster_id}: {count} survivors")

# Create scatter
fig.add_trace(go.Scatter(
    x=survivors["Tm_C"],
    y=survivors["latent_heat_kJ_kg"],
    mode="markers",
    name=f"Cluster {cid} ({len(survivors)} survivors)",
    ...
))

# Add vertical band for 42-70°C target
fig.add_vrect(x0=42, x1=70, fillcolor="lightblue", opacity=0.2, ...)

# Add horizontal line for L_required
fig.add_hline(y=l_required_mean, line_dash="dash", ...)
```

**Verification Block:**
- **Fingerprint match:** Survivors vs. cluster_profiles must be from same run
- **Survivor count per cluster:** Post-correction should be ~13-15 per cluster (39 total)
- **L_required values:** Post-correction should be 285–344 kJ/kg (pre-correction was 608–641, yielding 0 survivors)
- **Output:** `03_feasibility/pcm_feasibility_scatter.{png,html}`

**What you'll see:**
- Scatter of PCMs colored by cluster (survivors) or gray (non-survivors)
- Light blue vertical band showing 42–70°C melting-point target range
- Red dashed horizontal line at mean L_required threshold
- Hover: PCM name, exact Tm and latent heat values
- Interactive legend to toggle clusters on/off

**Why this matters:**
Visual confirmation that feasibility screening filtered meaningfully; shows which candidates are in target ranges vs. outliers.

---

### Plot 4: Survivors per Cluster

**File:** `04_pcm_survivors_per_cluster.py`

**What it does:**
Grouped bar chart comparing primary run (κ=0.7 fixed) vs. calibrated run (κ-optimized) survivor counts per cluster. Bars annotated with calibrated κ values.

**Key Code:**
```python
# Load both survivor files
survivors_primary = pd.read_csv("feasibility_survivors_rajasthan.csv")
survivors_calibrated = pd.read_csv("feasibility_survivors_rajasthan_kappa_calibrated.csv")

# Count per cluster
for cluster_id in clusters:
    primary_count = len(survivors_primary[survivors_primary["cluster_id"] == cluster_id])
    calibrated_count = len(survivors_calibrated[survivors_calibrated["cluster_id"] == cluster_id])
    
    # Get calibrated kappa value
    kappas = survivors_calibrated[survivors_calibrated["cluster_id"] == cluster_id]["calibrated_kappa"].unique()
    kappa_mean = np.mean(kappas) if len(kappas) > 0 else 0.0
    
    print(f"Cluster {cluster_id}: {primary_count} (primary) → {calibrated_count} (calibrated, κ={kappa_mean:.4f})")

# Create grouped bar chart
fig.add_trace(go.Bar(x=clusters, y=primary_counts, name="Primary (κ=0.7 fixed)", ...))
fig.add_trace(go.Bar(x=clusters, y=calibrated_counts, name="Calibrated (κ_optimized)", ...))
```

**Verification Block:**
- **Primary total:** Should match original kappa=0.7 screening
- **Calibrated total:** Post-correction should be 39; pre-correction was 20
- **Kappa values:** Printed and annotated on bars for each cluster
- **Output:** `03_feasibility/pcm_survivors_per_cluster.{png,html}`

**What you'll see:**
- Two bars per cluster (light blue for primary, dark blue for calibrated)
- Calibrated bar annotated with κ value and survivor count
- Title showing total counts (e.g., "39 total calibrated survivors")
- Hover shows exact counts for each bar

**Why this matters:**
Demonstrates that kappa calibration fine-tunes feasibility screening; post-correction calibration recovers viable candidates.

---

### Plot 5: Bump Chart (MCDM Method Agreement)

**File:** `05_bump_chart.py`

**What it does:**
Slopegraph showing each PCM candidate's rank across 5 methods: TOPSIS, PROMETHEE II, VIKOR, GRA, and Borda consensus. Generated once per cluster (3 total charts). Lines that stay flat = good method agreement; crossing lines = rank reversals.

**Key Code:**
```python
mcdm_df = pd.read_csv("mcdm_rankings_rajasthan.csv")

# Check for VIKOR sign-inversion bug (strong negative Spearman rho)
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]
    rho, pval = spearmanr(cluster_data["VIKOR_rank"], cluster_data["TOPSIS_rank"])
    
    if rho < -0.5:
        print(f"[WARN] Cluster {cluster_id}: Strong negative correlation (VIKOR bug?)")
    elif rho > 0.3:
        print(f"[OK] Cluster {cluster_id}: Positive correlation, no bug detected")

# Create bump chart for each cluster
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id].sort_values("borda_rank")
    
    fig = go.Figure()
    
    # Add traces for each method
    for method in ["TOPSIS_rank", "PROMETHEE_II_rank", "VIKOR_rank", "GRA_rank", "borda_rank"]:
        fig.add_trace(go.Scatter(
            x=["TOPSIS", "PROMETHEE", "VIKOR", "GRA", "Borda"],
            y=cluster_data[method],
            mode="lines+markers",
            name=cluster_data["pcm_id"],  # One line per PCM
            ...
        ))
    
    fig.write_html(f"bump_chart_cluster_{cluster_id}.html")
```

**Verification Block:**
- **Spearman rho (VIKOR vs. TOPSIS):** Should be >0.3 (positive correlation); if <-0.5, suggests sign-inversion bug
- **Method agreement:** Flat lines = strong agreement; crossing lines = weaker agreement
- **Output:** `04_mcdm_agreement/bump_chart_cluster_{0,1,2}.html` (3 separate HTML files)

**What you'll see:**
- 5 x-axis labels: TOPSIS, PROMETHEE II, VIKOR, GRA, Borda
- Multiple colored lines (one per PCM candidate)
- Lines that stay close together = good agreement
- Lines crossing heavily = rank reversals between methods
- Hover shows PCM name and exact rank at each method
- Interactive legend to show/hide candidates

**Why this matters:**
Shows whether top MCDM recommendations are robust across different weighting methods or fragile/method-dependent.

---

### Plot 6: Method Correlation Heatmap

**File:** `06_method_correlation_heatmap.py`

**What it does:**
Computes Spearman rank-correlation matrix between TOPSIS, PROMETHEE II, VIKOR, and GRA rankings. Generated once per cluster (3 heatmaps). Also identifies which method has lowest mean correlation (expected: GRA).

**Key Code:**
```python
mcdm_df = pd.read_csv("mcdm_rankings_rajasthan.csv")
methods = ["TOPSIS_rank", "PROMETHEE_II_rank", "VIKOR_rank", "GRA_rank"]

for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]
    
    # Compute 4x4 Spearman correlation matrix
    correlations = np.zeros((4, 4))
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                correlations[i, j] = 1.0
            else:
                rho, _ = spearmanr(cluster_data[method1], cluster_data[method2])
                correlations[i, j] = rho
    
    # Compute mean pairwise correlation per method (excluding diagonal)
    mean_corr_per_method = {}
    for i, method_name in enumerate(["TOPSIS", "PROMETHEE", "VIKOR", "GRA"]):
        corrs = correlations[i, :].copy()
        corrs[i] = np.nan  # Exclude diagonal
        mean_corr_per_method[method_name] = np.nanmean(corrs)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlations,
        x=["TOPSIS", "PROMETHEE", "VIKOR", "GRA"],
        y=["TOPSIS", "PROMETHEE", "VIKOR", "GRA"],
        colorscale="RdBu",
        zmid=0,
        ...
    ))
    
    fig.write_html(f"method_correlation_heatmap_cluster_{cluster_id}.html")
```

**Verification Block:**
- **Lowest correlation method:** Should be GRA (structural outlier in MCDM methodologies)
- **Correlation ranges:** Typically 0.3–0.8 pairwise
- **Output:** `04_mcdm_agreement/method_correlation_heatmap_cluster_{0,1,2}.html` (3 separate files)

**What you'll see:**
- 4×4 heatmap (methods vs. methods)
- Red (high correlation >0.5), blue (low correlation <0.3), white (near-zero)
- Diagonal always 1.0 (method vs. itself)
- Hover shows exact Spearman rho value
- Title shows cluster ID and any outlier method notes

**Why this matters:**
Quantifies method agreement; high correlations (>0.6) indicate robust consensus; low correlations suggest fragile recommendations.

---

### Plot 7: Monte Carlo Inclusion Probability

**File:** (Pre-existing from Phase 6 output)

**What it does:**
Copies and consolidates existing Phase 6 output showing the distribution of Monte Carlo inclusion probabilities for each candidate across 1000 runs. High bars = stable recommendations; low bars = fragile recommendations.

**Key Code:**
```python
# This is a pre-existing plot from Phase 6
# Plotting script copies it from: ../outputs/qc_montecarlo_inclusion_rajasthan.html

# Copy operation
shutil.copy2("../outputs/qc_montecarlo_inclusion_rajasthan.html",
             "../outputs/objective1_plots_rajasthan/05_montecarlo/qc_montecarlo_inclusion_rajasthan.html")
```

**Verification Block:**
- **File existence:** Must exist from Phase 6 run
- **Output:** `05_montecarlo/qc_montecarlo_inclusion_rajasthan.html` (copied from Phase 6)

**What you'll see:**
- Histogram of Monte Carlo inclusion probabilities (0–100%)
- Peaks near 100% = candidates that appear in Top-3 in most draws
- Candidates near 0% = fragile, rarely in Top-3
- Vertical lines marking clusters or quality thresholds
- Hover shows exact counts per bin

**Why this matters:**
Shows robustness of recommendations; Top-1 should have >80% inclusion probability.

---

### Plot 8: Rank-Reversal Frequency

**File:** `08_rank_reversal_frequency.py`

**What it does:**
Bar or violin plot showing the frequency with which candidates swap order across 1000 Monte Carlo draws. Cluster 0 (weak agreement, Kendall's W=0.388) should show HIGHER reversals; Clusters 1/2 (strong agreement, W≈0.63) should show LOWER reversals.

**Key Code:**
```python
mcdm_df = pd.read_csv("mcdm_rankings_rajasthan.csv")
profiles_df = pd.read_csv("cluster_profiles_rajasthan.csv")

# Verify the key metric: Kendall's W per cluster
kendalls_w_by_cluster = mcdm_df.groupby("cluster_id")["kendalls_w_cluster"].first().to_dict()
print(f"Kendall's W values:")
for cid in clusters:
    print(f"  Cluster {cid}: W = {kendalls_w_by_cluster[cid]:.4f}")

# Extract rank-reversal frequency (already computed in Phase 6)
print("\nMean rank-reversal frequency per cluster:")
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]
    freq = cluster_data["mc_rank_reversal_freq_cluster"].iloc[0]
    w = kendalls_w_by_cluster.get(cluster_id, np.nan)
    print(f"  Cluster {cluster_id}: freq = {freq:.3f} (Kendall's W = {w:.4f})")

# Prediction check: Cluster 0 freq > Clusters 1/2 freq?
if cluster_freq_data[0] > cluster_freq_data[1] and cluster_freq_data[0] > cluster_freq_data[2]:
    print(f"[OK] PASS: Cluster 0 has higher rank-reversal frequency (as expected)")

# Create bar or violin plot
fig = go.Figure()
for cluster_id in clusters:
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]
    # Extract per-candidate reversal frequencies
    fig.add_trace(go.Bar(
        x=cluster_data["pcm_id"],
        y=cluster_data["mc_rank_reversal_freq_in_cluster"],
        name=f"Cluster {cluster_id}",
        ...
    ))
```

**Verification Block:**
- **Kendall's W values:** Cluster 0 should be ~0.388 (low); Clusters 1/2 should be ~0.634–0.635 (high)
- **Reversal frequency relationship:** Cluster 0 freq > Clusters 1/2 freq (low agreement = more reversals)
- **Output:** `05_montecarlo/rank_reversal_frequency_rajasthan.html`

**What you'll see:**
- Bar chart (per cluster) or grouped bars (all clusters)
- Cluster 0 bars noticeably taller (higher reversal rates)
- Clusters 1/2 bars shorter (lower reversal rates)
- Y-axis: reversal frequency as % or fraction of 1000 draws
- Hover shows exact count and cluster membership

**Why this matters:**
Validates that weak method agreement (low Kendall's W) translates to instability in recommendations; provides quantitative evidence.

---

### Plot 9: MCDM vs. Physics Agreement

**File:** `09_mcdm_vs_physics_agreement.py`

**What it does:**
Scatter plot of MCDM consensus rank (Borda score) vs. simulated annual solar fraction per candidate. If MCDM is valid, there should be a positive trend: higher MCDM rank (lower Borda score) = higher solar fraction. Computes Spearman rho and plots trend line.

**Key Code:**
```python
mcdm_df = pd.read_csv("mcdm_rankings_rajasthan.csv")
physics_df = pd.read_csv("physics_validation_rajasthan.csv")
rho_df = pd.read_csv("spearman_rho_by_cluster_rajasthan.csv")

# Join MCDM and physics data
joined = physics_df.merge(
    mcdm_df[["cluster_id", "pcm_id", "borda_score"]],
    on=["cluster_id", "pcm_id"],
    how="inner"
)

# Compute Spearman rho per cluster and compare against audit baseline
pre_correction_rho = {0: -0.385, 1: +0.125, 2: -0.097}  # Audit documented

for cluster_id in clusters:
    cluster_data = joined[joined["cluster_id"] == cluster_id]
    
    # Compute Spearman correlation
    rho, pval = spearmanr(cluster_data["borda_score"], cluster_data["annual_solar_fraction"])
    
    expected_rho = pre_correction_rho.get(cluster_id, None)
    print(f"Cluster {cluster_id}: rho = {rho:+.3f} (p={pval:.3f})")
    if expected_rho:
        print(f"  Audit baseline (pre-correction): {expected_rho:+.3f}")
    
    # Create scatter plot with trend line
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cluster_data["borda_score"],
        y=cluster_data["annual_solar_fraction"],
        mode="markers",
        name=f"Cluster {cluster_id}",
        marker=dict(size=8, color=colors[cluster_id]),
        text=cluster_data["pcm_id"],
        hovertemplate="%{text}<br>Borda: %{x:.1f}<br>Solar Frac: %{y:.3f}<extra></extra>"
    ))
    
    # Add trend line (linear regression)
    z = np.polyfit(cluster_data["borda_score"], cluster_data["annual_solar_fraction"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(cluster_data["borda_score"].min(), cluster_data["borda_score"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode="lines",
        name=f"Trend (rho={rho:+.3f})",
        line=dict(color="red", dash="dash")
    ))
    
    fig.write_html(f"mcdm_vs_physics_agreement_cluster_{cluster_id}.html")
```

**Verification Block:**
- **Spearman rho per cluster:** Compare against pre-correction baseline (audit document)
- **Post-correction re-runs:** May show different rho; check `spearman_rho_by_cluster_rajasthan.csv` for updated baseline
- **Trend direction:** Should be negative (lower Borda score = better MCDM rank = higher solar fraction)
- **Output:** `06_physics_validation/mcdm_vs_physics_agreement_rajasthan.html`

**What you'll see:**
- Scatter plot per cluster (3 separate charts)
- X-axis: Borda score (lower = better MCDM rank)
- Y-axis: Annual solar fraction (simulated, higher = better)
- Red dashed trend line showing correlation direction
- Hover shows PCM name, exact Borda score, and solar fraction
- Title annotated with Spearman rho and p-value

**Why this matters:**
Validates that MCDM methodology produces rankings consistent with physics-based simulation; low or negative rho suggests MCDM weighting issues.

---

### Plot 10: Tank Profile (Day-Night Cycle)

**File:** (Awaiting implementation)

**Status:** BLOCKED — requires instrumentation of `physics_lib.py`

**What it will do (when implemented):**
Plots hourly time-series of tank water temperature (Tw), PCM capsule temperature (Tp), and melt fraction (0–1) over one representative day-night cycle. Shows how PCM charging (day) and discharging (night) modulates tank dynamics.

**Prerequisites:**
1. Modify `physics_lib.py` to add `save_timeseries=True` parameter
2. Return hourly arrays instead of just aggregate metrics
3. Run one simulation per cluster representative location

**Placeholder Code:**
```python
# Awaiting implementation
# def simulate_one_year(..., save_timeseries=False):
#     ...
#     if save_timeseries:
#         return {
#             ...,
#             "hourly_Tw": Tw_array,
#             "hourly_Tp": Tp_array,
#             "hourly_melt_frac": melt_array
#         }

# When ready:
# for cluster_id in clusters:
#     physics_output = simulate_one_year(..., save_timeseries=True)
#     
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
#                          subplot_titles=("Tank Temp (Tw)", "PCM Temp (Tp)", "Melt Fraction"))
#     
#     fig.add_trace(go.Scatter(y=physics_output["hourly_Tw"], ...), row=1, col=1)
#     fig.add_trace(go.Scatter(y=physics_output["hourly_Tp"], ...), row=2, col=1)
#     fig.add_trace(go.Scatter(y=physics_output["hourly_melt_frac"], ...), row=3, col=1)
```

**Output:** `06_physics_validation/tank_profile_cluster_{0,1,2}_rajasthan.html` (3 files)

**Why this matters:**
Visual confirmation of thermal dynamics; shows how PCM buffering extends availability during night hours.

---

### Plot 11: Summary Cards

**File:** `11_summary_cards.py`

**What it does:**
High-level summary figure (3-panel, one per cluster) showing Top-1 PCM recommendation with key properties: name, Tm, latent heat, MCDM confidence (MC inclusion %), and physics validation rho with pass/fail flag.

**Key Code:**
```python
physics_df = pd.read_csv("physics_validation_rajasthan.csv")
mcdm_df = pd.read_csv("mcdm_rankings_rajasthan.csv")
rho_df = pd.read_csv("spearman_rho_by_cluster_rajasthan.csv")

# Extract Top-1 per cluster (lowest Borda score = best)
top1_per_cluster = {}
for cluster_id in mcdm_df["cluster_id"].unique():
    cluster_data = mcdm_df[mcdm_df["cluster_id"] == cluster_id]
    top1_row = cluster_data.loc[cluster_data["borda_score"].idxmin()]
    
    top1_per_cluster[cluster_id] = {
        "pcm_id": top1_row["pcm_id"],
        "borda_score": top1_row["borda_score"],
        "mc_top3_inclusion_pct": top1_row["mc_top3_inclusion_pct"],
    }

# Get properties from PCM database
for cluster_id, info in top1_per_cluster.items():
    pcm_id = info["pcm_id"]
    pcm_row = pcm_db[pcm_db["pcm_id"] == pcm_id]
    
    properties = {
        "Tm_C": pcm_row["Tm_C"].values[0],
        "latent_heat_kJ_kg": pcm_row["latent_heat_kJ_kg"].values[0],
    }
    
    # Get physics validation (annual solar fraction)
    physics_row = physics_df[(physics_df["cluster_id"] == cluster_id) & 
                              (physics_df["pcm_id"] == pcm_id)]
    properties["solar_fraction"] = physics_row["annual_solar_fraction"].values[0]
    
    # Get Spearman rho (confidence in physics validation)
    rho_row = rho_df[rho_df["cluster_id"] == cluster_id]
    properties["spearman_rho"] = rho_row["spearman_rho"].values[0] if not rho_row.empty else np.nan
    
    # Flag: rho > 0.4 = "Validated", rho <= 0.4 = "NOT Validated"
    validation_status = "✓ Validated" if properties["spearman_rho"] > 0.4 else "⚠ NOT Validated"

# Create 3-panel matplotlib figure
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig)

for idx, (cluster_id, props) in enumerate(top1_per_cluster.items()):
    ax = fig.add_subplot(gs[0, idx])
    
    # Draw card background and text
    ax.add_patch(mpatches.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                     fill=True, facecolor="lightblue", 
                                     edgecolor="black", linewidth=2))
    
    ax.text(0.5, 0.85, f"Cluster {cluster_id}", ha="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.75, f"Recommended PCM: {props['pcm_id']}", ha="center", fontsize=12)
    ax.text(0.5, 0.60, f"Tm: {props['Tm_C']:.1f}°C", ha="center", fontsize=11)
    ax.text(0.5, 0.50, f"L: {props['latent_heat_kJ_kg']:.0f} kJ/kg", ha="center", fontsize=11)
    ax.text(0.5, 0.40, f"MC Inclusion: {props['mc_top3_inclusion_pct']:.1f}%", ha="center", fontsize=11)
    ax.text(0.5, 0.25, f"Solar Fraction: {props['solar_fraction']:.3f}", ha="center", fontsize=11)
    ax.text(0.5, 0.10, validation_status, ha="center", fontsize=10, 
            color="green" if validation_status.startswith("✓") else "red", weight="bold")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

fig.suptitle("Objective 1 Final Recommendations — Rajasthan", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("summary_cards_rajasthan.png", dpi=150, bbox_inches="tight")
```

**Verification Block:**
- **Top-1 names:** Must match MCDM rankings CSV (checked against `recommendation_cards_rajasthan.md`)
- **MC inclusion %:** Should be >80% for robust recommendations
- **Physics validation:** rho > 0.4 = "Validated"; rho ≤ 0.4 = "NOT Validated" (flag for follow-up)
- **Output:** `07_recommendation_summary/summary_cards_rajasthan.png`

**What you'll see:**
- 3 colored cards (one per cluster, arranged horizontally)
- Each card shows:
  - Cluster ID (title)
  - Top-1 PCM name (bold)
  - Tm (melting point)
  - Latent heat
  - MC inclusion probability (%)
  - Simulated annual solar fraction
  - ✓ Validated or ⚠ NOT Validated flag
- Professional publication-ready figure

**Why this matters:**
One-page summary of final recommendations; useful for thesis defense or presentations.

---

## PART B: Comparison Plots — Detailed Documentation

### Comparison Plot 1: Raw vs. Clean (5 Variables)

**File:** `comparison_phase2_5_raw_vs_clean.py`

**Status:** ⏳ TODO (not yet implemented)

**What it will do:**
Generalizes Plot 1 to show distributions of all 5 climate variables before/after cleaning:
- GHI (excluded from filtering, should be unchanged)
- T_amb (Hampel-filtered, should show tail-trimming)
- RHum (relative humidity, if filtered)
- W_spd (wind speed, if filtered)
- CSI (clear-sky index, if used)

**Expected Code Structure:**
```python
variables = ["era5_GHI", "era5_T_amb", "era5_RHum", "era5_W_spd", "era5_CSI"]

fig = make_subplots(rows=1, cols=5, subplot_titles=variables)

for idx, var in enumerate(variables, 1):
    ks_stat, ks_pval = ks_2samp(raw_df[var].dropna(), clean_df[var].dropna())
    
    fig.add_trace(go.Histogram(x=raw_df[var], name=f"{var} Raw", ...), row=1, col=idx)
    fig.add_trace(go.Histogram(x=clean_df[var], name=f"{var} Clean", ...), row=1, col=idx)
    
    # Annotate KS statistic on each subplot
```

**Expected Output:** `comparison_plots/phase2_5_raw_vs_clean/{histograms_5var.html}`

---

### Comparison Plot 2: Tier 1 vs. Tier 2

**File:** `comparison_phase3_tier1_vs_tier2.py`

**Status:** ⏳ TODO

**What it will do:**
Contrast two stratification methods for climate regimes:
- **Tier 1:** Diurnal gradient (T_max - T_min per day)
- **Tier 2:** DTR_true (day-to-month temperature range)

Shows how cluster assignments differ under each method.

---

### Comparison Plot 3: Tm_target_capped (Old vs. New)

**File:** `comparison_phase3_tmcap_old_vs_new.py`

**Status:** ✓ EXISTS (code written, awaiting deployment)

**What it does:**
Visualizes the 2026-08-11 methodology fix for Tm_target_capped:
- **Old basis (p05-day):** Single worst day's capacity → 40.8–49.5°C (implausibly low)
- **New basis (worst-month):** 30-day worst-month capacity → 51.1–55.2°C (realistic)

**Key Code:**
```python
sig_df = pd.read_csv("climate_signature_rajasthan.csv")

# Both columns already exist (audit trail)
old_col = "Tm_target_capped_C_p05day"
new_col = "Tm_target_capped_C"

old_vals = sig_df[old_col].dropna()
new_vals = sig_df[new_col].dropna()

print(f"Old basis: mean={old_vals.mean():.2f}°C, range={old_vals.min():.2f}–{old_vals.max():.2f}°C")
print(f"New basis: mean={new_vals.mean():.2f}°C, range={new_vals.min():.2f}–{new_vals.max():.2f}°C")

# Create scatter plot with y=x reference line
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=old_vals,
    y=new_vals,
    mode="markers",
    name="Observed shift",
    marker=dict(size=8, color="blue", opacity=0.6)
))

# Add y=x reference line (no change)
max_val = max(old_vals.max(), new_vals.max())
fig.add_trace(go.Scatter(
    x=[0, max_val],
    y=[0, max_val],
    mode="lines",
    name="No change (y=x)",
    line=dict(dash="dash", color="gray")
))

fig.update_layout(title="Phase 3: Tm_target_capped Methodology Revision (2026-08-11)")
```

**Output:** `comparison_plots/phase3_tmcap_old_vs_new/{scatter_plot.html}`

---

### Comparison Plot 4: Level A vs. Level B

**File:** `comparison_phase4_levelA_vs_levelB.py`

**Status:** ⏳ TODO

**What it will do:**
Show seasonal shift in cluster assignments (if Level A = winter, Level B = summer, for example). Tracks which data points change clusters across seasonal snapshots.

---

### Comparison Plot 5: L_required Before/After

**File:** `comparison_phase5_lrequired_before_after.py`

**Status:** ✓ EXISTS (code written, awaiting pre-correction backup)

**What it does:**
Demonstrates the impact of the 2026-08-31 L_required correction (SHARE_PCM=1.0 → 0.5):
- **Pre-correction:** L_required ~608–641 kJ/kg → 0 survivors (all PCMs fail)
- **Post-correction:** L_required ~285–344 kJ/kg → 39 survivors (diverse viable candidates)

**Key Code:**
```python
survivors_pre = pd.read_csv("feasibility_survivors_rajasthan_precorrection.csv")  # If exists
survivors_post = pd.read_csv("feasibility_survivors_rajasthan_kappa_calibrated.csv")

profiles = pd.read_csv("cluster_profiles_rajasthan.csv")

# Read L_required (post-correction) from profiles
l_required_post = profiles.set_index("cluster_id")["L_required_kJ_per_kg"].to_dict()

# If pre-correction profiles exist, read those too
# (Otherwise use audit-documented values: pre was ~608–641 kJ/kg)

# Count survivors per cluster
survivor_counts_pre = survivors_pre.groupby("cluster_id").size().to_dict()
survivor_counts_post = survivors_post.groupby("cluster_id").size().to_dict()

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=["Cluster 0", "Cluster 1", "Cluster 2"],
    y=[survivor_counts_pre.get(i, 0) for i in range(3)],
    name="Pre-correction (SHARE_PCM=1.0)",
    marker_color="lightcoral"
))

fig.add_trace(go.Bar(
    x=["Cluster 0", "Cluster 1", "Cluster 2"],
    y=[survivor_counts_post.get(i, 0) for i in range(3)],
    name="Post-correction (SHARE_PCM=0.5)",
    marker_color="lightgreen"
))

fig.update_layout(
    title="Phase 5: L_required Before/After Correction (2026-08-31)",
    yaxis_title="Number of Surviving Candidates",
    barmode="group"
)
```

**Output:** `comparison_plots/phase5_lrequired_before_after/{bar_chart.html}`

**Graceful Handling:**
If pre-correction file doesn't exist (expected, since Phase 5 was only run once post-correction):
```python
if survivors_pre_file is None:
    print("[OK] Gracefully skipping this comparison.")
    print("(Pre-correction data not retained; this is expected if Phase 5 ran only once.)")
    exit(0)
```

---

### Comparison Plot 6: VIKOR Sign-Inversion Bugfix

**File:** `comparison_phase6_vikor_bugfix_before_after.py`

**Status:** ⏳ TODO (requires historical backup)

**What it will do (if pre-bugfix data exists):**
Show rank reordering caused by VIKOR sign-inversion fix. Requires backup of MCDM rankings before the fix was applied.

**Graceful Handling:**
If backup doesn't exist:
```python
if not os.path.exists(VIKOR_BUGFIX_BACKUP):
    print("[OK] No pre-bugfix backup found. Skipping this comparison.")
    exit(0)
```

---

### Comparison Plot 7: PCM vs. Plain Tank

**File:** `comparison_phase7_pcm_vs_plaintank.py`

**Status:** ⏳ TODO (requires plain-tank simulation)

**What it will do:**
Compare annual solar fraction (simulated) for:
1. Full PCM-integrated system (current)
2. Plain tank (latent_heat=0, sensible-only)

Shows value added by PCM integration.

**Expected Code:**
```python
physics_df = pd.read_csv("physics_validation_rajasthan.csv")

# If plain-tank column exists, use it
if "annual_solar_fraction_plain_tank" in physics_df.columns:
    df_pcm = physics_df[["cluster_id", "pcm_id", "annual_solar_fraction"]]
    df_plain = physics_df[["cluster_id", "pcm_id", "annual_solar_fraction_plain_tank"]]
    
    # Compare per PCM
    df_comp = df_pcm.copy()
    df_comp["solar_fraction_pcm"] = df_pcm["annual_solar_fraction"]
    df_comp["solar_fraction_plain"] = df_plain["annual_solar_fraction_plain_tank"]
    df_comp["pcm_advantage"] = df_comp["solar_fraction_pcm"] - df_comp["solar_fraction_plain"]
```

---

### Comparison Plot 8: Penalty k=0.0 vs. k=0.3

**File:** `comparison_phase8_penalty_k0_vs_k3.py`

**Status:** ⏳ TODO (requires supercooling sweep output)

**What it will do:**
Compares feasibility survivor counts under two supercooling penalty scenarios:
- **k=0.0:** No supercooling penalty (lenient)
- **k=0.3:** 30% penalty factor applied (strict)

Shows how penalty tuning affects screening stringency.

---

## Architecture & Verification Pattern

### Per-Script Pattern

Each plotting script follows this structure:

```python
"""
XX_plot_name.py
=============
Brief description of what is plotted.
What is verified/checked.
"""

import os, sys, pandas as pd, numpy as np, plotly.graph_objects as go

# Configuration
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../outputs/objective1_plots_rajasthan/XX_name"
DATA_FILE = os.path.join(DATA_DIR, "filename.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

print("\n=== DATA VERIFICATION ===")
# Verify schema, columns, data types
# Load supporting files if needed

print("\n=== VERIFICATION BLOCK ===")
# Compute key metrics against audit-documented values
# Print PASS/WARN/INFO status
# This is the core of the audit

# Plot creation
fig = go.Figure()  # or plt.figure() for matplotlib
# ... add traces, configure layout, annotations

# Save
output_file = os.path.join(OUTPUT_DIR, "plot_name.html")
fig.write_html(output_file)
print(f"\n✓ Saved to: {output_file}")
```

### Verification Blocks

**Every plot includes a VERIFICATION BLOCK** (printed to console) that checks:
- **Data schema:** Required columns present, correct data types
- **Fingerprint staleness:** (for Phases 5–9) via `provenance_lib.py`
- **Key metrics:** Survivor counts, Kendall's W, correlation coefficients, etc., against audit-documented baselines
- **Expected ranges/patterns:** E.g., "Cluster 0's Kendall's W should be lower than Clusters 1/2"

**Example output:**
```
=== VERIFICATION BLOCK ===
KS statistic for GHI (raw vs clean): 0.0034 (p=0.9102)
  ✓ PASS: GHI distributions remain nearly unchanged

KS statistic for T_amb (raw vs clean): 0.1245 (p=0.0001)
  ✓ PASS: T_amb shows visible tail-trimming after cleaning

Total survivors (post-correction): 39
  [OK] PASS: Matches post-correction baseline (39)
```

### Provenance Checking

All Phase 5–9 plots import and use `provenance_lib.py`:

```python
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

# Check if survivors file matches current cluster_profiles
current_fp = fingerprint_id(file_fingerprint(CLUSTER_PROFILE_FILE))
if survivors_df["upstream_cluster_profile_fingerprint"].iloc[0] != current_fp:
    print("⚠ WARNING: Data is STALE (built against older cluster_profiles)")
    # But DON'T BLOCK — watermark the plot and continue
```

Staleness is detected but not fatal — plots are watermarked instead.

## Output Structure

```
outputs/objective1_plots_rajasthan/
├── 01_raw_vs_preprocessed/
│   └── ghi_tamb_distributions.html
├── 02_climate_regime_map/
│   └── climate_regime_map_rajasthan.html (copied from Phase 4 output)
├── 03_feasibility/
│   ├── pcm_feasibility_scatter.png
│   ├── pcm_feasibility_scatter.html
│   ├── pcm_survivors_per_cluster.png
│   └── pcm_survivors_per_cluster.html
├── 04_mcdm_agreement/
│   ├── bump_chart_cluster_0.html
│   ├── bump_chart_cluster_1.html
│   ├── bump_chart_cluster_2.html
│   ├── method_correlation_heatmap_cluster_0.html
│   ├── method_correlation_heatmap_cluster_1.html
│   └── method_correlation_heatmap_cluster_2.html
├── 05_montecarlo/
│   ├── qc_montecarlo_inclusion_rajasthan.html (copied from Phase 6)
│   ├── rank_reversal_frequency_rajasthan.html
│   └── rank_reversal_frequency_summary.html
├── 06_physics_validation/
│   ├── mcdm_vs_physics_agreement_rajasthan.html
│   └── tank_profile_cluster_{0,1,2}_rajasthan.html (awaiting instrumentation)
├── 07_recommendation_summary/
│   └── summary_cards_rajasthan.png
├── comparison_plots/
│   ├── phase2_5_raw_vs_clean/
│   ├── phase3_tier1_vs_tier2/
│   ├── phase3_tmcap_old_vs_new/
│   ├── phase4_levelA_vs_levelB/
│   ├── phase5_lrequired_before_after/
│   ├── phase6_vikor_bugfix_before_after/
│   ├── phase7_pcm_vs_plaintank/
│   └── phase8_penalty_k0_vs_k3/
└── PLOTTING_REPORT.json (summary from run_all_plots.py)
```

## Running Individual Plots

### All at once:
```bash
python run_all_plots.py
```

Generates `PLOTTING_REPORT.json` with status of each script.

### Individual plots:
```bash
python 01_raw_vs_preprocessed.py
python 05_bump_chart.py
python 11_summary_cards.py
# ... etc
```

Each script is independent and can be re-run.

To run a specific comparison plot:
```bash
python comparison_phase3_tmcap_old_vs_new.py
python comparison_phase5_lrequired_before_after.py
```

## Architecture & Conventions

### Per-Script Pattern

Each plotting script follows this structure:

```python
"""
XX_plot_name.py
=============
Brief description.
What is verified.
"""

import os, sys, pandas as pd, plotly.graph_objects as go
from provenance_lib import file_fingerprint, fingerprint_id

# Configuration
DATA_DIR = "./data/processed"
OUTPUT_DIR = "./outputs/objective1_plots_rajasthan/XX_name"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading...")
df = pd.read_csv(FILE)

print("\n=== DATA VERIFICATION ===")
# Verify columns, check fingerprints

print("\n=== VERIFICATION BLOCK ===")
# Compute key metrics against audit-documented values
# Print PASS/WARN status
# This is the core of the audit — every plot is verified

# Plot creation
fig = go.Figure()  # or matplotlib for PNG
# ... add traces, configure layout

# Save
fig.write_html(OUTPUT_FILE)
print(f"✓ Saved to: {OUTPUT_FILE}")
```

### Verification Blocks

**Every plot includes a VERIFICATION BLOCK** (printed to console) that checks:
- Data schema correctness (column names, data types)
- Fingerprint staleness (via `provenance_lib.py`)
- Key metrics against audit-documented baselines
- Expected ranges/patterns (e.g., Cluster 0's Kendall's W should be lower)

Example output:
```
=== VERIFICATION BLOCK ===
KS statistic for GHI (raw vs clean): 0.0034 (p=0.9102)
  ✓ PASS: GHI distributions remain nearly unchanged
KS statistic for T_amb (raw vs clean): 0.1245 (p=0.0001)
  ✓ PASS: T_amb shows visible tail-trimming after cleaning
```

This is the **actual point of the audit**, not an afterthought.

### Provenance Checking

All Phase 5–9 plots import and use `provenance_lib.py`:

```python
from provenance_lib import file_fingerprint, fingerprint_id, assert_fingerprint_match

# Check if survivors file matches current cluster_profiles
current_fp = fingerprint_id(file_fingerprint(CLUSTER_PROFILE_FILE))
if survivors_df["upstream_cluster_profile_fingerprint"].iloc[0] != current_fp:
    print("⚠ WARNING: Data is STALE (built against older cluster_profiles)")
    # But DON'T BLOCK — watermark the plot and continue
```

Staleness is detected but not fatal — plots are watermarked instead.

## Output Structure

```
outputs/objective1_plots_rajasthan/
├── 01_raw_vs_preprocessed/
│   └── ghi_tamb_distributions.html
├── 02_climate_regime_map/
│   └── climate_regime_map_rajasthan.html (copied from Phase 4 output)
├── 03_feasibility/
│   ├── pcm_feasibility_scatter.png
│   ├── pcm_feasibility_scatter.html
│   ├── pcm_survivors_per_cluster.png
│   └── pcm_survivors_per_cluster.html
├── 04_mcdm_agreement/
│   ├── bump_chart_cluster_0.html
│   ├── bump_chart_cluster_1.html
│   ├── bump_chart_cluster_2.html
│   ├── method_correlation_heatmap_cluster_0.html
│   ├── method_correlation_heatmap_cluster_1.html
│   └── method_correlation_heatmap_cluster_2.html
├── 05_montecarlo/
│   ├── qc_montecarlo_inclusion_rajasthan.html (copied from Phase 6)
│   ├── rank_reversal_frequency_rajasthan.html
│   └── rank_reversal_frequency_summary.html
├── 06_physics_validation/
│   ├── mcdm_vs_physics_agreement_rajasthan.html
│   └── tank_profile_cluster_{0,1,2}_rajasthan.html (awaiting instrumentation)
├── 07_recommendation_summary/
│   └── summary_cards_rajasthan.png
├── comparison_plots/
│   ├── phase2_5_raw_vs_clean/
│   ├── phase3_tier1_vs_tier2/
│   ├── phase3_tmcap_old_vs_new/
│   ├── phase4_levelA_vs_levelB/
│   ├── phase5_lrequired_before_after/
│   ├── phase6_vikor_bugfix_before_after/
│   ├── phase7_pcm_vs_plaintank/
│   └── phase8_penalty_k0_vs_k3/
└── PLOTTING_REPORT.json (summary from run_all_plots.py)
```

## Running the Plots

### All at once:
```bash
python run_all_plots.py
```

Generates `PLOTTING_REPORT.json` with status of each script.

### Individual plots:
```bash
python 01_raw_vs_preprocessed.py
python 05_bump_chart.py
# ... etc
```

Each script is independent and can be re-run.

---

## Known Issues & TODOs

### Plot 10 (Tank Profile) — BLOCKED

**Status:** Awaiting instrumentation of `physics_lib.py`

**Prerequisite Changes:**

The current `physics_lib.py` only returns aggregate metrics (annual solar fraction, efficiency, etc.). To enable Plot 10, it must be modified to return hourly time-series arrays:

```python
def simulate_one_year(..., save_timeseries=False):
    """
    Simulate one year of PCM-SWH operation.
    
    Parameters:
        ...
        save_timeseries (bool): If True, return hourly arrays instead of just aggregates
    
    Returns:
        dict with keys:
            "annual_solar_fraction": float (aggregate)
            "annual_efficiency": float (aggregate)
            ...
            # If save_timeseries=True, also include:
            "hourly_Tw": np.array of shape (8760,)  # Tank water temp [°C]
            "hourly_Tp": np.array of shape (8760,)  # PCM capsule temp [°C]
            "hourly_melt_frac": np.array of shape (8760,)  # PCM melt fraction [0-1]
    """
    ...
    if save_timeseries:
        return {
            ...,
            "hourly_Tw": Tw_array,
            "hourly_Tp": Tp_array,
            "hourly_melt_frac": melt_array
        }
    else:
        return {...}  # Just aggregates
```

**Once Implemented:**
Plot 10 can be generated straightforwardly:
```python
# For each cluster representative location
for cluster_id in clusters:
    physics_output = simulate_one_year(..., save_timeseries=True)
    
    # Extract arrays
    Tw = physics_output["hourly_Tw"]  # 8760 hours
    Tp = physics_output["hourly_Tp"]
    melt = physics_output["hourly_melt_frac"]
    
    # Plot as time-series
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Tank Temp (Tw)", "PCM Temp (Tp)", "Melt Fraction"))
    
    fig.add_trace(go.Scatter(y=Tw, name="Tw", ...), row=1, col=1)
    fig.add_trace(go.Scatter(y=Tp, name="Tp", ...), row=2, col=1)
    fig.add_trace(go.Scatter(y=melt, name="Melt Frac", ...), row=3, col=1)
    
    fig.write_html(f"tank_profile_cluster_{cluster_id}.html")
```

**Status:** Ready to implement; no blockers other than `physics_lib.py` changes.

---

### Comparison Plots (Part B) — Implementation Pipeline

**Status:** 8 comparison scripts remain to be implemented.

| Script | Status | Difficulty | Notes |
|--------|--------|------------|-------|
| `comparison_phase2_5_raw_vs_clean.py` | ⏳ TODO | Low | Generalize Plot 1; straightforward histogram expansion |
| `comparison_phase3_tier1_vs_tier2.py` | ⏳ TODO | Low | Compare two stratification methods |
| `comparison_phase3_tmcap_old_vs_new.py` | ✓ CODE READY | Low | Awaiting deployment; scatter plot with reference line |
| `comparison_phase4_levelA_vs_levelB.py` | ⏳ TODO | Medium | Track seasonal cluster reassignments; requires careful data joining |
| `comparison_phase5_lrequired_before_after.py` | ✓ CODE READY | Low | Awaiting pre-correction backup; includes graceful fallback |
| `comparison_phase6_vikor_bugfix_before_after.py` | ⏳ TODO | Medium | Requires pre-bugfix MCDM rankings backup; includes graceful skip if missing |
| `comparison_phase7_pcm_vs_plaintank.py` | ⏳ TODO | Medium | Requires plain-tank simulation or existing column in physics output |
| `comparison_phase8_penalty_k0_vs_k3.py` | ⏳ TODO | Low | Uses Phase 8 supercooling sweep CSV; bar chart comparison |

**Recommended Implementation Order:**
1. Low-difficulty plots first (easier to validate): phase2_5, phase3_tier1, phase3_tmcap, phase5, phase8
2. Medium-difficulty: phase4, phase6, phase7 (need data-joining or additional runs)

**All are independent:** Can be built incrementally without blocking Plot 1–11.

---

### HTML vs. PNG Output

**Current state:**
- HTML outputs (via `plotly`) are interactive: zoom, pan, hover, toggle series on/off
- PNG outputs (via `kaleido`) are static but publication-ready for thesis/presentations

**For consolidated export:**
- To embed multiple plots in a thesis: collect PNG outputs and include as figures
- To create a dashboard: use HTML outputs via Jupyter notebook or web server
- To create PowerPoint slides: can use `python-pptx` to programmatically assemble PNGs into slides

**Kaleido dependency:**
```bash
pip install kaleido
```
If not installed, PNG export fails gracefully with a warning; HTML still works.

---

### Staleness & Fingerprinting

**How it works:**

All Phase 5–9 data files carry a `upstream_cluster_profile_fingerprint` column that records the MD5 hash of the cluster_profiles used to generate them. Before plotting:

```python
from provenance_lib import file_fingerprint, fingerprint_id

current_fp = fingerprint_id(file_fingerprint("cluster_profiles_rajasthan.csv"))
stamped_fp = survivors_df["upstream_cluster_profile_fingerprint"].iloc[0]

if current_fp != stamped_fp:
    print(f"⚠ WARNING: Data is STALE")
    # Add watermark to plot, but DON'T BLOCK
```

**Graceful handling:**
- Plot generation continues even if data is stale
- A subtle watermark or title annotation flags the staleness for manual review
- User must manually re-run Phase 5–9 if they want fresh data

**Re-running phases:**
```bash
# In era5-rajasthan/
python phase5_feasibility_screening.py  # Updates survivors files
python phase6_mcdm_ranking.py  # Updates MCDM rankings
python phase7_physics_validation.py  # Simulates performance
python phase8_montecarlo_uncertainty.py  # Computes robustness
python phase9_recommendations.py  # Final ranking

# Then:
python plotting/run_all_plots.py  # Plots will auto-refresh against new data
```

---

## Dependencies

### Core Plotting Libraries

```
pandas >= 1.0        # Data manipulation
numpy >= 1.18        # Numerical arrays
scipy >= 1.5         # Statistical functions (KS test, Spearman correlation)
plotly >= 4.0        # Interactive HTML plots (Scatter, Heatmap, Bar, etc.)
matplotlib >= 3.3    # Static plots (PNG via savefig)
seaborn >= 0.11      # Statistical plotting (color palettes, etc.)
```

### Optional

```
kaleido >= 0.2       # PNG export of Plotly figures (optional but recommended)
```

### Installation

```bash
pip install pandas numpy scipy plotly matplotlib seaborn

# Optional: for PNG export
pip install kaleido
```

### Verification (Check that imports work)

```bash
python -c "import pandas, numpy, scipy, plotly, matplotlib, seaborn; print('OK')"
```

---

## Quick Reference: Plot Summary Table

| # | Plot | Type | Key Metric | What it Validates |
|---|------|------|------------|-------------------|
| 1 | Raw vs. Preprocessed | Histogram | KS stat (GHI <0.05, T_amb >0.1) | Data cleaning didn't over-filter |
| 2 | Climate Regime Map | Folium map | k=3, canonical order [0,1,2] | Geographic cluster separation |
| 3 | PCM Feasibility | Scatter | Survivors: 39 total (post-correction) | Melting point & latent heat suitability |
| 4 | Survivors per Cluster | Bar chart | Primary vs. calibrated, κ values | Kappa calibration effectiveness |
| 5 | Bump Chart | Slopegraph | Spearman rho(VIKOR,TOPSIS) >0.3 | MCDM method agreement (no sign-inversion bug) |
| 6 | Correlation Heatmap | Heatmap | GRA has lowest mean correlation | Structural method outlier |
| 7 | MC Inclusion Probability | Histogram | Top-1 inclusion >80% | Top recommendation stability |
| 8 | Rank-Reversal Frequency | Bar/violin | Cluster 0 freq > Clusters 1/2 | Relationship between Kendall's W and instability |
| 9 | MCDM vs. Physics | Scatter+trend | Spearman rho per cluster | MCDM methodology validity vs. simulation |
| 10 | Tank Profile | Time-series | Hourly Tw, Tp, melt | PCM charging/discharging dynamics (BLOCKED) |
| 11 | Summary Cards | 3-panel figure | Top-1 name, Tm, L, MC%, rho | Final recommendations (publication-ready) |
| — | Plot 7 Consolidation | HTML (copy) | MC inclusion + imputation | Augmented robustness analysis |

---

## Audit Reference

**Full specification:** `../docs/rajasthan/11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`

**Key sections:**
- **§2:** Per-plot detail, verification approach, expected outputs
- **§4:** Output directory structure
- **§5:** Prompt to Claude Code (implementation brief for this repo)
- **§6:** Detailed requirements per plot (data sources, column names, expected ranges)
- **§7:** Fingerprinting strategy & staleness detection

**See also:**
- `CLAUDE.md` (project instructions, PCM design-basis decision)
- `memory/` (persistent project notes)
- `sources/` (extracted research papers, referenced in methodology)

---

## FAQ

### Q: How do I run just Plot 1?
**A:**
```bash
python 01_raw_vs_preprocessed.py
```
Output goes to `../outputs/objective1_plots_rajasthan/01_raw_vs_preprocessed/ghi_tamb_distributions.html`

### Q: How do I run all plots?
**A:**
```bash
python run_all_plots.py
```
Generates all outputs and a summary `PLOTTING_REPORT.json`.

### Q: Can I run Plots 3–9 if Phases 5–9 data is stale?
**A:**
Yes. Plots will still generate but will include a staleness watermark. To refresh, re-run the Phase 5–9 scripts first, then re-run the plots.

### Q: What if a comparison plot's required file doesn't exist?
**A:**
Most comparison plots include graceful fallback logic. If the needed backup/archive file is missing, the script prints a message and exits cleanly (exit code 0) without erroring.

### Q: How do I add a new plot?
**A:**
1. Create `ZZ_plot_name.py` following the per-script pattern (docstring + DATA_VERIFICATION + VERIFICATION BLOCK + plot code)
2. Add fingerprint checks (if Phase 5–9 data is used)
3. Update this README with a new row in the status table and a detailed section
4. Add the script name to `run_all_plots.py`

### Q: Can I export plots to PowerPoint?
**A:**
Yes. Collect PNG outputs and use `python-pptx`:
```python
from pptx import Presentation
from pptx.util import Inches

prs = Presentation()
for png_file in png_files:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    left = Inches(0.5)
    top = Inches(0.5)
    pic = slide.shapes.add_picture(png_file, left, top, width=Inches(9))

prs.save("plots_presentation.pptx")
```

---

## Contact / Issues

For questions on plot generation or verification:
1. Check the **VERIFICATION BLOCK** output when running a script (console output)
2. Refer to this README's detailed plot section
3. Consult the audit specification in `../docs/rajasthan/11_OBJECTIVE1_PLOTTING_AUDIT_AND_PROMPT.md`
4. Check the CLAUDE.md file for project context and methodology

**Staleness or fingerprint warnings?** Re-run Phases 5–9 before plotting.
