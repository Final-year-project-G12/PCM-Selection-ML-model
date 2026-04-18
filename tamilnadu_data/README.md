# Tamil Nadu Multi-City Climate — Interactive Folium Map

## Overview

This Jupyter notebook creates a comprehensive **interactive geospatial climate analysis dashboard** for Tamil Nadu, India. It combines multi-layer Folium maps with interactive Plotly charts to visualize hourly climate data across 8 major cities throughout 2024.

**Purpose:** Analyze and visualize spatial and temporal patterns of solar irradiance (GHI), temperature, humidity, wind speed, precipitation, and cloud cover across Tamil Nadu.

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | ERA5 Climate Data (Copernicus Climate Data Store) |
| **File** | `data/era5_climate_tamilnadu_2024.csv` |
| **Time Period** | January 1 – December 31, 2024 |
| **Temporal Resolution** | Hourly (8,760 records per location) |
| **Spatial Coverage** | 8 major cities across Tamil Nadu, ~19 grid points |
| **Total Records** | ~140,000+ observations |

### Cities Covered
1. **Chennai** (#e74c3c - Red)
2. **Coimbatore** (#f39c12 - Orange)
3. **Madurai** (#2ecc71 - Green)
4. **Tiruchirappalli** (#3498db - Blue)
5. **Vellore** (#9b59b6 - Purple)
6. **Tirunelveli** (#1abc9c - Cyan)
7. **Thanjavur** (#e67e22 - Dark Orange)
8. **Cuddalore** (#e91e8c - Magenta)

### Climate Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| **GHI** | W/m² | Global Horizontal Irradiance (solar radiation) |
| **T_ambient** | °C | Ambient air temperature |
| **Wind_speed** | m/s | Wind speed at 10m height |
| **Precip** | mm | Precipitation accumulation |
| **RH** | % | Relative humidity |
| **Cloud_cover** | 0-1 | Cloud cover fraction |

---

## Notebook Structure

### **Cell 1: Install & Import Dependencies**
- Installs: `folium`, `branca`, `plotly`, `pandas`, `numpy`
- Imports visualization and data processing libraries
- Suppresses warnings for cleaner output
- **Output:** Library load confirmation

### **Cell 2: Load Data & Map Cities**
- Loads CSV file from `data/era5_climate_tamilnadu_2024.csv`
- Maps latitude/longitude coordinates to city names using `CITY_MAP` dictionary
- Extracts time components: `hour`, `month`, `DOY` (day of year), `date`
- Computes season labels: Winter, Pre-Monsoon, Monsoon, Post-Monsoon
- **Output:** Row counts, city list, date range, city-level summary statistics table

### **Cell 3: Pre-compute Aggregates & Helpers**
Prepares all aggregated datasets used by downstream visualizations:

| Aggregation | Purpose |
|-------------|---------|
| **city_stats** | Annual stats per city (mean, max values) |
| **grid_stats** | Point-level annual stats for heatmaps |
| **monthly** | Monthly aggregates by city |
| **hourly_pattern** | 24-hour average patterns by city |
| **seasonal** | Seasonal aggregates by city |
| **Normalization dict (G)** | Min/max ranges for color mapping |

**Key Functions:**
- `norm_series()` — Normalizes values to [0, 1] range for consistent color mapping
- `city_popup()` — Generates rich HTML tooltips for city markers (250px wide, 10 metrics)
- `grid_popup()` — Generates HTML tooltips for grid point markers (220px wide)

- **Output:** Confirmation message + counts of aggregates

### **Cell 4: Build the Multi-Layer Folium Map**
Creates the **primary interactive map** with 9 feature layers:

#### **Base Configuration**
- **Center:** Tamil Nadu (11.0°N, 78.5°E)
- **Zoom:** Level 7 (state-level view)
- **Canvas Mode:** `prefer_canvas=True` for performance

#### **Tile Layers** (switchable via layer control)
1. **CartoDB Dark Matter** (default) — Dark theme for visibility
2. **CartoDB Positron** — Light base map
3. **OpenStreetMap** — Standard OSM layer
4. **Satellite** — Esri satellite imagery

#### **Plugins** (built-in controls)
- **Fullscreen** — Toggle fullscreen mode (top-left)
- **MiniMap** — Overview map in corner
- **MousePosition** — Lat/lon coordinates display

#### **Feature Layers** (toggle in layer control)

| Layer | Description | Color Gradient | Default |
|-------|-------------|----------------|---------|
| **[GHI] Heatmap** | Global solar radiation | Navy → Blue → Orange → Red → White | ✓ ON |
| **[TEMP] Temperature** | Ambient temperature | Blue → Green → Yellow → Orange → Red | OFF |
| **[HUMIDITY] Humidity** | Relative humidity | Light → Blue → Dark Green | OFF |
| **[WIND] Wind Speed** | Wind velocity | Cyan → Blue → Teal | OFF |
| **[PRECIP] Precipitation** | Only rainy hours | Light Blue → Blue → Dark Blue | OFF |
| **[CLOUD] Cloud Cover** | Cloud fraction | Dark → Gray → Light | OFF |
| **[CITY] City Markers** | 8 city dots with abbreviations | City-specific colors | ✓ ON |
| **[GRID] Grid Points** | Individual measurement locations | GHI-based gradient | OFF |
| **[PEAK] Peak GHI** | Top 1% solar irradiance events | Orange/red stars | OFF |

#### **Heatmap Parameters**
- **Sampling:** Up to 15,000 points per layer (performance optimization)
- **Radius:** 18–24 pixels (blur radius varies by parameter)
- **Blur:** 15–20 pixels (edge smoothing)
- **Min Opacity:** 0.3–0.45 (transparency baseline)

- **Output:** Interactive HTML map rendered in notebook, print summary of rows and thresholds

### **Cell 5A: Temperature Heatmap by City**
**Standalone Map:** Focused visualization of temperature with city markers.
- **Heatmap:** Temperature gradient (blue cold → red hot)
- **Markers:** City centroids sized/colored by mean temperature
- **Interactive:** Popups show detailed stats per city
- **Output:** Rendered Folium map

### **Cell 5B: Humidity & Wind Rose Map**
**Standalone Map:** Relative humidity heatmap + wind vector representation.
- **Heatmap:** Humidity distribution (light → green → dark)
- **Wind Markers:** Circle radius ∝ wind speed per city
- **Colors:** City-specific (inherited from CITY_COLORS)
- **Output:** Rendered Folium map

### **Cell 5C: Precipitation & Cloud Cover Map**
**Standalone Map:** Cloud cover heatmap + precipitation at cities.
- **Heatmap:** Cloud coverage (dark gray → light)
- **Precip Circles:** Radius ∝ annual precipitation per city
- **Popups:** Show precipitation % and cloud mean
- **Output:** Rendered Folium map

### **Cell 6: Plotly Interactive Charts — City Comparisons**

#### **Chart 1: Radar Chart (Normalized City Profiles)**
- **Axes:** GHI, Temperature, Wind, RH, Precipitation (each 0–100)
- **Series:** One per city (8 overlapping radar charts)
- **Use:** Quick visual comparison of climate "fingerprints"
- **Interaction:** Zoom, pan, legend toggle

#### **Chart 2: Box Plot (Monthly GHI Distribution)**
- **X-axis:** Month (1–12)
- **Y-axis:** GHI (W/m²) — full distribution
- **Groups:** 8 cities color-coded
- **Use:** Identify variability and outliers by month
- **Interaction:** Hover for quartile details

#### **Chart 3: Line Chart (Daily Temperature Trends)**
- **X-axis:** Date (Jan 1 – Dec 31)
- **Y-axis:** Daily average temperature (°C)
- **Series:** One per city
- **Use:** Seasonal temperature patterns
- **Interaction:** Click legend to toggle cities

- **Output:** 3 interactive Plotly charts displayed inline

### **Cell 7: Plotly Interactive Charts — Hourly & Seasonal Patterns**

#### **Chart 4: Area Chart (Hourly GHI Pattern)**
- **X-axis:** Hour (0–23)
- **Y-axis:** Mean GHI (W/m²)
- **Series:** Stacked areas by city
- **Use:** Diurnal solar cycle per city
- **Pattern:** Sunrise ~6am, peak ~12–1pm, sunset ~6pm

#### **Chart 5: Grouped Bar Chart (Seasonal GHI)**
- **X-axis:** City
- **Y-axis:** GHI mean
- **Groups:** 4 seasons (Winter, Pre-Monsoon, Monsoon, Post-Monsoon)
- **Use:** Seasonal variation pattern
- **Colors:** Blue (Winter) → Orange (Pre-M) → Green (Monsoon) → Purple (Post-M)

#### **Chart 6: Heatmap (Monthly Precipitation)**
- **Rows:** Month (Jan–Dec)
- **Cols:** City
- **Color:** Intensity = precipitation (mm)
- **Use:** Identify monsoon periods by city
- **Interaction:** Hover for exact values

#### **Chart 7: Violin Plot (Wind Speed Distribution)**
- **X-axis:** City
- **Y-axis:** Wind speed (m/s)
- **Distribution:** Full shape + box plot overlay
- **Use:** Wind variability and extremes per city
- **Outliers:** Individual points shown as necessary

- **Output:** 4 interactive Plotly charts displayed inline

### **Cell 8: Summary Statistics & Export**

Prints formatted tables and exports data:

#### **Printed Output:**
1. **Annual City-Level Summary Table** (10 metrics per city)
   - GHI avg/max, T avg/min/max, Wind avg, RH avg, Precip total, Cloud avg

2. **Top 3 GHI Months** (statewide) with rankings

3. **Seasonal Precipitation** (mm per season, ranked)

4. **City Rankings** across 4 dimensions:
   - GHI rank (1–8)
   - Temperature rank (1–8)
   - Precipitation rank (1–8)
   - Wind rank (1–8)

#### **CSV Exports:**
- `tamil_nadu_city_stats.csv` — Annual city aggregates (10 columns, 8 rows)
- `tamil_nadu_monthly_data.csv` — Monthly city data (96 rows: 8 cities × 12 months)

### **Cell 9: Stats Panel Injection**
Adds a **fixed HTML overlay** to the main map (bottom-left corner):
- Semi-transparent dark background (rgba(15,19,30,0.93))
- Shows state-level summary:
  - City & grid point counts
  - Mean/max GHI, T_ambient, wind, RH, precipitation, cloud cover
- Non-interactive (pointer-events: none)

- **Output:** Map with overlay injected and displayed

### **Cell 10: Save & Display HTML**
- Saves interactive map as standalone **`tamil_nadu_climate_map.html`**
- Embeds all layers, styles, and data
- Displays via `IFrame` (650px height, 100% width)
- File size: ~3–5 MB (can be shared, opened in any browser)

- **Output:** Saved file path confirmation + embedded view

### **Cell 11: Download (Colab Support)**
- Attempts to trigger download on Google Colab
- Falls back to local file message if not in Colab
- **Output:** Status message

### **Cell 12: Matplotlib Monthly Charts**
Creates a **3×2 subplot grid** (dark-themed static image):

| Subplot | Parameter | Y-axis |
|---------|-----------|--------|
| (1,1) | GHI | W/m² |
| (1,2) | Temperature | °C |
| (2,1) | Wind Speed | m/s |
| (2,2) | Precipitation | mm |
| (3,1) | Relative Humidity | % |
| (3,2) | Cloud Cover | Fraction |

**Features:**
- 8 city lines + shaded fill areas
- Dark theme (#0f1419 background, #1a1f2e plot area)
- Grid lines, legend, month labels
- Saved as PNG: `tamil_nadu_monthly_params.png`

- **Output:** High-resolution chart (150 dpi, 16×14 inches)

---

## Features & Capabilities

### **Interactive Geospatial Analysis**
✓ Multi-layer heatmaps with dynamic sampling (15,000 pts max)
✓ City marker popups (7 metrics with dark HTML styling)
✓ Grid point markers with GHI colormapping
✓ Peak GHI event clusters (top 1%, up to 800 sampled)
✓ Tile layer switching (4 basemap options)
✓ Layer control with checkboxes
✓ Fullscreen, minimap, coordinates plugins

### **Statistical Visualizations**
✓ 7 interactive Plotly charts (radar, box, line, area, bar, heatmap, violin)
✓ City-level annual summary (12 metrics)
✓ Seasonal and monthly aggregates
✓ Hourly diurnal patterns (24-hour avg)
✓ Rankings across 4 climate dimensions

### **Data Export**
✓ Interactive HTML map (standalone, shareable)
✓ 2 CSV exports (city stats, monthly series)
✓ PNG monthly trends (16×14", 150 dpi)

### **Styling**
✓ Dark theme throughout (#0f1419 background, #1a1f2e plots)
✓ City-specific color palette (8 distinct hex codes)
✓ Consistent font (Arial, sans-serif)
✓ Professional gradient colormaps

---

## Requirements & Dependencies

### **Python Packages**
```
folium>=0.14.0       # Geospatial mapping
branca>=0.5.0        # Color management
plotly>=5.0.0        # Interactive charts
pandas>=1.3.0        # Data manipulation
numpy>=1.20.0        # Numerical computing
matplotlib>=3.3.0    # Static visualizations
```

### **Data File**
- **Path:** `data/era5_climate_tamilnadu_2024.csv`
- **Format:** CSV with headers: `timestamp`, `latitude`, `longitude`, `GHI_Wm2`, `T_ambient_C`, `Wind_speed_ms`, `Precip_mm`, `RH_percent`, `Cloud_cover_fraction`
- **Size:** ~10–15 MB (140,000+ rows)

### **Hardware**
- **Memory:** 2–4 GB RAM recommended (heatmap sampling reduces load)
- **Runtime:** ~2–5 minutes to execute all cells (depends on hardware)
- **Browser:** Any modern browser (Chrome, Firefox, Safari, Edge) for HTML output

---

## How to Run

### **Step 1: Install Dependencies**
```bash
pip install folium branca plotly pandas numpy matplotlib
```

### **Step 2: Prepare Data**
- Place `era5_climate_tamilnadu_2024.csv` in the `data/` subdirectory
- Ensure CSV has required columns (see above)

### **Step 3: Run Notebook**
```bash
jupyter notebook tamil_nadu_climate_map.ipynb
```

### **Step 4: Execute Cells Sequentially**
- **Cells 1–3:** ~10 seconds (setup & aggregation)
- **Cell 4:** ~30–60 seconds (main map, 9 layers)
- **Cells 5A–5C:** ~15 seconds each (standalone maps)
- **Cells 6–7:** ~5 seconds each (Plotly charts)
- **Cells 8–12:** ~30 seconds (stats, exports, matplotlib)

**Total Runtime:** ~3–5 minutes

### **Step 5: Explore Outputs**
- **Interactive maps:** Visible in notebook; can be toggled to fullscreen
- **CSV files:** Available in working directory
- **HTML file:** `tamil_nadu_climate_map.html` (standalone, shareable)
- **PNG file:** `tamil_nadu_monthly_params.png` (static chart)

---

## Configuration & Customization

### **Key Parameters**

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| `CSV_PATH` | Cell 2 | `data/era5_climate_tamilnadu_2024.csv` | Data file path |
| `MAX_HEAT` | Cell 4 | 15000 | Max pts per heatmap (performance) |
| `CITY_COLORS` | Cell 2 | 8 hex codes | City marker colors |
| `CITY_ABBR` | Cell 2 | 3-letter codes | City abbreviations on map |
| `zoom_start` | Cell 4 | 7 | Initial map zoom level |

### **Customization Tips**

1. **Change Data Source:**
   - Modify `CSV_PATH` to point to different ERA5 export
   - Ensure column names match expected format

2. **Adjust Heatmap Resolution:**
   - Increase `MAX_HEAT` for more detail (slower rendering)
   - Decrease for faster maps on slow machines

3. **Modify City Colors:**
   - Edit `CITY_COLORS` dict to use custom hex codes
   - Colors propagate to all visualizations

4. **Change Map Center/Zoom:**
   - Edit `location=[11.0, 78.5]` and `zoom_start=7` in Cell 4
   - Test different coordinates for different regions

5. **Adjust Heatmap Gradients:**
   - Modify gradient dicts (key = normalized 0–1, value = color hex)
   - Example: `gradient={0.0:'#blue', 0.5:'#green', 1.0:'#red'}`

---

## Output Files

| File | Type | Size | Purpose |
|------|------|------|---------|
| `tamil_nadu_climate_map.html` | HTML | 3–5 MB | Standalone interactive map |
| `tamil_nadu_city_stats.csv` | CSV | <10 KB | City-level annual summary |
| `tamil_nadu_monthly_data.csv` | CSV | <20 KB | Monthly city-level series |
| `tamil_nadu_monthly_params.png` | PNG | 200–300 KB | Static monthly trends chart |

---

## Interpretation Guide

### **GHI Heatmap (Cell 4, Layer 1)**
- **High values (white/red):** Strong solar resource zones
- **Peak GHI:** Typically 600–800 W/m² around noon
- **Daily pattern:** Zero at night, ramps up from ~6am, peaks ~1pm, ramps down by ~6pm
- **Seasonal:** Higher in summer (Apr–Jun), lower in monsoon (Jul–Sep)

### **Temperature Distribution (Cell 5A)**
- **Hottest:** Usually inland cities (Madurai, Tiruchirappalli) → 28–32°C annual
- **Coolest:** Elevated regions (Coimbatore, Vellore) → 24–28°C annual
- **Monsoon:** Slight cooling in Jun–Sep due to cloud cover

### **Wind Patterns (Cell 5B)**
- **Wind vectors (circles):** Size = mean wind speed
- **Higher wind:** Coastal cities (Chennai, Cuddalore) → 3–5 m/s
- **Lower wind:** Inland → 2–3 m/s
- **Seasonal surge:** Post-monsoon (Oct–Nov) shows elevated wind

### **Precipitation Hotspots (Cell 5C)**
- **Monsoon cities:** Tamil Nadu receives Oct–Dec northeast monsoon
- **Higher precip:** West-facing slopes near the coast → 800–1200 mm
- **Lower precip:** Inland plains → 400–600 mm
- **Rainy hours:** Only ~10–15% of 8,760 hourly records show precipitation

### **Peak GHI Events (Cell 4, Layer 9)**
- **Top 1% GHI:** Identified as clear-sky, high-noon events
- **Temporal:** Concentrated Apr–Oct (summer half-year)
- **Spatial:** Highest density inland (less cloud, high sun angle)
- **Cluster analysis:** Grouped by city color; star markers indicate locations

---

## Technical Notes

### **Performance Optimizations**
1. **Heatmap Sampling:** Limits 15,000 random points per layer to prevent browser lag
2. **Canvas Rendering:** `prefer_canvas=True` uses WebGL for faster rendering
3. **Feature Groups:** Layers grouped for efficient toggle (only visible layers render)
4. **Data Sampling:** Peak events limited to 800 samples (top 1% is ~1,400 events)

### **Color Mapping Strategy**
- **Normalization:** All parameters normalized to [0, 1] using global min/max
- **Consistent gradients:** Colors scale uniformly across heatmaps
- **Dark theme:** Dark background (#0f1419) reduces eye strain during exploration

### **Browser Compatibility**
- **Tested on:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Requires:** WebGL support, JavaScript enabled
- **File size:** ~5 MB (may be slow on slow internet)

### **Data Quality Assumptions**
- **No missing values imputed** (assumes clean ERA5 data)
- **Coordinate rounding:** Lat/lon rounded to 2 decimals (~1 km precision)
- **Time zones:** All times in India Standard Time (IST/UTC+5:30)

---

## Limitations & Future Enhancements

### **Current Limitations**
- Only 2024 data (single year; limited climate trend analysis)
- 8 cities sampled (not fine-grained county-level)
- Static city coordinates (no sub-city variation)
- No missing data handling (assumes perfect ERA5 extraction)

### **Potential Enhancements**
- Multi-year data for trend analysis
- Kriging interpolation for smooth spatial surfaces
- Wind speed + direction vectors (rosette plots)
- Machine learning clustering of micro-climates
- Real-time data integration (if ERA5 updated regularly)
- PDF export for reports
- Custom date range filtering UI

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Map not rendering** | Check `CSV_PATH` exists; run Cell 2 first |
| **Heatmap too sparse** | Increase `MAX_HEAT`; reduce to fewer cities |
| **Out of memory** | Reduce `MAX_HEAT` to 5000; use fewer heatmap layers |
| **Plotly charts blank** | Ensure `plotly>=5.0.0`; check internet connection |
| **Popup text tiny** | Zoom in; adjust font-size in popup HTML |
| **HTML file> 10 MB** | Remove some heatmap layers; resample data |

---

## Contact & Attribution

- **Data Source:** Copernicus Climate Data Store (ERA5 Dataset)
- **Visualization Libraries:** Folium, Plotly, Matplotlib, Branca
- **Region:** Tamil Nadu, India
- **Year:** 2024 (full calendar year)

---

**Last Updated:** March 2026
**Version:** 1.0
