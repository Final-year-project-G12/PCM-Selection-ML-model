# 07 — Phase 5 Audit: Feasibility Filtering

**Scripts**: `06_build_pcm_database.py`, `07_feasibility_filter.py`

**Status**: COMPLETE

## PCM database build (`06_build_pcm_database.py`)

### Sources
- **Manufacturer base**: MICE+RF+PMM cleaned dataset (`PCM_Properties_cleaned_mice_pmm_detailed.csv`)
  — 18 rows (Rubitherm RT + PLUSS savE product lines)
- **Literature additions**: 7 rows from Singh2025 Table 2 (fatty acids, eutectics, bio-based
  paraffins in the 42–70°C band)
- **Total**: **25 rows** in `pcm_database_assam.csv`

### 25-row PCM database (Tm range relevant to Assam's 44°C target)

| Name | Family | Tm (°C) | Latent heat (kJ/kg) |
|---|---|---|---|
| savE® HS36 | PLUSS savE | 35.0 | 163 |
| savE® OM35 | PLUSS savE | 35.0 | 171 |
| RT35 | Rubitherm RT | 35.0 | 160 |
| savE® OM37 | PLUSS savE | 38.0 | 186 |
| RT38 | Rubitherm RT | 38.0 | 170 |
| savE® OM39 | PLUSS savE | 39.0 | 229 |
| RT42 | Rubitherm RT | 41.0 | 165 |
| Myristic-Palmitic eutectic (58/42) | Eutectic | 42.6 | 169.7 |
| **RT44HC** | Rubitherm RT | **43.0** | **250** |
| savE® OM42 | PLUSS savE | 44.0 | 199 |
| C22H46 (docosane-class paraffin) | Paraffin | 44.5 | 249 |
| RT47 | Rubitherm RT | 46.0 | 160 |
| savE® OM46 | PLUSS savE | 47.0 | 177 |
| **RT45HC** | Rubitherm RT | **47.0** | **230** |
| RT50 | Rubitherm RT | 49.0 | 160 |
| savE® OM50 | PLUSS savE | 50.0 | 189 |
| savE® OM48 | PLUSS savE | 51.0 | 165 |
| Palmitic-Stearic eutectic (64.2/35.8) | Eutectic | 52.3 | 181.7 |
| Myristic acid | Fatty acid | 53.0 | 190 |
| RT54HC | Rubitherm RT | 54.0 | 200 |
| RT55 | Rubitherm RT | 55.0 | 170 |
| Palmitic acid | Fatty acid | 63.0 | 185.4 |
| RT64HC | Rubitherm RT | 64.0 | 250 |
| Paraffin wax (generic) | Paraffin | 64.0 | 173.6 |
| C30H62 (triacontane-class paraffin) | Paraffin | 65.5 | 252 |

Database still at 25 rows vs. the 40–60-row target — same gap as Rajasthan.

## Feasibility filter (`07_feasibility_filter.py`)

### 7 constraints applied

| # | Constraint | Parameters |
|---|---|---|
| 1 | Melting window | Tm ∈ [Tm_target − 6, Tm_target + 8] = [38, 52]°C; auto-relaxed +2K/step if <5 survive |
| 2 | Absolute band | Tm ∈ [42, 70]°C |
| 3 | Latent heat floor | L ≥ 0.7 × L_required (κ=0.7); relaxed per-cluster if needed |
| 4 | Cycling stability | ≥ 300 cycles if known; retained with flag if unknown |
| 5 | Corrosion veto | Exclude inorganic PCMs in clusters where HSI > global p75 |
| 6 | Supercooling veto | Exclude supercooling > 8K (known values only) |
| 7 | Safety keyword veto | Exclude "highly flammable", "extremely flammable", "toxic" |

### Assam-specific: Corrosion veto is load-bearing

Assam's climate is characterised by high humidity and a dominant monsoon signal. The **Humidity-Solar
Interaction (HSI) index** exceeds the global p75 threshold in at least the humid valley clusters.
This triggers the corrosion veto (Constraint 5), which excludes inorganic PCMs (salt hydrates,
eutectic salts) from those clusters. This is a **real differentiation** between Assam and Rajasthan:
in Rajasthan's dry climate, the corrosion veto was present but did not activate for most clusters.

### Survivors per cluster (from `feasibility_survivors_assam.csv`)

| Cluster | n_candidates (after κ-relaxation) |
|---|---|
| 0 | **6** |
| 1 | **6** |
| 2 | **8** |
| 3 | **8** |

Clusters 0 and 1 (northern/valley regimes with higher HSI-driven corrosion veto) have fewer survivors
(6 each), while Clusters 2 and 3 (Barak/western plains with moderately lower HSI) have 8.

### Latent-heat floor finding

With Tm_target = 44°C uniform and L_required ≈ 232–249 kJ/kg per cluster, the nominal κ=0.7 floor
(L ≥ 0.7 × L_required ≈ 162–174 kJ/kg) is satisfiable by several candidates (RT44HC at 250,
C22H46 at 249, RT45HC at 230, savE® OM42 at 199, etc.). Unlike Rajasthan (where L_required was
~610–643 kJ/kg creating a structural zero-survivor problem), **Assam's L_required is in a
reachable range** — the feasibility filter produces real survivors without exhausting all relax steps.

### Known issues

1. **κ-relaxation policy not settled**: Auto-relaxation (+2K/step, up to 4 steps) is applied if
   <5 candidates survive. This is an ad hoc pass, not a documented permanent policy.

2. **PCM database still undersized**: 25 rows vs. 40–60-row target. The corrosion veto + latent-heat
   floor combination means the effective pool is small (6–8 per cluster). Phase 6 MCDM and Phase 7
   physics results are both provisional pending database expansion.

3. **Literature PCM property gaps**: The 7 Singh2025 additions (fatty acids, eutectics, paraffins)
   have incomplete property coverage — thermal conductivity, density, specific heat are often not
   reported in the source data. These are carried forward with NaN values and affect criterion
   contribution calculations in Phase 6.
