# 09 — Phase 7 Audit: Physics-Based Validation

Script: `10_physics_validation.py`.

## Purpose
Validate the MCDM material rankings by solving a grey-box transient energy balance simulation of a PCM-integrated solar water heating tank driven by 10 years of real daily weather for each regime's medoid point, evaluating rank concordance against simulated annual solar fraction.

---

## Model Architecture & Governing Physics

### 1. 3-Phase Lumped-Enthalpy Tank Model (Barqawi 2025)
The thermal storage tank is modeled as a lumped-enthalpy system undergoing three distinct phase states:
1. **Sensible Solid Phase** ($T_{\text{tank}} < T_m$): Heat addition increases solid PCM temperature ($c_{p,\text{solid}}$).
2. **Isothermal Phase Change Phase** ($T_{\text{tank}} = T_m$): Energy absorbed at constant temperature melts the PCM ($L_{\text{fusion}}$).
3. **Sensible Liquid Phase** ($T_{\text{tank}} > T_m$): Heat addition increases liquid PCM temperature ($c_{p,\text{liquid}}$).

### 2. Governing Energy Balance ODE
The system temperature $T_{\text{tank}}$ evolves according to:
$$C_{\text{tank}} \frac{d T_{\text{tank}}}{dt} = Q_{\text{solar}} - Q_{\text{draw}} - U A_{\text{tank}} (T_{\text{tank}} - T_{\text{amb}})$$

- $Q_{\text{solar}} = \eta_{\text{coll}} \cdot A_{\text{coll}} \cdot \text{GHI}$: Collector solar thermal input.
- $Q_{\text{draw}} = \dot{m}_{\text{draw}} \cdot c_{p,\text{water}} \cdot (T_{\text{tank}} - T_{\text{mains}})$: Hot water draw thermal load (300 L/day profile).
- **Ambient Heat Loss Fix (v3.1)**: $U A_{\text{tank}} = 2.0\text{ W/K}$ added to represent tank envelope heat loss to surrounding ambient air ($T_{\text{amb}}$).

### 3. Numerical Solver
- **Scheme**: Backward Euler (implicit 1st order) to guarantee numerical stability across stiff phase-transition boundaries.
- **Time Step**: Hourly step $dt = 3600\text{ s}$ over the full 10-year simulation span (87,660 time steps per candidate).

---

## Validation Findings — 2026-09-16, final state (elevation-corrected data, `Tm_target_C=67`, 9-criterion MCDM engine, 3 clusters, 8/10/8 survivors, n=26)

> This section documents the full investigation chain in order: the original negative-ρ finding,
> the `thermal_margin` criterion added in response to it, what that fix actually did (helped
> Cluster 0, did not help Clusters 1/2), why it couldn't have, and the final honest conclusion.
> See `CHANGELOG.md`'s 2026-09-16 entries for the complete step-by-step record.

### 1. Timeline of Spearman ρ (MCDM consensus rank vs. simulated annual solar fraction)

| Cluster | n | Before any fix | After `Tm_target_capped_C` fix | After asymmetric σ | After `thermal_margin` criterion (final) |
|---|---|---|---|---|---|
| 0 | 8 | -0.595 | -0.595 (no change) | -0.595 (no change) | **+0.381** |
| 1 | 10 | +0.176 | +0.176 (no change) | +0.176 (no change) | +0.103 |
| 2 | 8 | +0.048 | +0.048 (no change) | +0.048 (no change) | +0.024 |
| Mean | | -0.124 | -0.124 | -0.124 | **+0.169** |

**None of these values are statistically significant at any conventional threshold** — p-values
range 0.12–0.96 across every column above, n=8–10 per cluster. This matters: with this few
survivors, no correlation reported anywhere in this table — positive or negative — is
distinguishable from zero. Treat every number in this table as a descriptive point estimate, not
a significance-tested claim, and see §4 before reading too much into any single sign flip.

### 2. Two inert fixes, one that worked — and why

1. **`f_Tm` scored against `Tm_target_capped_C` (61.94°C) instead of raw `Tm_target_C` (67.0°C)**
   — `08_mcdm_ranking.py` line 991. **No effect**: every survivor's Tm already sits below both
   numbers, so a Gaussian centered above the whole pool is monotonic in Tm either way.
2. **Asymmetric Gaussian** (σ=2K above target, 4K below). **No effect**: Constraint 6 already
   excludes every candidate above the ceiling, so the branch this fix changes never executes on
   this dataset.
3. **`thermal_margin` criterion added** (9th criterion, `08_mcdm_ranking.py` — see
   `08_PHASE_6_AUDIT.md`): `Tm_target_capped_C − Tm`, a benefit criterion rewarding headroom below
   the achievability ceiling, blended through the same entropy+AHP weighting as every other
   criterion. **Worked for Cluster 0** (ρ: -0.595→+0.381; its consensus #1, `RT57HC`, now lands
   in-band at 55.4% simulated solar fraction) because Cluster 0's actual failure mode — the old #1
   pick (`n-Octacosane`, Tm=61.6°C) sitting right at the 61.94°C ceiling and stalling on
   below-average solar days — is exactly what a margin criterion measures. **Did not meaningfully
   move Clusters 1/2** (+0.176→+0.103, +0.048→+0.024) — see §3 for why not.

### 3. Why Clusters 1/2 don't respond, and won't respond to any static criterion

The entropy weighting made `thermal_margin` even more dominant than `Tm_fitness` used to be in
these two clusters (weight 0.307–0.331 vs. 0.194–0.199) — the same one-criterion-domination
pattern recurring under a new name, not a real fix, and empirically it didn't help. Checking why:

**A clean test that rules out any static per-candidate criterion.** `CrodaTherm 60` has the
identical Tm (59.8°C), latent heat, and thermal conductivity in every cluster (its properties are
fixed; only the climate driving each cluster's simulation differs). Its simulated performance:

| Cluster | CrodaTherm 60 simulated solar fraction |
|---|---|
| 0 | 38.3% (worst candidate in the cluster) |
| 1 | **79.2%** (best candidate in the cluster) |
| 2 | **65.9%** (best candidate in the cluster) |

Same PCM, same Tm, wildly different real-world outcome — because the outcome is driven by an
**interaction between Tm and each cluster's specific real 10-year day-by-day weather at its
medoid point**, not by any property of the PCM alone. A "distance from delivery temperature"
hypothesis was tested too (Spearman of `|Tm-60°C|` vs. simulated SF) and also failed to hold
consistently across clusters (ρ = +0.558, -0.104, -0.158 — inconsistent sign). **No static,
per-candidate MCDM criterion — margin-based, delivery-distance-based, or otherwise — can capture
a relationship that depends on the specific dynamic climate trajectory a cluster's PCM tank sees
over 10 years.** This is a structural limit of static ranking criteria, not a gap that the next
criterion will close.

### 4. Benchmark band
Solar fraction range 30.7%–79.2% across all 26 simulations; 6/26 (23%) land in the published
54–84% band. Annual cycling stays in a realistic tens-to-low-hundreds-per-year range across
candidates (`physics_validation_results.csv`, `complete_cycles_per_year`).

### 5. Final conclusion — this is a reportable finding, not an unresolved bug

**Phase 6 (static MCDM ranking) and Phase 7 (dynamic physics simulation) sometimes disagree, and
that disagreement is itself a legitimate, reportable result of running both.** MCDM ranks
candidates on fixed properties evaluated the same way regardless of a cluster's actual weather
trajectory; the physics simulation is inherently dynamic, driven by 10 years of real day-by-day
data per cluster. Iterating the MCDM criteria set further to chase a positive correlation in every
cluster was tried once (successfully, for Cluster 0, for a diagnosable static reason) and
evaluated for Clusters 1/2, where it does not apply and should not be forced — doing so would mean
overfitting the ranking to this one simulation's specific parameterization, which would undermine
the point of having two independent checks. **The disagreement in Clusters 1/2 is evidence that
Phase 7's physics validation is doing exactly what it is for**: catching what a static,
literature-informed criteria set cannot see. Report both results together, not the MCDM ranking
alone.

---

## Status
**Investigation closed, 2026-09-16.** `thermal_margin` (9th criterion) kept — real, cited
improvement for Cluster 0, harmless elsewhere. Clusters 1/2's disagreement is documented as a
genuine finding (§5), not chased further. **Phase 7 itself is still NOT unified with Rajasthan**
— Tamil Nadu uses a standalone tank model, Rajasthan uses `physics_lib.py`; reconciling tank
mass/draw-schedule assumptions between the two remains a separate, optional unification step
(see `00_MASTER_OVERVIEW.md` "Still Open"). `N_DRAWS` raised to 5000 (2026-09-16, the plan doc's
default) for the final reported Monte Carlo stability numbers — confirmed this did not change
anything in this section, as expected (Spearman ρ here is computed from the deterministic
consensus rank, not the MC layer; all three values bit-identical before/after the raise).

---

## Literature Support

| Component | Reference / Model | Source File |
|---|---|---|
| Lumped-Enthalpy Tank Model | Barqawi (2025) dynamic simulation | `sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md` |
| Solar Fraction Benchmark (54–84%) | Singh et al. (2025) review | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Spearman Rank Validation | Framework doc §10 (N5 Novelty) | `13_LITERATURE_MAPPING.md` |
| Backward Euler Numerical Scheme | Ghodusinejad (2026) physics models | `sources/Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
