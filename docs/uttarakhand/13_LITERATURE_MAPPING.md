# 13 — Literature Mapping

> **Consolidation note.** Temporal- and spatial-processing justifications now live in
> `03_PHASE_1_AUDIT.md`; ERA5 de-accumulation, solar geometry, derived solar variables,
> cross-source validation and quality control now live in `04_PHASE_2_AUDIT.md`; the climate
> signature's feature-to-PCM-property mapping now lives in `05_PHASE_3_AUDIT.md`. This file records
> only what the Uttarakhand source files actually cite, and what must be added.

## Method

Every entry below was checked against the **contents of `era5-uttarakhand/`** — the Python scripts,
their docstrings and comments, and the four markdown files (`README.md`, `README_PREPROCESSING.md`,
`PREPROCESSING_STEPS.md`, `NEXT_STEPS.md`, `VERIFICATION_METHODOLOGY.md`). Nothing was asserted from
general knowledge and nothing was imported from another state's documentation.

The governing plan document (`Objective1_PCM_Climate_Framework_Plan_v3`, cited in-script as "plan
v3.0") is **not present inside `era5-uttarakhand/`**. Every plan reference is therefore recorded as
*"cited by the script"*, not verified against the document.

---

## The complete citation footprint of `era5-uttarakhand/`

This is the headline finding of this file, and it is short.

**Exactly three author-year literature citations appear anywhere in the Uttarakhand pipeline:**

| Citation | Where | Context |
|---|---|---|
| **Al-Mamun 2023** | `07b_charging_feasibility.py`, line 58 | "roughly consistent with Al-Mamun2023's cited FPC 25-100C operating band" — justifying the 70 °C `REFERENCE_GOOD_DAY_TEMP_C` ceiling |
| **Barqawi et al. 2025** | `10_physics_validation.py`, line 21 | "ODE structure — already extracted in your Sources/Barqawi2025DynamicSimulationPCM_SWH_summary.md" — 3-phase lumped-enthalpy tank simulation basis |
| **Avargani et al. 2021** | `04b_climate_signature.py` | Justifying the domestic SWH draw basis |

Every other methodological choice in `era5-uttarakhand/` either cites a software package (`pvlib`),
a plan section/table, or is un-cited in the code.

---

## Component -> implementation -> source mapping

| Component | Implementation | Supporting source | Status in `era5-uttarakhand/` |
|---|---|---|---|
| ERA5 reanalysis as climate backbone | Phase 1–2 | Hersbach et al. (2020), *QJRMS* | Plan-sourced; not cited in code |
| NASA POWER as cross-check | Phase 1–2 | NASA POWER project documentation | Plan-sourced; not cited in code |
| Sun-event times (sunrise/noon/sunset) | `00b_build_suntimes.py` | Reda & Andreas (2004), *Solar Energy* (SPA) | Implementation uses `pvlib.solarposition.sun_rise_set_transit_spa` with `method="spa"` pinned. Method is standard; citation **not present in code** |
| Clear-sky GHI | `02_combine_uttarakhand.py` | Ineichen & Perez (2002), *Solar Energy* | Implementation uses `pvlib.clearsky.ineichen` with default Linke turbidity. **Strong validation result** ($r = 0.9923$, MBE $+5.3\text{ W/m}^2$). Citation **not present in code** |
| Sun-event-aligned sampling | `00b`, `02` | Project-original sampling design | Uncited; novel framing |
| `elev_proxy` from surface pressure | `04b_climate_signature.py` | Barometric formula approximation | Uncited in code; standard physics |
| 13-step preprocessing sequence | `04_preprocess_uttarakhand.py` | Standard ML/data-science pipeline | Plan-sourced ("Section 5"); no literature citations in script |
| Hampel outlier filter | `04` step 4 | Pearson (2002) / Hampel (1974) | Standard method; uncited in code |
| Hierarchical imputation chain | `04` step 5 | MICE (van Buuren & Groothuis-Oudshoorn 2011) | Implementation uses `sklearn.experimental.enable_iterative_imputer` + `IterativeImputer`. Citation **not present in code** |
| Yeo-Johnson transform diagnostic | `04` step 8 | Yeo & Johnson (2000), *Biometrika* | Used as diagnostic only; uncited in code |
| Savitzky-Golay filter diagnostic | `04` step 9 | Savitzky & Golay (1964), *Anal. Chem.* | Used as diagnostic only; uncited in code |
| Gaussian Mixture Model clustering | `05_cluster_uttarakhand.py` | Standard GMM (Duda & Hart 1973; McLachlan & Peel 2000) | `sklearn.mixture.GaussianMixture` with `full` covariance, `n_init=5`/`n_init=10`. Uncited in code |
| BIC / Silhouette model selection | `05` | Schwarz (1978) / Rousseeuw (1987) | Standard heuristics; uncited in code |
| MICE + PMM PCM database imputation | `PCM_data/01_preprocess.py` | van Buuren (2018) | Applied upstream on the 55-row PCM database; uncited |
| Gaussian $T_m$-fitness transform | `08_mcdm_ranking.py` | Project-original fitness transformation | Plan-sourced ("Section 9.2"); uncited |
| Entropy weight calculation | `08` | Shannon (1948) | Standard entropy formula; uncited in code |
| AHP prior weighting | `08` | Saaty (1980) | Docstring: "an honest placeholder, not a claimed AHP result." Uncited |
| TOPSIS ranking | `08` | Hwang & Yoon (1981) | Implemented; uncited in code |
| Grey Relational Analysis ($\zeta = 0.5$) | `08` | Deng (1982) | Implemented; uncited in code |
| Borda-count consensus | `08` | Borda (1781) | Implemented; uncited in code |
| Kendall's W | `08` | Kendall & Babington Smith (1939) | Plan v3.0 §9.5 cited for interpretation; statistic uncited |
| Flat-plate collector 25–100 °C operating band | `07b` | **Al-Mamun 2023** | The pipeline's only substantive citation in Phase 5/7 |
| Annual solar fraction 54–84 % benchmark | `10_physics_validation.py` | plan Table 16; Barqawi 2025 | 92% of simulated runs land within this benchmark band |
| Grey-box lumped-enthalpy tank model | `10_physics_validation.py` | Barqawi et al. (2025) dynamic simulation | Implemented with implicit Backward Euler integration; see `09_PHASE_7_AUDIT.md` |

---

## Uttarakhand-specific literature note

**No Uttarakhand-specific or Himalayan-specific climate reference appears anywhere in `era5-uttarakhand/`.** The state-specific reasoning that does exist is stated as geographic domain knowledge in prose:

- Doon Valley vs Terai plains vs Chamoli/Pithoragarh high Himalaya elevation gradients (`05_cluster_uttarakhand.py` docstring).
- 1200 m flat altitude approximation for solar geometry (`02_combine_uttarakhand.py` comment).
- Monsoon JJA definition for northern India (`02_combine_uttarakhand.py` `SEASON_MAP`).

---

## Minimum citations to add before paper submission

To make the paper submission-ready, add BibTeX entries for:

1. **ERA5 Backbone**: Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
2. **NASA POWER**: Zhang, T., et al. (2014). NASA POWER release 8.
3. **Solar Geometry (pvlib / SPA)**: Reda, I., & Andreas, A. (2004). Solar position algorithm for solar radiation applications. *Solar Energy*, 76(5), 577–589.
4. **Ineichen Clear-Sky**: Ineichen, P., & Perez, R. (2002). A new airmass independent formulation for the Linke turbidity factor. *Solar Energy*, 73(3), 151–157.
5. **pvlib Software**: Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: a python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884.
6. **Physics Tank Model**: Barqawi, F. A. (2025). Dynamic simulation of PCM thermal storage for solar water heating. *Muthanna Journal of Engineering and Technology*, 13(3), 1–14.
7. **MCDM Frameworks**:
   - TOPSIS: Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making*. Springer-Verlag.
   - GRA: Deng, J. L. (1982). Control problems of grey systems. *Systems & Control Letters*, 1(5), 288–294.
   - MICE: van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.
