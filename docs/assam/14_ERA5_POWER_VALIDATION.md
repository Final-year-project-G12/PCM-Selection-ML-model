# 14 — ERA5 vs NASA POWER Validation (Assam)

## Status: COMPLETE (BACKBONE Decision)

The Assam pipeline now implements a formal cross-source validation step via `03b_agreement_analysis_assam.py`, bringing it into full architectural parity with the Rajasthan and Tamil Nadu pipelines.

## What was implemented
The script compares the `era5_GHI` generated after the `accum_to_flux()` correction against the `power_ALLSKY_SFC_SW_DWN` daily aggregates. It computes the Mean Bias Error (MBE), Root Mean Square Error (RMSE), and Pearson correlation ($r$) across all four seasons.

If the mean bias exceeds 10% of the true mean, the pipeline automatically selects the `QUANTILE_MAP` decision and fits empirical correction maps. Otherwise, it selects `BACKBONE`.

## Key Findings for Assam

> [!NOTE]
> **Excellent Agreement**
> The agreement analysis revealed that the Mean Bias Error (MBE) for Assam's daytime GHI is only **1.1%**. 

Because this error is well below the 10% threshold, the automated decision logic generated a **`BACKBONE`** decision (`bias_decision_assam.txt`). 

## Impact on Preprocessing — correction: the BACKBONE decision is not actually wired in

**Verified against the current script**: `04_preprocess_assam.py` never opens
`bias_decision_assam.txt` and has no conditional branch on it. Its step [7] (per-(point, season)
empirical quantile mapping of `era5_GHI` onto `power_ALLSKY_SFC_SW_DWN`) runs **unconditionally**
and always overwrites `era5_GHI` with the corrected series, regardless of the `BACKBONE` decision
computed in `03b_agreement_analysis_assam.py`. `quantile_maps_assam.joblib` (written only in the
`QUANTILE_MAP` branch of `03b`) is never loaded by anything downstream — `04` fits its own
quantile maps inline. So although the 1.1% MBE genuinely supports treating ERA5 as reliable, the
actual pipeline still applies an empirical correction to every row; it does not pass raw ERA5
through unmodified. This should be resolved one of two ways: either state that the quantile
correction is intentional (and drop the "BACKBONE bypass" framing), or make `04`'s step [7]
conditional on the decision file. As shipped, the two disagree.

The agreement analysis indicates strong empirical consistency between ERA5 and NASA POWER for Assam, with an MBE of 1.1%, well below the 10% decision threshold. This cross-source agreement should not be interpreted as proof against independent ground-truth measurements.
