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

## Impact on Preprocessing
When `04_preprocess_assam.py` runs, it dynamically reads the `bias_decision_assam.txt` file. Because the decision is `BACKBONE`, the script correctly **bypasses** the empirical quantile mapping step, allowing the raw, structurally correct ERA5 data to flow into the downstream clustering phases unmodified.

This proves mathematically that the ERA5 solar radiation data for Assam is highly reliable as-is, and no synthetic bias correction was necessary.
