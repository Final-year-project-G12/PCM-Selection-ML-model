"""
PCM_Properties.csv PREPROCESSING — MICE + RANDOM FOREST + PREDICTIVE MEAN MATCHING
====================================================================================
Upgrade over the KNN-based version: fills missing values using chained-equations
regression (MICE) with a Random Forest per column, refined with Predictive Mean
Matching (PMM) so every filled value is a REAL, previously-measured value donated
from the most physically-similar PCM — never a synthetic average.

CRITICAL DESIGN REQUIREMENT
----------------------------
Several properties (thermal conductivity liquid/solid, specific heat solid,
flammability) are missing across the ENTIRE Rubitherm RT-line (10/10 rows) and
only exist in the Pluss savE/OM-line (8/8 rows). A naive nearest-neighbour
search that includes "product line" as a distance feature can fail here: an
RT PCM's closest matches are always other RT PCMs, which are also missing the
value, so there's nothing to borrow (this is exactly what happened with the
first pass of the KNN script's categorical fill).

MICE+RF+PMM avoids this by construction:
  1. The Random Forest for a given column trains ONLY on rows where that
     column is actually observed — regardless of which product line they
     belong to. If only OM-series rows have it, the forest simply trains on
     those 8 rows and learns the relationship between the column and every
     other property (melting point, latent heat, density, Cp, type flags).
  2. It then predicts a value for EVERY row (RT included) from that row's
     OWN properties.
  3. PMM ranks candidate donors by how close the model's PREDICTIONS are to
     each other — not by raw feature distance — so the donor pool is
     whichever rows truly have the value, wherever they come from.

Every imputed numeric cell is logged with which manufacturer(s) supplied its
donor value, so cross-series borrowing can be verified directly in the output
rather than taken on faith.

Run: python3 preprocess_pcm_mice_rf_pmm.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# All paths are relative to this script's own location, not the current
# working directory, so `python3 01_preprocess.py` works no matter where
# it's launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

IN_PATH = os.path.join(DATA_DIR, "PCM_Properties_55records_42_70C_dense.csv")
OUT_LEAN = os.path.join(DATA_DIR, "PCM_Properties_cleaned_mice_pmm.csv")
OUT_DETAILED = os.path.join(DATA_DIR, "PCM_Properties_cleaned_mice_pmm_detailed.csv")
N_ITER = 8          # MICE refinement rounds
N_DONORS = 3         # PMM donor pool size per missing cell
RANDOM_STATE = 42

COLUMN_MAP = {
    "Product": "product",
    "Manufacturer": "manufacturer",
    "Type": "pcm_type",
    "Appearance": "appearance",
    "Melting Temperature (°C)": "Tm_melting",
    "Freezing/Congealing Temperature (°C)": "Tm_freezing",
    "Nucleation Temperature (°C)": "Tm_nucleation",
    "Latent Heat - Melting (kJ/kg)": "latent_heat_melting",
    "Latent Heat - Freezing (kJ/kg)": "latent_heat_freezing",
    "Heat Storage Capacity (Wh/kg)": "heat_storage_Wh_kg",
    "Density - Liquid (kg/m³)": "density_liquid",
    "Density - Solid (kg/m³)": "density_solid",
    "Specific Heat - Liquid (kJ/kgK)": "Cp_liquid",
    "Specific Heat - Solid (kJ/kgK)": "Cp_solid",
    "Thermal Conductivity - Liquid (W/mK)": "TC_liquid",
    "Thermal Conductivity - Solid (W/mK)": "TC_solid",
    "Thermal Conductivity - Both Phases (W/mK)": "TC_both",
    "Volume Expansion (%)": "volume_expansion",
    "Max Operating Temperature (°C)": "max_op_temp",
    "Flammability": "flammability",
    "Flash Point (°C)": "flash_point",
    "Number of Cycles Tested": "cycles_tested",
}

NUMERIC_COLS = [
    "Tm_melting", "Tm_freezing", "Tm_nucleation",
    "latent_heat_melting", "latent_heat_freezing", "heat_storage_Wh_kg",
    "density_liquid", "density_solid",
    "Cp_liquid", "Cp_solid",
    "TC_liquid", "TC_solid", "TC_both",
    "volume_expansion", "max_op_temp", "flash_point", "cycles_tested",
]

ROUND_MAP = {
    "Tm_melting": 1, "Tm_freezing": 1, "Tm_nucleation": 1,
    "latent_heat_melting": 0, "latent_heat_freezing": 0, "heat_storage_Wh_kg": 0,
    "density_liquid": 0, "density_solid": 0,
    "Cp_liquid": 2, "Cp_solid": 2,
    "TC_liquid": 3, "TC_solid": 3, "TC_both": 3,
    "volume_expansion": 1, "max_op_temp": 0, "flash_point": 0, "cycles_tested": 0,
}


def parse_messy_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    import re
    peak_match = re.search(r"peak:\s*([0-9.]+)", s, re.IGNORECASE)
    if peak_match:
        return float(peak_match.group(1))
    # New format seen in this dataset: "48 and 43" — a range using "and"
    # instead of a dash. Treated the same way as a dash-range: midpoint.
    and_match = re.match(r"^(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)$", s, re.IGNORECASE)
    if and_match:
        return (float(and_match.group(1)) + float(and_match.group(2))) / 2.0
    # Plain dash range, e.g. "43-37" or "38-43" or "43.6-46.0". This MUST be
    # matched with an anchored two-group pattern rather than a blind
    # findall(r"-?\d+...") — a naive findall misreads the separating dash as
    # a negative sign on the second number (e.g. "43-37" -> [43, -37]
    # instead of [43, 37]), silently corrupting every descending range and
    # roughly half of ascending ones too. This bug was masked in an earlier
    # dataset because every range there happened to carry a "peak: X"
    # annotation that short-circuited parsing before reaching this branch —
    # it is not safe to assume that will always be true.
    range_match = re.match(r"^\s*(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)", s)
    if range_match:
        try:
            return (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
        except ValueError:
            pass
    numbers = re.findall(r"-?\d+\.?\d*", s)
    if not numbers:
        return np.nan
    # (Dash-ranges are already handled above via range_match; anything
    # reaching here with multiple numbers is an unrecognized format — take
    # the first number rather than guessing at a midpoint.)
    return float(numbers[0])


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col + "_original_text"] = df[col]
            df[col] = df[col].apply(parse_messy_numeric)

    # This dataset's "Type" field is already a rich, chemically meaningful
    # category (e.g. "Organic n-alkane", "Organic fatty acid", "Organic
    # (RT-line)", "Organic/composite blend" — 11 distinct subtypes here,
    # vs. just "Organic"/"Inorganic" in the smaller dataset). Unlike the
    # earlier script, it is used as-is (not stripped to a broad category
    # plus a hand-rolled product-line flag) — the full text is one-hot
    # encoded directly as a predictor further down, preserving that extra
    # chemical-family signal for similarity matching.

    # Nucleation is only meaningful relative to each PCM's own freezing point
    # (degrees of supercooling) — imputed as that offset, not an absolute value.
    df["_nucleation_subcool"] = df["Tm_freezing"] - df["Tm_nucleation"]

    return df


def mice_rf_pmm_impute(df: pd.DataFrame):
    work = df.copy()
    numeric_cols = list(NUMERIC_COLS)  # 'Tm_nucleation' entries here operate on subcooling internally

    original_missing = {c: work[c].isna().copy() for c in numeric_cols}

    # Swap absolute nucleation temp for the relative subcooling degree
    working_values = work[numeric_cols].copy()
    working_values["Tm_nucleation"] = work["_nucleation_subcool"]
    original_missing["Tm_nucleation"] = working_values["Tm_nucleation"].isna()

    type_dummies = pd.get_dummies(work["pcm_type"], prefix="type").astype(float)
    manu_dummies = pd.get_dummies(work["manufacturer"], prefix="manu").astype(float)
    extra_predictors = pd.concat([type_dummies, manu_dummies], axis=1)

    # ---- MICE initialization: crude column-mean fill just to bootstrap ----
    imputed = working_values.fillna(working_values.mean())

    # Impute columns with the FEWEST missing values first (standard MICE
    # heuristic — stabilizes the early rounds before tackling harder columns)
    col_order = sorted(numeric_cols, key=lambda c: original_missing[c].sum())

    donor_log = {c: [] for c in numeric_cols}  # product -> list of donor manufacturers (final round)

    for iteration in range(N_ITER):
        for col in col_order:
            miss_mask = original_missing[col].values
            if not miss_mask.any():
                continue

            other_cols = [c for c in numeric_cols if c != col]
            X_all = pd.concat([imputed[other_cols], extra_predictors], axis=1).values
            y_all = imputed[col].values

            train_idx = ~miss_mask
            if train_idx.sum() < 3:
                continue  # not enough real data to fit a model for this column

            rf = RandomForestRegressor(
                n_estimators=300, max_depth=4, min_samples_leaf=2,
                random_state=RANDOM_STATE,
            )
            rf.fit(X_all[train_idx], y_all[train_idx])
            pred_all = rf.predict(X_all)

            donor_positions = np.where(train_idx)[0]
            donor_actual = y_all[donor_positions]       # true observed values only
            donor_pred = pred_all[donor_positions]

            for row_i in np.where(miss_mask)[0]:
                target_pred = pred_all[row_i]
                dists = np.abs(donor_pred - target_pred)
                k = min(N_DONORS, len(dists))
                nearest = np.argsort(dists)[:k]
                weights = 1.0 / (dists[nearest] + 1e-6)
                weights = weights / weights.sum()
                new_val = float(np.sum(weights * donor_actual[nearest]))
                imputed.iat[row_i, imputed.columns.get_loc(col)] = new_val

                if iteration == N_ITER - 1:
                    donor_product_rows = donor_positions[nearest]
                    # Tm_nucleation is internally a subcooling OFFSET
                    # (Tm_freezing - Tm_nucleation), not an absolute
                    # temperature. Translate both the predicted value and
                    # each donor's value back to real Tm_nucleation degrees
                    # here so this log matches what actually appears in the
                    # final output column, instead of a confusing raw offset.
                    if col == "Tm_nucleation":
                        display_predicted = float(work["Tm_freezing"].iloc[row_i] - max(new_val, 0))
                        display_donor_values = (
                            work["Tm_freezing"].iloc[donor_product_rows].values
                            - donor_actual[nearest]
                        ).tolist()
                    else:
                        display_predicted = new_val
                        display_donor_values = donor_actual[nearest].tolist()

                    donor_log[col].append({
                        "product": work.iloc[row_i]["product"],
                        "own_manufacturer": work.iloc[row_i]["manufacturer"],
                        "predicted_value": display_predicted,
                        "donor_manufacturers": work.iloc[donor_product_rows]["manufacturer"].tolist(),
                        "donor_products": work.iloc[donor_product_rows]["product"].tolist(),
                        "donor_values": display_donor_values,
                        "donor_weights": weights.tolist(),
                    })

    # ---- Flag columns where regression genuinely couldn't be fit ----
    # (fewer than 3 real observations in the WHOLE dataset — e.g. this
    # dataset has only 1 known Nucleation Temperature out of 55 rows). These
    # never went through the RF+PMM loop above, so `imputed[col]` for them
    # is still sitting at the crude initialization fill (the mean of
    # whatever handful of real values exist — with n=1 that's just that one
    # value, applied to every missing row). That's the best any method can
    # do with this little evidence, but it must be logged as low-confidence
    # rather than silently presented as an equal-quality MICE+RF+PMM result.
    low_confidence_cols = [c for c in numeric_cols if (~original_missing[c]).sum() < 3]
    for col in low_confidence_cols:
        known_positions = np.where(~original_missing[col].values)[0]
        if len(known_positions) == 0:
            continue  # genuinely zero data anywhere — cannot impute at all
        for row_i in np.where(original_missing[col].values)[0]:
            if col == "Tm_nucleation":
                # Use the MICE-refined Tm_freezing (`imputed`), not the raw
                # pre-imputation column (`work`) — Tm_freezing itself has
                # plenty of donor support (29/55 known, well above the n<3
                # threshold) and is already fully resolved by this point in
                # the loop, but `work["Tm_freezing"]` still holds NaN for
                # any row where freezing point was ALSO originally missing.
                # Reading the raw column here silently produced NaN for
                # every such row.
                donor_vals_abs = (
                    imputed["Tm_freezing"].iloc[known_positions].values
                    - working_values["Tm_nucleation"].iloc[known_positions].values
                )
                pred_abs = float(imputed["Tm_freezing"].iloc[row_i] - imputed["Tm_nucleation"].iloc[row_i])
            else:
                donor_vals_abs = working_values[col].iloc[known_positions].values
                pred_abs = float(imputed[col].iloc[row_i])
            donor_log[col].append({
                "product": work.iloc[row_i]["product"],
                "own_manufacturer": work.iloc[row_i]["manufacturer"],
                "predicted_value": pred_abs,
                "donor_manufacturers": work.iloc[known_positions]["manufacturer"].tolist(),
                "donor_products": work.iloc[known_positions]["product"].tolist(),
                "donor_values": donor_vals_abs.tolist(),
                "donor_weights": [1.0 / len(known_positions)] * len(known_positions),
                "low_confidence": True,
            })

    # ---- Write numeric results back, translating nucleation subcool -> absolute ----
    for col in numeric_cols:
        if col == "Tm_nucleation":
            continue
        work[col] = imputed[col].round(ROUND_MAP.get(col, 2))

    subcool_final = imputed["Tm_nucleation"].clip(lower=0)
    work["Tm_nucleation"] = (work["Tm_freezing"] - subcool_final).round(ROUND_MAP.get("Tm_nucleation", 1))

    # ---- Categorical columns: Random Forest classifier ----
    # Same cross-series guarantee: trains only on rows with a known label
    # (here, only savE), predicts for every row (RT included) from that
    # row's own physical properties.
    cat_cols = ["flammability", "appearance"]
    cat_imputed_flags = {}
    predictor_matrix = pd.concat([work[numeric_cols].drop(columns=["Tm_nucleation"]),
                                   work["Tm_nucleation"], extra_predictors], axis=1)

    cat_donor_log = {c: [] for c in cat_cols}

    for cat_col in cat_cols:
        cat_imputed_flags[cat_col] = work[cat_col].isna()
        known_mask = work[cat_col].notna().values
        if known_mask.sum() < 2 or cat_imputed_flags[cat_col].sum() == 0:
            continue
        clf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=1,
                                      random_state=RANDOM_STATE)
        clf.fit(predictor_matrix.values[known_mask], work.loc[known_mask, cat_col].values)
        missing_mask = ~known_mask
        preds = clf.predict(predictor_matrix.values[missing_mask])
        work.loc[missing_mask, cat_col] = preds

        # Log the most physically similar known-label PCMs as supporting
        # evidence for each prediction (Euclidean distance in the same
        # predictor space the classifier was trained on).
        X_known = predictor_matrix.values[known_mask]
        known_idx = np.where(known_mask)[0]
        for row_i, pred_label in zip(np.where(missing_mask)[0], preds):
            x_row = predictor_matrix.values[row_i]
            dists = np.linalg.norm(X_known - x_row, axis=1)
            k = min(N_DONORS, len(dists))
            nearest = np.argsort(dists)[:k]
            donor_rows = known_idx[nearest]
            cat_donor_log[cat_col].append({
                "product": work.iloc[row_i]["product"],
                "own_manufacturer": work.iloc[row_i]["manufacturer"],
                "predicted_value": pred_label,
                "donor_manufacturers": work.iloc[donor_rows]["manufacturer"].tolist(),
                "donor_products": work.iloc[donor_rows]["product"].tolist(),
                "donor_values": work.iloc[donor_rows][cat_col].tolist(),
            })

    cycles_status = np.where(
        df["cycles_tested_original_text"].astype(str).str.contains("Under Trial", case=False, na=False),
        "Under trial (estimate below)",
        np.where(original_missing["cycles_tested"], "Estimated via MICE-RF-PMM", "Reported by manufacturer"),
    )
    work["cycles_tested_status"] = cycles_status

    for col in numeric_cols:
        work[col + "_imputed"] = original_missing[col].values
    for col in cat_cols:
        work[col + "_imputed"] = cat_imputed_flags[col].values

    return work, donor_log, cat_donor_log, original_missing


def print_cross_series_audit(donor_log, df):
    """For every column that was missing in an ENTIRE product line, show
    exactly which manufacturer(s) the donated values actually came from."""
    print("=" * 78)
    print("CROSS-SERIES DONOR AUDIT")
    print("(proves donors reach across product lines when a whole line is missing)")
    print("=" * 78)
    for col, entries in donor_log.items():
        if not entries:
            continue
        cross_series_count = sum(
            1 for e in entries
            if e["own_manufacturer"] not in e["donor_manufacturers"]
        )
        print(f"\n[{col}]  {len(entries)} rows imputed, "
              f"{cross_series_count} used at least one cross-manufacturer donor")
        for e in entries[:3]:  # show a few examples
            print(f"   {e['product']:12s} ({e['own_manufacturer'][:6]})  <-  donors: "
                  f"{list(zip(e['donor_products'], [m[:6] for m in e['donor_manufacturers']]))}")


def build_provenance_table(donor_log, cat_donor_log):
    """One row per originally-missing cell: which value was predicted, and
    exactly which similar PCMs (with their real values and blend weights)
    supplied that prediction. This is the permanent, inspectable answer to
    'which similar PCMs were used to predict each not-reported value'."""
    rows = []
    for col, entries in donor_log.items():
        for e in entries:
            method = ("Insufficient data (fewer than 3 real values in the entire "
                      "dataset) — value carried from the only donor(s) available, "
                      "not a fitted MICE+RF+PMM prediction. Treat as low confidence."
                      if e.get("low_confidence") else
                      "MICE + Random Forest + PMM (weighted blend of real donor values)")
            row = {
                "product": e["product"],
                "manufacturer": e["own_manufacturer"],
                "property": col,
                "predicted_value": round(e["predicted_value"], 4),
                "confidence": "LOW (n<3 donors)" if e.get("low_confidence") else "Standard",
                "prediction_method": method,
            }
            for i in range(len(e["donor_products"])):
                row[f"donor_{i+1}_product"] = e["donor_products"][i]
                row[f"donor_{i+1}_manufacturer"] = e["donor_manufacturers"][i]
                row[f"donor_{i+1}_actual_value"] = round(e["donor_values"][i], 4)
                row[f"donor_{i+1}_weight"] = round(e["donor_weights"][i], 3)
            rows.append(row)

    for col, entries in cat_donor_log.items():
        for e in entries:
            row = {
                "product": e["product"],
                "manufacturer": e["own_manufacturer"],
                "property": col,
                "predicted_value": e["predicted_value"],
                "confidence": "Standard",
                "prediction_method": "Random Forest classifier (nearest known-label PCMs shown as evidence)",
            }
            for i in range(len(e["donor_products"])):
                row[f"donor_{i+1}_product"] = e["donor_products"][i]
                row[f"donor_{i+1}_manufacturer"] = e["donor_manufacturers"][i]
                row[f"donor_{i+1}_actual_value"] = e["donor_values"][i]
                row[f"donor_{i+1}_weight"] = ""
            rows.append(row)

    prov = pd.DataFrame(rows)
    donor_cols = sorted([c for c in prov.columns if c.startswith("donor_")])
    front_cols = ["product", "manufacturer", "property", "predicted_value", "confidence", "prediction_method"]
    return prov[front_cols + donor_cols]


def main():
    raw = load_raw(IN_PATH)
    check_cols = NUMERIC_COLS + ["flammability", "appearance"]
    before_missing = raw[check_cols].isna().sum()

    cleaned, donor_log, cat_donor_log, _ = mice_rf_pmm_impute(raw)
    after_missing = cleaned[check_cols].isna().sum()

    front = ["product", "manufacturer", "pcm_type"]

    # Group each property's three related columns together — value, whether
    # it was predicted, and exactly what the manufacturer's sheet said —
    # instead of scattering values/flags/original-text into three separate
    # blocks far apart in the file (which is what made the screenshot look
    # like a wall of missing data even though every value was filled).
    grouped = []
    for col in NUMERIC_COLS:
        grouped.append(col)
        if col + "_imputed" in cleaned.columns:
            grouped.append(col + "_imputed")
        if col + "_original_text" in cleaned.columns:
            grouped.append(col + "_original_text")
        if col == "cycles_tested" and "cycles_tested_status" in cleaned.columns:
            grouped.append("cycles_tested_status")

    grouped += ["flammability", "flammability_imputed", "appearance", "appearance_imputed"]

    ordered = front + grouped
    ordered = [c for c in ordered if c in cleaned.columns]
    ordered += [c for c in cleaned.columns if c not in ordered]
    cleaned = cleaned[ordered].drop(columns=["_nucleation_subcool"], errors="ignore")

    # Every remaining blank cell at this point is in a *_original_text audit
    # column, and it's blank because the manufacturer datasheet genuinely
    # said nothing there — not because imputation missed it (the actual
    # property columns above are already 100% filled). Make that explicit
    # rather than leaving an ambiguous empty cell in the CSV.
    for col in [c for c in cleaned.columns if c.endswith("_original_text")]:
        cleaned[col] = cleaned[col].fillna("Not reported by manufacturer")

    # The multi-column "detailed" file (value + imputed-flag + original-text
    # per attribute) is kept only as an optional audit artifact, not the
    # main deliverable — it was the source of the "two columns for one
    # attribute" confusion. Everyday use should be the lean file below,
    # which has exactly one clean, fully-numeric column per attribute.
    cleaned.to_csv(OUT_DETAILED, index=False)

    # Single clean column per attribute — no imputed-flag columns, no
    # original-text columns, no redundant raw-vs-parsed pairs. Every
    # not-reported value has already been replaced with its predicted
    # number (or predicted category, for flammability/appearance) directly
    # in these columns — nothing further to look up elsewhere.
    lean_cols = ["product", "manufacturer", "pcm_type", "appearance"] + NUMERIC_COLS + ["flammability"]
    lean_cols = [c for c in lean_cols if c in cleaned.columns]
    cleaned[lean_cols].to_csv(OUT_LEAN, index=False)

    print("=" * 78)
    print("MISSING VALUES — BEFORE vs AFTER (MICE + Random Forest + PMM)")
    print("=" * 78)
    summary = pd.DataFrame({"missing_before": before_missing, "missing_after": after_missing})
    summary = summary[summary["missing_before"] > 0]
    print(summary.to_string())
    print(f"\nTotal missing cells: {before_missing.sum()} -> {after_missing.sum()}")

    print_cross_series_audit(donor_log, raw)

    provenance = build_provenance_table(donor_log, cat_donor_log)
    OUT_PROVENANCE = os.path.join(DATA_DIR, "05_imputation_provenance.csv")
    provenance.to_csv(OUT_PROVENANCE, index=False)
    print(f"Saved provenance    -> {OUT_PROVENANCE}  ({len(provenance)} predicted cells, "
          f"each traced to its {N_DONORS} nearest donor PCMs)")

    print(f"\nSaved lean file      -> {OUT_LEAN}  (shape={cleaned[lean_cols].shape})")
    print(f"Saved detailed/audit -> {OUT_DETAILED}  (shape={cleaned.shape})")

    make_plots(raw, cleaned, donor_log, check_cols)


def make_plots(raw, cleaned, donor_log, check_cols):
    """Save the missingness heatmap, cross-series donor audit chart, an
    imputed-vs-reported sanity strip plot, and a correlation heatmap of the
    final numeric properties — all into the data/ folder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ---- 1. Missingness before/after ----
    # Figure height scales with row count so labels stay legible — a fixed
    # 5-inch height (fine for 18 rows) badly cramps 55 product-name labels.
    n_rows = len(raw)
    fig_h = max(5, 0.22 * n_rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_h))
    sns.heatmap(raw[check_cols].isna(), cbar=False, cmap="rocket_r",
                yticklabels=raw["product"], ax=axes[0])
    axes[0].set_title("Missing values — BEFORE")
    axes[0].tick_params(axis='y', labelsize=8)
    sns.heatmap(cleaned[check_cols].isna(), cbar=False, cmap="rocket_r",
                yticklabels=cleaned["product"], ax=axes[1])
    axes[1].set_title("Missing values — AFTER (MICE + RF + PMM)")
    axes[1].tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "01_missingness_before_after.png"), dpi=150)
    plt.close(fig)

    # ---- 2. Cross-series donor audit chart ----
    rows = []
    for col, entries in donor_log.items():
        if not entries:
            continue
        cross = sum(1 for e in entries if e["own_manufacturer"] not in e["donor_manufacturers"])
        rows.append({"column": col, "pct_cross_manufacturer": 100 * cross / len(entries),
                      "n_imputed": len(entries)})
    audit_df = pd.DataFrame(rows).sort_values("pct_cross_manufacturer", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(audit_df["column"], audit_df["pct_cross_manufacturer"], color="#6C63A6")
    for bar, n in zip(bars, audit_df["n_imputed"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"n={n}", va="center", fontsize=8)
    ax.set_xlabel("% of imputed rows using a cross-manufacturer donor")
    ax.set_title("Cross-series donor audit — MICE + RF + PMM")
    ax.set_xlim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "02_cross_series_donor_audit.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Imputed-vs-reported sanity strip plot for key columns ----
    key_cols = ["Tm_nucleation", "TC_liquid", "TC_solid", "Cp_solid", "heat_storage_Wh_kg"]
    key_cols = [c for c in key_cols if c in cleaned.columns]
    manufacturers = cleaned["manufacturer"].unique()
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(manufacturers)}

    fig, axes = plt.subplots(1, len(key_cols), figsize=(4 * len(key_cols), 5), sharey=False)
    for ax, col in zip(axes, key_cols):
        flag_col = col + "_imputed"
        for i, manu in enumerate(manufacturers):
            sub = cleaned[cleaned["manufacturer"] == manu]
            reported = sub[~sub[flag_col]]
            imputed = sub[sub[flag_col]]
            ax.scatter([manu[:5]] * len(reported), reported[col], color="#2A9D8F",
                       marker=markers[manu], label="Reported" if i == 0 else None, zorder=3)
            ax.scatter([manu[:5]] * len(imputed), imputed[col], color="#E76F51",
                       marker=markers[manu], label="Imputed" if i == 0 else None, zorder=3)
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=45)
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "03_imputed_vs_reported_sanity.png"), dpi=150)
    plt.close(fig)

    # ---- 4. Correlation heatmap of final numeric properties ----
    numeric_final = cleaned[NUMERIC_COLS].select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_final.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation matrix — final imputed dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "04_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    print("\nSaved plots -> data/01_missingness_before_after.png, "
          "02_cross_series_donor_audit.png, 03_imputed_vs_reported_sanity.png, "
          "04_correlation_heatmap.png")


if __name__ == "__main__":
    main()
