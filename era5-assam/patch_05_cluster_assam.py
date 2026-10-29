"""
patch_05_cluster_assam.py
Applies four exact-string edits to era5-assam/05_cluster_assam.py. Aborts without
writing if any anchor is not found exactly once. Writes 05_cluster_assam.py.bak first.

What it changes (clustering, K, features, seeds, assignments are NOT touched):
  1. Merges tm_cap_by_point_assam.csv (from 04c_charging_ceiling_assam.py) into the
     point table, so Tm_target_capped_C is the Rajasthan worst-month ceiling.
  2. Tm_target_capped_C: removes the silent `else TM_TARGET` fallback -> NaN + warning.
  3. HSI: removes the silent RH_mean proxy. Writes HSI_sunrise ONLY if the signatures
     file really has an 'HSI_sunrise' column; otherwise NaN + warning.
  4. Adds a loud completeness check on the finished profile.

Usage:  python patch_05_cluster_assam.py path/to/era5-assam/05_cluster_assam.py
"""
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
text = raw.replace("\r\n", "\n")

EDITS = [
 # 1. merge the 04c ceiling
 ('''    merged = pd.merge(assign_df, raw_df, on="point_id")
''',
  '''    merged = pd.merge(assign_df, raw_df, on="point_id")
    TM_CAP_FILE = BASE_DIR / "data" / "processed" / "tm_cap_by_point_assam.csv"
    if TM_CAP_FILE.exists():
        _cap = pd.read_csv(TM_CAP_FILE)[["point_id", "Tm_target_capped_C"]]
        if "Tm_target_capped_C" in merged.columns:
            log("  [NOTE] signatures file already had Tm_target_capped_C; replaced by "
                "04c_charging_ceiling_assam.py's worst-month value.")
            merged = merged.drop(columns=["Tm_target_capped_C"])
        merged = pd.merge(merged, _cap, on="point_id", how="left")
        log(f"  Merged worst-month charging ceiling from {TM_CAP_FILE.name}")
    elif "Tm_target_capped_C" in merged.columns:
        log("  [WARN] Using Tm_target_capped_C already present in the signatures file; its "
            "derivation is UNVERIFIED against the Rajasthan worst-month method. Run 04c first.")
    else:
        log("  [WARN] No Tm_target_capped_C available (run 04c_charging_ceiling_assam.py). "
            "Profile will carry NaN; Phase 6 will refuse to run.")
'''),
 # 2/3a. HSI proxy removal
 ('''        hsi_val = sub["HSI"].mean() if "HSI" in sub.columns else sub["RH_mean"].mean()
''',
  '''        hsi_sunrise_val = sub["HSI_sunrise"].mean() if "HSI_sunrise" in sub.columns else np.nan
        hsi_legacy_val = sub["HSI"].mean() if "HSI" in sub.columns else np.nan
'''),
 # 2. capped fallback removal
 ('''            "Tm_target_capped_C": (sub["Tm_target_capped_C"].mean()
                                    if "Tm_target_capped_C" in sub.columns else TM_TARGET),
''',
  '''            "Tm_target_capped_C": (sub["Tm_target_capped_C"].mean()
                                    if "Tm_target_capped_C" in sub.columns else np.nan),
'''),
 # 3b. HSI columns
 ('''            "HSI": hsi_val,
            "HSI_mean": hsi_val,
''',
  '''            "HSI_sunrise": hsi_sunrise_val,
            "HSI": hsi_legacy_val,
            "HSI_mean": hsi_legacy_val,
'''),
 # 4. completeness check
 ('''    profile_df = pd.DataFrame(profile_rows)
''',
  '''    profile_df = pd.DataFrame(profile_rows)
    for _col in ("Tm_target_capped_C", "HSI_sunrise"):
        if profile_df[_col].isna().any():
            log(f"  [WARN] cluster profile column {_col} is NaN (upstream value missing; "
                f"NOT substituted by a proxy).")
'''),
]

for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times (need exactly 1):\n{old[:120]}")
for old, new in EDITS:
    text = text.replace(old, new)

shutil.copy(path, str(path) + ".bak")
path.write_bytes(text.replace("\n", nl).encode("utf-8"))
print(f"Patched {path} ({len(EDITS)} edits). Backup: {path}.bak")