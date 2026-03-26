# Issues Log — `climate` branch

> Bugs were present in **both** `data_1/` and `data_2/` scripts. All have been fixed.

## `data_2/era5_combine_to_csv.py`

---

### ✅ [FIXED] BUG-01 — Wrong working directory for file glob (Critical)
**Severity:** 🔴 Critical  
**Line:** 242  
**Description:** `glob.glob("era5_2024_*.nc")` used a relative path, causing immediate failure (`❌ No era5_2024_*.nc files found.`) when the script was run from any directory other than `data_2/`.  
**Fix:** Added `SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))` at module level and changed glob to `glob.glob(os.path.join(SCRIPT_DIR, "era5_2024_*.nc"))`.

---

### ✅ [FIXED] BUG-02 — `ssrd` not deaccumulated before GHI conversion (Silent data error)
**Severity:** 🔴 Critical  
**Line:** 173–175  
**Description:** ERA5 `ssrd` is a cumulative accumulated quantity (J/m²), not an instantaneous flux. Dividing raw `ssrd` / 3600 without first differencing consecutive rows produces incorrect (inflated) GHI values that grow monotonically through the day.  
**Fix:** Sort by time, apply `.diff().clip(lower=0)` to deaccumulate, then divide by 3600 to get W/m².

---

### ✅ [FIXED] BUG-03 — Engine fallback opens file twice and discards result
**Severity:** 🟡 Medium  
**Lines:** 54–60  
**Description:** The engine-detection loop called `xr.open_dataset(filepath, engine=eng)` but discarded the returned dataset, then fell through to open the file a second time. On Windows this can cause file-handle contention and is wasteful.  
**Fix:** Changed the loop body to `return xr.open_dataset(filepath, engine=eng)` directly.

---

### ✅ [FIXED] BUG-04 — Month key extraction fragile for `data_2` filenames
**Severity:** 🟡 Medium  
**Lines:** 254–257  
**Description:** `os.path.basename(f).split('_')[2][:2]` relied on positional splitting that coincidentally worked for `01`–`12` but would silently misfill keys for any unexpected naming pattern.  
**Fix:** Replaced with `re.match(r'era5_\d{4}_(\d{2})', ...)` for explicit, robust month extraction.

---

### ✅ [FIXED] BUG-05 — Output CSV written to CWD instead of script directory
**Severity:** 🟡 Medium  
**Line:** 323  
**Description:** `output = 'era5_climate_coimbatore_2024.csv'` wrote the CSV to the current working directory, not to `data_2/`.  
**Fix:** Changed to `output = os.path.join(SCRIPT_DIR, 'era5_climate_coimbatore_2024.csv')`.

---

## `data_2/era5_download.py`

---

### ✅ [FIXED] BUG-06 — Downloaded files scattered to CWD
**Severity:** 🟡 Medium  
**Lines:** 12–13, 66–69  
**Description:** `output_nc` and `output_zip` used bare filenames, causing all downloaded `.nc` / `.zip` files to land in the current working directory instead of `data_2/`. Extracted ZIP contents were also renamed relative to CWD.  
**Fix:** Added `SCRIPT_DIR` and prefixed all output paths with `os.path.join(SCRIPT_DIR, ...)`. ZIP extraction target changed from `'.'` to `SCRIPT_DIR`.

---

## Summary

| ID | File | Severity | Status |
|---|---|---|---|
| BUG-01 | `era5_combine_to_csv.py` | 🔴 Critical | ✅ Fixed |
| BUG-02 | `era5_combine_to_csv.py` | 🔴 Critical | ✅ Fixed |
| BUG-03 | `era5_combine_to_csv.py` | 🟡 Medium | ✅ Fixed |
| BUG-04 | `era5_combine_to_csv.py` | 🟡 Medium | ✅ Fixed |
| BUG-05 | `era5_combine_to_csv.py` | 🟡 Medium | ✅ Fixed |
| BUG-06 | `era5_download.py` | 🟡 Medium | ✅ Fixed |