# 09 — ERA5 Data Pipeline: Deep Audit

This is the mandatory audit checkpoint: the deaccumulation story. Everything else in this file is
secondary to understanding this one function correctly, because it determined whether the entire
downstream pipeline (Phases 3–6) was built on physically valid GHI data.

## The variable transformation table

See `02_DATA_SOURCES_AND_VARIABLES.md` for the full table. This file focuses on *why* the
accumulated-field handling looks the way it does.

## Accumulated ERA5 variables: what the code assumes vs. what it found

`ssrd` (solar radiation), `strd` (thermal radiation), `tp` (precipitation) are ERA5's canonically
"accumulated" fields. The classic ERA5/MARS convention is that these are **cumulative since the last
forecast reset** (00Z or 12Z), requiring a `diff()` against the previous hour to recover an
hourly flux, with a special case at the first hour after each reset (where the raw value already *is*
the flux, since there's nothing to diff against).

### What the pipeline originally assumed

An earlier function, `deaccumulate()`, implemented exactly that: `diff()` against the previous
downloaded hour, with `reset_mask = hour.isin([1, 13])` treating hours 1 and 13 as the post-reset
special case. `01_download_era5_rajasthan.py` was built around this — it deliberately downloads each
target hour's immediate predecessor (`ACCUM_HOURS = INSTANT_HOURS ∪ {(h-1) mod 24 for h in
INSTANT_HOURS}`) specifically to feed this diff.

### What was actually found

`03b_agreement_analysis.py` flagged the ERA5-vs-POWER GHI comparison as physically implausible:
median ERA5 GHI ~2 W/m² against NASA POWER's ~37 W/m² at the same instants, noon Pearson r≈0.01.
Tracing back to the raw NetCDF (checked across 11 (year, month) samples spanning 2016–2025, every
season) showed **34–44% of consecutive-hour raw values were *lower* than their predecessor within
the same nominal accumulation cycle** — impossible for a genuine cumulative-since-reset field (which
can only increase monotonically until the next reset). The conclusion: **each downloaded hour, for
this specific pipeline's CDS request configuration, is already its own ~1-hour accumulated value,
not a running total.**

### The fix — `accum_to_flux()`, `02_combine_rajasthan.py`

```python
def accum_to_flux(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    return s.clip(lower=0)
```
**No diffing at all.** The entire operational content is a stateless clip-to-nonnegative. No
reset-hour special case, no month/day-boundary handling, no memory of adjacent time steps. The
function was deliberately **renamed** away from `deaccumulate()` specifically so a future edit would
not casually reintroduce a diff step without first re-running the raw-vs-diffed comparison that
justified this change.

### Verification of the fix

Skipping the diff reproduces physically correct GHI, with seasonal peaks tracking Rajasthan's known
climatology (~900 W/m² pre-monsoon, ~700 W/m² monsoon, ~650 W/m² winter — physically sensible
magnitudes for this latitude/season). Post-fix, solar-noon ERA5-vs-POWER agreement is **MBE=10.95
W/m², RMSE=113.8 W/m², Pearson r=0.810** (n=1,168,960) — a categorical improvement from the pre-fix
state (r≈0.01, near-zero GHI). See `14_ERA5_POWER_VALIDATION.md` for the full statistics and the
decision this enabled.

### What is asserted but not independently re-verifiable from the code alone

The 34–44%/11-samples claim is documented only in the function's own docstring, referencing an
external investigation that is not itself reproducible from any code present in the repository — no
script exists that re-runs this specific raw-vs-diffed diagnostic on demand. **Recommendation for
the write-up**: either preserve the ad hoc diagnostic script (if one exists outside this session's
read scope) or note this as an investigation that should be re-run and captured in a script before
final publication, so the claim is independently reproducible, not only narratively documented.

## Does deaccumulation risk negative values, and is clipping hiding a real bug?

Yes to both questions, addressed directly: `accum_to_flux()`'s `clip(lower=0)` would silently absorb
a genuinely negative raw value regardless of cause. Given the confirmed finding that raw values are
already per-hour fluxes (not diffs), a negative raw value would indicate either a genuine ERA5
data artifact or a download/parsing issue — the clip cannot distinguish these from ordinary
noise-floor negative values near zero irradiance (which are common and benign in solar radiation
products). The pipeline does not log *how often* the clip actually fires or by how much, which would
be a cheap, valuable addition (a printed count of clipped rows and their magnitude) for
transparency in the methodology write-up.

## Unit conversion correctness

`GHI = accum_to_flux(ssrd)/3600` (J/m² → W/m², assuming the raw value is a 1-hour Joule
accumulation — consistent with the "already ~1-hour accumulated" finding). `LW_down` identical
treatment. `precipitation = accum_to_flux(tp)×1000` (m → mm). These are standard, correct
conversions **given** the now-verified "already per-hour" premise.

## The one unresolved inconsistency: `avg_sdirswrf` / DNI surrogate

`avg_sdirswrf` is populated from whichever of `msdwswrf`, `fdir`, or `msdrswrf` matches first in the
downloaded file, and receives only `.clip(0)` — **never divided by 3600, regardless of which name
matched.** These are not interchangeable ERA5 fields: `fdir` is an accumulated field (needs the same
`/3600` treatment `ssrd`/`strd` get); `msdwswrf`/`msdrswrf` are mean-rate fields (already W/m², no
conversion needed). The code's uniform treatment is only correct if the field that actually matched
is always one of the mean-rate variants — this was not independently verified against the raw NetCDF
variable names actually present in the downloaded files during this audit. **This is the pipeline's
one remaining plausible unit-error risk and should be checked before DNI is presented as
fully validated** — see `13_SOLAR_DERIVED_VARIABLES.md` and `20_IMPLEMENTATION_ISSUES.md` item 8.

## Literature support

Hersbach et al. (2020), *QJRMS* 146(730), "The ERA5 global reanalysis" — the ERA5 product citation.
ECMWF's own IFS documentation on accumulated-field conventions is the correct Tier-1 citation for
what the *classic* MARS convention is (used here to explain what the code originally assumed); this
audit did not independently fetch ECMWF's documentation to verify the specific claim that *this
pipeline's particular CDS request* returns non-cumulative hourly values — that conclusion rests on
the project's own empirical investigation (the 11-sample, 34–44% finding), which is internally
consistent with the observed downstream results (physically plausible GHI magnitudes, r=0.81
agreement with an independent source) but should be described in a write-up as an *empirically
determined* pipeline-specific behavior, not asserted as a general ERA5/CDS API fact applicable to
every possible request configuration.
