# 13 — Reproducibility Audit

## The two dominant reproducibility risks, both more severe than Rajasthan's

1. **Filename mismatch across the entire folder.** Any attempt to reproduce this pipeline by running
   files in their on-disk name order (`python 00_unzip_accum.py` etc.) will silently execute the
   wrong script. This is the single most severe reproducibility hazard found across either state's
   audit — worse than any issue found in Rajasthan, because it affects *discoverability*, not just
   correctness: a new reader cannot even identify the right entry point without the correspondence
   table in `01_FILENAME_CORRESPONDENCE.md`. **Fix (already recommended by the project's own
   README): rename every file before doing anything else.**
2. **Never executed.** Reproducibility in the conventional sense (can a second person get the same
   result) is not yet a meaningful question for this pipeline — there is no first result to
   reproduce. The relevant reproducibility question right now is narrower: *is the code itself,
   read cold, sufficient to produce a correct result once run?* — and the answer is "mostly yes,
   with the Phase 3 `L_required` fix required first."

## Standard checklist

| Item | Status | Notes |
|---|---|---|
| Random seeds | **PASS** | `random_state=42` used consistently across imputation, clustering, MCDM — same convention as Rajasthan |
| Dataset version | **PARTIAL** | Same gap as Rajasthan — no pinned ERA5 product version/pull-date manifest |
| Geographic coordinates | **PASS (design)** | Deterministic GADM+WorldPop+ERA5-aligned-grid algorithm, same method as Rajasthan |
| API parameters | **PASS** | Version-controlled in `.py` files (once correctly identified/renamed) |
| Time ranges | **PASS** | 2016-01-01..2025-12-31, consistent with Rajasthan |
| Dependency versions | **FAIL** | No `requirements.txt`/lockfile found under `tamilnadu/` |
| Environment | **FAIL** | Same gap as Rajasthan |
| Output naming | **N/A yet** | No outputs exist to assess |
| Logging | **PASS (design)** | Same `StatusTracker` pattern as Rajasthan for the two downloader scripts |
| Preprocessing rules | **PASS** | Deterministic, code-defined 13-step pipeline, well-documented |

## The filename-mislabeling root cause, and what it implies for future work

The project's own README attributes the mismatch to a browser's duplicate-download auto-suffixing
(`name (1).ext`, `name (2).ext`) combined with the files being re-associated with the wrong original
name afterward — plausible and consistent with the `(N)` suffix pattern seen on almost every file.
**This means the underlying content is trustworthy** (each file's actual code/data is presumably
what was intended, just mislabeled) — the risk is purely about *which file a human or script picks
up by name*, not about corrupted or fabricated content. Still, given this has already happened once
in this project, the same failure mode (bulk one-at-a-time downloads from a chat/canvas UI) should be
avoided for any future file transfers into this project, or files should be renamed immediately upon
receipt rather than left with browser-assigned names.

## Recommended fixes, in order

1. **Rename every file per `01_FILENAME_CORRESPONDENCE.md`** — zero-risk per the project's own
   timestamp analysis, and a prerequisite for anyone else being able to use this pipeline correctly.
2. **Fix the `L_required` bug** in (the file that should be renamed to) `04b_climate_signature.py`.
3. **Run the pipeline once, end-to-end**, checking the deaccumulation assumption and the GMM
   covariance-type risk explicitly as part of that first run (see `11_IMPLEMENTATION_ISSUES.md`
   items 3–4).
4. **Add a `requirements.txt`** — same recommendation as Rajasthan.
