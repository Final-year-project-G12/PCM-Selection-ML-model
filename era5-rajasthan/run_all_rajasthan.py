"""
run_all_rajasthan.py
=============================================================================
Runs every stage of the Rajasthan pipeline, in the correct dependency
order, in one invocation — via `subprocess`, exactly as if you'd typed
`python <script>.py` for each one yourself in sequence (no shared Python
process, no import side effects between stages — each script starts fresh,
same as running it manually).

REQUIRED LIBRARIES: none beyond the standard library (subprocess/pathlib/
argparse/time) — this file itself has no third-party dependencies. Each
individual stage script has its own requirements (see that script's own
docstring); this runner doesn't install anything, it just invokes
`python <script>.py` with whatever interpreter you ran this with.

WHAT THIS DOES NOT RUN, AND WHY:

  A. RAW-DATA ACQUISITION (SETUP) — run only with --include-setup:
     00_unzip_accum.py, 00a_build_population_grid.py, 00b_build_suntimes.py,
     00c_attach_elevation.py, 01_download_era5_rajasthan.py,
     01b_download_nasapower.py
     — ONE-TIME steps: they hit external APIs (CDS/ERA5, NASA POWER), need
     credentials (.cdsapirc) this runner has no way to verify, can take
     hours, and would re-download/overwrite the raw archive this whole
     pipeline is built on if re-run casually. NOT part of the "re-run the
     pipeline" cycle. Pass --include-setup to include them (they run first,
     in their own correct order) — read that flag's own warning below.

  B. NEVER RUN by this runner (not a --flag away — genuinely excluded):
     - config.py, cluster_lib.py, signature_lib.py, physics_lib.py,
       provenance_lib.py, run_all_rajasthan.py — import-only library
       modules and this file itself, not runnable stages.
     - 05c_explore_interactive.py — a Streamlit app; launch it with
       `streamlit run 05c_explore_interactive.py`, not plain `python`.
     - 03b_validate_quality_fix_rajasthan.py — deprecated (kept on disk
       with a banner): both its premises stopped holding once Phase 3
       started reading 04_preprocess_rajasthan.py's output.
     - check_supercooling_data.py, check_supercooling_K.py — throwaway
       read-only diagnostics from the Phase 8 field-identification work;
       both hardcode absolute d:/Final Year Project/... paths, so they are
       not portable tooling. Run by hand if you need them.
     - plotting/__init__.py — an empty package marker, not a script.
     - PLOTSV2/fix_unicode_issues.py — a one-off text-repair utility for
       mojibake in already-generated plot files; run by hand if needed.
     - PLOTSV2/interactive_plots/*.py — Streamlit / interactive-explorer
       apps (00e/00f/03e/03f/04e/04f + 03b_interactive_raw_qa); launch
       with `streamlit run <file>`, not plain `python`.

  C. THE OBJECTIVE-1 PLOT SET — run only with --with-plots:
     PLOTSV2/run_all_plots_v2.py  (the maintained plots orchestrator).
     It runs, each in its own process so one failure doesn't take the
     rest down: generate_rajasthan_plots.py (13 objective-1 figures),
     the four verify_0{1,2,3,4}_*_rajasthan.py suites,
     phase1_data_collection_rajasthan.py + phase3_climate_signature_
     rajasthan.py, the four comparison_*/09_mcdm_vs_physics_agreement.py
     cross-step plots, then build_plots_folder_rajasthan.py to assemble
     the curated PLOTSV2/Plots/ folder. Delegating to it (rather than
     re-listing 20 scripts here) keeps the two runners from drifting.
     Non-blocking and runs LAST, after the core chain + --with-optional.
     Phase 1's figure reads the ~1.4 GB raw points CSV, so this adds a
     few minutes.

ORDER AND WHY (CORE stages — required, this runner STOPS at the first
failure since every later stage reads an earlier stage's output):
  1. 02_combine_rajasthan.py            — raw ERA5 events -> climate_
                                           rajasthan_points.csv
  2. 02b_build_daily_aggregates.py      — raw NASA POWER hourly ->
                                           daily_aggregates_rajasthan*.csv
                                           (independent of step 1 — reads
                                           raw NASA POWER directly — but
                                           sequenced after it for a single
                                           linear log to read)
  3. 04_preprocess_rajasthan.py         — Phase 2.5: BOUNDS screen + SZA>=90
                                           night-mask + per-season quantile
                                           map (ERA5->POWER, persisted) +
                                           Hampel + 4-stage/MICE impute ->
                                           data/preprocessed/
                                           rajasthan_cleaned_physical.csv
                                           (converged onto the Tamil Nadu
                                           04_preprocess contract; replaces
                                           03b_quality_check_rajasthan.py,
                                           now diagnostic-only)
  4. 04b_climate_signature.py           — Phase 3: two-tier signature ->
                                           climate_signature_rajasthan.csv
                                           (real filename — the former
                                           "04_climate_signature_rajasthan.py"
                                           entry named a nonexistent file,
                                           so Phase 3 was silently SKIPped
                                           by this runner)
  5. 05_cluster_rajasthan.py            — Phase 4: GMM clustering ->
                                           cluster_profiles_rajasthan.csv
                                           (canonical-relabeled, see that
                                           script's 2026-08-11 fix)
  6. 07_feasibility_filter.py           — Phase 5: 8-constraint filter ->
                                           feasibility_survivors_*.csv
  7. 08_mcdm_ranking.py                 — Phase 6: 4-method MCDM + Monte
                                           Carlo -> mcdm_full_rankings.csv,
                                           mcdm_topk_by_cluster.csv,
                                           monte_carlo_stability.csv,
                                           mcdm_method_agreement.csv
                                           (UNIFIED with Tamil Nadu 2026-09-08)
  8. 10_physics_validation.py           — Phase 7: physics simulation ->
                                           physics_validation_*.csv
                                           (RENUMBERED 2026-09-08 to match
                                           Tamil Nadu — was
                                           09_physics_validation_rajasthan.py)
  9. 09_recommendation_cards.py         — Phase 8 deliverable: pure
                                           aggregation -> recommendation_
                                           cards_rajasthan.md
                                           (was 10_recommendation_cards_rajasthan.py;
                                           numbered 09 but runs LAST, after
                                           Phase 7 — same as Tamil Nadu)

  Steps 6-9 (Phase 5-8) each independently hard-fail via provenance_lib.py
  if their input was built from a DIFFERENT on-disk cluster_profiles_
  rajasthan.csv than the one currently present — running this script
  start-to-finish in one pass is exactly what keeps that chain consistent
  (see provenance_lib.py's module docstring for the 2026-08-11 bug this
  guards against).

  NOTE on step 1: 02_combine_rajasthan.py is the heaviest stage (it
  processes a multi-GB combined ERA5-events file) and its output
  climate_rajasthan_points.csv changes only when the raw ERA5/POWER
  archive changes. It IS a core step (re-added 2026-09-09; it had been
  silently commented out, mirroring a bug already fixed in
  run_all_tamilnadu.py). To re-run the pipeline WITHOUT redoing it, use
  --from 02b_build_daily_aggregates.py (or any later stage).

POST-PHASE analysis (run AFTER the core chain, non-blocking — a late
failure must not invalidate the completed Phase 5-8 deliverables):
  11_seasonal_pcm_sensitivity.py    — post-Phase-6 seasonal re-ranking:
                                       does each cluster's #1 PCM flip by
                                       season? (reads 08's mcdm_full_
                                       rankings.csv + 06's PCM DB)
  08_phase8_supercooling_sweep.py   — Phase 8 sensitivity sweep: re-runs
                                       the Phase 7 experiment with a
                                       supercooling penalty on h_p at
                                       k in [0.0,0.1,0.2,0.3] and reports
                                       how per-cluster Spearman rho moves
                                       (reads 06's + 05's outputs; imports
                                       physics_lib.py). Its own provenance
                                       check hard-fails on a cluster_
                                       profiles mismatch, same as the core
                                       chain.

OPTIONAL / DIAGNOSTIC stages (run only with --with-optional, AFTER the
core chain — none of steps 1-9 read their output, so they cannot break
the core chain; each one's failure is logged and does NOT stop the run):
  00d_population_grid_viz.py (population-grid sample-point map — needs
    00a's population_grid_points.csv),
  03_verify_climate_csv.py, 03_plots_raw.py, 03b_agreement_analysis.py,
  03b_quality_check_rajasthan.py (SUPERSEDED as Phase 2.5 — kept as a
    standalone diagnostic that writes its own climate_rajasthan_points_
    clean.csv + quality_report_rajasthan.{md,json}; nothing in the core
    chain reads them),
  03b_quality_check_plots_rajasthan.py, 03b_interactive_raw_qa.py,
  04c_postprocess_plots.py, 04c_interactive_postprocess_qc.py,
  05b_cluster_interactive.py, 05d_plots_comprehensive.py

HOW TO RUN:
  python run_all_rajasthan.py                 # core pipeline only (default)
  python run_all_rajasthan.py --with-optional  # core + QC/diagnostic scripts
  python run_all_rajasthan.py --with-plots     # core + the full PLOTSV2 plot set
  python run_all_rajasthan.py --with-optional --with-plots   # everything
  python run_all_rajasthan.py --dry-run        # print the order, run nothing
  python run_all_rajasthan.py --include-setup  # ALSO run the 00/01 raw-data
                                                # acquisition scripts FIRST —
                                                # read the warning above and
                                                # in --include-setup's own
                                                # section below before using
  python run_all_rajasthan.py --from 05_cluster_rajasthan.py
                                                # resume the CORE chain
                                                # starting at a given script
                                                # (skips everything before
                                                # it) — useful after fixing
                                                # one stage without wanting
                                                # to re-run everything
                                                # upstream of it again

This can take a while end to end — 02_combine_rajasthan.py alone processes
a multi-GB combined events file, and 08 (Monte Carlo) + 10 (full-year
hourly simulation) are each minutes. Expect several minutes to ~30+ minutes
for the full core chain depending on machine speed; this is not a hang.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# One-time raw-data acquisition — excluded by default, see docstring.
SETUP_SCRIPTS = [
    "00_unzip_accum.py",
    "00a_build_population_grid.py",
    "00b_build_suntimes.py",
    "00c_attach_elevation.py",
    "01_download_era5_rajasthan.py",
    "01b_download_nasapower.py",
]

# Required, sequential, stop-on-failure — every later stage reads an
# earlier one's output (see docstring for the full dependency reasoning).
CORE_SCRIPTS = [
    "02_combine_rajasthan.py",           # Phase 1 final step — re-added 2026-09-09; it had
                                         # been silently commented out (same bug already fixed
                                         # in run_all_tamilnadu.py). Heaviest stage; use
                                         # --from 02b_build_daily_aggregates.py to skip it when
                                         # the raw ERA5/POWER archive hasn't changed.
    "02b_build_daily_aggregates.py",
    "04_preprocess_rajasthan.py",       # Phase 2.5 — was 03b_quality_check_rajasthan.py;
                                         # converged onto the Tamil Nadu 04_preprocess contract
                                         # (BOUNDS + SZA night-mask + per-season quantile map
                                         # persisted + 4-stage/MICE imputation).
    "04b_climate_signature.py",         # Phase 3 — real filename; the old
                                         # "04_climate_signature_rajasthan.py" entry named a file
                                         # that does not exist, so run_script() silently SKIPped
                                         # Phase 3 in the core chain.
    "05_cluster_rajasthan.py",
    "05a_level_b_regime_shift_rajasthan.py",   # Phase 4 Level B — regime
                                         # shift. Extracted 2026-09-08 from
                                         # 05_cluster_rajasthan.py. Rebuilds
                                         # its own per-point-per-season Tier-1
                                         # signature, so it does not read
                                         # Level A's output and nothing below
                                         # reads its output.
    "07_feasibility_filter.py",          # real filename — the old
                                         # "07_feasibility_filter_rajasthan.py" entry named a file
                                         # that does not exist, so run_script() silently SKIPped
                                         # Phase 5 (same bug class as the former Phase 3 entry).
    "08_mcdm_ranking.py",               # real filename — ditto; the old
                                         # "08_mcdm_ranking_rajasthan.py" entry was silently SKIPped.
    # Phase 7 then Phase 8 — RENUMBERED 2026-09-08 to match Tamil Nadu's
    # convention exactly (physics validation = 10, recommendation cards =
    # 09; cards still run AFTER physics because they aggregate its output).
    "10_physics_validation.py",         # was 09_physics_validation_rajasthan.py
    "09_recommendation_cards.py",       # was 10_recommendation_cards_rajasthan.py
]

# Post-Phase supplementary analysis — run after the core chain, non-
# blocking (a late failure must not invalidate the completed deliverables).
# 11 mirrors run_all_tamilnadu.py's counterpart; 08_phase8_supercooling_
# sweep.py is Phase 8's sensitivity analysis (re-runs Phase 7 with a
# supercooling h_p penalty at k in [0.0,0.1,0.2,0.3]) — it has no Tamil
# Nadu twin and was previously missing from this runner entirely.
POST_PHASE6_SCRIPTS = [
    "11_seasonal_pcm_sensitivity.py",
    "08_phase8_supercooling_sweep.py",
]

# Core scripts that run IN core order but must NOT abort the chain on
# failure. 05a (Level B regime shift) is a supplementary temporal analysis
# sequenced in Phase 4 order; nothing downstream reads its output, so a
# failure there must not cost the completed Phase 5-8 deliverables. Matches
# how run_all_tamilnadu.py marks its equivalent required=False.
NON_BLOCKING_CORE = {"05a_level_b_regime_shift_rajasthan.py"}

# The full Objective-1 plot set — run only with --with-plots. Delegates to
# the maintained PLOTSV2 orchestrator so the two runners can't drift; it
# fans out to ~20 plot scripts (13 objective-1 figures, 4 verify suites,
# 2 phase figures, 4 cross-step comparisons, + the curated-folder builder).
# Non-blocking, runs LAST. NOT covered: PLOTSV2/interactive_plots/*.py
# (Streamlit apps — `streamlit run <file>`) and PLOTSV2/fix_unicode_issues.py
# (one-off repair utility).
PLOT_SCRIPTS = [
    "PLOTSV2/run_all_plots_v2.py",
]

# QC/plotting/diagnostic — run after the core chain, continue-on-failure,
# nothing in CORE_SCRIPTS reads any of these scripts' output.
OPTIONAL_SCRIPTS = [
    "00d_population_grid_viz.py",               # population-grid sample-point map; reads 00a's
                                                 # population_grid_points.csv. Was missing from
                                                 # this runner entirely (added 2026-09-09).
    "03_verify_climate_csv.py",
    "03_plots_raw.py",                          # raw-data QA (static PNG) — direct port of
                                                 # era5-tamilnadu/03_plots_raw.py; output to PLOTSV2/raw/
    "03b_agreement_analysis.py",
    "03b_quality_check_rajasthan.py",           # SUPERSEDED as Phase 2.5 by
                                                 # 04_preprocess_rajasthan.py; kept here as a
                                                 # standalone diagnostic (writes its own
                                                 # climate_rajasthan_points_clean.csv + quality
                                                 # report; nothing in the core chain reads them).
    "03b_quality_check_plots_rajasthan.py",
    # Tamil-Nadu plot-script ports (Group A) — direct ports of
    # era5-tamilnadu/{03_plots_raw,03b_interactive_raw_qa,04c_postprocess_plots,
    # 04c_interactive_postprocess_qc}.py, output to PLOTSV2/<subfolder>/.
    # 05c_explore_interactive.py is excluded here: it is a Streamlit app, run
    # with `streamlit run 05c_explore_interactive.py`, not plain `python`.
    "03b_interactive_raw_qa.py",
    "04c_postprocess_plots.py",
    "04c_interactive_postprocess_qc.py",
    # 04f_signature_interactive.py deleted 2026-09-08 — non-core read-only
    # signature explorer, no downstream dependents (same call as dropping the
    # other plotting scripts from the core chain). Its 1:1 Tamil Nadu twin
    # (04d_signature_interactive.py) was deleted in the same pass.
    "05b_cluster_interactive.py",
    "05d_plots_comprehensive.py",
    # 03b_validate_quality_fix_rajasthan.py removed: it diffed a climate
    # signature built from climate_rajasthan_points_clean.csv against the
    # pre-quality-check version and internally re-ran a script name
    # ("04_climate_signature_rajasthan.py") that does not exist. Neither
    # premise holds now that Phase 3 reads 04_preprocess_rajasthan.py's
    # output. Kept on disk with a deprecation banner; not run here.
]


def run_script(name, stop_on_failure):
    path = BASE_DIR / name
    if not path.exists():
        print(f"  [SKIP] {name} — file not found at {path}")
        return "skipped", 0.0

    print("\n" + "=" * 68)
    print(f"  RUNNING: {name}")
    print("=" * 68)
    t0 = time.time()
    # Run each script from its OWN directory — matters for the PLOTSV2/
    # scripts; a no-op for the top-level stages (they all sit in BASE_DIR).
    result = subprocess.run([sys.executable, str(path)], cwd=str(path.parent))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] {name} finished in {elapsed:.1f}s")
        return "ok", elapsed

    print(f"\n  [FAILED] {name} exited with code {result.returncode} after {elapsed:.1f}s")
    if stop_on_failure:
        print(f"\n  Stopping — every stage after {name} in the core chain reads its output, "
              f"so continuing would run against stale/missing data.")
    return "failed", elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Rajasthan pipeline end-to-end, in order.")
    parser.add_argument("--with-optional", action="store_true",
                         help="Also run the QC/diagnostic scripts after the core chain.")
    parser.add_argument("--with-plots", action="store_true",
                         help="Also run the full PLOTSV2 plot set (via PLOTSV2/run_all_plots_v2.py) "
                              "LAST, non-blocking. Adds a few minutes (Phase 1's figure reads the "
                              "~1.4 GB raw points CSV).")
    parser.add_argument("--include-setup", action="store_true",
                         help="Also run the one-time raw-data download/setup scripts FIRST. "
                              "These hit external APIs and need credentials — see this file's "
                              "own docstring before using this flag.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved run order and exit without running anything.")
    parser.add_argument("--from", dest="from_script", default=None,
                         help="Resume the CORE chain starting at this script name "
                              "(e.g. 05_cluster_rajasthan.py), skipping everything before it. "
                              "Does not affect --include-setup / --with-optional / --with-plots stages.")
    args = parser.parse_args()

    core = list(CORE_SCRIPTS)
    if args.from_script:
        if args.from_script not in core:
            print(f"ERROR: --from {args.from_script!r} is not one of the core scripts: {core}")
            sys.exit(2)
        core = core[core.index(args.from_script):]

    setup = list(SETUP_SCRIPTS) if args.include_setup else []
    optional = list(OPTIONAL_SCRIPTS) if args.with_optional else []
    plots = list(PLOT_SCRIPTS) if args.with_plots else []

    print("=" * 68)
    print("  RAJASTHAN PIPELINE — RUN ORDER")
    print("=" * 68)
    if setup:
        print("\n  SETUP (one-time, external APIs — --include-setup was passed):")
        for s in setup:
            print(f"    {s}")
    print("\n  CORE (required, stop-on-first-failure):")
    for s in core:
        tag = "  (non-blocking)" if s in NON_BLOCKING_CORE else ""
        print(f"    {s}{tag}")
    print("\n  POST-PHASE-6 (supplementary, continue-on-failure):")
    for s in POST_PHASE6_SCRIPTS:
        print(f"    {s}")
    if optional:
        print("\n  OPTIONAL / DIAGNOSTIC (continue-on-failure):")
        for s in optional:
            print(f"    {s}")
    else:
        print("\n  (Optional/diagnostic scripts skipped — pass --with-optional to include them.)")
    if plots:
        print("\n  PLOTS (continue-on-failure, runs LAST):")
        for s in plots:
            print(f"    {s}   (fans out to the full PLOTSV2 set)")
    else:
        print("\n  (Plot set skipped — pass --with-plots to include it.)")

    if args.dry_run:
        print("\n--dry-run: exiting without running anything.")
        return

    t_start = time.time()
    log = []

    for name in setup:
        status, elapsed = run_script(name, stop_on_failure=True)
        log.append((name, status, elapsed))
        if status == "failed":
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    for name in core:
        status, elapsed = run_script(name, stop_on_failure=name not in NON_BLOCKING_CORE)
        log.append((name, status, elapsed))
        if status == "failed" and name in NON_BLOCKING_CORE:
            print(f"\n  [NON-BLOCKING] {name} failed but is not required by any later "
                  f"stage — continuing.")
            continue
        if status == "failed":
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    # Post-Phase-6 supplementary analysis, non-blocking — mirrors how
    # run_all_tamilnadu.py sequences 11_seasonal_pcm_sensitivity.py last.
    for name in POST_PHASE6_SCRIPTS:
        status, elapsed = run_script(name, stop_on_failure=False)
        log.append((name, status, elapsed))

    for name in optional:
        status, elapsed = run_script(name, stop_on_failure=False)
        log.append((name, status, elapsed))

    # Full PLOTSV2 plot set — LAST, non-blocking. Delegated to the PLOTSV2
    # orchestrator so this runner and that one can't drift.
    for name in plots:
        status, elapsed = run_script(name, stop_on_failure=False)
        log.append((name, status, elapsed))

    print_summary(log, time.time() - t_start)


def print_summary(log, total_elapsed):
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    for name, status, elapsed in log:
        tag = {"ok": "OK    ", "failed": "FAILED", "skipped": "SKIP  "}[status]
        print(f"  [{tag}] {name:45s} {elapsed:7.1f}s")
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    n_failed = sum(1 for _, s, _ in log if s == "failed")
    if n_failed:
        print(f"  {n_failed} stage(s) FAILED — see the log above for the first failure's output.")
    else:
        print("  All stages completed (or were intentionally skipped).")
    print("=" * 68)


if __name__ == "__main__":
    main()
