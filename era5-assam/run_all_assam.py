"""
run_all_assam.py
=============================================================================
Runs every stage of the Assam pipeline, in the correct dependency order, in
one invocation — via `subprocess`, exactly as if you'd typed
`python <script>.py` for each one yourself in sequence (no shared Python
process, no import side effects between stages — each script starts fresh,
same as running it manually). Mirrors era5-tamilnadu/run_all_tamilnadu.py
and era5-rajasthan/run_all_rajasthan.py in structure and flags.

REQUIRED LIBRARIES: none beyond the standard library (subprocess/pathlib/
argparse/time) — this file itself has no third-party dependencies. Each
individual stage script has its own requirements; this runner doesn't
install anything, it just invokes `python <script>.py` with whatever
interpreter you ran this with.

WHY ASSAM IS DIFFERENT FROM TAMIL NADU / RAJASTHAN — TWO PARALLEL PCM
CHAINS (read this before running unattended):

  Per docs/assam/00_MASTER_OVERVIEW.md, Assam's clustering was re-locked
  from an exploratory K=4 model to a final, audited K=3 model AFTER the PCM
  database/feasibility/MCDM scripts had already been run and audited once.
  Rather than silently overwrite that historical run, the repo keeps BOTH
  chains on disk as separate files:

    HISTORICAL K=4 chain (plain filenames — feeds Phase 9 physics + the
    recommendation cards):
      06_build_pcm_database.py      -> pcm_database_assam.csv (25 rows)
      07_feasibility_filter.py      -> feasibility_survivors_assam.csv
                                        (8 PCM survivors under K=4)
      08_mcdm_ranking.py            -> mcdm_topk_assam.csv,
                                        mcdm_full_scores_assam.csv
                                        (TOPSIS/GRA/PROMETHEE II/VIKOR +
                                        5,000-draw Monte Carlo; RT44HC #1
                                        across all 4 clusters)
      09_recommendation_cards.py    -> recommendation_cards_assam.md
                                        (reads the historical mcdm_topk +
                                        feasibility_survivors, PLUS Phase 9's
                                        physics results)

    FINAL K=3 GOVERNED chain (_final filenames — the locked, audited
    pipeline actually reported as Assam's Phase 5-8 result):
      06_build_pcm_database_final.py -> pcm_database_final.csv (58 rows,
                                        deduplicated, strict provenance)
      07_feasibility_filter_final.py -> pcm_feasibility_by_cluster.csv +
                                        pcm_feasibility_summary.csv
                                        (n_confirmed=[0,0,0]; 1 conditional
                                        candidate, n-Tetracosane C24)
      08_mcdm_ranking_final.py      -> Phase 7 eligibility governance:
                                        formal MCDM NOT PERFORMED
                                        (n_confirmed < 2 in every cluster)
      08b_monte_carlo_stability_final.py -> Phase 8 governance: Monte Carlo
                                        SKIPPED (n_draws=0), same reason

  Both chains are REQUIRED — this is not "historical = dead code". Phase 9
  (10_physics_validation.py) simulates the historical 8-PCM K=4 survivor
  set, and Phase 10 (10_validation_comparison.py) explicitly compares the
  historical MCDM ranking against that physics result (Spearman rho =
  -0.52 to -0.64, verdict "NOT PHYSICALLY SUPPORTED" — an intentional,
  documented finding, not a bug). The final K=3 chain is run independently
  alongside it to produce the governed, audit-clean Phase 5-8 record. This
  runner therefore runs BOTH 06/07/08 (historical) AND
  06_final/07_final/08_final/08b_final (governed) before Phase 9.

  05b_swh_design_specification.py is INCLUDED in the core chain (Phase 4):
  it reads 05_cluster_assam.py's cluster_assignments_assam.csv and writes
  the SWH design specification (Tm_target, L_required per cluster) that
  07_feasibility_filter_final.py's INPUTS list requires.

WHAT THIS DOES NOT RUN, AND WHY:

  A. RAW-DATA ACQUISITION (SETUP) — run only with --include-setup:
     00a_build_population_grid.py, 00b_build_suntimes.py,
     01_download_era5_assam.py, 01b_download_nasapower.py,
     00_unzip_accum.py (run LAST within setup — fixes any ZIP-disguised
     .nc files produced by the ERA5 download, right before 02_combine
     reads them; same ordering rationale as run_all_tamilnadu.py).
     — ONE-TIME steps: they hit external APIs (CDS/ERA5, NASA POWER), need
     credentials (.cdsapirc) this runner has no way to verify, and can take
     hours. Pass --include-setup to include them.

  B. NEVER RUN by this runner (not a --flag away — genuinely excluded):
     - config.py — import-only library module, not a runnable stage.

  C. run only with --with-optional (QC/diagnostic scripts; nothing in the
     core chain reads their output):
     03_plots_raw.py — raw-data QA, run before cleaning.

REQUIRED PLOTS / VERIFICATION SET — run only with --with-plots (mirrors
Rajasthan's --with-plots delegating to a plot orchestrator; Assam has no
single PLOTSV2 entry point so this runner lists them individually, in the
order documented by PLOTS_GUIDE.md, each in its own subprocess so one
failure doesn't take the rest down). Non-blocking, runs AFTER the core
chain and --with-optional:
  1. generate_assam_plots.py             — 13 Objective-1 figures
  2. comparison_plots_assam.py           — 9 cross-pipeline comparison charts
  3. generate_era5_nasa_comparison_plots.py — ERA5 vs NASA POWER cross-source plots
  4. generate_missing_assam_diagrams.py  — parity diagrams (pop grid, PCA scree,
                                            diurnal physics, agreement plot)
  5. verify_01_preprocessing_assam.py    — Verification Suite 01 (preprocessing QA)
  6. verify_02_clustering_assam.py       — Verification Suite 02 (clustering QA)
  7. verify_03_feasibility_assam.py      — Verification Suite 03 (feasibility QA)
  8. verify_04_ranking_assam.py          — Verification Suite 04 (MCDM/MC QA)
  9. generate_phase11_figures.py         — the 2 final K=3 thesis figures
                                            (regime map + PCA projection)
  10. build_plots_assam_ppt.py           — assembles the curated PPT-ready
                                            plots_assam_ppt/ folder from
                                            everything generated above
                                            (run LAST — it copies existing
                                            plot files, so it must run after
                                            every other plot script)

POST-PHASE-10 VERIFICATION (non-blocking regression suites — a late
failure must not invalidate the completed Phase 5-10 deliverables; these
are also what final_project_verification.py re-runs as its own item 10,
kept here individually too so each suite's own pass/fail is visible):
  verify_phase5_phase6.py, verify_phase7.py, verify_phase8.py,
  verify_phase9.py, verify_phase10.py

ORDER AND WHY (CORE stages — required, this runner STOPS at the first
required failure since every later stage reads an earlier stage's output):
  1.  02_combine_assam.py             — Phase 2: merges ERA5 + NASA POWER
                                         at sun-events -> climate_assam_
                                         points.csv
  2.  02b_build_daily_aggregates_assam.py — Phase 2: true daily integrals
                                         from full NASA POWER hourly cache
                                         -> daily_aggregates_assam.csv
  3.  03b_agreement_analysis_assam.py — Phase 2: ERA5 vs POWER agreement,
                                         BACKBONE decision (1.1% GHI MBE)
  4.  04_preprocess_assam.py          — Phase 2.5: physical bounds + QC +
                                         IsolationForest -> parquet files +
                                         assam_cleaned_physical.csv
  5.  04b_climate_signature.py        — Phase 3: 19-index signature vector
                                         -> climate_signatures_raw.csv
  6.  05_cluster_assam.py             — Phase 3+4: locked K=3 GMM
                                         clustering -> cluster_assignments_
                                         assam.csv, cluster_profiles_
                                         assam.csv
  7.  05b_swh_design_specification.py — Phase 4: SWH design spec (Tm_target,
                                         L_required per cluster)
  8.  06_build_pcm_database.py        — Phase 5 (historical K=4 chain) ->
                                         pcm_database_assam.csv
  9.  06_build_pcm_database_final.py  — Phase 5 (final K=3 governed chain)
                                         -> pcm_database_final.csv
  10. 07_feasibility_filter.py        — Phase 5/6 (historical) ->
                                         feasibility_survivors_assam.csv
  11. 07_feasibility_filter_final.py  — Phase 6 (governed) -> pcm_
                                         feasibility_by_cluster.csv,
                                         pcm_feasibility_summary.csv
  12. 08_mcdm_ranking.py              — Phase 6/7 (historical) -> mcdm_
                                         topk_assam.csv, mcdm_full_scores_
                                         assam.csv
  13. 08_mcdm_ranking_final.py        — Phase 7 (governed): eligibility
                                         governance, NOT PERFORMED (n<2)
  14. 08b_monte_carlo_stability_final.py — Phase 8 (governed): SKIPPED
                                         (n_draws=0)
  15. 10_physics_validation.py        — Phase 9: 10-year sub-hourly grey-box
                                         simulation of the historical 8-PCM
                                         survivor set across the 3 final K=3
                                         medoids -> physics_validation_
                                         results_assam.csv (numbered 10 but
                                         runs before 09_recommendation_cards
                                         — same convention as Tamil Nadu/
                                         Rajasthan)
  16. 09_recommendation_cards.py      — Phase 8 deliverable: aggregates
                                         Phases 4-7 + Phase 9 physics into
                                         recommendation_cards_assam.md
  17. 10_validation_comparison.py     — Phase 10: dual-level MCDM-vs-physics
                                         comparison -> validation_comparison
                                         outputs, verdict "NOT PHYSICALLY
                                         SUPPORTED"
  18. generate_phase11_consolidation.py — Phase 11: builds final_output_
                                         manifest.csv + final_outputs/
                                         tables + report
  19. final_project_verification.py   — Phase 11: master verification suite
                                         (re-runs verify_phase5_phase6.py
                                         through verify_phase10.py itself)

HOW TO RUN:
  python run_all_assam.py                  # core pipeline only (default)
  python run_all_assam.py --with-optional   # core + 03_plots_raw.py
  python run_all_assam.py --with-plots      # core + the full plots/verification set
  python run_all_assam.py --with-optional --with-plots   # everything
  python run_all_assam.py --dry-run         # print the order, run nothing
  python run_all_assam.py --include-setup   # ALSO run the raw-data
                                             # acquisition scripts FIRST —
                                             # read the warning above before
                                             # using this flag
  python run_all_assam.py --from 05_cluster_assam.py
                                             # resume the CORE chain
                                             # starting at a given script
                                             # (skips everything before it)

This can take a while end to end — 08_mcdm_ranking.py runs a 5,000-draw
Monte Carlo, and 10_physics_validation.py simulates 10 full years
sub-hourly (dt=300s) for 8 PCMs across 3 medoids (24 simulation runs).
Expect anywhere from several minutes to well over 30 minutes for the full
core chain depending on machine speed; this is not a hang.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# One-time raw-data acquisition — excluded by default, see docstring.
# 00_unzip_accum.py runs LAST within setup: it fixes ZIP-disguised .nc
# files that CDS sometimes produces, right before 02_combine reads them.
SETUP_SCRIPTS = [
    "00a_build_population_grid.py",
    "00b_build_suntimes.py",
    "01_download_era5_assam.py",
    "01b_download_nasapower.py",
    "00_unzip_accum.py",
]

# Required, sequential, stop-on-failure — every later stage reads an
# earlier one's output (see docstring for the full dependency reasoning,
# including why BOTH the historical K=4 chain (06/07/08) and the final
# K=3 governed chain (06_final/07_final/08_final/08b_final) must run).
CORE_SCRIPTS = [
    "02_combine_assam.py",
    "02b_build_daily_aggregates_assam.py",
    "03b_agreement_analysis_assam.py",
    "04_preprocess_assam.py",
    "04b_climate_signature.py",
    "05_cluster_assam.py",
    "05b_swh_design_specification.py",
    # --- Historical K=4 chain (feeds Phase 9 physics + recommendation cards) ---
    "06_build_pcm_database.py",
    "07_feasibility_filter.py",
    "08_mcdm_ranking.py",
    # --- Final K=3 governed chain (the locked, audited Phase 5-8 record) ---
    "06_build_pcm_database_final.py",
    "07_feasibility_filter_final.py",
    "08_mcdm_ranking_final.py",
    "08b_monte_carlo_stability_final.py",
    # --- Phase 9-11 ---
    "10_physics_validation.py",
    "09_recommendation_cards.py",
    "10_validation_comparison.py",
    "generate_phase11_consolidation.py",
    "final_project_verification.py",
]

# QC/diagnostic — run after the core chain, continue-on-failure, nothing
# in CORE_SCRIPTS reads its output.
OPTIONAL_SCRIPTS = [
    "03_plots_raw.py",
]

# Full plots + verification set — run only with --with-plots, AFTER the
# core chain and --with-optional. Order follows PLOTS_GUIDE.md exactly;
# build_plots_assam_ppt.py runs LAST because it copies files the earlier
# scripts generate. Each runs in its own process so one failure doesn't
# take the rest down.
PLOT_SCRIPTS = [
    "generate_assam_plots.py",
    "comparison_plots_assam.py",
    "generate_era5_nasa_comparison_plots.py",
    "generate_missing_assam_diagrams.py",
    "verify_01_preprocessing_assam.py",
    "verify_02_clustering_assam.py",
    "verify_03_feasibility_assam.py",
    "verify_04_ranking_assam.py",
    "generate_phase11_figures.py",
    "build_plots_assam_ppt.py",
]

# Post-Phase-10 regression/verification suites — non-blocking. Also what
# final_project_verification.py re-runs internally as its own item 10;
# kept here individually so each suite's own pass/fail is visible in the log.
POST_PHASE10_SCRIPTS = [
    "verify_phase5_phase6.py",
    "verify_phase7.py",
    "verify_phase8.py",
    "verify_phase9.py",
    "verify_phase10.py",
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
    result = subprocess.run([sys.executable, str(path)], cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] {name} finished in {elapsed:.1f}s")
        return "ok", elapsed

    print(f"\n  [FAILED] {name} exited with code {result.returncode} after {elapsed:.1f}s")
    if stop_on_failure:
        print(f"\n  Stopping — every required stage after {name} in the core chain reads "
              f"its output, so continuing would run against stale/missing data.")
    return "failed", elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Assam pipeline end-to-end, in order.")
    parser.add_argument("--with-optional", action="store_true",
                         help="Also run the QC/diagnostic scripts (03_plots_raw.py) after the core chain.")
    parser.add_argument("--with-plots", action="store_true",
                         help="Also run the full plots + verification set (Objective-1 figures, "
                              "comparison charts, ERA5/NASA cross-source plots, verify_01..04 suites, "
                              "Phase 11 figures, and the curated PPT folder builder) after the core "
                              "chain, non-blocking, LAST.")
    parser.add_argument("--include-setup", action="store_true",
                         help="Also run the one-time raw-data download/setup scripts FIRST. "
                              "These hit external APIs and need credentials — see this file's "
                              "own docstring before using this flag.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved run order and exit without running anything.")
    parser.add_argument("--from", dest="from_script", default=None,
                         help="Resume the CORE chain starting at this script, skipping everything "
                              "before it (e.g. --from 05_cluster_assam.py). A leading './' or '.\\' "
                              "or a directory prefix is accepted — only the filename is matched. "
                              "Does not affect --include-setup, --with-optional, or --with-plots stages.")
    args = parser.parse_args()

    core = list(CORE_SCRIPTS)
    if args.from_script:
        from_name = Path(args.from_script).name
        if from_name not in core:
            print(f"ERROR: --from {args.from_script!r} is not one of the core scripts: {core}")
            sys.exit(2)
        core = core[core.index(from_name):]

    setup = list(SETUP_SCRIPTS) if args.include_setup else []
    optional = list(OPTIONAL_SCRIPTS) if args.with_optional else []
    plots = list(PLOT_SCRIPTS) + list(POST_PHASE10_SCRIPTS) if args.with_plots else []

    print("=" * 68)
    print("  ASSAM PIPELINE — RUN ORDER")
    print("=" * 68)
    if setup:
        print("\n  SETUP (one-time, external APIs — --include-setup was passed):")
        for s in setup:
            print(f"    {s}")
    print("\n  CORE (required; stop-on-first-failure):")
    for name in core:
        print(f"    {name}")
    if optional:
        print("\n  OPTIONAL / DIAGNOSTIC (continue-on-failure):")
        for s in optional:
            print(f"    {s}")
    else:
        print("\n  (Optional/diagnostic scripts skipped — pass --with-optional to include them.)")
    if plots:
        print("\n  PLOTS + VERIFICATION (continue-on-failure, runs LAST):")
        for s in plots:
            print(f"    {s}")
    else:
        print("\n  (Plots/verification set skipped — pass --with-plots to include it.)")

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
        status, elapsed = run_script(name, stop_on_failure=True)
        log.append((name, status, elapsed))
        if status == "failed":
            print_summary(log, time.time() - t_start)
            sys.exit(1)

    for name in optional:
        status, elapsed = run_script(name, stop_on_failure=False)
        log.append((name, status, elapsed))

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
