"""
Assemble the curated 6-phase "ppt" folder for Rajasthan.

Groups the paper-facing figures by pipeline phase. The data/plots/ppt/ tree
is flat - figures sit directly in each phase folder (single state, no need
for a per-state subfolder), the same layout Assam's build_plots_assam_ppt.py
writes:

    data/plots/ppt/
      1 Data collection/                                   *.png
      2 Data Preprocessing/                                *.png
      3 Climate Feature Engineering (Climate Signature)/   *.png
      4 Climate Region Discovery (Clustering)/             *.png
      5 PCM Suitability Evaluation (MCDA)/                 *.png
      6 PCM Recommendation and Output/                     *.png

The project-root `Plots/` tree (written only with --mirror) keeps its
one-subfolder-per-state layout (.../<phase>/Rajasthan/), so the states line
up file-for-file there. Target filenames are the source filenames, which are
the same in every state (../../docs/PLOT_NAMING.md).

    python build_plots_folder_rajasthan.py            # build data/plots/ppt/ (flat)
    python build_plots_folder_rajasthan.py --mirror   # also copy into the
                                                      # project-root Plots/ tree

Run the generators first (../03_plots_raw.py, run_all_plots_v2.py) or this
will just report the missing sources; nothing here computes a figure.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = "Rajasthan"
PLOTS = os.path.join(HERE, "..", "data", "plots")
DEST_ROOT = os.path.join(PLOTS, "ppt")
# ...\PCM-Selection-ML-model\era5-rajasthan\PLOTSV2 -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MIRROR_ROOT = os.path.join(PROJECT_ROOT, "Plots")

# (phase folder, [(source relative to data/plots, target filename), ...])
PHASES = [
    ("1 Data collection", [
        ("raw/A_point_map.png",                          "A_point_map.png"),
        ("raw/C_era5_vs_power.png",                      "C_era5_vs_power.png"),
        ("raw/F_yearly_trend.png",                       "F_yearly_trend.png"),
    ]),
    ("2 Data Preprocessing", [
        ("objective1/01_raw_vs_preprocessed_radiation.png",       "01_raw_vs_preprocessed_radiation.png"),
        ("verify_preprocessing/02_data_completeness.png",                   "02_data_completeness.png"),
        ("verify_preprocessing/05_correlation_analysis.png",                "05_correlation_analysis.png"),
        ("verify_preprocessing/06_data_quality_metrics.png",                "06_data_quality_metrics.png"),
        ("verify_preprocessing/07_preprocessing_summary.png",               "07_preprocessing_summary.png"),
    ]),
    ("3 Climate Feature Engineering (Climate Signature)", [
        ("signatures/point_signature_map.png",                "point_signature_map.png"),
        ("signatures/signature_correlation_heatmap.png",      "signature_correlation_heatmap.png"),
        ("signatures/signature_distributions.png",            "signature_distributions.png"),
    ]),
    ("4 Climate Region Discovery (Clustering)", [
        ("verify_clustering/01_elbow_curves.png",                           "01_elbow_curves.png"),
        ("verify_clustering/02_silhouette_plot.png",                        "02_silhouette_plot.png"),
        ("verify_clustering/05_cluster_profiles.png",                       "05_cluster_profiles.png"),
        ("verify_clustering/06_cluster_sizes.png",                          "06_cluster_sizes.png"),
    ]),
    ("5 PCM Suitability Evaluation (MCDA)", [
        ("objective1/03_melting_point_vs_latent_heat.png",        "03_melting_point_vs_latent_heat.png"),
        ("verify_feasibility/04_constraint_analysis.png",                   "04_constraint_analysis.png"),
        ("objective1/04_feasible_candidates_highlighted.png",     "04_feasible_candidates_highlighted.png"),
        ("objective1/05_pcm_survivors_per_cluster.png",           "05_pcm_survivors_per_cluster.png"),
        ("verify_feasibility/05_property_distributions.png",                "05_property_distributions.png"),
        # Three per-cluster bump charts where Uttarakhand has one pooled chart
        # - MCDM ranks are assigned within a cluster, so pooling them puts three
        # different candidates at rank 1 on one pair of axes. See PLOTS_GUIDE.md §3.
        ("objective1/07_bump_chart_ranks_cluster_0.png",          "07_bump_chart_ranks_cluster_0.png"),
        ("objective1/07_bump_chart_ranks_cluster_1.png",          "07_bump_chart_ranks_cluster_1.png"),
        ("objective1/07_bump_chart_ranks_cluster_2.png",          "07_bump_chart_ranks_cluster_2.png"),
        ("objective1/08_method_rank_correlation_heatmap.png",     "08_method_rank_correlation_heatmap.png"),
        ("objective1/10_rank_reversal_violin_bar.png",            "10_rank_reversal_violin_bar.png"),
    ]),
    ("6 PCM Recommendation and Output", [
        ("objective1/11_agreement_plot.png",                      "11_agreement_plot.png"),
        ("objective1/12_tank_temperature_melt_fraction.png",      "12_tank_temperature_melt_fraction.png"),
        ("objective1/13_recommended_pcm_summary.png",             "13_recommended_pcm_summary.png"),
    ]),
]


def build(dest_root, label, flat=False):
    copied, missing = 0, []
    print(f"\n{'=' * 68}\n  Building: {dest_root}\n{'=' * 68}")
    for phase, files in PHASES:
        # flat: figures sit directly in the phase folder (data/plots/ppt layout).
        # non-flat: one <state> subfolder per phase (project-root Plots/ layout).
        dest_dir = os.path.join(dest_root, phase) if flat else os.path.join(dest_root, phase, STATE)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n  {phase}/" if flat else f"\n  {phase}/{STATE}/")
        for src_rel, target_name in files:
            src = os.path.join(PLOTS, src_rel.replace("/", os.sep))
            if not os.path.exists(src):
                print(f"    [MISSING] {src_rel}")
                missing.append(src_rel)
                continue
            shutil.copy2(src, os.path.join(dest_dir, target_name))
            note = "" if os.path.basename(src) == target_name else f"   (from {os.path.basename(src)})"
            print(f"    {target_name}{note}")
            copied += 1
    print(f"\n  {label}: {copied} copied, {len(missing)} missing")
    return copied, missing


if __name__ == "__main__":
    mirror = "--mirror" in sys.argv

    total_expected = sum(len(f) for _, f in PHASES)
    copied, missing = build(DEST_ROOT, "data/plots/ppt", flat=True)

    if mirror:
        if not os.path.isdir(MIRROR_ROOT):
            print(f"\n  [SKIP] mirror target not found: {MIRROR_ROOT}")
        else:
            build(MIRROR_ROOT, "project-root Plots")

    print("\n" + "=" * 68)
    print(f"  DONE - {copied}/{total_expected} figures in {DEST_ROOT}")
    if missing:
        print("\n  Missing sources - run the generator that produces them first:")
        for m in missing:
            print(f"    {m}")
        print("      raw/             -> ../03_plots_raw.py")
        print("      everything else  -> run_all_plots_v2.py")
    print("\n  Not reproducible from data: the Uttarakhand phase-4 folder also holds")
    print("  'Clustering.jpeg', a hand-made conceptual diagram with no source script.")
    print("=" * 68)
    raise SystemExit(0 if not missing else 1)
