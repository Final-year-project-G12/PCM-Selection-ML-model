"""
Assemble the curated 6-phase "Plots" folder for Rajasthan.

Mirrors the layout of the project-root `Plots/` folder, which groups the
paper-facing figures by pipeline phase with one subfolder per state:

    Plots/
      1 Data collection/                                    Rajasthan/
      2 Data Preprocessing/                                 Rajasthan/
      3 Climate Feature Engineering (Climate Signature)/    Rajasthan/
      4 Climate Region Discovery (Clustering)/              Rajasthan/
      5 PCM Suitability Evaluation (MCDA)/                  Rajasthan/
      6 PCM Recommendation and Output/                      Rajasthan/

Target filenames follow the Uttarakhand folder exactly, so the three states'
subfolders line up file-for-file.

    python build_plots_folder_rajasthan.py            # build PLOTSV2/Plots/
    python build_plots_folder_rajasthan.py --mirror   # also copy into the
                                                      # project-root Plots/ tree

Run the generators first (run_all_plots_v2.py, phase1_..., phase3_...) or this
will just report the missing sources; nothing here computes a figure.

NOTE ON PHASE 2 NUMBERING: Uttarakhand's verify_preprocessing writes
05_data_quality_metrics / 06_correlation_analysis, while Tamil Nadu's and
Rajasthan's write those two the other way round (05_correlation_analysis /
06_data_quality_metrics). The curated folder uses Uttarakhand's numbering, so
those two are copied under swapped names on purpose - see PHASES below.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = "Rajasthan"
DEST_ROOT = os.path.join(HERE, "Plots")
# ...\PCM-Selection-ML-model\era5-rajasthan\PLOTSV2 -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MIRROR_ROOT = os.path.join(PROJECT_ROOT, "Plots")

# (phase folder, [(source relative to PLOTSV2, target filename), ...])
PHASES = [
    ("1 Data collection", [
        ("phase1_data_collection/A_point_map.png",                          "A_point_map.png"),
        ("phase1_data_collection/C_era5_vs_power.png",                      "C_era5_vs_power.png"),
        ("phase1_data_collection/F_yearly_trend.png",                       "F_yearly_trend.png"),
    ]),
    ("2 Data Preprocessing", [
        ("rajasthan_objective1/01_raw_vs_preprocessed_radiation.png",       "01_raw_vs_preprocessed_radiation.png"),
        ("verify_preprocessing/02_data_completeness.png",                   "02_data_completeness.png"),
        # swapped on purpose - see module docstring
        ("verify_preprocessing/06_data_quality_metrics.png",                "05_data_quality_metrics.png"),
        ("verify_preprocessing/05_correlation_analysis.png",                "06_correlation_analysis.png"),
        ("verify_preprocessing/07_preprocessing_summary.png",               "07_preprocessing_summary.png"),
    ]),
    ("3 Climate Feature Engineering (Climate Signature)", [
        ("phase3_climate_signature/point_signature_map.png",                "point_signature_map.png"),
        ("phase3_climate_signature/signature_correlation_heatmap.png",      "signature_correlation_heatmap.png"),
        ("phase3_climate_signature/signature_distributions.png",            "signature_distributions.png"),
    ]),
    ("4 Climate Region Discovery (Clustering)", [
        ("verify_clustering/01_elbow_curves.png",                           "01_elbow_curves.png"),
        ("verify_clustering/02_silhouette_plot.png",                        "02_silhouette_plot.png"),
        ("verify_clustering/05_cluster_profiles.png",                       "05_cluster_profiles.png"),
        ("verify_clustering/06_cluster_sizes.png",                          "06_cluster_sizes.png"),
    ]),
    ("5 PCM Suitability Evaluation (MCDA)", [
        ("rajasthan_objective1/03_melting_point_vs_latent_heat.png",        "03_melting_point_vs_latent_heat.png"),
        ("verify_feasibility/04_constraint_analysis.png",                   "04_constraint_analysis.png"),
        ("rajasthan_objective1/04_feasible_candidates_highlighted.png",     "04_feasible_candidates_highlighted.png"),
        ("rajasthan_objective1/05_pcm_survivors_per_cluster.png",           "05_pcm_survivors_per_cluster.png"),
        ("verify_feasibility/05_property_distributions.png",                "05_property_distributions.png"),
        # Three per-cluster bump charts where Uttarakhand has one pooled chart
        # - MCDM ranks are assigned within a cluster, so pooling them puts three
        # different candidates at rank 1 on one pair of axes. See PLOTS_GUIDE.md §3.
        ("rajasthan_objective1/07_bump_chart_ranks_cluster_0.png",          "07_bump_chart_ranks_cluster_0.png"),
        ("rajasthan_objective1/07_bump_chart_ranks_cluster_1.png",          "07_bump_chart_ranks_cluster_1.png"),
        ("rajasthan_objective1/07_bump_chart_ranks_cluster_2.png",          "07_bump_chart_ranks_cluster_2.png"),
        ("rajasthan_objective1/08_method_rank_correlation_heatmap.png",     "08_method_rank_correlation_heatmap.png"),
        ("rajasthan_objective1/10_rank_reversal_violin_bar.png",            "10_rank_reversal_violin_bar.png"),
    ]),
    ("6 PCM Recommendation and Output", [
        ("rajasthan_objective1/11_agreement_plot.png",                      "11_agreement_plot.png"),
        ("rajasthan_objective1/12_tank_temperature_melt_fraction.png",      "12_tank_temperature_melt_fraction.png"),
        ("rajasthan_objective1/13_recommended_pcm_summary.png",             "13_recommended_pcm_summary.png"),
    ]),
]


def build(dest_root, label):
    copied, missing = 0, []
    print(f"\n{'=' * 68}\n  Building: {dest_root}\n{'=' * 68}")
    for phase, files in PHASES:
        dest_dir = os.path.join(dest_root, phase, STATE)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n  {phase}/{STATE}/")
        for src_rel, target_name in files:
            src = os.path.join(HERE, src_rel.replace("/", os.sep))
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
    copied, missing = build(DEST_ROOT, "PLOTSV2/Plots")

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
        print("      phase1_data_collection/  -> phase1_data_collection_rajasthan.py")
        print("      phase3_climate_signature/-> phase3_climate_signature_rajasthan.py")
        print("      everything else          -> run_all_plots_v2.py")
    print("\n  Not reproducible from data: the Uttarakhand phase-4 folder also holds")
    print("  'Clustering.jpeg', a hand-made conceptual diagram with no source script.")
    print("=" * 68)
    raise SystemExit(0 if not missing else 1)
