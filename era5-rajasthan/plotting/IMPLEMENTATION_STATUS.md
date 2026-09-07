# Superseded — historical record

This was a progress tracker for the retired `plotting/` folder. It listed scripts
as "TODO / BLOCKED" and "READY TO IMPLEMENT" against a plan that has since been
completed by a different implementation.

**Current status of the Rajasthan plot set lives in
[`../PLOTSV2/PLOTS_GUIDE.md`](../PLOTSV2/PLOTS_GUIDE.md).** All 13 objective-1
plots, 4 verification suites, 2 phase suites and 8 comparison plots generate.

The one architectural idea from this document that PLOTSV2 kept is the
**verification-block pattern** — each script loads its inputs, checks the required
columns exist, compares key metrics against audit-documented baselines and prints
PASS/WARN/INFO rather than failing silently. `09_mcdm_vs_physics_agreement.py` is
the clearest surviving example: it checks per-cluster Spearman ρ against both the
audit values and the stored `spearman_rho_by_cluster_rajasthan.csv`.

Provenance/staleness detection also survived: every downstream CSV carries
`upstream_cluster_profile_fingerprint`, and a mismatch against
`cluster_profiles_rajasthan.csv` means the plots are stale (`PLOTS_GUIDE.md` §6).

See [`README.md`](README.md) for why this folder was retired.
