"""
pcm_shared_config.py
=============================================================================
ONE definition of the cross-state numeric constants that Phase 2 and Phase 3
of every state pipeline (era5-rajasthan, era5-tamilnadu, ...) must agree on.

WHY THIS EXISTS
---------------
Each state keeps its OWN config.py for paths (data/ lives inside the state
folder, filenames carry the state name) — that stays per-state on purpose.
But the *numeric design basis* below was previously copy-pasted into each
state's scripts independently, so a value could silently drift between
Rajasthan and Tamil Nadu and break cross-state comparability (Objective 1's
whole premise). These live here once; every state's config.py imports and
re-exports them, so `from config import SHARE_PCM` keeps working unchanged
in the scripts while there is only one source of truth.

This module has NO path logic and imports nothing from the pipeline — it is
safe to import from any state folder via `BASE_DIR.parent` on sys.path.

Pure deduplication: none of these values changed when this file was created.
"""

# --- Population-grid coverage (Phase 1/2, 00a_build_population_grid.py) ------
# Minimal set of population points whose cumulative population covers at least
# this fraction of the state total.
COVERAGE_TARGET = 0.875

# --- ERA5 <-> NASA POWER sun-event time matching (Phase 2, 02_combine_*.py) --
# A NASA POWER hourly row is accepted as the match for a sun-event instant
# only if it is within this many hours of the true event time.
MAX_MATCH_HOURS = 3

# --- PCM-facing design basis (Phase 3, 04b_climate_signature.py; also read by
#     Phase 5 feasibility and Level B seasonal analysis) --------------------
# Indian domestic SWH delivery target and heat-exchanger approach temperature
# (midpoint of the framework doc's stated 5-8 K range). Their sum is the
# constant base melting-point target for an indirect system.
T_DELIVERY_C = 50.0
DT_APPROACH_C = 7.0
TM_TARGET_C = T_DELIVERY_C + DT_APPROACH_C   # -> 57.0 C

# Placeholder PCM bed mass used to convert the night-discharge energy floor
# into a per-kg latent-heat requirement. A sizing placeholder, not a design
# decision — kept identical across states so L_required is comparable.
ASSUMED_PCM_MASS_KG = 50.0

# --- Phase 3 PCA block (04b_climate_signature.py) -------------------------
# Number of principal components retained from the 8-column temperature/
# elevation PCA block. PINNED to a fixed integer (not a data-determined
# `n_components=0.95` threshold) so every state's climate_signature_*.csv
# has the SAME PC1..PCn / PC1_z..PCn_z columns and Phase 4's cross-region
# 05_cluster_regions.py can concatenate them directly. 4 was the
# data-determined count for Rajasthan at the 95%-variance threshold, so
# pinning here leaves Rajasthan's output numerically unchanged; for a more
# collinear block (e.g. Tamil Nadu, where PC1..PC3 already exceed 95%) the
# extra low-variance component is near-noise but keeps the schema aligned.
PCA_N_COMPONENTS = 4

# PCM's literature-anchored fractional share of total overnight thermal
# delivery in a combined sensible+latent tank (tank sensible heat + concurrent
# daytime charging supply the rest). Central estimate 0.5; published range
# 0.4-0.78 (Zhao 2022, Huang 2020, Abdelsalam 2020, Kozelj 2021). See
# CLAUDE.md 3.1 and docs/*/07_PHASE_5_AUDIT.md (OPTION A, 2026-08-31).
SHARE_PCM = 0.5

# --- Single shared open-gap note (Phase 3) ---------------------------------
# T_mains_est_C is currently `Ta_mean - 2.0` in BOTH states' 04b scripts.
# This is NOT a published correlation — it is a flat offset kept only for
# cross-state consistency. Before either state's L_required_kJ_per_kg (which
# depends on it, and which drives the Phase 5 latent-heat gate) is presented
# as final, replace it with a real ground-temperature-lag correlation
# (Kusuda & Achenbach-style annual-lag model, mains inlet ~ damped/lagged
# annual air-temperature wave). One gap, one fix, both states — do not paper
# over it with two independently-tuned offsets.
T_MAINS_EST_C_TODO = (
    "T_mains_est_C = Ta_mean - 2.0 is a placeholder, not a published "
    "correlation. Replace with a Kusuda & Achenbach-style ground-temperature "
    "annual-lag model before L_required is presented as final. Shared gap "
    "across all state pipelines — see pcm_shared_config.T_MAINS_EST_C_TODO."
)
