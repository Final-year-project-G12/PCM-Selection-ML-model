# 10 — Comparison to the Rajasthan Pipeline

Both pipelines are explicitly designed to be method-comparable (both are read for a shared,
eventual multi-state clustering run). This file consolidates the differences a reader would need to
know before treating the two states' results as directly comparable.

## Execution status — the single biggest difference

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Has been run end-to-end | **Yes** — full 10-year dataset exists on disk | **No** — no `data/` folder exists at all |
| Points | 320 (confirmed) | ~133 (expected, unconfirmed) |
| Furthest phase with real output | Phase 6 (MCDM rankings exist, self-tagged provisional) | None — every phase is code-only |

Every comparison below is therefore between **Rajasthan's measured behavior** and **Tamil Nadu's
code-level design** — not two measured results.

## Filename integrity

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Scripts correctly named | Yes (`era5-rajasthan/`) | **No** — every file mismatched (see `01_FILENAME_CORRESPONDENCE.md`) |
| A shared "vestigial" mislabeled folder exists for both | `until phase 4/` (renamed to `tamilnadu/` between sessions) | **is** this folder |

Note: the Rajasthan audit's Phase 5 section referenced a "vestigial TN-branch script" inside a
mislabeled `until phase 4/` folder — that folder **is this Tamil Nadu pipeline**, confirmed by file
hash comparison during this audit. The PCM literature-row additions and imputation script found
during the Rajasthan audit's Phase 5 research were, in fact, Tamil Nadu's files all along.

## Deaccumulation

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Function name | `accum_to_flux()` | `deaccumulate()` |
| Logic | Stateless clip, **no diffing** — found empirically that raw hourly values are already per-hour flux | True `diff()`-based deaccumulation with reset-hour override at hours 1/13 UTC |
| History | A bug was found (old diff-based approach was wrong for this pipeline's actual CDS behavior) and fixed | No bug found in this audit; **but also never empirically re-verified against TN's actual CDS download the way Rajasthan's was** |

**This is not "TN copied Rajasthan's fixed version" or "TN still has Rajasthan's old bug"** — it's a
genuinely different function altogether, and whether it's correct for Tamil Nadu's specific CDS
request behavior is an open question that can only be answered by running the pipeline once and
checking (same diagnostic Rajasthan used: look for consecutive-hour raw values that decrease within
a nominal accumulation cycle, which would indicate the download is already flux, not cumulative).

## Climate signature (`L_required_kJ_per_kg`) — the headline finding of this audit

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Formula | Corrected: `300 kg total × 4.186 × (50−T_mains) / 50` (Avargani et al. 2021, total-volume basis) | **Unfixed**: `(60/1000/60) × 4.186 × (50−T_mains) × 3600 × 7 / 50` (pre-correction rate-based basis) |
| Resulting order of magnitude | ~600–650 kJ/kg (a deliberately strict ceiling) | ~50–55 kJ/kg (roughly 12× smaller) |
| Consequence | Feasibility filter fails almost everything (self-diagnosed, documented, κ-relaxation policy in place) | Feasibility filter would pass far more than it should (silently, no self-diagnosis exists for this in TN's code) |

Tamil Nadu's `04b_climate_signature.py` is running the exact formula Rajasthan's own code comments
identify as a units-confusion bug. This should be fixed by porting Rajasthan's corrected formula
before Tamil Nadu's pipeline is run.

## Clustering

| | Rajasthan | Tamil Nadu |
|---|---|---|
| GMM covariance type | `diag` (fixed from `full`, documented bug) | `full` (no issue found or fixed — never run to check) |
| k selection | Automated 3-tier cascade (`suggest_k()`), no manual override used | **Manually hardcoded** `K_FINAL=5`, pending a real first run |
| Silhouette band | 0.15–0.35 | 0.15–0.40 (wider, explicitly justified for single-state scope) |
| Metrics computed | BIC, AIC, silhouette, Davies-Bouldin, Calinski-Harabasz, bootstrap-ARI | BIC, silhouette, Davies-Bouldin, Calinski-Harabasz (no AIC, no bootstrap-ARI) |
| Population weighting in fit | No (by design) | No (by design) — identical rationale |
| External classification validation | Stubbed with explicit `None` placeholders and a cited TODO (Beck et al. 2018) | **No structure at all** — not even a stub |
| Level B (seasonal) clustering | Exists | **Does not exist** |

## PCM database & feasibility filtering

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Candidate count | 18–25 (same underlying manufacturer table + 7 literature rows) | Same — 18 + 7 = 25 |
| Imputation method | Hand-rolled MICE+RF+3-donor PMM-like blend | Same design, `N_DONORS=3` confirmed identical |
| Feasibility constraints | 8 (5 original + charging feasibility + corrosion veto + safety) | **5 only** — the 3 additions are honestly documented as absent (data doesn't exist yet), not silently skipped |
| Result at nominal thresholds | 0 survivors (self-diagnosed, ceiling too strict) | Unknown — will likely be too permissive given the `L_required` bug |

## MCDM ranking

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Methods | TOPSIS, PROMETHEE II, VIKOR, GRA (+ optional CoCoSo, off by default) | **TOPSIS + GRA only** |
| Monte Carlo | Yes, 1000 draws (documented deviation from the framework doc's 5000) | **None** — confirmed zero code, not disabled |
| AHP | `AHP_PAIRWISE_MATRIX = None`, bare stub, working eigenvector code never invoked | Also not real elicitation, but presented as an explicitly-labeled, renormalized "honest placeholder" — marginally more transparent framing |
| Consensus mechanism | Borda + Copeland + Kendall's W | Borda + Kendall's W only (no Copeland) |
| Criteria count | 8 | 5 (corrosion, cost, supercooling excluded, disclosed) |
| Bugs found & fixed | VIKOR sign inversion, entropy-weight inflation, kappa-inequality inversion (all dated 2026-08-11) | None found — but also never run, so latent bugs of this kind cannot be ruled out the way "found and fixed" implies active testing did occur for Rajasthan |

## Elevation

| | Rajasthan | Tamil Nadu |
|---|---|---|
| Source | Real per-point ERA5 geopotential (`00c_attach_elevation.py`) | Flat 150 m constant, no attachment step exists |
| Fallback semantics | 300 m used only if real elevation is missing/NaN for a point | 150 m used unconditionally for every point |
| PCA pseudo-elevation | N/A (real elevation used directly) | Separate `elev_proxy = mean(P_atm)/1013.25`, never reconciled with the flat 150 m value used elsewhere |

## What this comparison implies for the eventual 4-state combined analysis

Both pipelines' docstrings repeatedly state the goal of directly comparable cross-state output for a
future combined clustering run (`05_cluster_regions.py`/`05_cluster_tamilnadu.py`'s multi-region
variant). **As currently implemented, the two states' pipelines are not yet at parity** — Rajasthan
is further along (real data, more MCDM methods, more feasibility constraints, real elevation,
external-validation stub structure) while Tamil Nadu has a more rigorous Phase-2 preprocessing layer
(the 13-step pipeline has no Rajasthan equivalent) but a currently-incorrect Phase-3 formula and a
narrower Phase 6. Before combining both states' signature files for a real multi-state clustering
run, at minimum: (1) fix Tamil Nadu's `L_required` formula, (2) run Tamil Nadu's pipeline once
end-to-end, (3) decide whether to bring Tamil Nadu's MCDM stack up to Rajasthan's 4-method+Monte
Carlo level first, or accept an asymmetric comparison and disclose it explicitly.
