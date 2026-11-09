# CORRECTIONS — Objective 1 (Tamil Nadu)

**Applied 2026-11-09.** Every change below is a **documentation** change. No
script was modified, no pipeline was re-run, and **no Objective 1 number
changed.** The outputs in `data/processed/` are bit-identical to the 2026-09-16
run both before and after this pass.

> **Read this first, because it sets expectations.** These corrections do not
> and cannot make Objective 1's results better. A weak correlation does not
> become strong because its description is fixed. What they do is make every
> claim in the repository match what the code actually computed — which is what
> makes the work defensible in a review, and what stops Objective 3 from
> inheriting a wrong number. That is the entire benefit. Anyone expecting the
> headline results to improve should stop here: they are unchanged.

---

## Verification first

The corrections were triggered by an external review listing items 1.1–1.7.
**Each was checked against the code and data before any edit.** Four of the
seven were already correctly handled in this repository and needed no
substantive correction; the review had been written from the summary `.docx`
reports, not from the audit docs or the source.

| review item | verdict after checking | action |
|---|---|---|
| **1.1** Physics validation weak, not significant | **Already documented.** `09_PHASE_7_AUDIT.md` §1 already stated "None of these values are statistically significant at any conventional threshold — p-values range 0.12–0.96". The *audit* was honest; three *summary* docs were not. | Propagated the caveat; killed stale numbers. **C-2, C-3, C-7** |
| **1.2** Temperature specs contradictory (67 / 65 / 45 °C) | **Substantially false.** These are four different physical quantities, not competing values for one. `TM_TARGET_C = 67 °C` never binds — Constraint 6 screens on `Tm_target_capped_C` (mean 61.45 °C) and every shortlisted PCM melts at 54.0–58.0 °C. O2's held-out peaks were 63.4–64.0 °C, inside the 65 °C limit. **No safety risk, and O3 inherits none.** One *real* incoherence found underneath it, which the review had not identified. | Master table + the real finding. **C-5** |
| **1.3** GMM config differs across states | **Out of scope.** Only Tamil Nadu exists in this repository. Rajasthan / Assam / Uttarakhand O1 pipelines are not here, so nothing in this tree can claim or deny cross-state uniformity. | Added an explicit scope statement. **C-8** |
| **1.4** External validation incomplete | **True, and already documented in one place — but flatly contradicted in another.** `06_PHASE_4_AUDIT.md` line 35 said Köppen is "NOT WIRED IN"; line 18 of the *same file* described it as delivered with an output CSV. `data/raw/koppen/` is empty. | Fixed the contradiction. **C-1** |
| **1.5** AHP weights unclear | **Already documented in code.** `08_mcdm_ranking.py` lines 111–117 already said "NOT a formally elicited AHP pairwise comparison matrix" and gave the 50/50 formula. Not surfaced in the docs. | Stated it in the docs. **C-6** |
| **1.6** Population weighting may be double-counted | **False — verified.** `sklearn.mixture.GaussianMixture` accepts no `sample_weight` and none is passed; the GMM fit is population-blind. Population enters once at point *selection* and once at profile *aggregation* — different operations on different objects. | Documented the audit so it is not re-suspected. **C-9** |
| **1.7** MCDM citations too thin | **True.** The whole four-method stack was cited to Chen et al. (2025), which is a Taguchi+GRA paper and does not originate TOPSIS, PROMETHEE II or VIKOR. | Added six primary method sources. **C-4** |

---

## The corrections

### C-1 — Köppen validation: internal contradiction removed

`06_PHASE_4_AUDIT.md`. Line 18 presented the Köppen-Geiger external validation
as delivered, with an ARI/NMI and a `koppen_validation_tamilnadu.csv`
contingency table. Line 35 of the same document said it is **not wired in** and
that the raster is absent. Verified: `data/raw/koppen/` exists and is **empty**;
no `koppen_validation_tamilnadu.csv` exists anywhere in the tree.

Line 18 now states that the code is ported but **not delivered in this copy**,
that no ARI/NMI against Köppen is available, and that there is therefore **no
external cluster validation** for Tamil Nadu. The previously-circulated
ARI = 0.0672 / NMI = 0.1499 figures must not be quoted.

### C-2 — "verify that rankings correlate" → "test whether they correlate"

`00_MASTER_OVERVIEW.md`, Phase 7 row of the research-gap table. It claimed the
simulation exists "to verify that MCDM rankings correlate with physical solar
fraction". It tests that; it did not confirm it. Now carries the actual
ρ = +0.169 and the non-significance.

### C-3 — Stale physics numbers removed from the readiness report

`12_FINAL_READINESS_REPORT.md` still quoted the **pre-unification k=5** run:
mean ρ = +0.177 over five clusters and 24/59 simulations in band. Current values
are ρ = **+0.169** over three clusters and **6/26** in band. Three lines fixed —
the transparency bullet, the reasoning paragraph, and the verdict line, which
still read "CLEAN RE-RUN INTO CANONICAL TREE PENDING" although that re-run
completed 2026-09-16.

### C-4 — Primary MCDM method citations added

`13_LITERATURE_MAPPING.md`. Added Hwang & Yoon (1981) for TOPSIS, Brans &
Vincke (1985) for PROMETHEE II, Opricovic & Tzeng (2004) for VIKOR, de Borda
(1781) / Saari (2001) for Borda aggregation, and Shannon (1948) / Zeleny (1982)
for entropy weighting. Chen et al. (2025) is retained but scoped to GRA/Taguchi
and to the PCM-SWH *application*, which is what it actually supports.

### C-5 — Master temperature table, and the one real incoherence

`00_MASTER_OVERVIEW.md`, new section. Tabulates all seven temperature
quantities with value, meaning, source file, and whether it binds. Shows why
67 °C vs 65 °C is not a hazard.

**The genuine finding, which the review missed:** Objective 1 sizes latent heat
for a **60 °C** delivery service (`Q_night = 300 L × cp × (60 − T_mains)`,
Avargani et al. 2021) while Objective 2 scores delivered energy at **45 °C**.
Both are defensible alone; they are not the same service. A PCM selected as
optimal at 60 °C need not be optimal at 45 °C, and nothing in either objective
quantifies that gap.

This is recorded as **open**, not resolved. It does not invalidate O2: its
results are paired differences against a plain tank measured on the same 45 °C
metric, so the *comparison* holds. What is unquantified is whether O1 would have
shortlisted different PCMs had it sized against 45 °C.

### C-6 — Weighting method stated honestly

`00_MASTER_OVERVIEW.md`, new section.
`w_j = 0.5·w_entropy_j + 0.5·w_literature_prior_j`. **No AHP elicitation was
ever performed**; the eigenvector AHP code in `08_mcdm_ranking.py` is
unexercised and carries its own `TODO`. Describe as "50/50 blend of
Shannon-entropy objective weights and literature-informed priors", never as
"AHP-weighted".

### C-7 — README validation claims corrected

`README.md`. The status section reported the headline results without the
non-significance caveat or the missing external validation. Both added, with a
pointer here.

### C-8 — Scope statement

`00_MASTER_OVERVIEW.md`. This tree is Tamil Nadu only (k=3, `diag`).
Cross-state claims about GMM configuration cannot be made from it.

### C-9 — Population weighting audit recorded

`00_MASTER_OVERVIEW.md`. Table of where population weighting is and is not
applied, with the finding that clustering is population-blind and nothing is
double-counted.

---

## Objective 1 → Objective 2 handoff: verified intact

The request was that Objective 1's final results be the ones Objective 2 uses.
**This was already true and is now verified by hash.** Objective 2 holds a
frozen copy of Objective 1's outputs in
`objective2_design_optimization/data/objective1/`. All eight handoff files are
SHA-256 identical to the live Objective 1 outputs:

| file | status |
|---|---|
| `cluster_assignments_tamilnadu.csv` | identical |
| `cluster_profiles_tamilnadu.csv` | identical |
| `climate_signature_tamilnadu.csv` | identical |
| `feasibility_survivors_by_cluster_kappa_calibrated.csv` | identical |
| `pcm_database_tamilnadu.csv` | identical |
| `mcdm_full_rankings.csv` | identical |
| `mcdm_topk_by_cluster.csv` | identical |
| `physics_validation_results.csv` | identical |

Because this pass changed no Objective 1 output, **the handoff needs no refresh
and Objective 2 needs no re-run.** If Objective 1 is ever genuinely re-run,
re-copy these eight files and re-verify the hashes before trusting any
Objective 2 result.

---

## Why nothing further was corrected in Objective 1 (assessed 2026-11-09)

A second pass checked whether any remaining item could be closed **without**
either a network download or a re-run that changes published numbers. **None
can.** Each was checked against the data on disk, not assumed:

| item | why it cannot be closed safely now |
|---|---|
| Köppen external validation | `data/raw/koppen/` is **empty**. Needs `Beck_KG_V1_present_0p0083.tif`, a ~1 GB global raster, downloaded from outside. Nothing in the tree can substitute for it. |
| `monsoon_index` is a proxy | Checked the downloaded NASA POWER JSONs: their parameters are `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS10M` — **no precipitation field**. So a real precipitation index needs a fresh POWER download. And `monsoon_index` (currently JJAS GHI / annual GHI, `04b_climate_signature.py:236`) is a **Tier-2 signature feature**, so replacing it would move the climate signature, the GMM clustering, and every result downstream. Design-changing, deferred. |
| MCDM/physics agreement | Not a defect to repair. `09_PHASE_7_AUDIT.md` §3 proves no static per-candidate criterion can capture it. Chasing it further would overfit the ranking to one simulation. |
| n = 8–10 per cluster | Only a larger PCM database fixes this, which changes the feasibility pool and every ranking. Design-changing. |
| 60 °C / 45 °C service mismatch | Closing it means re-sizing `L_required` against 45 °C and re-running Phases 3–8, which can change the PCM shortlist itself and cascade through all of Objective 2. Deliberately deferred so Objective 3 can proceed against a frozen contract. |
| Phase 7 not unified with Rajasthan | A refactor of the tank model; would change Phase 7 numbers. |
| Seasonal sensitivity yields no comparison | A true consequence of `Tm_target_C = 67`, already documented as such. Not a bug. |

**So Objective 1's documentation is now consistent with its code and data, and
every remaining item is gated on new data or on an accepted change of results.**
The nine corrections above changed no number; none of the items in this table
can be done while that remains a requirement.

---

## Still open in Objective 1 (not fixed here)

These are real gaps, listed so they are not mistaken for completed work.

1. **No external cluster validation.** Köppen raster never sourced; NBC/ECBC
   stubbed. Closing it needs `Beck_KG_V1_present_0p0083.tif` in
   `data/raw/koppen/` and a re-run of Phase 4.
2. **MCDM/physics agreement stays weak and non-significant.** Deliberately not
   chased further — `09_PHASE_7_AUDIT.md` §3 proves no static per-candidate
   criterion can capture it (`CrodaTherm 60`, identical properties, scores
   38 % / 79 % / 66 % across the three clusters). More criteria tuning would
   overfit the ranking to one simulation.
3. **n is small.** 8–10 survivors per cluster. No correlation computed on this
   pool can reach significance; expanding the PCM database is the only fix.
4. **The 60 °C / 45 °C service mismatch** (C-5) is documented, not resolved.
5. **`monsoon_index` is a proxy** — NASA POWER precipitation never downloaded.
6. **Phase 7 is not unified with Rajasthan** — Tamil Nadu uses a standalone
   tank model, Rajasthan uses `physics_lib.py`.
7. **Seasonal PCM sensitivity produces no comparison** under
   `Tm_target_C = 67`: every (cluster, season) cell has fewer than 2 survivors.
   A legitimate consequence of the corrected target, not a script failure.

---

## What changed in Objective 2 in this pass

Only a pointer. `objective2_design_optimization/OVERVIEW.md` now links here so
that its inherited "physics-validated" framing is read against C-2/C-3 above.

**Objective 2's own corrections are recorded separately** in
`../../objective2_design_optimization/docs/CORRECTIONS.md` (applied 2026-11-09,
same pass): the surrogate-winner conflict, the 30 % cap and "global optimum"
language, the CFD-as-validation framing, the Pareto-front/selection separation,
the stage-evidence audit and the verification-terminology standard.
