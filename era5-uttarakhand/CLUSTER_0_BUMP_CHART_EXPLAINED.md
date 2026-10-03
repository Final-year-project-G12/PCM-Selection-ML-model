# Cluster 0 Bump Chart — Explained

This is a candidate-by-candidate walkthrough of `data/plots/uttarakhand_objective1/07_bump_chart_ranks_cluster0.png` (and its interactive `.html` twin), tying every zigzag on the chart back to the actual criterion values and the actual formula each MCDM method uses. Read `PHASE_6_MCDM_RANKING_EXPLAINED.md` first for the general method/weighting background — this file assumes it.

**Cluster 0** (K = 4 run) — warm plains, 10 points, ~320 m mean elevation, `Tm_target = 57.0°C`, 29 feasibility survivors, Kendall's W = 0.796 (fairly strong four-method agreement, but with real, explainable exceptions below).

> **Cluster 1 (mid-hills, 23 points) produces exactly the same chart.** It has the same `Tm_target`, the same 29 survivors and therefore identical ranks and scores, so everything below applies to `07_bump_chart_ranks_cluster1.png` as well. Only `L_required` differs (118 vs 132 kJ/kg), and that is a feasibility floor, not a ranking criterion.

---

## The chart's Top-5, in numbers

| PCM | TOPSIS rank | GRA rank | PROMETHEE rank | VIKOR rank | **Consensus** |
|---|---|---|---|---|---|
| PureTemp 58 | 1 | 7 | 4 | 1 | **1** |
| n-Octacosane (C28) | 9 | 3 | 2 | 4 | **2** |
| PlusICE A58 | 2 | 8 | 5 | 6 | **3** |
| n-Hexacosane (C26) | 6 | 1 | 3 | 12 | 4 |
| RT57HC | 7 | 2 | 1 | 13 | 5 |

The underlying criterion values driving all of this (raw units, plus the derived scores each method actually consumes):

| PCM | Tm (°C) | f_Tm | Latent heat (kJ/kg) | ρH (MJ/m³) | TC (W/mK) | Cycles confidence | TOPSIS score | GRA grade | VIKOR Q |
|---|---|---|---|---|---|---|---|---|---|
| PureTemp 58 | 58.0 | **0.969** | 225 | 200.3 | 0.200 | 0.972 | **0.653** (#1) | 0.657 | **0.063** (#1, lowest=best) |
| n-Octacosane (C28) | 61.6 | **0.516** | **253** (#1) | **230.2** (#1) | **0.267** (#1) | 0.969 | 0.592 | 0.690 | 0.139 |
| PlusICE A58 | 58.0 | 0.969 | 215 | 195.7 | 0.220 | 0.969 | 0.636 | 0.645 | 0.152 |
| n-Hexacosane (C26) | 56.5 | **0.992** (#1) | 256 (#1 overall) | 197.1 | 0.238 | 0.953 | 0.595 | **0.726** (#1) | 0.434 |
| RT57HC | 56.5 | 0.992 | 240 | 216.0 | 0.239 | 0.953 | 0.593 | 0.711 | 0.436 |

(`Tm_target` = 57.0°C, so `f_Tm` peaks near 1.0 the closer Tm sits to 57.)

---

## Why PureTemp 58 wins TOPSIS and VIKOR outright, but only places 7th in GRA

**TOPSIS** (`s_minus / (s_plus + s_minus)`, Euclidean distance to the per-criterion best/worst
point) rewards a candidate for being close to the ideal on *every* axis simultaneously — and
because the distance is squared, having no single bad criterion matters more than having one
spectacular one. PureTemp 58 is near-best on `f_Tm` (0.969) and cycling confidence (0.972),
and merely decent (not weak, not exceptional) on the rest — the most balanced profile of any
Top-5 candidate. That's exactly what minimizes Euclidean distance to the ideal point, giving
it the highest TOPSIS score (0.653) of all 29 survivors.

**VIKOR**'s Q blends an average-gap term (S) with a worst-single-gap term (R) 50/50. Because
PureTemp 58 has no criterion it's badly exposed on, both its S and R are the lowest in the
cluster — Q = 0.063, the best (lowest) of all 29, and comfortably so.

**GRA**, though, uses a *coefficient* per criterion — `(delta_min + ζ·delta_max) / (delta + ζ·delta_max)`
— not a distance. This formula is mathematically more forgiving of an isolated large gap (it's
bounded below by roughly ζ/(1+ζ) even in the worst case) than it is generous toward a merely
"good, not great" profile: it rewards candidates whose *strongest* criteria sit exactly at the
reference maximum. PureTemp 58's thermal conductivity (0.200 W/mK) is the weakest of the
Top-5 — n-Octacosane's is 33% higher, RT57HC's and n-Hexacosane's are ~19-20% higher — and
GRA's profile-similarity math weighs that gap more heavily than TOPSIS's squared-distance sum
does relative to PureTemp 58's other strengths. Six candidates (n-Hexacosane, RT57HC,
n-Octacosane, RT60, savE® OM55, Palmitic-stearic/EG) beat it on GRA specifically — all of
them stronger on at least one of TC or latent heat, the two criteria where PureTemp 58 is
merely mid-pack.

**Net effect on the chart:** PureTemp 58's line starts at rank 1 (TOPSIS), dives to rank 7
(GRA), recovers to rank 4 (PROMETHEE), and returns to rank 1 (VIKOR) — a deep V, but it still
wins the Borda consensus because "1, 7, 4, 1" beats every competitor's combined score.

---

## Why n-Octacosane (C28) is TOPSIS's 9th choice but GRA/PROMETHEE's 2nd-3rd — and still the consensus runner-up

n-Octacosane's melting point (61.6°C) is 4.6°C above the 57°C target — by far the worst
`f_Tm` (0.516) of the Top-10. On its own, that should sink it. But it's simultaneously the
single best candidate in the entire cluster on **three separate criteria**: highest latent
heat (253 kJ/kg), highest volumetric energy density (230.2 MJ/m³), and highest thermal
conductivity (0.267 W/mK).

- **GRA rank 3, PROMETHEE rank 2**: both methods are compensatory weighted-sum schemes over
  the five criteria — three max-of-cluster criteria outweigh one badly-off criterion in a
  linear/coefficient sum, especially since GRA's coefficient formula (see above) doesn't let
  even the worst gap collapse a candidate's grade to near-zero.
- **TOPSIS rank 9**: TOPSIS's *squared* Euclidean distance to the ideal point punishes that
  same large `f_Tm` gap more severely — squaring amplifies the one big deviation relative to
  a linear/coefficient treatment, pulling its TOPSIS score (0.592) well behind candidates with
  more balanced (if less extreme) profiles, even though its raw score isn't dramatically lower
  than PureTemp 58's 0.653.
- **VIKOR rank 4**: VIKOR's R term is *literally* "your single worst weighted gap" — n-Octacosane's
  badly-off `f_Tm` shows up directly and undiluted in R, which is exactly the mechanism VIKOR
  was designed around. It still doesn't fall further than rank 4 because its S (average gap)
  is excellent, and Q blends both 50/50.

**Net effect on the chart:** n-Octacosane's line does the opposite of PureTemp 58's — it
starts low (rank 9, its TOPSIS penalty for the off-target melting point), then climbs to
rank 2–3 for GRA/PROMETHEE (its three best-in-cluster strengths compensating), and settles at
rank 4 for VIKOR (the regret penalty reasserting itself, but only partially). It still reaches
consensus rank 2 because it's strong across enough methods to outweigh the one bad one.

---

## The general lesson this chart is illustrating

A PCM's position on this chart isn't arbitrary noise — it's a direct readout of **how each
method's formula treats a single weak criterion**:

- A candidate that's *uniformly good, nothing outstanding, nothing weak* (PureTemp 58) does
  best under distance-based/regret-based methods (TOPSIS, VIKOR) and worst under
  profile-similarity methods (GRA) that reward criteria sitting exactly at the cluster
  maximum.
- A candidate with *one clear weakness and several standout strengths* (n-Octacosane) is the
  mirror image: penalized hardest by the squared-distance and pure-regret methods (TOPSIS,
  VIKOR), rewarded most by the compensatory coefficient/pairwise methods (GRA, PROMETHEE).

Kendall's W = 0.796 for this cluster means that, in aggregate across all 29 survivors, this
tension isn't severe enough to erase the ranking's usefulness — but the two clearest zigzags
on the chart (PureTemp 58 and n-Octacosane) are exactly the two candidates where the tension
is sharpest, which is why they're worth explaining individually rather than just quoting
"Kendall's W = 0.796" and moving on.
