# 12 — Literature Mapping

Largely identical situation to Rajasthan (same framework doc, same `Sources/` folder, same
`references.bib`/`.claude/references.md`) — this file notes only what's specific to Tamil Nadu's
implementation, and cross-references `docs/era5_rajasthan/17_LITERATURE_MAPPING.md` for the shared
methodology-citation gaps (ERA5/pvlib/MCDM-method-origin papers not yet in the project bibliography).

## Citations independently spot-verified by this project's own `FIXES.md`

Before this audit began, a prior review already independently verified two of the framework doc's
most load-bearing citations directly against the source papers:

- **Oluah, Akinlabi & Njoku (2020)**, "Selection of phase change material for improved performance of
  Trombe wall systems using the entropy weight and TOPSIS methodology," *Energy and Buildings* 217,
  DOI: 10.1016/j.enbuild.2020.109967 — verified finding: thermal conductivity received 72.12% entropy
  weight, confirming why entropy weighting alone (without an AHP blend) is untrustworthy. Used
  identically in both Rajasthan and Tamil Nadu's MCDM scripts as the domination-threshold comparator.
- **The *Building and Environment* (2024) India climate-classification study**, DOI:
  10.1016/j.buildenv.2024.112057 — verified finding: silhouette 0.21 vs. −0.2 for the existing NBC
  classification, peaking 0.3 at k=6. Confirms both states' silhouette acceptance bands (Rajasthan
  0.15–0.35, TN single-state 0.15–0.40) are literature-grounded, not arbitrary.
- **Avargani, Norton, Rahimi & Karimi (2021)**, *J. Energy Storage* — "300 L at 60±2°C for 7h" —
  the design basis both states' `L_required` derivations are *supposed* to use. Rajasthan's does;
  Tamil Nadu's currently does not (see `05_PHASE_3_AUDIT.md`).

## Singh et al. (2025) — direct, correctly-applied citation

Table 2 of Singh et al. (2025), *Solar Energy Materials and Solar Cells* 293, supplies the 7
literature PCM rows in both states' PCM databases (myristic acid, palmitic acid, two eutectics,
generic paraffin, C22H46, C30H62) — confirmed identical row-for-row between the Rajasthan and Tamil
Nadu audits, consistent with the PCM database being a shared, state-independent resource.

## Gaps specific to Tamil Nadu's implementation

- **`HSI` (heat-stress index) in `04b_climate_signature.py`** has no cited source — unlike
  Rajasthan's correctly-attributed `HSI_sunrise` (Thom 1959 THI), Tamil Nadu's formula
  (`RH_mean × fraction of readings within 3°C dew-point depression`) does not appear in Thom's
  original THI or any other source identified during this audit. Either source it or explicitly
  label it as an original, uncalibrated index in any write-up.
- **`07b_charging_feasibility.py`'s constants** (`REFERENCE_GOOD_DAY_TEMP_C=70`,
  `MIN_ACHIEVABLE_TEMP_C=42`) are loosely anchored to Al-Mamun et al. (2023)'s cited flat-plate-
  collector operating band (25–100°C) — a real citation, but explicitly labeled in-code as a "stated
  assumption, not a measured value," and correctly so.

## Shared gap with Rajasthan (repeated here for completeness)

No dedicated methodology citations for: Reda & Andreas (2004, SPA), Ineichen & Perez (2002,
clear-sky model), Holmgren et al. (2018, pvlib), Hwang & Yoon (1981, TOPSIS), Deng (1982, GRA), van
Buuren & Groothuis-Oudshoorn (2011, MICE) are present in `references.bib`/`.claude/references.md` —
see `docs/era5_rajasthan/17_LITERATURE_MAPPING.md` for the full recommended addition list, which
applies identically to a Tamil Nadu methodology write-up.
