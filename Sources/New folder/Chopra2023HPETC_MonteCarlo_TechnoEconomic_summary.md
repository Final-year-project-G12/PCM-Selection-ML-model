# Technical & Financial Feasibility Assessment of Heat Pipe Evacuated Tube Collector for Water Heating Using Monte Carlo Technique for Buildings

**Authors:** K. Chopra, V.V. Tyagi, Sakshi Popli, A.K. Pandey  
**Year:** 2023  
**Journal/Conference:** Energy, Vol. 267, Article 126338  
**DOI/Link:** https://doi.org/10.1016/j.energy.2022.126338  
**IEEE Citation:** K. Chopra et al., "Technical & financial feasibility assessment of heat pipe evacuated tube collector for water heating using Monte Carlo technique for buildings," Energy, vol. 267, p. 126338, 2023, doi: 10.1016/j.energy.2022.126338.

---

## 1. One-Line Summary
This study couples a **heat-pipe evacuated-tube collector (HP-ETC)** thermal model with **Monte Carlo uncertainty** and **genetic-algorithm optimization** across **five Indian climate zones**, finding mean **LCWH = 5.14 INR/kWh**, **NPV = 663,788 INR**, **PP = 5.84 years**, with **Zone-V (Ahmedabad)** most favorable and optimized cases cutting LCWH **~25–33%** and PP **~37–47%**.

---

## 2. Problem Being Solved
- India residential sector supplies **~80%** of national hot-water demand — prime target for SWH.
- **HP-ETC** SWH is efficient but under-adopted vs thermosyphon ETC due to **high capital cost**, overheating, scaling, and optimistic deterministic economic models.
- Conventional techno-economic studies use fixed inputs, ignoring uncertainty in irradiance, efficiency, tariffs, and finance — biasing investment decisions.
- Need **probabilistic** feasibility tool for HP-ETC deployment across India's climate zones.

---

## 3. Key Contributions
1. **HP-ETC (HPT-ETCS)** performance + **multi-energy/economic** cost model for domestic SWH.
2. **Monte Carlo Technique (MCT)** — triangular distributions on key inputs; **N** simulation trials for LCWH, NPV, PP.
3. **Five-zone India analysis** (Zones I–V) with city-level solar radiation (**Ahmedabad 5.615** vs **Srinagar 4.695 kWh/m²/day**).
4. **Sensitivity analysis:** solar radiation drives **79.47%** of LCWH uncertainty; thermal efficiency **15.65%**.
5. **Single-objective GA optimization** (PIKAIA in EES) for LCWH, NPV, PP — **33.46%** LCWH reduction vs base case (LCWHOL).
6. Policy insight: prioritize high **electricity-price** regions; subsidies could accelerate HP-ETC penetration.

---

## 4. Methodology
### 4a. System model
- **HP-ETC** domestic SWH: fixed orientation (latitude tilt, azimuth 0°), **6.32 m²** aperture/collector, **67** evacuated tubes.
- Hot water: **60 L/day/person**, **6** persons/house, **60 °C** delivery, **15-year** life.
- Thermal efficiency triangular **51–69%**, mean **60%**; degradation **1%/year**.

### 4b. Economic metrics
- **LCWH** — levelized cost of water heating (INR/kWh).
- **NPV** — net present value over 15 years (INR).
- **PP** — payback period (years).
- Matrix formulation **(7)** linking annual energy, costs, loan payments across **N** years.

### 4c. MCT procedure
- Sample each uncertain input from preset distributions → compute outputs → build probability histograms.
- Compare against grid electricity **6.50–28.05 INR/kWh** over system life.

### 4d. Optimization
- **GA:** 9 individuals, 64 generations, crossover **0.85**, mutation **0.005–0.25**.
- Decision variables: capital cost \(C_0\), debt-equity ratio, interest, O&M, irradiance \(I_T\), \(\eta_{th}\), electricity price, discount rate.
- Three cases: **LCWHOL**, **NPVOL**, **PPOL**.

---

## 5. PCM Details (if applicable)
N/A — study focuses on **HP-ETC** sensible water heating, not latent PCM storage.
- Authors suggest **high-boiling-point nanofluids** to mitigate overheating/scaling — indirect link to your PCM-TES SWH (latent buffer replaces oversizing).

---

## 6. AI / ML / Control Details (if applicable)
N/A — no machine learning.
- **Genetic Algorithm (GA)** for economic optimization (metaheuristic, not predictive AI).
- **Monte Carlo** for uncertainty — analogous to robust policy evaluation under climate/economic noise (related to RG5).

---

## 7. Solar / Climate Data Details (if applicable)
- **Geography:** **Five Indian climatic zones** (I cold to V hot/dry); cities include **Srinagar, Delhi, Mumbai, Chennai, Ahmedabad** (Table 3).
- **Solar variable:** daily average irradiance **3.50–6.98 kWh/m²/day**, mean **5.24**.
- **India resource:** **~5×10³ trillion kWh/year** national solar endowment; **4–7 kWh/m²/day** average.
- **Hot water demand:** residential **80%** of sector demand; commercial **13%**, industrial **6%**.
- **Project mapping:** Coimbatore/Jaisalmer/Kochi zone analogs — high radiation (Jaisalmer ≈ Zone-V) favors lower LCWH/PP; humid/coastal zones need larger collector area.

---

## 8. Key Results & Numbers
- **Mean LCWH:** **5.14 INR/kWh** (90.65% probability between **3.80–6.00**).
- **Mean NPV:** **663,788.48 INR** — **100%** probability NPV > 0 in India scenario.
- **Mean payback:** **5.84 years** — **100%** certainty PP < 15 years.
- **Zone-V:** **lowest** LCWH and PP, **least collector area** required; **Zone-III:** highest LCWH/PP, lowest NPV.
- **Ahmedabad:** **5.615 kWh/m²/day** (max); **Srinagar:** **4.695 kWh/m²/day** (min among selected).
- **Sensitivity:** solar radiation **79.47%** of LCWH variance; \(\eta_{th}\) **15.65%**; capital cost marginal.
- **η_th sweep 45–73%:** LCWH **6.32 → 4.38 INR/kWh**; NPV **581,231 → 717,797 INR**; PP **7.65 → 4.63 years**.
- **Electricity price 4–8 INR/kWh:** NPV **308,977 → 886,337 INR**; PP **9.18 → 4.56 years**.
- **Discount rate 4–10%:** NPV **741,259 → 417,440 INR** (\(R^2=0.9913\)).
- **GA optimization vs base:** LCWH **−33.46% / −25.34% / −26.93%**; NPV **+9% / +28.43% / +26.35%**; PP **−37.76% / −41.43% / −47.37%** (LCWHOL/NPVOL/PPOL).
- **Optimized \(C_0\):** **30,140–31,145 INR/m²** vs base **32,500**; \(\eta_{th}\) up to **69%**.

---

## 9. Baseline Comparison
| Case | LCWH | NPV (INR) | PP (years) |
|------|------|-----------|------------|
| Grid electricity only | 6.50–28.05 INR/kWh | — | — |
| HP-ETC mean (MCT) | **5.14** | **663,788** | **5.84** |
| Thermosyphon ETC (market default) | Higher LCWH implied | Lower NPV implied | Longer PP implied |
| GA-optimized LCWHOL | **~−33%** vs base | **~+9–28%** | **~−38–47%** |
| Low \(\eta_{th}=45\%\) | 6.32 INR/kWh | 581,232 | 7.65 |
| High \(\eta_{th}=73\%\) | 4.38 INR/kWh | 717,797 | 4.63 |

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **simulation-only** techno-economic model (EES + MCT + GA).
- **Modeled hardware:** HP-ETC, **67** tubes, **6.32 m²** aperture, fixed tilt = latitude.
- **Demand side:** **360 L/day** hot water (6×60 L), **60 °C**.
- **Financial:** bank loan **15 years**, interest **8–10%**, debt-equity **50–90%**.
- Comparable to your project's **ETC + storage tank** architecture; add PCM capex in extended NPV model.

---

## 11. Limitations Acknowledged by Authors
- HP-ETC **initial cost** and maintenance remain barriers despite favorable LCWH.
- **Overheating** if oversized or low flow; vacuum tube failure **>100 °C**.
- **Scaling** on heat-pipe condenser in hard-water regions.
- Model assumes triangular distributions — real tariffs/policy may differ.
- Does not include **PCM**, smart control, or dynamic demand profiles.

---

## 12. Direct Relevance to My Project
- **RG1:** **Indirect** — no control, but shows value of efficiency gains from better operation (η_th 45→73% cuts PP).
- **RG2:** **Relevant** — HP-ETC is closest commercial analog to your collector loop; PCM+AI adds differentiation beyond this economic study.
- **RG3:** **Highly relevant** — **60 L/day/person**, **60 °C**, **6 occupants** = explicit household demand model for reward shaping.
- **RG4:** **Relevant** — India field economics; validate prototype against **60%** mean η_th assumption.
- **RG5:** **Highly relevant** — **MCT** framework for uncertain **GHI, tariffs, efficiency** maps to climate-adaptive optimization under ERA5/NASA POWER variability across **Coimbatore, Jaisalmer, Kochi**.

---

## 13. Equations to Reuse or Adapt
**Seasonal energy demand (conceptual from model):**
\[
Q_{annual} = \sum_{\tau} \dot{Q}_{load}(\tau) - \sum_{\tau} \eta_{th} A_{col} G(\tau)
\]

**NPV:**
\[
NPV = \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t} - C_{cap}
\]

**LCWH (levelized cost):**
\[
LCWH = \frac{\sum_t (I_t + O\&M_t + fuel_t)}{(1+r)^t}{\Big/}{\sum_t \frac{E_{thermal,t}}{(1+r)^t}}
\]

**Payback:** smallest \(t\) where cumulative savings > \(C_{cap}\).

**Sensitivity index:** fraction of output variance attributable to input \(x_i\) (MCT correlation).

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Mehmood et al., heat-pipe ETC SWH natural gas backup, *Energy Rep.*, 2019** — HP-ETC performance baseline [7].
2. **Duraivel et al. / Indian SWH techno-economic studies** — regional economics context.
3. **TRNSYS/MATLAB economic SWH models** — deterministic predecessors [10,19].
4. **MNRE India solar zone maps** — climatic zoning [3].
5. **Singh et al., PCM-SWH review, 2025** — PCM complement to HP-ETC economics.

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite **80%** residential hot-water share and HP-ETC anti-freeze/high-performance rationale for India.
- **Section II:** Position Chopra as **probabilistic techno-economic** reference for ETC-SWH vs your PCM-intelligent control contribution.
- **Section III:** Use **60 L/person/day, 60 °C** demand profile in grey-box and DRL reward (meet evening draw).
- **Section IV:** Map test cities to zones; target **η_th ≥ 60%** and PP **<6 years** when adding PCM+control capex.
- **Section V:** Report **LCWH/NPV/PP** improvement from intelligent PCM control vs base HP-ETC (**5.14 INR/kWh** benchmark).

---
