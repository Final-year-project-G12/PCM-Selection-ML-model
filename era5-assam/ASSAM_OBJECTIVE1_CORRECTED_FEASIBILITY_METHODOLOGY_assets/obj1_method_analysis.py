"""ISOLATED analysis for the corrected Assam Objective-1 feasibility methodology (scratch; no project file is written or modified).
Parts: M medoid verification | W weather-source consistency | E energy-balance charging screen + sensitivity | S sequential rate-limited charge/discharge model | C classification
Run: python obj1_method_analysis.py   (from any cwd)   Outputs -> scratchpad/obj1_method/
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
O1 = r"M:\Final_year_pro\PCM-Selection-ML-model\era5-assam"
O2 = r"M:\Final_year_pro\Objective---2\objective2-assam\objective2-assam-fpc"
SCR = r"C:\Users\hp\AppData\Local\Temp\claude\0189d894ab42dcbb\de02d978-3298-4895-a1f7-5f4bfe890c04\scratchpad"
OUT = os.path.join(SCR, "obj1_method"); os.makedirs(OUT, exist_ok=True)
R = {}

# ============================ M. medoid verification ============================
sig = pd.read_csv(O1 + r"\data\processed\climate_signatures_raw.csv")
asg = pd.read_csv(O1 + r"\data\processed\clustering\cluster_assignments_assam.csv")
grid = pd.read_csv(O1 + r"\data\processed\population_grid_points.csv")
FEAT = ["GHI_mean", "Ta_mean", "DTR", "RH_mean", "wind_mean"]
mg = asg[["point_id", "cluster", "max_membership_prob"]].merge(sig[["point_id"] + FEAT], on="point_id")
X = mg[FEAT].values; Xn = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
med = {}
for c in sorted(mg.cluster.unique()):
    idx = np.where(mg.cluster.values == c)[0]; Xs = Xn[idx]
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(2)); sd = D.sum(1)
    cen = np.sqrt(((Xs - Xs.mean(0)) ** 2).sum(1))
    med[int(c)] = dict(n_points=int(len(idx)),
                       distance_sum_medoid=mg.point_id.iloc[idx[int(sd.argmin())]],      # Assam O1 10_physics_validation.derive_true_medoids
                       centroid_nearest=mg.point_id.iloc[idx[int(cen.argmin())]],         # Rajasthan physics_lib.find_medoid definition
                       max_membership=mg.point_id.iloc[idx[int(mg.max_membership_prob.values[idx].argmax())]])  # 09_recommendation_cards / TN O2
    pid = med[int(c)]["distance_sum_medoid"]; r = grid[grid.point_id == pid].iloc[0]
    med[int(c)].update(lat=float(r.lat), lon=float(r.lon), elev=float(r.elevation_m))
    # rank of hard-coded id under the distance-sum criterion
R["medoids"] = med
hard = {0: "ASP_0003", 1: "ASP_0036", 2: "ASP_0080"}
R["medoid_hardcoded_in_08_08b"] = {c: dict(id=hard[c], is_distance_sum_medoid=(med[c]["distance_sum_medoid"] == hard[c]), cluster_of_id=int(asg[asg.point_id == hard[c]].cluster.iloc[0])) for c in hard}

# ============================ W. weather source consistency =====================
sys.path.insert(0, O2)
from src.config_loader import load_config
from src.io_weather import load_regime_weather
cfg = load_config()
W = {c: load_regime_weather(c, cfg) for c in (0, 1, 2)}
cons = {}
import glob
for c in (0, 1, 2):
    pid = hard[c]; fs = sorted(glob.glob(O1 + rf"\data\raw\nasapower\power_{pid}_*.json"))
    if not fs: cons[c] = "no raw json"; continue
    ghi, t2 = {}, {}
    for f in fs:
        d = json.load(open(f))["properties"]["parameter"]; ghi.update(d["ALLSKY_SFC_SW_DWN"]); t2.update(d["T2M"])
    s_g = pd.Series(ghi); s_t = pd.Series(t2)
    s_g.index = pd.to_datetime(s_g.index, format="%Y%m%d%H"); s_t.index = pd.to_datetime(s_t.index, format="%Y%m%d%H")
    h = W[c].data; ts = pd.to_datetime(h.timestamp_utc, utc=True).dt.tz_localize(None)
    a = pd.Series(h.GHI_Wm2.values, index=ts); b = pd.Series(h.T_amb_C.values, index=ts)
    j = pd.concat([a.rename("pk"), s_g.rename("raw")], axis=1).dropna(); jt = pd.concat([b.rename("pk"), s_t.rename("raw")], axis=1).dropna()
    cons[c] = dict(raw_files=len(fs), n_joined_ghi=int(len(j)), ghi_max_abs_diff=float((j.pk - j.raw).abs().max()), ghi_frac_equal=float(((j.pk - j.raw).abs() < 1e-6).mean()),
                   n_joined_T=int(len(jt)), T_max_abs_diff=float((jt.pk - jt.raw).abs().max()), T_frac_equal=float(((jt.pk - jt.raw).abs() < 1e-6).mean()),
                   package_point_id=W[c].point_id, package_lat=W[c].lat, package_lon=W[c].lon)
R["weather_consistency"] = cons

from src.pcm_geometry import calculate_mass_and_volume_bounds
try:
    mb = calculate_mass_and_volume_bounds(); R["m_pcm_bounds_M1"] = {k: (v if not isinstance(v, dict) else v) for k, v in mb.items() if 'm_pcm' in k or 'V_pcm' in k}
except Exception as ex:
    R["m_pcm_bounds_M1"] = repr(ex)
# ============================ common inputs =====================================
db = pd.read_csv(O1 + r"\data\processed\pcm\pcm_database_final.csv")
src = pd.read_csv(r"M:\Final_year_pro\PCM-Selection-ML-model\PCM_data\data\PCM_Properties_cleaned_mice_pmm_detailed.csv")
fl = src[["product", "flash_point", "flash_point_imputed", "max_op_temp", "max_op_temp_imputed", "flammability", "flammability_imputed"]].rename(columns={"product": "product_name"})
db = db.merge(fl[["product_name", "flash_point", "flash_point_imputed", "max_op_temp", "max_op_temp_imputed"]], on="product_name", how="left")
db["product_name"] = db.product_name.str.replace("\ufffd", "").str.replace("Â", "").str.replace("®", "")
CP_W = 4186.0
TS_MIN, TS_MAX = db.loc[db.TC_status == "Reported", "TC_solid_W_mK"].min(), db.loc[db.TC_status == "Reported", "TC_solid_W_mK"].max()
TL_MIN, TL_MAX = db.loc[db.TC_status == "Reported", "TC_liquid_W_mK"].min(), db.loc[db.TC_status == "Reported", "TC_liquid_W_mK"].max()
R["k_reported_range"] = dict(solid=[float(TS_MIN), float(TS_MAX)], liquid=[float(TL_MIN), float(TL_MAX)])

def prep_weather(c):
    d = W[c].data.copy()
    d["day"] = pd.factorize(d.ist_date)[0]
    G = d.pivot_table(index="day", columns="hour_ist", values="GHI_Wm2").values
    Ta = d.pivot_table(index="day", columns="hour_ist", values="T_amb_C").values
    Tm = d.groupby("day").T_mains_C.first().values
    dates = d.groupby("day").ist_date.first().values
    mon = pd.to_datetime(dates).month.values
    return G, Ta, Tm, mon

# ============================ E. energy-balance charging screen =================
def useful_day(G, Ta, Tin, FRta, FRUL):
    q = FRta * G - FRUL * (Tin - Ta)
    q = np.where((G >= 10.0) & (q > 0), q, 0.0)
    return q.sum(1) / 1000.0       # kWh per m2 per day (hourly rows)

def wm_frac(ok, mon):
    return float(min(ok[mon == m].mean() for m in range(1, 13)))

ES = []
base = dict(A=4.0, m=50.0, FRta=0.72, FRUL=4.5, appr=6.0, eta=1.0)
variants = [("base", {}), ("A=2", dict(A=2.0)), ("A=5", dict(A=5.0)), ("A=6", dict(A=6.0)), ("m=25", dict(m=25.0)), ("m=100", dict(m=100.0)),
            ("FR(ta)=0.65,FRUL=5.0", dict(FRta=0.65, FRUL=5.0)), ("FR(ta)=0.80,FRUL=3.5", dict(FRta=0.80, FRUL=3.5)), ("approach=4", dict(appr=4.0)), ("approach=8", dict(appr=8.0)), ("eta_loop=0.85", dict(eta=0.85))]
WD = {c: prep_weather(c) for c in (0, 1, 2)}
for c in (0, 1, 2):
    G, Ta, Tmn, mon = WD[c]
    for Tm in (47, 52, 56, 60, 64):
        for name, ch in variants:
            p = {**base, **ch}
            Eu = useful_day(G, Ta, Tm + p["appr"], p["FRta"], p["FRUL"]) * p["A"] * p["eta"]       # kWh/day delivered to PCM loop
            for t0name, need in (("latent_only(L=200)", np.full(len(Tmn), p["m"] * 200.0 / 3600.0)), ("from_mains(L=200,cps=2)", p["m"] * (2.0 * (Tm - Tmn) + 200.0) / 3600.0)):
                ok = Eu >= need
                ES.append(dict(cluster=c, Tm=Tm, variant=name, need=t0name, frac_days=float(ok.mean()), worst_month=wm_frac(ok, mon), p10_Eu=float(np.quantile(Eu, .1)), mean_Eu=float(Eu.mean())))
ES = pd.DataFrame(ES); ES.to_csv(OUT + r"\energy_screen_sensitivity.csv", index=False)
# required area-to-mass ratio quantiles
AR = []
for c in (0, 1, 2):
    G, Ta, Tmn, mon = WD[c]
    for Tm in (47, 52, 56, 60, 64):
        Eu1 = useful_day(G, Ta, Tm + 6.0, 0.72, 4.5)                     # kWh/m2/day
        for t0name, need in (("latent_only", 200.0 / 3600.0 * np.ones(len(Tmn))), ("from_mains", (2.0 * (Tm - Tmn) + 200.0) / 3600.0)):
            ar = np.where(Eu1 > 1e-9, need / np.maximum(Eu1, 1e-9), np.inf)       # m2 per kg PCM for a full melt that day
            AR.append(dict(cluster=c, Tm=Tm, need=t0name, A_per_kg_q50=float(np.quantile(ar, .5)), A_per_kg_q80=float(np.quantile(ar, .8)), A_per_kg_q90=float(np.quantile(ar, .9)),
                           worstmonth_median=float(max(np.median(ar[mon == m]) for m in range(1, 13)))))
pd.DataFrame(AR).to_csv(OUT + r"\area_per_kg_required.csv", index=False)
# collector operating point
OP = []
for c in (0, 1, 2):
    G, Ta, Tmn, mon = WD[c]
    for Tm in (47, 56, 64):
        Tin = Tm + 6.0; gcrit = 4.5 * (Tin - Ta) / 0.72
        daylight = G >= 10
        OP.append(dict(cluster=c, Tm=Tm, hours_G_ge_Gcrit_mean=float(((G >= gcrit) & daylight).sum(1).mean()), hours_G_ge_10_mean=float(daylight.sum(1).mean()),
                       eta_at_600=float(0.72 - 4.5 * (Tin - np.median(Ta[:, 10:15])) / 600.0), frac_GHI_energy_outside_h8_18=float(1 - G[:, 8:19].sum() / G.sum())))
pd.DataFrame(OP).to_csv(OUT + r"\collector_operating_point.csv", index=False)

# ============================ S. sequential rate-limited model ==================
# per-element state: x (liquid fraction); elements = scenario x candidate
SCEN = {}
def add(name, **kw): SCEN[name] = {**dict(A=4.0, m=50.0, FRta=0.72, FRUL=4.5, appr=6.0, delta=0.020, share=0.5, eta=1.0, kmode="nom", Mw=50.0), **kw}
add("NOM"); add("OPT", A=6.0, FRta=0.80, FRUL=3.5, delta=0.010, share=0.4, eta=1.0, kmode="opt"); add("PES", A=2.0, FRta=0.65, FRUL=5.0, delta=0.030, share=0.7, eta=0.85, kmode="pes")
for k, v in [("A=2", dict(A=2.0)), ("A=5", dict(A=5.0)), ("A=6", dict(A=6.0)), ("m=25", dict(m=25.0)), ("m=100", dict(m=100.0)), ("FR_pes", dict(FRta=0.65, FRUL=5.0)), ("FR_opt", dict(FRta=0.80, FRUL=3.5)),
             ("appr=4", dict(appr=4.0)), ("appr=8", dict(appr=8.0)), ("delta=10mm", dict(delta=0.010)), ("delta=30mm", dict(delta=0.030)), ("share=0.4", dict(share=0.4)), ("share=0.7", dict(share=0.7)),
             ("share=1.0", dict(share=1.0)), ("eta=0.85", dict(eta=0.85)), ("k_pes", dict(kmode="pes")), ("k_opt", dict(kmode="opt")), ("Mw=100(O2 200L/d)", dict(Mw=100.0))]:
    add("S:" + k, **v)
snames = list(SCEN); NC = len(db)
E = len(snames) * NC
par = {k: np.zeros(E) for k in ("A", "m", "FRta", "FRUL", "appr", "delta", "share", "eta", "Mw", "Tm", "L", "rho", "ks", "kl", "SC", "k_flag_nonreported")}
scen_idx = np.zeros(E, int); cand_idx = np.zeros(E, int)
for si, sn in enumerate(snames):
    s = SCEN[sn]
    for j in range(NC):
        e = si * NC + j; r = db.iloc[j]; scen_idx[e] = si; cand_idx[e] = j
        for k in ("A", "m", "FRta", "FRUL", "appr", "delta", "share", "eta", "Mw"): par[k][e] = s[k]
        par["Tm"][e] = r.Tm_C; par["L"][e] = r.latent_heat_kJ_kg * 1000.0; par["rho"][e] = r.density_solid_kg_m3 if pd.notna(r.density_solid_kg_m3) else np.nan
        par["SC"][e] = r.supercooling_K if pd.notna(r.supercooling_K) else 0.0
        ks, kl = r.TC_solid_W_mK, r.TC_liquid_W_mK
        nonrep = (str(r.TC_status) != "Reported"); par["k_flag_nonreported"][e] = float(nonrep)
        if nonrep or pd.isna(ks):
            ks = {"nom": (ks if pd.notna(ks) else (TS_MIN + TS_MAX) / 2), "pes": TS_MIN, "opt": TS_MAX}[s["kmode"]]
            kl = {"nom": (kl if pd.notna(kl) else (TL_MIN + TL_MAX) / 2), "pes": TL_MIN, "opt": TL_MAX}[s["kmode"]]
            if s["kmode"] == "nom" and pd.isna(r.TC_solid_W_mK): ks, kl = (TS_MIN + TS_MAX) / 2, (TL_MIN + TL_MAX) / 2   # missing k: mid-range of reported (flagged)
        par["ks"][e], par["kl"][e] = ks, kl
valid = np.isfinite(par["rho"]) & np.isfinite(par["L"]) & np.isfinite(par["Tm"])
Af = np.where(valid, par["m"] / (np.where(valid, par["rho"], 1.0) * par["delta"]), 0.0)      # total contact area (two faces of a slab of half-thickness delta)
mL = par["m"] * par["L"]
Tout_eff = np.minimum(50.0, par["Tm"] - par["appr"])         # A: 50 C when Tm >= 50 + approach; B: preheat target
pathA = (par["Tm"] - par["appr"]) >= 50.0
S_MIN = 5e-4; DT = 900.0
def run_cluster(c):
    G, Ta, Tmn, mon = WD[c]; D = G.shape[0]
    x = np.full(E, 0.5); xref = x.copy()
    ok = np.zeros((D, E), bool); cov_e = np.zeros((D, E), np.float32); cov_m = np.zeros((D, E), np.float32); xpre = np.zeros((D, E), np.float32)
    for d in range(D):
        Tw = Tmn[d]
        duty = np.minimum(par["share"] * par["Mw"] * CP_W * np.maximum(50.0 - Tw, 0.0), par["Mw"] * CP_W * np.maximum(Tout_eff - Tw, 0.0))
        Twm = 0.5 * (Tw + Tout_eff)
        dTd = np.maximum(par["Tm"] - par["SC"] - Twm, 0.0)
        for h in range(24):
            if h in (7, 19):
                xr = x.copy(); deliv = np.zeros(E)
                if h == 19: xpre[d] = x
                for s in range(4):
                    film = np.maximum((xr - x) * par["delta"], S_MIN)
                    Q = np.where(valid, Af * par["ks"] * dTd / film, 0.0)
                    dq = np.minimum(np.minimum(Q * DT, duty / 4.0), x * mL)
                    dq = np.maximum(dq, 0.0); deliv += dq; x = x - dq / np.where(mL > 0, mL, 1.0)
                cv = np.where(duty > 0, np.minimum(deliv / np.where(duty > 0, duty, 1.0), 1.0), 1.0)
                if h == 7: cov_m[d] = cv
                else: cov_e[d] = cv
            elif 8 <= h <= 18:
                g = G[d, h]
                if h == 8: xref = x.copy()
                if g < 10.0: continue
                Tin = par["Tm"] + par["appr"]
                for s in range(4):
                    Qc = par["A"] * np.maximum(par["FRta"] * g - par["FRUL"] * (Tin - Ta[d, h]), 0.0) * par["eta"]
                    film = np.maximum((x - xref) * par["delta"], S_MIN)
                    Qr = np.where(valid, Af * par["kl"] * par["appr"] / film, 0.0)
                    dq = np.minimum(Qc, Qr) * DT
                    x = np.minimum(1.0, x + dq / np.where(mL > 0, mL, 1.0))
            elif h == 8:
                xref = x.copy()
        ok[d] = (cov_m[d] >= 0.999) & (cov_e[d] >= 0.999)
    sp = 5
    res = pd.DataFrame(dict(scen=[snames[i] for i in scen_idx], cand=cand_idx))
    res["R_all"] = ok[sp:].mean(0)
    res["R_wm"] = [min(ok[sp:][mon[sp:] == m, e].mean() for m in range(1, 13)) for e in range(E)]
    res["cov_eve_mean"] = cov_e[sp:].mean(0); res["cov_morn_mean"] = cov_m[sp:].mean(0); res["x_pre_eve_mean"] = xpre[sp:].mean(0)
    res["evaluable"] = valid
    res["pathA"] = pathA
    return res
RES = {}
t0 = time.time()
for c in (0, 1, 2):
    RES[c] = run_cluster(c); RES[c]["cluster"] = c
    print("cluster", c, "done", round(time.time() - t0), "s", flush=True)
allr = pd.concat(RES.values(), ignore_index=True)
allr["pcm_id"] = db.pcm_id.values[allr.cand.values]; allr["name"] = db.product_name.values[allr.cand.values]; allr["Tm"] = db.Tm_C.values[allr.cand.values]
allr.to_csv(OUT + r"\sequential_results_all_scenarios.csv", index=False)
json.dump(R, open(OUT + r"\verification.json", "w"), indent=1, default=str)
print(json.dumps(R["medoids"], indent=1, default=str)); print(json.dumps(R["medoid_hardcoded_in_08_08b"], default=str)); print(json.dumps(R["weather_consistency"], indent=1, default=str))
print("k range", R["k_reported_range"]); print(R["m_pcm_bounds_M1"])
