"""ISOLATED (scratch): candidate-level cluster-specific classification from the scenario results + evidence categories. Writes only to scratchpad/obj1_method/."""
import sys, os, json
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
O1 = r"M:\Final_year_pro\PCM-Selection-ML-model\era5-assam"
SCR = r"C:\Users\hp\AppData\Local\Temp\claude\0189d894ab42dcbb\de02d978-3298-4895-a1f7-5f4bfe890c04\scratchpad"
OUT = os.path.join(SCR, "obj1_method")
db = pd.read_csv(O1 + r"\data\processed\pcm\pcm_database_final.csv")
src = pd.read_csv(r"M:\Final_year_pro\PCM-Selection-ML-model\PCM_data\data\PCM_Properties_cleaned_mice_pmm_detailed.csv")
db = db.merge(src[["product", "flash_point", "flash_point_imputed", "max_op_temp", "max_op_temp_imputed"]].rename(columns={"product": "product_name"}), on="product_name", how="left")
db["name"] = db.product_name.str.replace("\ufffd", "").str.replace("Â", "").str.replace("®", "")
seq = pd.read_csv(OUT + r"\sequential_results_all_scenarios.csv")
R_THR = 0.5     # Rajasthan-consistent: duty met on the median day of the worst calendar month (see methodology doc)
APPR = 6.0; T_HOT = 75.0
FA = ["Myristic", "Palmitic", "Stearic", "Lauric"]

def cat_safety(r):
    fr = (str(r.flash_point_imputed) == "False")
    flam_rep = str(r.flammability_status) == "Reported"
    if fr and r.flash_point <= T_HOT: return "CONFIRMED_UNACCEPTABLE", f"reported flash point {r.flash_point:g} C <= {T_HOT:g} C"
    if flam_rep and str(r.flammability).lower() in ("highly flammable", "extremely flammable", "toxic"): return "CONFIRMED_UNACCEPTABLE", "reported high-severity hazard"
    if fr and r.flash_point >= 93.0:
        return "CONDITIONAL_ON_CONTROLS", f"manufacturer flash point {r.flash_point:g} C (>=93 C, far above max operating {T_HOT:g} C); flammability flag {r.flammability}/{r.flammability_status}; SDS + enclosure review needed"
    if flam_rep and str(r.flammability).lower() == "no": return "CONDITIONAL_ON_CONTROLS", "manufacturer flag flammable=No (Reported); flash point not sourced (imputed)"
    return "UNKNOWN", "flammability and flash point are imputed/missing"

def cat_corr(r):
    if any(f in r["name"] for f in FA): return "CONDITIONAL_ON_CONTAINMENT", "fatty acid: literature indicates Cu/Al attack (not cited in repo); needs stainless/polymer containment and coupon evidence"
    return "UNKNOWN", "no material-specific compatibility evidence in the database (no corrosion column)"

def cat_cyc(r):
    if str(r.cycles_status) == "Reported": return ("CONFIRMED", f"{int(r.cycles_tested)} cycles reported") if r.cycles_tested >= 300 else ("FAIL", f"{int(r.cycles_tested)} cycles < 300")
    return "UNKNOWN", f"cycles {r.cycles_status.lower()}"

def cat_sc(r):
    if pd.notna(r.supercooling_K) and str(r.supercooling_status) == "Reported": return ("CONFIRMED_DERIVED", f"{r.supercooling_K:g} K (Tm_melt - Tm_freeze, both reported; not a measured supercooling)") if r.supercooling_K <= 8 else ("FAIL", f"{r.supercooling_K:g} K > 8 K")
    return "UNKNOWN", f"supercooling {str(r.supercooling_status).lower()}"

MEAN_MAINS = {}
sys.path.insert(0, r"M:\Final_year_pro\Objective---2\objective2-assam\objective2-assam-fpc")
from src.config_loader import load_config as _lc
from src.io_weather import load_regime_weather as _lw
for _c in (0, 1, 2): MEAN_MAINS[_c] = float(_lw(_c, _lc()).data.groupby("ist_date").T_mains_C.first().mean())
rows = []
sc = {s: seq[seq.scen == s].set_index(["cluster", "cand"]) for s in ("NOM", "OPT", "S:k_pes", "PES")}
for c in (0, 1, 2):
    for j, r in db.iterrows():
        pathA = (r.Tm_C - APPR) >= 50.0
        pw = "A_DIRECT_DELIVERY (Tm >= 50 + approach)" if pathA else "B_PREHEAT (auxiliary boost required)"
        saf, saf_r = cat_safety(r); cor, cor_r = cat_corr(r); cyc, cyc_r = cat_cyc(r); sub, sub_r = cat_sc(r)
        nom, opt, kp = sc["NOM"].loc[(c, j)], sc["OPT"].loc[(c, j)], sc["S:k_pes"].loc[(c, j)]
        evaluable = bool(nom.evaluable)
        L_imp = str(r.latent_heat_status) != "Reported"; k_state = str(r.TC_status)
        reasons = []; status = None
        hard = []
        if not (42.0 <= r.Tm_C <= 70.0): hard.append(f"Tm {r.Tm_C:g} C outside the 42-70 C SWH band")
        if cyc == "FAIL": hard.append("cycling " + cyc_r)
        if sub == "FAIL": hard.append("supercooling " + sub_r)
        if saf == "CONFIRMED_UNACCEPTABLE": hard.append("safety " + saf_r)
        if hard: status = "REJECTED"; reasons = hard
        elif not evaluable or L_imp:
            status = "UNVERIFIED"
            if not evaluable: reasons.append("solid density missing: charge/discharge screen cannot be evaluated")
            if L_imp: reasons.append("latent heat is imputed, not reported: storage capacity unverified")
            if str(r.TC_status) == "Missing": reasons.append("thermal conductivity missing")
            if evaluable and not bool(opt.R_wm >= R_THR): status = "REJECTED"; reasons = ["fails the duty screen even at the most favourable design corner"]
        elif not (opt.R_wm >= R_THR):
            status = "REJECTED"; reasons = [f"duty reliability (worst-month median day) {opt.R_wm:.2f} < {R_THR} even at the most favourable corner (A=6 m2, delta=10 mm, share 0.4, FR(ta)=0.80/FRUL=3.5, k at the reported maximum)"]
        else:
            nom_ok = bool(nom.R_wm >= R_THR); kp_ok = bool(kp.R_wm >= R_THR)
            ev_ok = (cyc == "CONFIRMED" and sub in ("CONFIRMED_DERIVED",) and cor == "CONFIRMED" and saf == "CONFIRMED_ACCEPTABLE" and k_state == "Reported")
            if nom_ok and kp_ok and ev_ok: status = "FEASIBLE"
            else:
                status = "CONDITIONAL"
                if not nom_ok: reasons.append(f"passes only at the favourable design corner (nominal R_wm {nom.R_wm:.2f}); needs larger A/m and/or thinner PCM layer")
                elif not kp_ok and k_state != "Reported": reasons.append("passes at nominal only if the imputed conductivity is right (fails at the lowest reported k)")
                if cyc != "CONFIRMED": reasons.append("cycling evidence " + cyc_r)
                if cor != "CONFIRMED": reasons.append("corrosion: " + cor_r)
                if saf != "CONFIRMED_ACCEPTABLE": reasons.append("safety: " + saf_r)
                if sub != "CONFIRMED_DERIVED": reasons.append("supercooling " + sub_r)
        # storage capacity (analytic, cluster mean mains)
        Dd = 2 * 0.5 * 50.0 * 4.186 * (50 - MEAN_MAINS[c])
        cap_ratio = 50.0 * r.latent_heat_kJ_kg / Dd
        rows.append(dict(cluster=c, capacity_ratio_nominal=cap_ratio, pcm_id=int(r.pcm_id), name=r["name"], Tm_C=r.Tm_C, L_kJ_kg=r.latent_heat_kJ_kg, L_status=r.latent_heat_status,
                         k_solid=r.TC_solid_W_mK, k_liquid=r.TC_liquid_W_mK, k_status=k_state, rho_solid=r.density_solid_kg_m3, rho_status=r.density_solid_status,
                         cycles=cyc, cycles_detail=cyc_r, supercooling=sub, supercooling_detail=sub_r, corrosion=cor, corrosion_detail=cor_r, safety=saf, safety_detail=saf_r,
                         pathway=pw, charge_mean_melt_before_evening=float(nom.x_pre_eve_mean), charge_R_wm_NOM=float(nom.R_wm), duty_R_all_NOM=float(nom.R_all),
                         discharge_cov_eve=float(nom.cov_eve_mean), discharge_cov_morn=float(nom.cov_morn_mean), R_wm_OPT=float(opt.R_wm), R_wm_kpes=float(kp.R_wm),
                         flash_point_C=(r.flash_point if str(r.flash_point_imputed) == "False" else np.nan), flash_imputed=str(r.flash_point_imputed), max_op_temp_C=r.max_op_temp, max_op_imputed=str(r.max_op_temp_imputed),
                         status=status, reason="; ".join(reasons)))
T = pd.DataFrame(rows)
T.to_csv(OUT + r"\candidate_status_all_clusters.csv", index=False)
counts = {}
for c in (0, 1, 2):
    t = T[T.cluster == c]
    counts[c] = {"all": t.status.value_counts().to_dict(), "A": t[t.pathway.str.startswith("A")].status.value_counts().to_dict(), "B": t[t.pathway.str.startswith("B")].status.value_counts().to_dict(),
                 "nominal_design_pass": int(((t.charge_R_wm_NOM >= R_THR) & t.status.isin(["CONDITIONAL", "FEASIBLE"])).sum()), "n_pathA": int(t.pathway.str.startswith("A").sum()), "n_pathB": int(t.pathway.str.startswith("B").sum())}
# status differences between clusters
piv = T.pivot(index="pcm_id", columns="cluster", values="status")
counts["candidates_with_cluster_dependent_status"] = int((piv.nunique(axis=1) > 1).sum())
counts["cluster_dependent_ids"] = piv[piv.nunique(axis=1) > 1].index.tolist()
# R_THR sensitivity on NOM pass counts
sens = []
for thr in (0.25, 0.5, 0.75, 0.9):
    for c in (0, 1, 2):
        t = T[(T.cluster == c) & T.status.isin(["CONDITIONAL", "FEASIBLE"])]
        for pw in ("A", "B"):
            u = t[t.pathway.str.startswith(pw)]
            sens.append(dict(threshold=thr, cluster=c, pathway=pw, n_candidates=len(u), nominal_pass=int((u.charge_R_wm_NOM >= thr).sum()), kpes_pass=int((u.R_wm_kpes >= thr).sum()), opt_pass=int((u.R_wm_OPT >= thr).sum())))
pd.DataFrame(sens).to_csv(OUT + r"\threshold_sensitivity.csv", index=False)
# scenario tornado
tor = []
for s in sorted(seq.scen.unique()):
    for c in (0, 1, 2):
        x = seq[(seq.scen == s) & (seq.cluster == c) & seq.evaluable]
        for pw, flag in (("A", True), ("B", False)):
            y = x[x.pathA == flag]
            tor.append(dict(scenario=s, cluster=c, pathway=pw, n_eval=len(y), n_pass_Rwm_ge_0p5=int((y.R_wm >= R_THR).sum()), median_R_all=float(y.R_all.median())))
pd.DataFrame(tor).to_csv(OUT + r"\scenario_sensitivity_counts.csv", index=False)
# pathway boundary by approach
bd = [dict(approach_K=a, n_pathA=int((db.Tm_C - a >= 50).sum() - (db.Tm_C > 70).sum()), Tm_min_for_direct_delivery=50 + a) for a in (2, 3, 4, 5, 6, 7, 8)]
pd.DataFrame(bd).to_csv(OUT + r"\pathway_boundary_by_approach.csv", index=False)
# preheat fraction (Path B), cluster-specific
import importlib
sys.path.insert(0, r"M:\Final_year_pro\Objective---2\objective2-assam\objective2-assam-fpc")
from src.config_loader import load_config
from src.io_weather import load_regime_weather
cfg = load_config(); pre = []
for c in (0, 1, 2):
    d = load_regime_weather(c, cfg).data; Tw = d.groupby("ist_date").T_mains_C.first().values
    for Tm in (44, 47, 49, 52, 55):
        Tp = Tm - APPR; f = (Tp - Tw) / (50.0 - Tw)
        pre.append(dict(cluster=c, Tm=Tm, T_pre=Tp, f_pre_mean=float(np.mean(f)), f_pre_cold=float(np.mean(f[Tw < np.quantile(Tw, 0.1)])), aux_kWh_per_50kg_draw_mean=float(np.mean(50 * 4.186 * (50 - Tp) / 3600.0))))
pd.DataFrame(pre).to_csv(OUT + r"\preheat_fraction.csv", index=False)
# latent / capacity derivation
lat = []
for c in (0, 1, 2):
    d = load_regime_weather(c, cfg).data; Tw = d.groupby("ist_date").T_mains_C.first()
    mon = pd.to_datetime(Tw.index).month; Twv = Tw.values
    for basis, Mw in (("Obj1 spec 50+50 L", 50.0), ("Obj2 baseline 100+100 kg", 100.0)):
        for share in (0.4, 0.5, 0.7, 1.0):
            Dday = 2 * share * Mw * 4.186 * (50 - Twv)            # kJ/day (two draws)
            for xuse in (1.0, 0.8):
                Lmin_mean = Dday.mean() / (50.0 * xuse); Lmin_p95 = np.quantile(Dday, .95) / (50.0 * xuse)
                n_below = int(((db.latent_heat_kJ_kg < Lmin_mean)).sum()); n_below95 = int((db.latent_heat_kJ_kg < Lmin_p95).sum())
                lat.append(dict(cluster=c, basis=basis, share=share, x_use=xuse, D_day_MJ_mean=float(Dday.mean() / 1000.0), L_min_kJ_kg_m50_mean=float(Lmin_mean), L_min_p95=float(Lmin_p95), n_cand_below_mean=n_below, n_cand_below_p95=n_below95,
                                m_req_kg_at_L200_mean=float(Dday.mean() / (200.0 * xuse)), m_req_kg_at_L200_p95=float(np.quantile(Dday, .95) / (200.0 * xuse))))
pd.DataFrame(lat).to_csv(OUT + r"\latent_requirement_sensitivity.csv", index=False)
json.dump(counts, open(OUT + r"\status_counts.json", "w"), indent=1, default=str)
print(json.dumps(counts, indent=1, default=str))
print(T.groupby(["cluster", "status"]).size().unstack())
print(T[T.status == "REJECTED"][["cluster", "name", "Tm_C", "reason"]].to_string())
print(T[(T.cluster == 0) & (T.status == "UNVERIFIED")][["name", "reason"]].to_string())
print(pd.DataFrame(bd))
