"""
Verification Script 1: Preprocessing and QC - Tamil Nadu
Output: data/plots/verify_preprocessing/
Run AFTER: 04_preprocess_tamilnadu.py
Checks distributions, completeness, correlations, engineered features.
"""
import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW  = os.path.join(BASE,"data","processed","climate_tamilnadu_points.csv")
PRE  = os.path.join(BASE,"data","preprocessed","tamilnadu_cleaned_physical.csv")
OUT  = os.path.join(BASE,"data","plots","verify_preprocessing")
os.makedirs(OUT, exist_ok=True)

print("Loading data (first 500k rows of raw for speed)...")
try:
    raw = pd.read_csv(RAW, nrows=500000)
    pre = pd.read_csv(PRE, nrows=200000)
    print(f"  Raw: {raw.shape}  Pre: {pre.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}"); raise SystemExit(1)

KEY = ["era5_T_amb","era5_RHum","era5_W_spd","era5_P_atm","era5_GHI","era5_precipitation"]
av  = [v for v in KEY if v in pre.columns]

print(f"Core vars: {av}")

# 1. Distributions
fig,axes=plt.subplots(len(av),1,figsize=(12,3*len(av)))
if len(av)==1: axes=[axes]
for i,var in enumerate(av):
    vals=pre[var].dropna()
    axes[i].hist(vals,bins=60,alpha=0.8,color="steelblue",edgecolor="white")
    axes[i].set_title(f"{var}  mean={vals.mean():.2f}  std={vals.std():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")
    axes[i].set_ylabel("Frequency"); axes[i].grid(alpha=0.3)
plt.suptitle("Tamil Nadu - Climate Variable Distributions (Preprocessed)",fontsize=14,y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"01_climate_distributions.png"),dpi=150); plt.close(); print("  01_climate_distributions.png")

# 2. Data Completeness
fig,ax=plt.subplots(figsize=(12,6))
cov=100*(1-pre[av].isna().sum()/len(pre))
bars=ax.bar(range(len(av)),cov.values,color="steelblue",edgecolor="black")
ax.set_xticks(range(len(av))); ax.set_xticklabels(av,rotation=45,ha="right")
ax.set_ylabel("Data Coverage (%)"); ax.set_title("Tamil Nadu - Data Completeness (Preprocessed)")
ax.axhline(95,color="green",ls="--",lw=2,label=">95% Good"); ax.axhline(90,color="orange",ls="--",lw=2,label=">90% Fair"); ax.axhline(80,color="red",ls="--",lw=2,label="<90% Poor")
ax.set_ylim(0,105); ax.legend(); ax.grid(alpha=0.3,axis="y")
for b in bars:
    h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h,f"{h:.1f}%",ha="center",va="bottom",fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"02_data_completeness.png"),dpi=150); plt.close(); print("  02_data_completeness.png")

# 3. Statistical Summary
fig,ax=plt.subplots(figsize=(12,6))
sd=[{"Variable":v,"Mean":pre[v].mean(),"Std":pre[v].std()} for v in av]
pd.DataFrame(sd).set_index("Variable")[["Mean","Std"]].plot(kind="bar",ax=ax,width=0.8,color=["steelblue","coral"])
ax.set_title("Tamil Nadu - Statistical Summary of Climate Variables"); ax.set_ylabel("Value"); ax.grid(alpha=0.3,axis="y")
plt.xticks(rotation=45,ha="right"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"03_statistical_summary.png"),dpi=150); plt.close(); print("  03_statistical_summary.png")

# 4. Engineered features
eng=[c for c in pre.columns if any(x in c for x in ["_lag","_roll","_delta"])]
if eng:
    fig,axes=plt.subplots(min(3,len(eng)),1,figsize=(12,4*min(3,len(eng))))
    if len(eng)==1: axes=[axes]
    for i,col in enumerate(eng[:3]):
        d=pre[col].dropna(); axes[i].hist(d,bins=50,alpha=0.8,color="purple",edgecolor="white")
        axes[i].set_title(f"{col}  mean={d.mean():.3f}  std={d.std():.3f}  n={len(d)}"); axes[i].set_ylabel("Freq"); axes[i].grid(alpha=0.3)
    plt.suptitle("Engineered Features (Lag/Rolling/Delta)",fontsize=14); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"04_feature_engineering.png"),dpi=150); plt.close(); print("  04_feature_engineering.png")
else: print("  no engineered features found")

# 5. Correlation heatmap
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(pre[av].corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,ax=ax,vmin=-1,vmax=1,cbar_kws={"label":"Pearson r"})
ax.set_title("Tamil Nadu - Climate Variable Correlations (Preprocessed)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"05_correlation_analysis.png"),dpi=150); plt.close(); print("  05_correlation_analysis.png")

# 6. Data Quality Metrics
fig,ax=plt.subplots(figsize=(12,6))
qm={"Total Records":len(pre),"Complete Cases":len(pre.dropna()),"Core Climate Vars":len(av),"Engineered Features":len(eng),"Variables":len(pre.columns)}
y=np.arange(len(qm)); bars=ax.barh(y,list(qm.values()),color="steelblue",edgecolor="black")
ax.set_yticks(y); ax.set_yticklabels(list(qm.keys())); ax.set_xlabel("Count"); ax.set_title("Tamil Nadu - Data Quality Metrics (Preprocessed)")
ax.grid(alpha=0.3,axis="x")
for b in bars:
    w=b.get_width(); ax.text(w,b.get_y()+b.get_height()/2,str(int(w)),ha="left",va="center",fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"06_data_quality_metrics.png"),dpi=150); plt.close(); print("  06_data_quality_metrics.png")

# 7. Summary text
fig,ax=plt.subplots(figsize=(12,8))
txt=f"PREPROCESSING VERIFICATION SUMMARY (Tamil Nadu)\n{'='*60}\nInput records: {len(raw):,}\nOutput records: {len(pre):,}\nData retention: {100*len(pre)/len(raw):.1f}%\nInput dims: {raw.shape[1]}   Output dims: {pre.shape[1]}\nEngineered features: {len(eng)}\n\nData Quality:\n"
for v in av:
    cov_v=100*(1-pre[v].isna().sum()/len(pre)); ok="OK" if cov_v>95 else ("Fair" if cov_v>90 else "WARN")
    txt+=f"  {v}: {cov_v:.1f}%  [{ok}]\n"
ax.text(0.05,0.95,txt,transform=ax.transAxes,fontsize=10,va="top",fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.8)); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"07_preprocessing_summary.png"),dpi=150); plt.close(); print("  07_preprocessing_summary.png")

print("\nAll verification plots saved to:", OUT)
