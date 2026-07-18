"""任务3: 修订数据源消融实验 (P4阶段, 简化RF)"""
import pandas as pd, numpy as np, json, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

DATA_PATH = r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv"
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "yield"
META = ["Year", "Zone", "latitude", "longitude", TARGET]

SOIL_KW = ["Sand", "Clay", "SOC"]
METEO_KW = ["Tmean", "PPT", "SM", "GDD", "FDD", "VPD", "VPD_max"]
RS_KW = ["NDVI", "NDVI_max", "NDWI", "NIRv", "EVI", "EVI_max"]
CUM_KW = ["Cum_PPT", "Cum_FDD", "Cum_VPD"]
DELTA_KW = ["Delta_NDVI", "Delta_NDWI", "Delta_SM"]
COMP_KW = ["WUE", "Decoupling_Stress", "Thermal_Efficiency", "Hydrothermal_Balance", "Drought_Vulnerability", "Fertility_Vigor"]

GROUPS = ["Soil-Raw", "Meteorology-Raw", "RS-Raw", "Raw-All", "Derived-Only", "Full-All"]
STAGES = [("P4", ["P1","P2","P3","P4"])]

RF = dict(n_estimators=100, random_state=42, n_jobs=-1, max_features="sqrt", min_samples_leaf=1, verbose=0)

def get_features(df, prefixes, group):
    cols = df.columns.tolist()
    r = []
    for c in cols:
        if c in META: continue
        is_soil = c in SOIL_KW
        is_meteo = any(f"{p}_{k}" == c for p in prefixes for k in METEO_KW)
        is_rs = any(f"{p}_{k}" == c for p in prefixes for k in RS_KW)
        is_cum = any(f"{k}_{p}" == c for p in prefixes for k in CUM_KW)
        is_delta = any(f"{k}_{p}" == c for p in ["P2","P3","P4"] if p in prefixes for k in DELTA_KW)
        is_comp = any(f"{p}_{k}" == c for p in prefixes for k in COMP_KW)

        if group == "Soil-Raw" and is_soil: r.append(c)
        elif group == "Meteorology-Raw" and is_meteo: r.append(c)
        elif group == "RS-Raw" and is_rs: r.append(c)
        elif group == "Raw-All" and (is_soil or is_meteo or is_rs): r.append(c)
        elif group == "Derived-Only" and (is_cum or is_delta or is_comp): r.append(c)
        elif group == "Full-All" and (is_soil or is_meteo or is_rs or is_cum or is_delta or is_comp): r.append(c)
    return sorted(set(r))

def metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0: return {k: np.nan for k in ["R2","RMSE","MAE","MAPE","RRMSE","d_index"]}
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return dict(R2=r2_score(yt, yp), RMSE=rmse, MAE=mean_absolute_error(yt, yp),
                MAPE=np.mean(np.abs((yt-yp)/np.where(yt==0,np.nan,yt)))*100,
                RRMSE=rmse/np.mean(yt)*100,
                d_index=(1 - np.sum((yp-yt)**2)/np.sum((np.abs(yp-np.mean(yt))+np.abs(yt-np.mean(yt)))**2)) if np.sum((np.abs(yp-np.mean(yt))+np.abs(yt-np.mean(yt)))**2)!=0 else 0)

print("=" * 60)
print("  修订数据源消融实验 (P4 only, RF n_est=100)")
print("=" * 60)

df = pd.read_csv(DATA_PATH).replace(-9999, np.nan).dropna(subset=[TARGET])
years = sorted(df["Year"].dropna().unique().astype(int))
print(f"[INFO] {len(years)} 折 LOYO, 样本: {len(df)}, 耗时约 5-10 min")

results, preds, feat_map = [], [], {}
n_total = len(STAGES) * len(GROUPS) * len(years)
n_done = 0

for slabel, sp in STAGES:
    for grp in GROUPS:
        fcols = get_features(df, sp, grp)
        feat_map[f"{slabel}_{grp}"] = fcols
        print(f"\n[STAGE] {slabel} | [GROUP] {grp} | 特征: {len(fcols)}")
        y_grp_t, y_grp_p = [], []
        yearly = {k: [] for k in ["R2","RMSE","MAE","MAPE","RRMSE","d_index"]}
        for ty in years:
            n_done += 1
            train = df[df["Year"] != ty]
            test = df[df["Year"] == ty]
            avail = [c for c in fcols if c in train.columns]
            if not avail: continue
            Xtr = train[avail].fillna(train[avail].median())
            Xte = test[avail].fillna(Xtr.median())
            ytr = train[TARGET].values
            yte = test[TARGET].values
            rf = RandomForestRegressor(**RF)
            rf.fit(Xtr, ytr)
            yp = rf.predict(Xte)
            m = metrics(yte, yp)
            for k in yearly: yearly[k].append(m[k])
            y_grp_t.extend(yte); y_grp_p.extend(yp)
            preds.append(pd.DataFrame({"Year": [int(ty)]*len(yte), "yield_true": yte, "yield_pred": yp,
                                       "stage": slabel, "feature_group": grp}))
            print(f"  [{n_done}/{n_total}] Year={ty} R2={m['R2']:.3f} MAPE={m['MAPE']:.1f}%", flush=True)
        s = dict(stage=slabel, feature_group=grp, n_features=len(fcols), n_years=len(yearly["R2"]))
        for k in yearly:
            vals = [v for v in yearly[k] if not np.isnan(v)]
            s[k] = float(np.mean(vals)) if vals else np.nan
        results.append(s)
        print(f"  Avg: R2={s['R2']:.3f} RRMSE={s['RRMSE']:.1f}% MAPE={s['MAPE']:.1f}%")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUTPUT_DIR, "rf_ablation_revised_overall.csv"), index=False)
pd.concat(preds, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "rf_ablation_revised_predictions.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "rf_ablation_revised_features.json"), "w", encoding="utf-8") as f:
    json.dump(feat_map, f, indent=2, ensure_ascii=False)

for metric in ["MAPE", "RRMSE"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp in GROUPS:
        sub = rdf[rdf["feature_group"]==grp].sort_values("stage")
        ax.plot(sub["stage"], sub[metric], marker="o", label=grp, linewidth=2)
    ax.set_xlabel("Stage"); ax.set_ylabel(f"{metric} (%)"); ax.set_title(f"Revised Ablation - {metric}")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, f"rf_ablation_revised_{metric.lower()}.png"), dpi=150)
    plt.close()

print("\n完成!")
