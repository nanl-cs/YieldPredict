"""
任务2：原始特征与派生特征敏感性实验
目的：验证派生特征是否真正改善跨年预测，而不仅在SHAP中表现重要。

统一要求：
- 使用同一份P4数据
- 使用完全相同的LOYO划分
- XGBRFRegressor + Optuna MOBO
- 固定随机种子42
- 所有预处理/特征筛选/调参仅在训练年份内部完成

组：
  Raw: 仅原始变量+三项土壤变量（~55维）
  Raw+Cumulative+Delta: Raw + 累积量 + 阶段差分（~76维）
  Full: 全部100维特征

输出：
  derived_feature_sensitivity_by_year.csv
  derived_feature_sensitivity_overall.csv
  derived_feature_sensitivity_predictions.csv
  derived_feature_sensitivity_best_params.csv
  derived_feature_sensitivity_selected_features.json
"""
import pandas as pd
import numpy as np
import os
import warnings
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRFRegressor
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================== 配置 ========================
DATA_PATH = r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv"
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTUNA_TRIALS = 50
RANDOM_SEED = 42
META_COLS = ["Year", "Zone", "latitude", "longitude", "yield"]
STAGES = ["P1", "P2", "P3", "P4"]

# ======================== 特征分组定义 ========================
# 原始构成变量
RAW_METEO = ["Tmean", "PPT", "SM", "GDD", "FDD", "VPD", "VPD_max"]
RAW_RS = ["NDVI", "NDVI_max", "NDWI", "NIRv", "EVI", "EVI_max"]
SOIL = ["Sand", "Clay", "SOC"]

# 累积量 (不含 Cum_GDD，已被黑名单排除)
CUM_VARS = ["Cum_PPT", "Cum_FDD", "Cum_VPD"]

# 阶段差分
DELTA_VARS = ["Delta_NDVI", "Delta_NDWI", "Delta_SM"]

# 六大复合代理特征
COMPOSITE_KEYWORDS = [
    "WUE", "Decoupling_Stress", "Thermal_Efficiency",
    "Hydrothermal_Balance", "Drought_Vulnerability", "Fertility_Vigor",
]

def build_feature_list(df, group):
    """根据分组名构建特征列表"""
    all_cols = [c for c in df.columns if c not in META_COLS]
    selected = set()

    # Raw/All 都包含的基本变量
    if group in ("raw", "raw_cum_delta", "full", "full_wue",
                 "full_hydrothermal", "full_drought", "full_all_ratio",
                 "full_raw_only"):
        selected.update(SOIL)
        for stage in STAGES:
            for var in RAW_METEO + RAW_RS:
                col = f"{stage}_{var}"
                if col in all_cols:
                    selected.add(col)

    # 累积量和差分
    if group in ("raw_cum_delta", "full", "full_wue",
                 "full_hydrothermal", "full_drought", "full_all_ratio",
                 "full_raw_only"):
        for stage in STAGES:
            for var in CUM_VARS:
                col = f"{var}_{stage}"
                if col in all_cols:
                    selected.add(col)
        for stage in ["P2", "P3", "P4"]:
            for var in DELTA_VARS:
                col = f"{var}_{stage}"
                if col in all_cols:
                    selected.add(col)

    # 复合代理特征
    if group in ("full", "full_wue", "full_hydrothermal", "full_drought",
                 "full_all_ratio", "full_raw_only"):
        for stage in STAGES:
            for kw in COMPOSITE_KEYWORDS:
                col = f"{stage}_{kw}"
                if col in all_cols:
                    selected.add(col)

    # 剔除特定类型
    if group in ("full_raw_only", "full_all_ratio"):
        for stage in STAGES:
            for kw in COMPOSITE_KEYWORDS:
                if kw != "Fertility_Vigor" or group == "full_raw_only":
                    col = f"{stage}_{kw}"
                    selected.discard(col)
    if group == "full_wue":
        for stage in STAGES:
            selected.discard(f"{stage}_WUE")
    if group == "full_hydrothermal":
        for stage in STAGES:
            selected.discard(f"{stage}_Hydrothermal_Balance")
    if group == "full_drought":
        for stage in STAGES:
            selected.discard(f"{stage}_Drought_Vulnerability")

    return sorted(selected)

# ======================== 指标计算 ========================
def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mean_true = np.mean(y_true)
    rrmse = (rmse / mean_true) * 100 if mean_true != 0 else 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    den = np.sum((np.abs(y_pred - mean_true) + np.abs(y_true - mean_true)) ** 2)
    d_index = 1 - (np.sum((y_pred - y_true) ** 2) / den) if den != 0 else 0
    return {
        "R2": round(float(r2), 3), "RMSE": round(float(rmse), 3),
        "RRMSE(%)": round(float(rrmse), 3), "MAE": round(float(mae), 3),
        "MAPE(%)": round(float(mape), 3), "d-index": round(float(d_index), 3),
    }

# ======================== MOBO目标函数 ========================
def objective(trial, X_df, y_np, groups):
    active_features = []
    for col in X_df.columns:
        if trial.suggest_categorical(f"mask_{col}", [True, False]):
            active_features.append(col)
    if len(active_features) == 0:
        return float("inf"), len(X_df.columns)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "tree_method": "gpu_hist", "random_state": RANDOM_SEED, "n_jobs": -1,
    }

    logo = LeaveOneGroupOut()
    mae_scores = []
    X_np_subset = X_df[active_features].values
    for train_idx, val_idx in logo.split(X_np_subset, y_np, groups):
        X_tr, X_val = X_np_subset[train_idx], X_np_subset[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        model = XGBRFRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))
    return np.mean(mae_scores), len(active_features)

# ======================== 主实验 ========================
GROUPS_TO_RUN = [
    "raw", "raw_cum_delta", "full",
    "full_wue", "full_hydrothermal", "full_drought", "full_all_ratio",
]

print(f"{'='*70}")
print(f">>> 特征敏感性实验启动")
print(f"    特征分组: {GROUPS_TO_RUN}")
print(f"    Optuna寻优次数: {OPTUNA_TRIALS}")
print(f"{'='*70}")

df = pd.read_csv(DATA_PATH)
years = sorted(df["Year"].unique())

all_by_year = []
all_overall = []
all_predictions = []
all_best_params = []
all_selected_features = {}

for group in GROUPS_TO_RUN:
    feature_cols = build_feature_list(df, group)
    print(f"\n--- [{group}] 特征数: {len(feature_cols)} ---")
    print(f"  前10: {feature_cols[:10]}")

    all_y_true_group, all_y_pred_group = [], []
    fold_results = []

    for test_year in years:
        train_df = df[df["Year"] != test_year]
        test_df = df[df["Year"] == test_year]
        X_train_df = train_df[feature_cols]
        y_train_np = train_df["yield"].values
        groups_train = train_df["Year"].values
        X_test_df = test_df[feature_cols]
        y_test_np = test_df["yield"].values

        print(f"    [Year {test_year}] Optuna搜索中...", end="", flush=True)

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=TPESampler(seed=RANDOM_SEED),
        )
        func = lambda trial: objective(trial, X_train_df, y_train_np, groups_train)
        study.optimize(func, n_trials=OPTUNA_TRIALS)

        pareto_front = study.best_trials
        best_trial = sorted(pareto_front, key=lambda t: t.values[0])[0]

        best_params = {k: v for k, v in best_trial.params.items() if not k.startswith("mask_")}
        best_params.update({"tree_method": "gpu_hist", "random_state": RANDOM_SEED, "n_jobs": -1})

        selected_features = [col for col in feature_cols if best_trial.params.get(f"mask_{col}", False)]

        X_train_final = X_train_df[selected_features].values
        X_test_final = X_test_df[selected_features].values

        final_model = XGBRFRegressor(**best_params)
        final_model.fit(X_train_final, y_train_np)
        y_pred = final_model.predict(X_test_final)

        all_y_true_group.extend(y_test_np)
        all_y_pred_group.extend(y_pred)

        m = calculate_metrics(y_test_np, y_pred)
        m["test_year"] = str(int(test_year))
        m["feature_group"] = group
        m["n_features_total"] = len(feature_cols)
        m["n_features_selected"] = len(selected_features)
        fold_results.append(m)

        all_predictions.append(pd.DataFrame({
            "Year": test_df["Year"].values,
            "Zone": test_df["Zone"].values,
            "yield_true": y_test_np,
            "yield_pred": y_pred,
            "feature_group": group,
        }))

        best_param_record = {"test_year": str(int(test_year)), "feature_group": group}
        for k in ["max_depth", "colsample_bynode", "subsample", "n_estimators"]:
            best_param_record[k] = best_params.get(k, None)
        best_param_record["selected_features"] = json.dumps(selected_features)
        best_param_record["n_features_selected"] = len(selected_features)
        best_param_record["inner_mae"] = float(best_trial.values[0])
        best_param_record["inner_n_features"] = int(best_trial.values[1])
        all_best_params.append(best_param_record)

        all_selected_features[f"{group}_year{int(test_year)}"] = selected_features

        print(f" 完成. MAE_sel={len(selected_features)}/{len(feature_cols)} RRMSE={m['RRMSE(%)']:.2f}%")

    global_m = calculate_metrics(all_y_true_group, all_y_pred_group)
    global_m["test_year"] = "Overall"
    global_m["feature_group"] = group
    global_m["n_features_total"] = len(feature_cols)
    global_m["n_features_selected"] = "N/A"
    fold_results.append(global_m)
    all_overall.append(global_m)

    print(f"  [{group}] Overall: R2={global_m['R2']:.3f} RRMSE={global_m['RRMSE(%)']:.2f}% MAPE={global_m['MAPE(%)']:.2f}%")

    all_by_year.extend([r for r in fold_results if r["test_year"] != "Overall"])

# ======================== 导出 ========================
pd.DataFrame(all_by_year).to_csv(os.path.join(OUTPUT_DIR, "derived_feature_sensitivity_by_year.csv"), index=False)
pd.DataFrame(all_overall).to_csv(os.path.join(OUTPUT_DIR, "derived_feature_sensitivity_overall.csv"), index=False)
pd.concat(all_predictions, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "derived_feature_sensitivity_predictions.csv"), index=False)
pd.DataFrame(all_best_params).to_csv(os.path.join(OUTPUT_DIR, "derived_feature_sensitivity_best_params.csv"), index=False)

# JSON序列化selected_features
json_safe = {}
for k, v in all_selected_features.items():
    json_safe[k] = list(v)
with open(os.path.join(OUTPUT_DIR, "derived_feature_sensitivity_selected_features.json"), "w") as f:
    json.dump(json_safe, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(">>> 敏感性实验完成")
print("    所有输出在:", OUTPUT_DIR)
print(f"{'='*70}")
