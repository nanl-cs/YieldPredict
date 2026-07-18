"""
任务4：导出P4最终模型逐年指标和逐样本预测
基于XGBRFRegressor + Optuna MOBO，导出每个测试年份的预测值和指标。
"""
import pandas as pd
import numpy as np
import os
import warnings
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRFRegressor
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv"
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTUNA_TRIALS = 50
RANDOM_SEED = 42
META_COLS = ["Year", "Zone", "latitude", "longitude", "yield"]

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

print(f"{'='*70}")
print(">>> P4 最终模型 MOBO 优化 + 预测导出")
print(f"{'='*70}")

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in META_COLS]
years = sorted(df["Year"].unique())

all_metrics = []
all_predictions = []
all_best_params = []

for test_year in years:
    train_df = df[df["Year"] != test_year]
    test_df = df[df["Year"] == test_year]
    X_train_df = train_df[feature_cols]
    y_train_np = train_df["yield"].values
    groups_train = train_df["Year"].values
    X_test_df = test_df[feature_cols]
    y_test_np = test_df["yield"].values

    print(f"[Year {test_year}] Optuna搜索中...", end="", flush=True)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: objective(trial, X_train_df, y_train_np, groups_train), n_trials=OPTUNA_TRIALS)

    best_trial = sorted(study.best_trials, key=lambda t: t.values[0])[0]
    best_params = {k: v for k, v in best_trial.params.items() if not k.startswith("mask_")}
    best_params.update({"tree_method": "gpu_hist", "random_state": RANDOM_SEED, "n_jobs": -1})
    selected_features = [col for col in feature_cols if best_trial.params.get(f"mask_{col}", False)]

    X_train_final = X_train_df[selected_features].values
    X_test_final = X_test_df[selected_features].values

    final_model = XGBRFRegressor(**best_params)
    final_model.fit(X_train_final, y_train_np)
    y_pred = final_model.predict(X_test_final)

    m = calculate_metrics(y_test_np, y_pred)
    m["Test_Year"] = str(int(test_year))
    m["Used_Features"] = len(selected_features)
    all_metrics.append(m)

    all_predictions.append(pd.DataFrame({
        "Year": test_df["Year"].values,
        "Zone": test_df["Zone"].values,
        "latitude": test_df["latitude"].values,
        "longitude": test_df["longitude"].values,
        "yield_true": y_test_np,
        "yield_pred": y_pred,
        "error": y_test_np - y_pred,
    }))

    bp = {"Test_Year": str(int(test_year)), "n_features_selected": len(selected_features),
          "inner_MAE": float(best_trial.values[0]), "inner_n_features": int(best_trial.values[1])}
    bp.update({k: best_params.get(k) for k in ["n_estimators", "max_depth", "colsample_bynode", "subsample"]})
    all_best_params.append(bp)

    print(f" 完成. RRMSE={m['RRMSE(%)']:.2f}% MAPE={m['MAPE(%)']:.2f}%")

# Overall
all_y_true = np.concatenate([p["yield_true"].values for p in all_predictions])
all_y_pred = np.concatenate([p["yield_pred"].values for p in all_predictions])
global_m = calculate_metrics(all_y_true, all_y_pred)
global_m["Test_Year"] = "Overall_Pooled"
global_m["Used_Features"] = "N/A"
all_metrics.append(global_m)

print(f"\nOverall: R2={global_m['R2']:.3f} RRMSE={global_m['RRMSE(%)']:.2f}% MAPE={global_m['MAPE(%)']:.2f}%")

pd.DataFrame(all_metrics).to_csv(os.path.join(OUTPUT_DIR, "P4_Final_Model_ByYear.csv"), index=False)
pd.concat(all_predictions, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "P4_Final_Model_Predictions.csv"), index=False)
pd.DataFrame(all_best_params).to_csv(os.path.join(OUTPUT_DIR, "P4_Final_Model_BestParams.csv"), index=False)

print("导出完成。")
