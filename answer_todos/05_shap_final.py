"""
任务5：使用最终确定的特征体系重新运行SHAP
训练全局最优模型，计算SHAP值，导出矩阵和图表。
"""
import pandas as pd
import numpy as np
import os
import warnings
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv"
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTUNA_TRIALS = 30
RANDOM_SEED = 42
META_COLS = ["Year", "Zone", "latitude", "longitude", "yield"]

def objective(trial, X_train_np, y_train_np):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.6),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "tree_method": "gpu_hist", "random_state": RANDOM_SEED, "n_jobs": -1,
    }
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_np, y_train_np, test_size=0.2, random_state=RANDOM_SEED)
    model = XGBRFRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

print("=" * 60)
print(">>> SHAP分析 (最终模型)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in META_COLS]
X = df[feature_cols]
y = df["yield"]
X_np = X.values
y_np = y.values

print("[INFO] 第一步：全局贝叶斯优化...")
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_SEED))
study.optimize(lambda trial: objective(trial, X_np, y_np), n_trials=OPTUNA_TRIALS)

best_params = study.best_params
best_params.update({"tree_method": "gpu_hist", "random_state": RANDOM_SEED, "n_jobs": -1})
print(f"  最佳参数: {best_params}")

print("[INFO] 第二步：全量数据训练最终模型...")
final_model = XGBRFRegressor(**best_params)
final_model.fit(X_np, y_np)

print("[INFO] 第三步：计算SHAP值...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# 导出矩阵
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Values_Matrix.csv"), index=False)
X.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Feature_Matrix.csv"), index=False)

full_export = df[META_COLS].copy()
full_export["Predicted_Yield"] = final_model.predict(X_np)
full_export.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Metadata_and_Predictions.csv"), index=False)

ev = explainer.expected_value
if isinstance(ev, np.ndarray):
    ev = ev[0]
with open(os.path.join(OUTPUT_DIR, "SHAP_Expected_Value.txt"), "w") as f:
    f.write(f"Expected Value (Baseline Yield): {ev}\n")

# 平均绝对SHAP排序
mean_abs = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Mean_Absolute_Importance.csv"), index=False)

# Top15方向性摘要
top15 = importance_df.head(15)["feature"].tolist()
dir_rows = []
for feat in top15:
    if feat in X.columns:
        shap_col = shap_values[:, X.columns.get_loc(feat)]
        feat_col = X[feat]
        mask = ~np.isnan(shap_col) & ~np.isnan(feat_col)
        if mask.sum() > 10:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(feat_col[mask], shap_col[mask])
            dir_rows.append({"feature": feat, "spearman_r": round(corr, 4),
                             "mean_SHAP": round(float(np.mean(np.abs(shap_col[mask]))), 6)})

pd.DataFrame(dir_rows).to_csv(os.path.join(OUTPUT_DIR, "SHAP_Top15_Direction_Summary.csv"), index=False)

# SHAP图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "SHAP_Importance_Bar.pdf"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False, max_display=15)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "SHAP_Summary_Dot.pdf"), dpi=300, bbox_inches="tight")
plt.close()

print("\n分析完成。所有输出在:", OUTPUT_DIR)
