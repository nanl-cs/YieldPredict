"""
任务6-8: 稳定性分析（按Zone评估误差 + 残差分析 + Bootstrap置信区间）
使用已有预测结果（从results目录的csv文件或从最终模型导出）
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ======================== 配置 ========================
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 尝试从现有结果文件加载预测值，若没有则从answer_todos找
PRED_PATH = r"D:\uv_py\xgb\answer_todos\outputs\P4_Final_Model_Predictions.csv"
FALLBACK_PRED = r"D:\uv_py\xgb\results\mobo\P4_Nested_MOBO_Metrics_Fast.csv"
METRICS_PATH = r"D:\uv_py\xgb\results\mobo\Progressive_LOYO_Overall_Results.csv"

# ======================== 数据加载 ========================
print("[INFO] 加载数据...")
df_data = pd.read_csv(r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv")

# 尝试加载预测值
has_predictions = os.path.exists(PRED_PATH)
if has_predictions:
    pred_df = pd.read_csv(PRED_PATH)
    print(f"  从 {PRED_PATH} 加载逐样本预测 (n={len(pred_df)})")
else:
    print(f"  警告: 未找到逐样本预测文件 {PRED_PATH}")
    print(f"  脚本04_export_predictions.py尚未运行，将使用简化版分析")

# ======================== 任务6: 按Zone和YearxZone评估误差 ========================
print("\n" + "=" * 60)
print("  任务6: 按Zone评估P4预测误差")
print("=" * 60)

if has_predictions:
    # 按Zone
    zone_results = []
    for zone in sorted(pred_df["Zone"].unique()):
        zp = pred_df[pred_df["Zone"] == zone]
        if len(zp) == 0:
            continue
        yt, yp = zp["yield_true"].values, zp["yield_pred"].values
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
        mape = np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
        rrmse = rmse / np.mean(yt) * 100 if np.mean(yt) != 0 else 0
        mean_obs = np.mean(yt)
        num = np.sum((yp - yt) ** 2)
        den = np.sum((np.abs(yp - mean_obs) + np.abs(yt - mean_obs)) ** 2)
        d_index = 1 - num / den if den != 0 else 0
        zone_results.append({
            "Zone": int(zone), "n_samples": len(zp),
            "R2": round(r2, 3), "RMSE": round(rmse, 2),
            "RRMSE(%)": round(rrmse, 2), "MAE": round(mae, 2),
            "MAPE(%)": round(mape, 2), "d-index": round(d_index, 3),
        })
    df_zone = pd.DataFrame(zone_results)
    df_zone.to_csv(os.path.join(OUTPUT_DIR, "P4_Metrics_ByZone.csv"), index=False)
    print("  已保存: P4_Metrics_ByZone.csv")
    print(df_zone.to_string())

    # 按 Year x Zone
    yz_results = []
    for year in sorted(pred_df["Year"].unique()):
        for zone in sorted(pred_df["Zone"].unique()):
            yzp = pred_df[(pred_df["Year"] == year) & (pred_df["Zone"] == zone)]
            if len(yzp) == 0:
                continue
            yt, yp = yzp["yield_true"].values, yzp["yield_pred"].values
            mae = np.mean(np.abs(yt - yp))
            rmse = np.sqrt(np.mean((yt - yp) ** 2))
            mape = np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
            yz_results.append({
                "Year": int(year), "Zone": int(zone), "n_samples": len(yzp),
                "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE(%)": round(mape, 2),
            })
    df_yz = pd.DataFrame(yz_results)
    df_yz.to_csv(os.path.join(OUTPUT_DIR, "P4_Metrics_ByYear_Zone.csv"), index=False)
    print("  已保存: P4_Metrics_ByYear_Zone.csv")
else:
    print("  跳过(需先运行04_export_predictions.py)")

# ======================== 任务7: 误差分布与产量区间偏差 ========================
print("\n" + "=" * 60)
print("  任务7: 残差分析")
print("=" * 60)

if has_predictions:
    yt = pred_df["yield_true"].values
    yp = pred_df["yield_pred"].values
    errors = yt - yp

    # 按产量分位数分组
    q33 = np.percentile(yt, 33.3)
    q67 = np.percentile(yt, 66.7)
    low_mask = yt <= q33
    mid_mask = (yt > q33) & (yt <= q67)
    high_mask = yt > q67

    quantile_results = []
    for label, mask in [("Low", low_mask), ("Mid", mid_mask), ("High", high_mask)]:
        e = errors[mask]
        quantile_results.append({
            "yield_quantile": label, "n": int(mask.sum()),
            "mean_error": round(float(np.mean(e)), 2),
            "MAE": round(float(np.mean(np.abs(e))), 2),
            "RMSE": round(float(np.sqrt(np.mean(e**2))), 2),
            "MAPE(%)": round(float(np.mean(np.abs(e / np.where(yt[mask] == 0, np.nan, yt[mask]))) * 100), 2),
        })
    df_quantile = pd.DataFrame(quantile_results)
    df_quantile.to_csv(os.path.join(OUTPUT_DIR, "P4_Metrics_ByYieldQuantile.csv"), index=False)
    print("  已保存: P4_Metrics_ByYieldQuantile.csv")
    print(df_quantile.to_string())

    # 残差摘要
    residual_summary = {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "skewness_error": float(pd.Series(errors).skew()),
        "mean_yield_true": float(np.mean(yt)),
        "mean_yield_pred": float(np.mean(yp)),
        "systematic_bias": float(np.mean(yt) - np.mean(yp)),
    }
    pd.DataFrame([residual_summary]).to_csv(os.path.join(OUTPUT_DIR, "P4_Residual_Summary.csv"), index=False)
    print("  已保存: P4_Residual_Summary.csv")
    print(f"  系统性偏差(真实-预测): {residual_summary['systematic_bias']:.2f}")

    # 预测散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(yt[::10], yp[::10], alpha=0.1, s=1, color="steelblue")
    minv, maxv = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
    ax.set_xlabel("True Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title("Predicted vs True")
    ax.text(0.05, 0.95, f"R={np.corrcoef(yt, yp)[0,1]:.3f}\nN={len(yt)}",
            transform=ax.transAxes, verticalalignment="top", fontsize=10)

    ax = axes[1]
    ax.scatter(yp[::10], errors[::10], alpha=0.1, s=1, color="coral")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Yield")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.set_title("Residuals vs Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "P4_Predictions_Scatter.png"), dpi=150)
    plt.close()
    print("  已保存: P4_Predictions_Scatter.png")

    # 残差分布图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(errors), color="orange", linestyle="--", linewidth=2, label=f"Mean={np.mean(errors):.1f}")
    ax.set_xlabel("Residual (True - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "P4_Residual_Distribution.png"), dpi=150)
    plt.close()
    print("  已保存: P4_Residual_Distribution.png")
else:
    print("  跳过(需先运行04_export_predictions.py)")

# ======================== 任务8: Bootstrap置信区间 ========================
print("\n" + "=" * 60)
print("  任务8: Bootstrap置信区间")
print("=" * 60)

if has_predictions:
    np.random.seed(42)
    n_bootstrap = 1000
    n_samples = len(yt)

    metrics_bootstrap = {"RMSE": [], "MAE": [], "MAPE(%)": [], "RRMSE(%)": []}

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt_b, yp_b = yt[idx], yp[idx]
        mae = np.mean(np.abs(yt_b - yp_b))
        rmse = np.sqrt(np.mean((yt_b - yp_b) ** 2))
        mape = np.mean(np.abs((yt_b - yp_b) / np.where(yt_b == 0, np.nan, yt_b))) * 100
        rrmse = rmse / np.mean(yt_b) * 100 if np.mean(yt_b) != 0 else 0
        metrics_bootstrap["RMSE"].append(rmse)
        metrics_bootstrap["MAE"].append(mae)
        metrics_bootstrap["MAPE(%)"].append(mape)
        metrics_bootstrap["RRMSE(%)"].append(rrmse)

    ci_results = []
    for metric, values in metrics_bootstrap.items():
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        mean_val = np.mean(values)
        ci_results.append({
            "metric": metric, "mean": round(mean_val, 3),
            "CI_lower_2.5%": round(ci_low, 3), "CI_upper_97.5%": round(ci_high, 3),
        })
    df_ci = pd.DataFrame(ci_results)
    df_ci.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_metric_confidence_intervals.csv"), index=False)
    print("  已保存: bootstrap_metric_confidence_intervals.csv")
    print(df_ci.to_string())
else:
    print("  跳过(需先运行04_export_predictions.py)")

print("\n[INFO] 分析脚本完成。输出在:", OUTPUT_DIR)
