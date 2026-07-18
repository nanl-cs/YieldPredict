"""
任务1：派生特征数值诊断
目的：判断除法型派生特征是否产生极端值，以及模型是否可能受到少量异常样本支配。

检查的特征：
  WUE, Decoupling_Stress, Thermal_Efficiency, Hydrothermal_Balance,
  Drought_Vulnerability, Fertility_Vigor, Cum_VPD, Delta_SM, Delta_NDVI, Delta_NDWI

输出：
  derived_feature_summary_overall.csv
  derived_feature_summary_by_year.csv
  derived_feature_summary_by_zone.csv
  derived_feature_edge_case_counts.csv
  derived_feature_spearman.csv
  derived_feature_distributions/ (直方图+箱线图)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ======================== 配置 ========================
DATA_PATH = r"D:\uv_py\xgb\data\P4_Cleaned_Dataset.csv"
OUTPUT_DIR = r"D:\uv_py\xgb\answer_todos\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "derived_feature_distributions"), exist_ok=True)

META_COLS = ["Year", "Zone", "latitude", "longitude", "yield"]
STAGES = ["P1", "P2", "P3", "P4"]

DERIVED_KEYWORDS = [
    "WUE", "Decoupling_Stress", "Thermal_Efficiency", "Hydrothermal_Balance",
    "Drought_Vulnerability", "Fertility_Vigor",
]
CUM_DELTA_KEYWORDS = ["Cum_PPT", "Cum_FDD", "Cum_VPD",
                       "Delta_NDVI", "Delta_NDWI", "Delta_SM"]

# ======================== 加载数据 ========================
print("[INFO] 加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"  数据维度: {df.shape}")
print(f"  年份: {sorted(df['Year'].unique())}")
print(f"  分区: {sorted(df['Zone'].unique())}")

# ======================== 辅助函数 ========================
PERC_KEYS = ["count", "missing", "zero_count", "mean", "std", "min",
             "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]

def compute_stats(series):
    vals = series.dropna().values
    if len(vals) == 0:
        return {k: np.nan for k in PERC_KEYS}
    return {
        "count": len(vals),
        "missing": series.isna().sum(),
        "zero_count": int((vals == 0).sum()),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "p1": float(np.percentile(vals, 1)),
        "p5": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }

def get_derived_features(df):
    all_cols = [c for c in df.columns if c not in META_COLS]
    derived = []
    for col in all_cols:
        for kw in DERIVED_KEYWORDS + CUM_DELTA_KEYWORDS:
            if kw in col and col not in derived:
                derived.append(col)
    return derived

# ======================== 1. 整体统计 ========================
print("\n[INFO] 计算整体统计量...")
derived_cols = get_derived_features(df)
print(f"  派生特征总数: {len(derived_cols)}")

overall_stats = []
for col in derived_cols:
    stats = compute_stats(df[col])
    stats["feature"] = col
    overall_stats.append(stats)

df_overall = pd.DataFrame(overall_stats)
df_overall = df_overall[["feature"] + [c for c in df_overall.columns if c != "feature"]]
df_overall.to_csv(os.path.join(OUTPUT_DIR, "derived_feature_summary_overall.csv"), index=False)
print("  已保存: derived_feature_summary_overall.csv")

# ======================== 2. 按年份统计 ========================
print("\n[INFO] 按年份统计...")
years = sorted(df["Year"].unique())
year_stats = []
for year in years:
    df_year = df[df["Year"] == year]
    for col in derived_cols:
        stats = compute_stats(df_year[col])
        stats["feature"] = col
        stats["year"] = int(year)
        stats["n_samples"] = len(df_year)
        year_stats.append(stats)

df_year = pd.DataFrame(year_stats)
cols_order = ["feature", "year", "n_samples"] + [c for c in df_year.columns if c not in ["feature", "year", "n_samples"]]
df_year = df_year[cols_order]
df_year.to_csv(os.path.join(OUTPUT_DIR, "derived_feature_summary_by_year.csv"), index=False)
print("  已保存: derived_feature_summary_by_year.csv")

# ======================== 3. 按Zone统计 ========================
print("\n[INFO] 按Zone统计...")
zones = sorted(df["Zone"].unique())
zone_stats = []
for zone in zones:
    df_zone = df[df["Zone"] == zone]
    for col in derived_cols:
        stats = compute_stats(df_zone[col])
        stats["feature"] = col
        stats["zone"] = int(zone)
        stats["n_samples"] = len(df_zone)
        zone_stats.append(stats)

df_zone = pd.DataFrame(zone_stats)
cols_order = ["feature", "zone", "n_samples"] + [c for c in df_zone.columns if c not in ["feature", "zone", "n_samples"]]
df_zone = df_zone[cols_order]
df_zone.to_csv(os.path.join(OUTPUT_DIR, "derived_feature_summary_by_zone.csv"), index=False)
print("  已保存: derived_feature_summary_by_zone.csv")

# ======================== 4. 边缘情况统计 ========================
print("\n[INFO] 边缘情况统计...")
edge_cases = []

for stage in STAGES:
    sm_col = f"{stage}_SM"
    if sm_col in df.columns:
        for thresh in [0.01, 0.02]:
            n = int((df[sm_col] < thresh).sum())
            edge_cases.append({"condition": f"SM < {thresh}", "stage": stage,
                               "n_samples": n, "pct_total": round(n / len(df) * 100, 2)})

for stage in STAGES:
    tmean_col = f"{stage}_Tmean"
    if tmean_col in df.columns:
        n = int((df[tmean_col] <= 0).sum())
        edge_cases.append({"condition": "Tmean <= 0", "stage": stage,
                           "n_samples": n, "pct_total": round(n / len(df) * 100, 2)})

clay_vals = df["Clay"].dropna()
edge_cases.append({"condition": "Clay range (used in Drought_Vulnerability)",
                   "stage": "All", "n_samples": len(clay_vals), "pct_total": 100.0})

for col in derived_cols:
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    p99 = np.percentile(vals, 99)
    p1 = np.percentile(vals, 1)
    p50 = np.percentile(vals, 50)
    if p99 > 0 and p50 > 0:
        ratio = p99 / p50
        if ratio > 10:
            edge_cases.append({"condition": f"p99/p50 ratio > 10", "stage": col,
                               "n_samples": int(ratio), "pct_total": round(p99, 2)})

df_edge = pd.DataFrame(edge_cases)
df_edge.to_csv(os.path.join(OUTPUT_DIR, "derived_feature_edge_case_counts.csv"), index=False)
print("  已保存: derived_feature_edge_case_counts.csv")

# ======================== 5. Spearman 相关矩阵 ========================
print("\n[INFO] 计算Spearman相关矩阵...")
analysis_cols = derived_cols + ["Sand", "Clay", "SOC"]
for stage in STAGES:
    for var in ["Tmean", "PPT", "SM", "GDD", "FDD", "VPD", "VPD_max",
                "NDVI", "NDVI_max", "NDWI", "NIRv", "EVI", "EVI_max"]:
        col_name = f"{stage}_{var}"
        if col_name in df.columns and col_name not in analysis_cols:
            analysis_cols.append(col_name)

df_corr = df[analysis_cols].dropna()
print(f"  分析变量数: {len(analysis_cols)}, 有效样本: {len(df_corr)}")

spearman_mat = pd.DataFrame(index=analysis_cols, columns=analysis_cols, dtype=float)
for i, col_i in enumerate(analysis_cols):
    for j, col_j in enumerate(analysis_cols):
        if j > i:
            if col_i in derived_cols or col_j in derived_cols:
                corr, _ = spearmanr(df_corr[col_i], df_corr[col_j], nan_policy="omit")
                spearman_mat.loc[col_i, col_j] = round(corr, 4)
                spearman_mat.loc[col_j, col_i] = round(corr, 4)
            else:
                spearman_mat.loc[col_i, col_j] = 0
                spearman_mat.loc[col_j, col_i] = 0
        elif j == i:
            spearman_mat.loc[col_i, col_j] = 1.0

spearman_mat.to_csv(os.path.join(OUTPUT_DIR, "derived_feature_spearman.csv"))
print("  已保存: derived_feature_spearman.csv（含派生特征与原始变量间Spearman相关）")

# ======================== 6. 分布图 ========================
print("\n[INFO] 生成分布图...")
plot_dir = os.path.join(OUTPUT_DIR, "derived_feature_distributions")

# 6a. 六大复合代理特征的直方图
for kw in DERIVED_KEYWORDS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{kw} Distribution by Stage", fontsize=14)
    for idx, stage in enumerate(STAGES):
        ax = axes[idx // 2, idx % 2]
        col = f"{stage}_{kw}"
        if col in df.columns:
            vals = df[col].dropna()
            p99 = np.percentile(vals, 99)
            p1_val = np.percentile(vals, 1)
            clip_vals = vals[(vals >= p1_val) & (vals <= p99)]
            if len(clip_vals) > 0:
                ax.hist(clip_vals, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
                ax.axvline(np.median(vals), color="red", linestyle="--", label=f"Median={np.median(vals):.3f}")
                ax.axvline(np.mean(vals), color="orange", linestyle="--", label=f"Mean={np.mean(vals):.3f}")
            ax.set_title(f"{stage} (n={len(vals)} clipped to p1-p99)")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{kw}_histogram.png"), dpi=150)
    plt.close()

# 6b. 六大复合代理特征按年份箱线图
for kw in DERIVED_KEYWORDS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{kw} Boxplot by Year", fontsize=14)
    for idx, stage in enumerate(STAGES):
        ax = axes[idx // 2, idx % 2]
        col = f"{stage}_{kw}"
        if col in df.columns:
            data_by_year = [df[df["Year"] == y][col].dropna().values for y in years]
            ax.boxplot(data_by_year, labels=[str(int(y)) for y in years])
            ax.set_title(f"{stage}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{kw}_boxplot_by_year.png"), dpi=150)
    plt.close()

# 6c. Cum_VPD 直方图
for stage in STAGES:
    col = f"Cum_VPD_{stage}"
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = df[col].dropna()
        p99 = np.percentile(vals, 99)
        p1_val = np.percentile(vals, 1)
        clip_vals = vals[(vals >= p1_val) & (vals <= p99)]
        ax.hist(clip_vals, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(np.median(vals), color="red", linestyle="--", label=f"Median={np.median(vals):.1f}")
        ax.set_title(f"Cum_VPD_{stage}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Cum_VPD_{stage}_histogram.png"), dpi=150)
        plt.close()

# 6d. Delta 变量按年份箱线图
for kw in ["Delta_NDVI", "Delta_NDWI", "Delta_SM"]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{kw} Boxplot by Year", fontsize=14)
    for idx, stage in enumerate(["P2", "P3", "P4"]):
        ax = axes[idx]
        col = f"{kw}_{stage}"
        if col in df.columns:
            data_by_year = [df[df["Year"] == y][col].dropna().values for y in years]
            ax.boxplot(data_by_year, labels=[str(int(y)) for y in years])
            ax.set_title(f"{stage}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{kw}_boxplot_by_year.png"), dpi=150)
    plt.close()

print("  已保存: 分布图到 derived_feature_distributions/")

# ======================== 7. 极端值摘要报告 ========================
print("\n" + "=" * 70)
print("  派生特征极端值快速诊断")
print("=" * 70)

for col in derived_cols:
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    p99 = np.percentile(vals, 99)
    p1_val = np.percentile(vals, 1)
    p50 = np.percentile(vals, 50)
    mean_val = np.mean(vals)
    std_val = np.std(vals)

    flags = []
    if std_val > 100 * abs(mean_val) and mean_val != 0:
        flags.append(f"EXTREME_SPREAD: std({std_val:.1f}) >> |mean|({mean_val:.1f})")
    if p99 > 0 and p50 > 0 and p99 / p50 > 20:
        flags.append(f"HEAVY_TAIL: p99/p50={p99/p50:.1f}")
    if p50 > 0 and p1_val > 0 and p50 / p1_val > 20:
        flags.append(f"LEFT_TAIL: p50/p1={p50/p1_val:.1f}")
    zero_pct = (vals == 0).sum() / len(vals) * 100
    if zero_pct > 5:
        flags.append(f"MANY_ZEROS: {zero_pct:.1f}%")

    if flags:
        print("  [!] " + col + ": " + ";".join(flags))

print("\n[INFO] 完成。所有输出在:", OUTPUT_DIR)
