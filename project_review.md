# 冬小麦产量预测 — 多源数据回归项目

## 目录结构

```
xgb/
│
├── data/                          ← 所有输入数据
│   ├── raw/                       ← 原始多源数据（按 Zone × 年份存放）
│   │   └── zone{1..5}/Academic_Zone*_*.csv
│   ├── merged/                    ← 逐年合并数据（01拼数据步骤产物）
│   │   └── Merged_Features_2016~2021.csv
│   ├── P4_Cleaned_Dataset.csv     ← 最终清洗数据集（100维特征，213MB，Git LFS 追踪）
│   └── 物候期具体时间.xlsx          ← 辅助参考文件
│
├── code_dataset/                  ← 实验 Notebook（主要工作区）
│   ├── evaluate_performance.ipynb     ← MOBO 训练（XGBRF + Optuna 双目标优化）
│   ├── model_comparison.ipynb         ← 多模型对比（MLR / SVR / XGBoost / RF+MOBO）
│   ├── optuna_MOBO.ipynb             ← MOBO 实验历史版本
│   ├── shap_explanation_and_export.ipynb ← SHAP 分析（全局模型 + 矩阵导出）
│   ├── 消融.ipynb                     ← 消融实验（Static/气象/RS 特征组）
│   ├── run_ablation.py               ← 消融实验快捷执行脚本
│   └── shap/
│       └── plot_dependence.ipynb    ← SHAP 偏依赖图
│
├── results/                       ← 所有实验输出
│   ├── mobo/                     ← MOBO 训练结果（P4_Nested_*.csv）
│   ├── comparison/                ← 模型对比汇总表
│   ├── ablation/                  ← 消融实验指标与图表
│   └── shap/                      ← SHAP 矩阵与可视化（大矩阵被 .gitignore 排除）
│
├── pre_done/                      ← [归档] 原始流水线（00~06 按编号排列）
│   ├── 00分区对齐.js
│   ├── 01拼数据+数据处理.ipynb       ← 从 origin/ → merged_dataset/ + 特征工程
│   ├── 02P1.ipynb                  ← P1~P5 各阶段独立训练
│   ├── 03减少静态.ipynb             ← 静态特征筛选
│   ├── 04_01_prepare_P4_data.ipynb  ← 最终数据集整理
│   ├── 05_evaluate_performance.ipynb ← 原始版 MOBO 训练
│   └── 06_shap_explanation_and_export.ipynb ← 原始版 SHAP 分析
│
├── paper/                         ← 参考论文（Git 不追踪）
├── main.py                        ← 占位入口
├── pyproject.toml / uv.lock       ← Python 依赖（uv 管理）
└── project_review.md              ← 本文件
```

## 数据流水线

```
origin/zone*/Academic_Zone*_*.csv   (原始数据)
  │
  ├── 00分区对齐.js → 坐标对齐
  └── 01拼数据+数据处理 → merged_dataset/ (逐年合并)
                       → engineered_features/ (缺失，需重跑01生成)
  │
  ├── 02P1 → 各阶段独立 RF 训练（确定最优阶段=P4）
  ├── 03减少静态 → XGBRF 渐进训练（剔除冗余静态特征）
  └── 04_prepare_P4_data → P4_Cleaned_Dataset.csv (最终数据)
  │
  ▼
P4_Cleaned_Dataset.csv  ──→  05_MOBO训练 (code_dataset/evaluate_performance)
                         ├─→  06_模型对比 (code_dataset/model_comparison)
                         ├─→  07_消融实验 (code_dataset/消融)
                         └─→  08_SHAP分析 (code_dataset/shap_explanation_and_export)
```

## 运行方式

所有 notebook 从 `code_dataset/` 目录打开，数据路径已配置为 `../data/P4_Cleaned_Dataset.csv`。

执行顺序：05_MOBO → 06_模型对比 / 07_消融实验 → 08_SHAP分析（消融实验可独立运行）。

## 技术栈

- Python 3.10+
- XGBoost (XGBRFRegressor, GPU 加速)
- Optuna (TPESampler, 多目标贝叶斯优化)
- SHAP (TreeExplainer)
- Scikit-learn (MLR, SVR, RF baseline)
- CuPy (GPU 矩阵运算)
- uv (Python 依赖管理)

## Git 注意事项

- `P4_Cleaned_Dataset.csv` 使用 Git LFS 追踪
- SHAP 大矩阵（SHAP_Feature_Matrix.csv, SHAP_Values_Matrix.csv）被 .gitignore 排除
- `.venv/` 和 `paper/` 不上传
