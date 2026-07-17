import json, os

os.chdir(r"D:\uv_py\xgb\code_dataset")

replacements = {
    "evaluate_performance.ipynb": [
        ('"P4_Cleaned_Dataset.csv"', '"../data/P4_Cleaned_Dataset.csv"'),
        ('"P4_Nested_MOBO_Metrics_Fast.csv"', '"../results/mobo/P4_Nested_MOBO_Metrics_Fast.csv"'),
        ('"P4_Nested_MOBO_Metrics.csv"', '"../results/mobo/P4_Nested_MOBO_Metrics.csv"'),
        ('"P4_Nested_Metrics.csv"', '"../results/mobo/P4_Nested_Metrics.csv"'),
    ],
    "model_comparison.ipynb": [
        ('"P4_Cleaned_Dataset.csv"', '"../data/P4_Cleaned_Dataset.csv"'),
        ('"P4_Nested_MLR_Metrics.csv"', '"../results/comparison/P4_Nested_MLR_Metrics.csv"'),
        ('"P4_Nested_SVR_Metrics.csv"', '"../results/comparison/P4_Nested_SVR_Metrics.csv"'),
        ('"P4_Nested_XGBoost_Metrics.csv"', '"../results/comparison/P4_Nested_XGBoost_Metrics.csv"'),
        ('"P4_Model_Comparison_Summary.csv"', '"../results/comparison/P4_Model_Comparison_Summary.csv"'),
        ('"P4_Nested_MOBO_Metrics_Fast.csv"', '"../results/mobo/P4_Nested_MOBO_Metrics_Fast.csv"'),
    ],
    "optuna_MOBO.ipynb": [
        ('"P4_Cleaned_Dataset.csv"', '"../data/P4_Cleaned_Dataset.csv"'),
        ('"P4_Nested_MOBO_Metrics_Fast.csv"', '"../results/mobo/P4_Nested_MOBO_Metrics_Fast.csv"'),
        ('"P4_Nested_MOBO_Metrics.csv"', '"../results/mobo/P4_Nested_MOBO_Metrics.csv"'),
        ('"P4_Nested_Metrics.csv"', '"../results/mobo/P4_Nested_Metrics.csv"'),
    ],
    "shap_explanation_and_export.ipynb": [
        ('"P4_Cleaned_Dataset.csv"', '"../data/P4_Cleaned_Dataset.csv"'),
        ('"SHAP_Values_Matrix.csv"', '"../results/shap/SHAP_Values_Matrix.csv"'),
        ('"SHAP_Feature_Matrix.csv"', '"../results/shap/SHAP_Feature_Matrix.csv"'),
        ('"SHAP_Metadata_and_Predictions.csv"', '"../results/shap/SHAP_Metadata_and_Predictions.csv"'),
        ('"SHAP_Expected_Value.txt"', '"../results/shap/SHAP_Expected_Value.txt"'),
        ('"SHAP_Importance_Bar.pdf"', '"../results/shap/SHAP_Importance_Bar.pdf"'),
        ('"SHAP_Summary_Dot.pdf"', '"../results/shap/SHAP_Summary_Dot.pdf"'),
    ],
    "shap\\plot_dependence.ipynb": [
        ('"SHAP_Values_Matrix.csv"', '"../../results/shap/SHAP_Values_Matrix.csv"'),
        ('"SHAP_Feature_Matrix.csv"', '"../../results/shap/SHAP_Feature_Matrix.csv"'),
    ],
    "消融.ipynb": [
        ("'P4_Cleaned_Dataset.csv'", "'../data/P4_Cleaned_Dataset.csv'"),
        ("'../results_消融实验'", "'../results/ablation'"),
    ],
}

for filename, reps in replacements.items():
    filepath = os.path.join(r"D:\uv_py\xgb\code_dataset", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    for old, new in reps:
        content = content.replace(old, new)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"[OK] {filename} ({len(reps)} replacements)")
