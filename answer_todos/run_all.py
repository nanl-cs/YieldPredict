"""
一键运行所有待办代码任务。
用法：python run_all.py [--light|--full]

  --light: 仅运行轻型诊断分析脚本 (01, 06)
  --full:  运行重型实验脚本 (02-05, 需要GPU和较长时间)
  (默认):  仅运行轻型脚本（01已完成）
"""
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"D:\uv_py\xgb\.venv\Scripts\python.exe"

LIGHT_SCRIPTS = [
    "01_derived_diagnostics.py",
    "06_analysis.py",
]

HEAVY_SCRIPTS = [
    "02_feature_sensitivity.py",
    "03_ablation_revised.py",
    "04_export_predictions.py",
    "05_shap_final.py",
    "06_analysis.py",  # 重型实验后重新运行分析
]

def run_script(name):
    path = os.path.join(SCRIPTS_DIR, name)
    if not os.path.exists(path):
        print(f"[SKIP] {name} 不存在")
        return
    print(f"\n{'='*60}\n>>> 运行: {name}\n{'='*60}")
    subprocess.run([PYTHON, path], cwd=SCRIPTS_DIR)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "light"

    if mode == "--light":
        print("=== 轻型模式：运行诊断与分析脚本 ===")
        scripts = [s for s in LIGHT_SCRIPTS if s != "01_derived_diagnostics.py"]
        for s in scripts:
            run_script(s)
    elif mode == "--full":
        print("=== 重型模式：运行全部实验脚本 (预计数小时) ===")
        for s in HEAVY_SCRIPTS:
            run_script(s)
    else:
        print(f"用法: python run_all.py [--light|--full]")

if __name__ == "__main__":
    main()
