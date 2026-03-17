# config.py
# 全局配置：模型名称、文件路径、超参数。
# 所有模块都从这里读取配置，避免硬编码。

from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

NORMAL_CSV = "normal.csv"
INDUCED_CSV = "induced.csv"
RESULTS_CSV = "results_all.csv"

STATES_DIR = Path("states")
FIGURES_DIR = Path("figures")

# 后期层数：计算 LateSlope / LateMean 时使用最后 k 层
LATE_K = 8
