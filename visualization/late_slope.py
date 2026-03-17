# visualization/late_slope.py
# 图3：后期稳定化分布（LateSlope & LateMean）。
# 用箱线图对比正确组和错误组在维度 3 后期指标上的分布差异：
#   LateSlope：后 k 层 UpdateNorm 的线性回归斜率（越负 = 越快收敛）
#   LateMean： 后 k 层 UpdateNorm 的均值（越小 = 越稳定）

import matplotlib.pyplot as plt

from config import FIGURES_DIR, LATE_K


def plot_late_slope_distribution(all_records):
    FIGURES_DIR.mkdir(exist_ok=True)

    correct = [r for r in all_records if r["is_correct"]]
    wrong = [r for r in all_records if not r["is_correct"]]

    if not correct or not wrong:
        print("[fig3] Not enough samples. Skipping.")
        return

    c_slopes = [r["update_norm_late_slope"] for r in correct]
    w_slopes = [r["update_norm_late_slope"] for r in wrong]
    c_means = [r["update_norm_late_mean"] for r in correct]
    w_means = [r["update_norm_late_mean"] for r in wrong]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Dim 3: Late-Layer Stabilization  (last {LATE_K} layers)", fontsize=13)

    axes[0].boxplot([c_slopes, w_slopes], labels=["Correct", "Wrong"])
    axes[0].set_title("LateSlope of UpdateNorm")
    axes[0].set_ylabel("Slope  (more negative → faster convergence)")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([c_means, w_means], labels=["Correct", "Wrong"])
    axes[1].set_title("LateMean of UpdateNorm")
    axes[1].set_ylabel("Mean ‖Δh‖ in late layers")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_late_slope.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
