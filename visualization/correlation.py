# visualization/correlation.py
# 图4：三指标之间的 Spearman 相关热力图。
# 验证互补性：中等相关 (|ρ| < 0.5) 说明三个维度捕捉了不同信息；
# 高度相关 (|ρ| > 0.8) 则说明存在冗余。
#
# 输入应为 sample-level records。
# 默认主分析对象为整段生成内容。

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import FIGURES_DIR

METRIC_LABELS = ["Mismatch\n(whole response)", "AttnDrift\n(whole response)", "UpdateNorm\nlate_mean"]
METRIC_KEYS = ["mismatch_mean", "attn_drift_mean", "update_norm_late_mean"]


def plot_correlation_matrix(sample_records):
    FIGURES_DIR.mkdir(exist_ok=True)

    data = np.array([[r[k] for k in METRIC_KEYS] for r in sample_records])
    n = len(METRIC_KEYS)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = spearmanr(data[:, i], data[:, j])
            corr[i, j] = rho

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Spearman Correlation Between Three Metrics  (Whole Response Average)", fontsize=13)

    try:
        import seaborn as sns
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=-1, vmax=1,
                    xticklabels=METRIC_LABELS, yticklabels=METRIC_LABELS, ax=ax)
    except ImportError:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(METRIC_LABELS)
        ax.set_yticklabels(METRIC_LABELS)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_correlation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
