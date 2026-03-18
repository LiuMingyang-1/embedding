# visualization/group_average.py
# 图2：分组平均层曲线。
# 将所有正确回答和错误回答样本的“整段生成内容平均曲线”分别按层取均值，
# 画出带 std 阴影的对比曲线，展示两组在三个维度上的系统性差异。

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR


def plot_group_average(sample_records):
    FIGURES_DIR.mkdir(exist_ok=True)

    correct = [r for r in sample_records if r["is_correct"]]
    wrong = [r for r in sample_records if not r["is_correct"]]

    if not correct or not wrong:
        print("[fig2] Not enough samples in one group. Skipping.")
        return

    panels = [
        ("mismatch_curve",    "Mismatch\n(1 − cos(Δh, context))", "Dim 1: Update Direction Mismatch"),
        ("attn_drift_curve",  "AttnDrift (JS²)",                  "Dim 2: Attention Structure Drift"),
        ("update_norm_curve", "UpdateNorm (‖Δh‖)",               "Dim 3: Representation Stabilization"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Group Average Layer Curves  (Whole Response Average)", fontsize=14)

    for ax, (key, ylabel, title) in zip(axes, panels):
        c_arr = np.array([r[key] for r in correct])
        w_arr = np.array([r[key] for r in wrong])

        c_mean, c_std = c_arr.mean(axis=0), c_arr.std(axis=0)
        w_mean, w_std = w_arr.mean(axis=0), w_arr.std(axis=0)

        x_c = np.arange(1, len(c_mean) + 1)
        x_w = np.arange(1, len(w_mean) + 1)

        ax.plot(x_c, c_mean, color="steelblue", label=f"Correct (n={len(correct)})")
        ax.fill_between(x_c, c_mean - c_std, c_mean + c_std, alpha=0.2, color="steelblue")

        ax.plot(x_w, w_mean, color="tomato", linestyle="--", label=f"Wrong (n={len(wrong)})")
        ax.fill_between(x_w, w_mean - w_std, w_mean + w_std, alpha=0.2, color="tomato")

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Layer")
    plt.tight_layout()
    path = FIGURES_DIR / "fig2_group_average.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
