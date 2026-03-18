# visualization/single_example.py
# 图1：单样例对比。
# 各选一个非幻觉样本和幻觉样本，
# 画出整段生成内容按 token 平均后的层曲线，直观展示两组的内部状态差异。

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR


def plot_single_example(sample_records):
    FIGURES_DIR.mkdir(exist_ok=True)

    non_hallucinated = next((r for r in sample_records if not r["has_hallucination"]), None)
    hallucinated = next((r for r in sample_records if r["has_hallucination"]), None)

    if non_hallucinated is None or hallucinated is None:
        print("[fig1] Need at least one non-hallucination sample and one hallucination sample. Skipping.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(
        "Single Example Comparison  "
        f"(Whole Response Average, non-hallucination id={non_hallucinated['id']}, hallucination id={hallucinated['id']})",
        fontsize=14,
    )

    panels = [
        ("mismatch_curve",    "Mismatch\n(1 − cos(Δh, context))", "Dim 1: Update Direction Mismatch"),
        ("attn_drift_curve",  "AttnDrift (JS²)",                  "Dim 2: Attention Structure Drift"),
        ("update_norm_curve", "UpdateNorm (‖Δh‖)",               "Dim 3: Representation Stabilization"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        nh_curve = non_hallucinated[key]
        h_curve = hallucinated[key]
        x_nh = np.arange(1, len(nh_curve) + 1)
        x_h = np.arange(1, len(h_curve) + 1)
        ax.plot(x_nh, nh_curve, color="steelblue", label="Non-hallucination")
        ax.plot(x_h, h_curve, color="tomato", linestyle="--", label="Hallucination")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Layer")
    plt.tight_layout()
    path = FIGURES_DIR / "fig1_single_example.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
