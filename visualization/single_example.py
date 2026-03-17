# visualization/single_example.py
# 图1：单样例配对对比。
# 选取同一 id 中 normal=正确、induced=错误的一对样本，
# 画出答案 token 在各层的三个指标曲线，直观展示幻觉 vs 非幻觉的内部状态差异。

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR


def plot_single_example(all_records, results_df):
    FIGURES_DIR.mkdir(exist_ok=True)

    # Find a pair: same id, normal=correct, induced=wrong
    pair_id = None
    for sid in results_df["id"].unique():
        sub = results_df[results_df["id"] == sid]
        normal_ok = sub[(sub["type"] == "normal") & sub["is_correct"]]
        induced_fail = sub[(sub["type"] == "induced") & ~sub["is_correct"]]
        if not normal_ok.empty and not induced_fail.empty:
            pair_id = sid
            break

    if pair_id is None:
        print("[fig1] No ideal pair (normal correct, induced wrong) found. Skipping.")
        return

    correct_rec = next((r for r in all_records if r["id"] == pair_id and r["is_correct"]), None)
    wrong_rec = next((r for r in all_records if r["id"] == pair_id and not r["is_correct"]), None)

    if correct_rec is None or wrong_rec is None:
        print("[fig1] Missing records for chosen pair. Skipping.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Single Example Comparison  (id={pair_id})", fontsize=14)

    panels = [
        ("mismatch_curve",    "Mismatch\n(1 − cos(Δh, context))", "Dim 1: Update Direction Mismatch"),
        ("attn_drift_curve",  "AttnDrift (JS²)",                  "Dim 2: Attention Structure Drift"),
        ("update_norm_curve", "UpdateNorm (‖Δh‖)",               "Dim 3: Representation Stabilization"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        c_curve = correct_rec[key]
        w_curve = wrong_rec[key]
        x_c = np.arange(1, len(c_curve) + 1)
        x_w = np.arange(1, len(w_curve) + 1)
        ax.plot(x_c, c_curve, color="steelblue", label="Correct (normal)")
        ax.plot(x_w, w_curve, color="tomato", linestyle="--", label="Wrong (induced)")
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
