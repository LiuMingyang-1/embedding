# visualization/single_example.py
# 图1：单样例配对对比。
# 选取同一 id 中 normal=正确、induced=错误的一对样本，
# 画出整段生成内容按 token 平均后的层曲线，直观展示幻觉 vs 非幻觉的内部状态差异。

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR


def plot_single_example(sample_records):
    FIGURES_DIR.mkdir(exist_ok=True)

    records_by_key = {(r["id"], r["type"]): r for r in sample_records}

    # Find a pair: same id, normal=correct, induced=wrong
    pair_id = None
    correct_rec = None
    wrong_rec = None
    for sid in sorted({r["id"] for r in sample_records}):
        normal_rec = records_by_key.get((sid, "normal"))
        induced_rec = records_by_key.get((sid, "induced"))
        if normal_rec and induced_rec and normal_rec["is_correct"] and not induced_rec["is_correct"]:
            pair_id = sid
            correct_rec = normal_rec
            wrong_rec = induced_rec
            break

    if pair_id is None:
        print("[fig1] No ideal pair (normal correct, induced wrong) found. Skipping.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Single Example Comparison  (Whole Response Average, id={pair_id})", fontsize=14)

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
