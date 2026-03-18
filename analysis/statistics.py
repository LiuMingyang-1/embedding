# analysis/statistics.py
# 统计检验：验证三个维度在幻觉 vs 非幻觉样本之间是否有显著差异，以及是否互补。
#
# run_statistics:
#   1. Mann-Whitney U 检验：每个指标在正确/错误回答组之间的差异（p-value）
#   2. Spearman 相关：三指标之间的相关性
#      |ρ| < 0.5 → 互补；|ρ| > 0.8 → 冗余
#
# 注意：输入应为 sample-level records（每个回答一条）。
# 当前主分析对象为整段生成内容，对所有生成内容 token 取平均。

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


def run_statistics(all_records):
    correct = [r for r in all_records if r["is_correct"]]
    wrong = [r for r in all_records if not r["is_correct"]]

    print("\n" + "=" * 65)
    print("STATISTICS")
    print("=" * 65)
    print(f"Correct samples: {len(correct)},  Wrong samples: {len(wrong)}")

    if not correct or not wrong:
        print("Not enough samples for statistics.")
        return

    metrics = [
        ("mismatch_mean",          "Mismatch (whole response)       "),
        ("attn_drift_mean",        "AttnDrift (whole response)      "),
        ("update_norm_late_mean",  "UpdateNorm late mean            "),
        ("update_norm_late_slope", "UpdateNorm late slope           "),
    ]

    print("\n--- Mann-Whitney U Test (correct vs wrong) ---")
    print(f"{'Metric':<38} {'Correct':>8} {'Wrong':>8} {'p-value':>10} {'':>5}")
    print("-" * 73)

    for key, label in metrics:
        c_vals = [r[key] for r in correct]
        w_vals = [r[key] for r in wrong]
        _, p = mannwhitneyu(c_vals, w_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{label} {np.mean(c_vals):>8.4f} {np.mean(w_vals):>8.4f} {p:>10.4f} {sig:>5}")

    print("\n--- Spearman Correlation Between Metrics (sample-level) ---")
    m1 = [r["mismatch_mean"] for r in all_records]
    m2 = [r["attn_drift_mean"] for r in all_records]
    m3 = [r["update_norm_late_mean"] for r in all_records]

    r12, p12 = spearmanr(m1, m2)
    r13, p13 = spearmanr(m1, m3)
    r23, p23 = spearmanr(m2, m3)

    print(f"  Mismatch  vs AttnDrift:    ρ = {r12:+.3f}  (p={p12:.3f})")
    print(f"  Mismatch  vs UpdateNorm:   ρ = {r13:+.3f}  (p={p13:.3f})")
    print(f"  AttnDrift vs UpdateNorm:   ρ = {r23:+.3f}  (p={p23:.3f})")
    print("\n  |ρ| < 0.5 → complementary   |ρ| > 0.8 → redundant")
    print("=" * 65)
