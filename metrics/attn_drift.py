# metrics/attn_drift.py
# 维度 2：注意力结构漂移（AttnDrift）
#
# 关注：某个 token 的注意力分布在相邻层之间的变化幅度。
# 幻觉假说：产生幻觉时，注意力在中后层快速"锁定"，漂移值突然变小，说明模式固化。
#
# AttnDrift_i(t) = JS(A_i(t), A_{i-1}(t))
#   A_i(t) = 第 i 层对 token t 的注意力分布，按 head 取均值，只保留 t 之前的位置。
#   JS = Jensen-Shannon 散度（取平方使其成为真正的距离度量）

import numpy as np
from scipy.spatial.distance import jensenshannon


def _get_attn_dist(attentions, layer_idx, token_pos):
    """提取第 layer_idx 层 token_pos 的注意力分布（归一化概率向量）。"""
    attn_avg = attentions[layer_idx].mean(dim=0)         # (seq_len, seq_len)
    attn_row = attn_avg[token_pos, :token_pos + 1].numpy().astype(np.float64)
    attn_row = np.maximum(attn_row, 0)
    s = attn_row.sum()
    return attn_row / s if s > 1e-10 else np.ones(len(attn_row)) / len(attn_row)


def compute_attn_drift(attentions, token_pos):
    """
    Args:
        attentions: tuple of (num_heads, seq_len, seq_len), length = num_layers
        token_pos:  int

    Returns:
        np.array of shape (num_layers - 1,), JS² divergence between adjacent layers
    """
    num_layers = len(attentions)
    drifts = np.zeros(num_layers - 1)
    dists = [_get_attn_dist(attentions, i, token_pos) for i in range(num_layers)]

    for i in range(num_layers - 1):
        p, q = dists[i], dists[i + 1]
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)))
        q = np.pad(q, (0, max_len - len(q)))
        drifts[i] = jensenshannon(p, q) ** 2

    return drifts
