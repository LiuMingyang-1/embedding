# metrics/mismatch.py
# 维度 1：状态更新方向偏离（Mismatch）
#
# 关注：模型在更新某个 token 的表示时，更新方向是否偏离了上下文聚合方向。
# 幻觉假说：产生幻觉时，更新方向更多由模型内部记忆驱动，而非上下文，导致 Mismatch 偏高。
#
# Mismatch_i(t) = 1 - cos(Δh_i(t), c_i(t))
#   Δh_i(t) = h_i(t) - h_{i-1}(t)            实际更新向量
#   c_i(t)  = Σ_j α_{i,t,j} · h_{i-1}(j)     上下文聚合方向（attention 加权的上一层表示）
#
# 注意：attention 用第 i 层的权重（按 head 取均值），hidden states 用第 i-1 层。

import numpy as np
import torch


def compute_mismatch(hidden_states, attentions, token_pos):
    """
    Args:
        hidden_states: tuple of (seq_len, hidden_dim), length = num_layers + 1
        attentions:    tuple of (num_heads, seq_len, seq_len), length = num_layers
        token_pos:     int, position of the target token in the full sequence

    Returns:
        np.array of shape (num_layers,), one value per layer
    """
    num_layers = len(hidden_states) - 1
    mismatches = np.zeros(num_layers)

    for i in range(1, num_layers + 1):
        h_curr = hidden_states[i][token_pos]       # (hidden_dim,)
        h_prev_all = hidden_states[i - 1]          # (seq_len, hidden_dim)
        delta_h = h_curr - h_prev_all[token_pos]

        # Context vector: attention-weighted sum of h_{i-1}
        attn_avg = attentions[i - 1].mean(dim=0)   # (seq_len, seq_len)
        attn_row = attn_avg[token_pos, :]           # (seq_len,)
        context = (attn_row.unsqueeze(1) * h_prev_all).sum(dim=0)  # (hidden_dim,)

        norm_delta = torch.norm(delta_h)
        norm_ctx = torch.norm(context)
        if norm_delta < 1e-8 or norm_ctx < 1e-8:
            mismatches[i - 1] = 1.0  # undefined → neutral midpoint
        else:
            cos_sim = torch.dot(delta_h, context) / (norm_delta * norm_ctx)
            mismatches[i - 1] = 1.0 - cos_sim.item()

    return mismatches
