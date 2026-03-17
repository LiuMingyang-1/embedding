# metrics/update_norm.py
# 维度 3：表示稳定化程度（UpdateNorm）
#
# 关注：某个 token 的 hidden state 在逐层之间的更新幅度，以及后期是否快速收敛。
# 幻觉假说：产生幻觉时，模型对该 token 的表示更早进入稳定状态（后期更新幅度急剧下降），
# 意味着模型更早"决定"了这个词，没有继续整合上下文信息。
#
# U_i(t) = ||h_i(t) - h_{i-1}(t)||_2
# LateSlope: 后 k 层 UpdateNorm 的线性回归斜率（越负 = 后期收敛越快）
# LateMean:  后 k 层 UpdateNorm 的均值

import numpy as np
import torch
from config import LATE_K


def compute_update_norm(hidden_states, token_pos):
    """
    Args:
        hidden_states: tuple of (seq_len, hidden_dim), length = num_layers + 1
        token_pos:     int

    Returns:
        np.array of shape (num_layers,)
    """
    num_layers = len(hidden_states) - 1
    norms = np.zeros(num_layers)
    for i in range(1, num_layers + 1):
        delta = hidden_states[i][token_pos] - hidden_states[i - 1][token_pos]
        norms[i - 1] = torch.norm(delta).item()
    return norms


def compute_late_slope(update_norms, k=LATE_K):
    """后 k 层的线性回归斜率。越负 → 后期收敛越快。"""
    tail = update_norms[-k:]
    if len(tail) < 2:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def compute_late_mean(update_norms, k=LATE_K):
    """后 k 层的均值。"""
    return float(np.mean(update_norms[-k:]))
