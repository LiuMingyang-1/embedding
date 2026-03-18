# metrics/__init__.py
# 三个内部动态维度的指标计算子包。
# mismatch:    维度 1 — 更新方向与 attention-context 的不对齐程度
# attn_drift:  维度 2 — 相邻层注意力分布漂移（JS 散度）
# update_norm: 维度 3 — 逐层表示更新幅度及后期稳定化程度
