# visualization/__init__.py
# 可视化子包，生成 4 张诊断图保存到 figures/ 目录。
# single_example: 图1 — 同一 id 的 normal(正确) vs induced(错误) 单样例层曲线对比
# group_average:  图2 — 所有正确 token vs 所有错误 token 的分组平均层曲线
# late_slope:     图3 — LateSlope / LateMean 分布（箱线图）
# correlation:    图4 — 三指标之间的 Spearman 相关热力图
