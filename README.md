# Hallucination Internal Dynamics Demo

验证假说：当 LLM 产生幻觉时，模型内部的三个动态维度会同时出现异常，且彼此互补。

## 当前数据格式

当前版本使用单一数据文件 `dataset.csv`，每行一个样本，不再预设：

- 预期幻觉内容
- normal / induced 二分类
- 配对结构

唯一必需列：

| 列名 | 说明 |
|------|------|
| `id` | 样本唯一标识，必须唯一 |
| `prompt_text` | 直接发给模型的用户问题或提示 |
| `reference_answer` | 参考答案，仅供人工标注时参考，不参与分析分组 |

示例：

```csv
id,prompt_text,reference_answer
1,What is the capital of Australia?,Canberra
2,Many people believe Australia's capital is Sydney due to its fame. What is the capital of Australia? Answer directly.,Canberra
```

你只需要把自己的数据按这个格式放进 [dataset.csv](/Users/lmy/Documents/code/embedding/dataset.csv)。

## 标注方式

模型生成后会导出 `results_all.csv`，你只需要填写：

- `manual_has_hallucination = 1/true/yes`：有幻觉
- `manual_has_hallucination = 0/false/no`：无幻觉

最终统计和画图都只按“是否幻觉”分组。

## 当前版本在做什么

- 样本级标签：是否幻觉，由你在 `results_all.csv` 中人工填写 `manual_has_hallucination`
- 分析对象：整段生成内容
- 聚合方式：对生成内容中的所有内容 token 分别计算三种指标，再在 token 维度上取平均，得到 sample-level 曲线和统计量

这版先回答两个问题：

1. 幻觉样本和非幻觉样本在整段生成内容层面，三条曲线有没有系统性区别？
2. 三个维度之间是不是互补，而不是高度冗余？

## 三个内部动态维度

### 维度 1：更新-上下文不对齐（Mismatch）

```text
Mismatch_i(t) = 1 - cos(Δh_i(t), c_i(t))

Δh_i(t) = h_i(t) - h_{i-1}(t)
c_i(t)  = Σ_j α_{i,t,j} · h_{i-1}(j)
```

直觉：该指标衡量某层对 token 的表示更新，是否与该层 attention 聚合出的上下文向量一致。
`Mismatch` 越小，说明更新方向与该上下文向量越接近；`Mismatch` 越大，说明两者越不一致。
它描述的是一种“更新-上下文对齐度”，而不是对“是否依赖上下文”或“是否幻觉”的直接判定。

### 维度 2：注意力结构漂移（AttnDrift）

```text
AttnDrift_i(t) = JS²(A_i(t), A_{i-1}(t))
```

直觉：正常情况下注意力逐层演化；幻觉时中后层可能更早固化。

### 维度 3：表示稳定化（UpdateNorm）

```text
U_i(t) = ||h_i(t) - h_{i-1}(t)||_2

LateSlope = 后 k 层 UpdateNorm 的线性回归斜率
LateMean  = 后 k 层 UpdateNorm 的均值
```

直觉：如果模型过早“拍板”，后期更新会更快收敛。

## 运行方式

```bash
# 阶段 A：云上跑（需 GPU）
python3 main.py --stage extract

# 只先试跑前 20 条，并且会在每个 batch 后刷新 results_all.csv
python3 main.py --stage extract --limit 20

# 阶段 B：人工标注
# 在 results_all.csv 里填写 manual_has_hallucination

# 阶段 C：本地跑（CPU 即可）
python3 main.py --stage analyze
```

## 主要产物

- `states/sample_<id>.pt`：每个样本的 hidden states、attentions 和生成 token 位置
- `results_all.csv`：生成结果和人工幻觉标签
- `figures/fig1_single_example.png`：一个非幻觉样本 vs 一个幻觉样本
- `figures/fig2_group_average.png`：两组平均层曲线
- `figures/fig3_late_slope.png`：后期稳定化分布
- `figures/fig4_correlation.png`：三指标相关热力图

## 文件结构

```text
embedding/
├── config.py
├── dataset.csv
├── main.py
├── extraction/
├── metrics/
├── analysis/
├── visualization/
├── normal.csv
├── induced.csv
└── eval_normal.py
```

其中 `normal.csv`、`induced.csv`、`eval_normal.py` 属于旧版本遗留文件，当前主流程不再依赖。
