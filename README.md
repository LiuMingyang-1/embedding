# Hallucination Internal Dynamics Demo

验证假说：**当 LLM 产生幻觉时，模型内部的三个动态维度会同时出现异常，且彼此互补。**

---

## 研究目标

不追求指标好看，先回答三个最小问题：

1. 幻觉样本和非幻觉样本，三个维度的层间曲线有没有明显区别？
2. 三个维度捕捉的是不同的东西，而不是同一信号的重复？
3. 在错误答案词附近，这些异常会不会更集中？

---

## 实验设计

### 数据

15 对配对 factual QA（`normal.csv` + `induced.csv`），每对同一问题两个版本：

| 版本 | 策略 | 预期 |
|------|------|------|
| normal | 直接提问 | 模型大概率答对 |
| induced | 在问题前加误导性前缀 | 提高出错概率，诱发幻觉 |

示例：

> **normal**: What is the capital of Australia?
>
> **induced**: Many people believe Australia's capital is Sydney due to its fame. What is the capital of Australia? Answer directly.

两个 CSV 通过 `id`（1–15）对齐，同一 `id` 代表同一知识点的两个版本。

### 分组逻辑

最终分组**不按 normal/induced**，而是按实际回答正误：

- 正确回答的答案 token → **无幻觉组**
- 错误回答的答案 token → **幻觉组**

induced 的作用只是提高出错概率，保证有足够的幻觉样本。

### 模型

`Qwen/Qwen2.5-7B-Instruct`，float16，需 GPU 运行推理阶段。

---

## 三个内部动态维度

### 维度 1：更新方向偏离（Mismatch）

```
Mismatch_i(t) = 1 - cos(Δh_i(t), c_i(t))

Δh_i(t) = h_i(t) - h_{i-1}(t)           # 实际更新向量
c_i(t)  = Σ_j α_{i,t,j} · h_{i-1}(j)    # attention 加权的上下文聚合方向
```

**直觉**：正常情况下，表示更新应该朝着上下文信息整合的方向走。幻觉发生时，模型更多被内部记忆驱动，更新方向偏离上下文，Mismatch 偏高。

### 维度 2：注意力结构漂移（AttnDrift）

```
AttnDrift_i(t) = JS²(A_i(t), A_{i-1}(t))

A_i(t) = 第 i 层 token t 的注意力分布（按 head 取均值）
```

**直觉**：正常情况下注意力随层逐渐演化；幻觉词在中后层注意力会快速"锁定"，相邻层漂移值突然变小。

### 维度 3：表示稳定化（UpdateNorm）

```
U_i(t) = ||h_i(t) - h_{i-1}(t)||_2

LateSlope = 后 k 层 UpdateNorm 的线性回归斜率
LateMean  = 后 k 层 UpdateNorm 的均值
```

**直觉**：幻觉词的表示更早进入稳定（不再更新），说明模型过早"决定"了这个词，没有继续整合上下文。

---

## 文件结构

```
embedding/
│
├── config.py                    # 全局配置：模型名、文件路径、超参数
├── main.py                      # 入口脚本，--stage extract|analyze|all
│
├── extraction/                  # 推理与状态提取
│   ├── __init__.py
│   ├── model.py                 # 模型加载 + 贪婪解码生成
│   ├── prompt.py                # chat template prompt 构造
│   ├── states.py                # forward pass 提取 hidden states/attentions
│   └── correctness.py          # 正误判断 + 答案 token 位置定位
│
├── metrics/                     # 三个维度的指标计算
│   ├── __init__.py
│   ├── mismatch.py              # 维度 1: 更新方向偏离
│   ├── attn_drift.py            # 维度 2: 注意力结构漂移（JS²）
│   └── update_norm.py           # 维度 3: 表示稳定化 + LateSlope/LateMean
│
├── analysis/                    # 汇总计算与统计
│   ├── __init__.py
│   ├── compute.py               # 对每个样本的答案 token 计算全部指标
│   └── statistics.py            # Mann-Whitney U 检验 + Spearman 相关
│
├── visualization/               # 可视化（4 张图）
│   ├── __init__.py
│   ├── single_example.py        # 图1: 单样例配对层曲线对比
│   ├── group_average.py         # 图2: 正确/错误组平均层曲线
│   ├── late_slope.py            # 图3: LateSlope / LateMean 分布箱线图
│   └── correlation.py           # 图4: 三指标 Spearman 相关热力图
│
├── normal.csv                   # 15 条直接提问的 factual QA
├── induced.csv                  # 15 条含误导前缀的配对问题
└── eval_normal.py               # 早期验证脚本（已被 main.py 取代）
```

---

## 运行方式

```bash
# 阶段 A：云上跑（需 GPU，约 14GB+ 显存）
python main.py --stage extract

# 阶段 B：本地跑（CPU 即可）
# 先把云上生成的 states/ 目录和 results_all.csv 下载到本地
python main.py --stage analyze

# 一次性全跑
python main.py --stage all
```

### 依赖

```
torch
transformers
pandas
numpy
scipy
matplotlib
seaborn      # 可选，相关热力图更好看
```

---

## 输出产物

| 产物 | 说明 |
|------|------|
| `states/` | 30 个 `.pt` 文件，每个样本的完整 hidden states 和 attentions |
| `results_all.csv` | 所有样本的生成结果、正误标签 |
| `figures/fig1_single_example.png` | 单样例配对层曲线对比 |
| `figures/fig2_group_average.png` | 正确/错误组平均层曲线 |
| `figures/fig3_late_slope.png` | 后期稳定化分布箱线图 |
| `figures/fig4_correlation.png` | 三指标互补性相关热力图 |
| 终端输出 | Mann-Whitney U p-value + Spearman ρ |

---

## 预期现象

| 维度 | 幻觉词预期表现 |
|------|--------------|
| Mismatch | 中后层偏高，更新方向偏离上下文 |
| AttnDrift | 中后层漂移突然减小（注意力固化） |
| UpdateNorm | LateSlope 更负，后期更快收敛 |

三个维度如果**中度相关而非高度相关**（|ρ| < 0.5），说明互补性成立，值得进一步发展为完整方法。
