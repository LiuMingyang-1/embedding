# Hallucination Internal Dynamics Demo

验证假说：**当 LLM 产生幻觉时，模型内部的三个动态维度会同时出现异常，且彼此互补。**

---

## 当前版本在做什么

当前实现先走一个最稳的 V1：

- **样本级标签**：是否幻觉，由你在 `results_all.csv` 里人工填写 `manual_is_correct`
- **分析对象**：不是单个答案 token，而是**整段生成内容**
- **聚合方式**：对生成内容中的所有内容 token 分别计算三种指标，再在 token 维度上取平均，得到 sample-level 曲线和统计量

这样做的目标不是最精细，而是先把整条流程稳定跑通，先回答：

1. 正确回答和错误回答，在整段生成内容层面，三条曲线有没有系统性区别？
2. 三个维度之间是不是互补，而不是高度冗余？

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

最终分组**不按 normal/induced**，而是按你在 `results_all.csv` 中人工填写的 `manual_is_correct`：

- `manual_is_correct = true/1/yes/correct` → **正确组**
- `manual_is_correct = false/0/no/wrong` → **错误组**

`induced` 的作用只是提高出错概率，保证有足够的错误样本；最终统计不依赖自动字符串匹配判断对错。

### 模型

`Qwen/Qwen2.5-7B-Instruct`，float16，需 GPU 运行推理阶段。

---

## 三个内部动态维度

下面三个指标都是**token 级**定义的；当前版本会先对整段生成内容中的每个 token 分别计算，再在 token 维度上求平均。

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

**直觉**：正常情况下注意力随层逐渐演化；幻觉生成时，中后层可能更早出现注意力模式固化，相邻层漂移值下降。

### 维度 3：表示稳定化（UpdateNorm）

```
U_i(t) = ||h_i(t) - h_{i-1}(t)||_2

LateSlope = 后 k 层 UpdateNorm 的线性回归斜率
LateMean  = 后 k 层 UpdateNorm 的均值
```

**直觉**：如果模型对某段生成过早“拍板”，后期更新会更快收敛，表现为 `LateSlope` 更负、`LateMean` 更低。

### 当前 sample-level 聚合

对每个样本，先找出生成内容中的所有内容 token：

```
T_response = {t1, t2, ..., tn}
```

然后对每个 token 分别计算三条层曲线，再在 token 维度上取平均：

```
MismatchCurve_i(sample)   = mean_t Mismatch_i(t)
AttnDriftCurve_i(sample) = mean_t AttnDrift_i(t)
UpdateNormCurve_i(sample)= mean_t UpdateNorm_i(t)
```

`LateMean` / `LateSlope` 也先按 token 计算，再在整段生成内容上取平均。

---

## 文件结构

```
embedding/
│
├── config.py                    # 全局配置：模型名、文件路径、超参数
├── main.py                      # 入口脚本，--stage extract|analyze
│
├── extraction/                  # 推理与状态提取
│   ├── __init__.py
│   ├── model.py                 # 模型加载 + 贪婪解码生成
│   ├── prompt.py                # chat template prompt 构造
│   ├── states.py                # forward pass 提取 hidden states/attentions，保存整段生成内容位置
│   └── correctness.py           # token 位置选择：当前保存整段生成内容；答案 span 逻辑保留给后续版本
│
├── metrics/                     # 三个维度的指标计算
│   ├── __init__.py
│   ├── mismatch.py              # 维度 1: 更新方向偏离
│   ├── attn_drift.py            # 维度 2: 注意力结构漂移（JS²）
│   └── update_norm.py           # 维度 3: 表示稳定化 + LateSlope/LateMean
│
├── analysis/                    # 汇总计算与统计
│   ├── __init__.py
│   ├── compute.py               # 对整段生成内容的所有 token 计算指标并聚合到 sample-level
│   └── statistics.py            # Mann-Whitney U 检验 + Spearman 相关
│
├── visualization/               # 可视化（4 张图）
│   ├── __init__.py
│   ├── single_example.py        # 图1: 单样例配对层曲线对比（整段生成内容平均）
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
# 生成 states/ 和待人工标注的 results_all.csv
python3 main.py --stage extract

# 阶段 B：人工标注
# 在 results_all.csv 里填写 manual_is_correct 列
# 可用值：1/0, true/false, yes/no, correct/wrong

# 阶段 C：本地跑（CPU 即可）
# 先把云上生成并标注完成的 states/ 目录和 results_all.csv 下载到本地
python3 main.py --stage analyze
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

说明：

- `extract` 阶段需要 GPU 和模型权重
- `analyze` 阶段只读取已保存的 `states/*.pt` 和 `results_all.csv`
- 建议在切到当前版本后重新跑一次 `extract`，让 state 文件包含新的 `response_token_positions`

---

## 输出产物

| 产物 | 说明 |
|------|------|
| `states/` | 30 个 `.pt` 文件，每个样本的完整 hidden states、attentions 和整段生成内容 token 位置 |
| `results_all.csv` | 所有样本的生成结果、人工标签列，以及生成内容 token 数等辅助字段 |
| `figures/fig1_single_example.png` | 单样例配对层曲线对比 |
| `figures/fig2_group_average.png` | 正确/错误组平均层曲线 |
| `figures/fig3_late_slope.png` | 后期稳定化分布箱线图 |
| `figures/fig4_correlation.png` | 三指标互补性相关热力图 |
| 终端输出 | Mann-Whitney U p-value + Spearman ρ |

---

## 当前限制

- 当前版本是**整段生成内容平均**，不是幻觉 onset 定位。
- 如果回答里模板 token 很多，比如 `The capital is ...`，真实错误信号可能会被整体平均稀释。
- 这版适合先看“整体趋势是否存在”，不适合回答“幻觉究竟从哪个 token 开始出现”。

---

## 后续规划

当前优先级是先把整句分析跑通。之后计划按下面顺序迭代：

1. **幻觉锚点分析**：不再只看整段平均，而是显式定位错误内容 token 或答案起始位置。
2. **局部窗口分析**：围绕锚点做前后若干 token 的局部曲线或热图，观察幻觉形成过程。
3. **整句视角 + 锚点视角并行**：保留当前整句平均作为全局视图，同时增加细粒度视图，避免只看单点或只看整句。
4. **更稳的标注方式**：必要时把 `hallucination_start` 或 `answer_span` 也加入人工标注，而不是完全依赖启发式。

---

## 预期现象

在当前“整段生成内容平均”的版本里，预期看到的是**整体趋势**而不是单个错误词的尖峰：

| 维度 | 错误回答组预期表现 |
|------|------------------|
| Mismatch | 中后层整体更高，更新方向更偏离上下文 |
| AttnDrift | 中后层整体更低，注意力更早固化 |
| UpdateNorm | `LateSlope` 更负，`LateMean` 更低，后期更快收敛 |

三个维度如果**中度相关而非高度相关**（|ρ| < 0.5），说明互补性成立，值得进一步发展为完整方法。
