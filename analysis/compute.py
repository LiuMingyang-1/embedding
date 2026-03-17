# analysis/compute.py
# 对单个样本的每个答案 token 计算全部三个维度的指标，返回结构化 record 列表。
# 加载所有 states/*.pt 文件后，对每个样本调用 compute_sample_metrics，
# 汇总结果供可视化和统计分析使用。

from pathlib import Path
import torch

from config import STATES_DIR
from extraction.correctness import identify_answer_tokens
from metrics.mismatch import compute_mismatch
from metrics.attn_drift import compute_attn_drift
from metrics.update_norm import compute_update_norm, compute_late_slope, compute_late_mean


def compute_sample_metrics(data, tokenizer):
    """
    对单个样本计算所有答案 token 的三个维度指标。

    Args:
        data:      torch.load 加载的 dict（来自 states/*.pt）
        tokenizer: HuggingFace tokenizer

    Returns:
        list of dict，每个 dict 对应一个答案 token 的全部指标
    """
    hidden_states = data["hidden_states"]
    attentions = data["attentions"]
    token_ids = data["token_ids"]
    prompt_len = data["prompt_len"]
    sample_is_correct = data["is_correct"]

    positions = identify_answer_tokens(
        tokenizer, token_ids, prompt_len,
        data["correct_answer"], data["expected_hallucination"],
        sample_is_correct,
    )

    records = []
    for pos in positions:
        if pos >= token_ids.shape[0]:
            continue

        mismatch = compute_mismatch(hidden_states, attentions, pos)
        attn_drift = compute_attn_drift(attentions, pos)
        update_norm = compute_update_norm(hidden_states, pos)

        records.append({
            "id": data["id"],
            "type": data["type"],
            "is_correct": sample_is_correct,
            "token_pos": pos,
            # Full layer curves (for plotting)
            "mismatch_curve": mismatch,
            "attn_drift_curve": attn_drift,
            "update_norm_curve": update_norm,
            # Scalar summaries (for statistics)
            "mismatch_mean": float(mismatch.mean()),
            "attn_drift_mean": float(attn_drift.mean()),
            "update_norm_late_mean": compute_late_mean(update_norm),
            "update_norm_late_slope": compute_late_slope(update_norm),
        })

    return records


def load_all_records(tokenizer):
    """
    从 states/ 目录加载所有样本，计算指标，返回 record 列表。
    """
    state_files = sorted(STATES_DIR.glob("*.pt"))
    if not state_files:
        raise FileNotFoundError(f"No .pt files in '{STATES_DIR}'. Run --stage extract first.")

    all_records = []
    print(f"Processing {len(state_files)} state files...")

    for path in state_files:
        print(f"  {path.name} ...", end=" ", flush=True)
        data = torch.load(path, map_location="cpu")
        records = compute_sample_metrics(data, tokenizer)
        all_records.extend(records)
        print(f"{len(records)} token(s)" if records else "no answer tokens found")

    return all_records
