# analysis/compute.py
# 对单个样本的每个生成内容 token 计算全部三个维度的指标，返回结构化 record 列表。
# 加载所有 states/sample_*.pt 文件后，对每个样本调用 compute_sample_metrics，
# 再把整段生成内容按 token 聚合成 sample-level record，供可视化和统计分析使用。

import numpy as np
import torch

from config import STATES_DIR
from metrics.mismatch import compute_mismatch
from metrics.attn_drift import compute_attn_drift
from metrics.update_norm import compute_update_norm, compute_late_slope, compute_late_mean


def _get_response_info(data):
    positions = data.get("response_token_positions")
    if positions is not None:
        positions = list(positions)
        first_position = data.get("response_first_token_pos")
        if first_position is None and positions:
            first_position = positions[0]
        return {
            "positions": positions,
            "first_position": first_position,
            "source": data.get("response_token_source", "saved"),
            "text": data.get("response_text", data.get("model_response", "")),
        }

    generated_token_ids = data.get("generated_token_ids")
    prompt_len = data.get("prompt_len")
    if generated_token_ids is not None and prompt_len is not None:
        if hasattr(generated_token_ids, "tolist"):
            generated_token_ids = generated_token_ids.tolist()
        positions = [prompt_len + idx for idx in range(len(generated_token_ids))]
        first_position = positions[0] if positions else None
        return {
            "positions": positions,
            "first_position": first_position,
            "source": "generated_token_ids_fallback",
            "text": data.get("model_response", ""),
        }

    return {
        "positions": [],
        "first_position": None,
        "source": "missing_response_tokens",
        "text": data.get("model_response", ""),
    }


def _parse_manual_label(value):
    if value is None:
        return None

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    if text in {"1", "true", "t", "yes", "y", "hallucination", "hallucinated"}:
        return True
    if text in {"0", "false", "f", "no", "n", "non-hallucination", "not_hallucination"}:
        return False

    raise ValueError(
        f"Unsupported manual_has_hallucination value: {value!r}. "
        "Use 1/0, true/false, yes/no, or hallucination/non-hallucination."
    )


def build_manual_label_lookup(results_df):
    if "manual_has_hallucination" not in results_df.columns:
        raise ValueError(
            "results_all.csv is missing the 'manual_has_hallucination' column. "
            "Run --stage extract again, then fill that column before --stage analyze."
        )

    labels = {}
    missing = []
    seen_keys = set()

    for row in results_df.itertuples(index=False):
        key = int(row.id)
        if key in seen_keys:
            raise ValueError(f"Duplicate row in results_all.csv for id={key}.")
        seen_keys.add(key)

        try:
            label = _parse_manual_label(getattr(row, "manual_has_hallucination"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid manual_has_hallucination value for id={key}: "
                f"{getattr(row, 'manual_has_hallucination')!r}"
            ) from exc

        if label is None:
            missing.append(key)
            continue

        labels[key] = label

    if missing:
        preview = ", ".join(f"id={sid}" for sid in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise ValueError(
            "results_all.csv has unlabeled rows in 'manual_has_hallucination'. "
            f"Fill every row before analyzing. Missing: {preview}{suffix}"
        )

    return labels


def compute_sample_metrics(data):
    """
    对单个样本计算所有生成内容 token 的三个维度指标。

    Args:
        data:      torch.load 加载的 dict（来自 states/*.pt）

    Returns:
        list of dict，每个 dict 对应一个生成内容 token 的全部指标
    """
    hidden_states = data["hidden_states"]
    attentions = data["attentions"]
    token_ids = data["token_ids"]
    sample_has_hallucination = data["has_hallucination"]

    expected_num_attn_layers = len(hidden_states) - 1
    actual_num_attn_layers = len(attentions)
    if actual_num_attn_layers != expected_num_attn_layers:
        raise ValueError(
            f"State file for id={data['id']} has inconsistent layer counts: "
            f"{len(hidden_states)} hidden-state tensors but {actual_num_attn_layers} attention tensors. "
            "This usually means extraction did not actually save attention weights. "
            "Rerun --stage extract after forcing an attention implementation that returns attentions."
        )

    response_info = _get_response_info(data)
    positions = response_info["positions"]
    first_position = response_info["first_position"]

    records = []
    for idx, pos in enumerate(positions):
        if pos >= token_ids.shape[0]:
            continue

        mismatch = compute_mismatch(hidden_states, attentions, pos)
        attn_drift = compute_attn_drift(attentions, pos)
        update_norm = compute_update_norm(hidden_states, pos)

        records.append({
            "id": data["id"],
            "has_hallucination": sample_has_hallucination,
            "token_pos": pos,
            "token_index_in_response": idx,
            "is_first_response_token": pos == first_position,
            "response_token_source": response_info["source"],
            "response_text": response_info["text"],
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


def aggregate_sample_records(all_records):
    grouped = {}

    for record in all_records:
        key = record["id"]
        grouped.setdefault(key, []).append(record)

    sample_records = []
    for sample_id, records in grouped.items():
        records = sorted(records, key=lambda r: r["token_index_in_response"])
        response_mismatch_curve = np.stack([r["mismatch_curve"] for r in records]).mean(axis=0)
        response_attn_drift_curve = np.stack([r["attn_drift_curve"] for r in records]).mean(axis=0)
        response_update_norm_curve = np.stack([r["update_norm_curve"] for r in records]).mean(axis=0)

        sample_records.append({
            "id": sample_id,
            "has_hallucination": records[0]["has_hallucination"],
            "num_response_tokens": len(records),
            "response_token_source": records[0]["response_token_source"],
            "response_text": records[0]["response_text"],
            "response_token_positions": [r["token_pos"] for r in records],
            "response_first_token_pos": records[0]["token_pos"],
            "mismatch_curve": response_mismatch_curve,
            "attn_drift_curve": response_attn_drift_curve,
            "update_norm_curve": response_update_norm_curve,
            "mismatch_mean": float(np.mean([r["mismatch_mean"] for r in records])),
            "attn_drift_mean": float(np.mean([r["attn_drift_mean"] for r in records])),
            "update_norm_late_mean": float(np.mean([r["update_norm_late_mean"] for r in records])),
            "update_norm_late_slope": float(np.mean([r["update_norm_late_slope"] for r in records])),
        })

    return sorted(sample_records, key=lambda r: r["id"])


def load_all_records(label_lookup):
    """
    从 states/ 目录加载所有样本，计算指标，返回 record 列表。
    """
    state_files = sorted(STATES_DIR.glob("sample_*.pt"))
    if not state_files:
        raise FileNotFoundError(f"No .pt files in '{STATES_DIR}'. Run --stage extract first.")

    all_records = []
    seen_keys = set()
    print(f"Processing {len(state_files)} state files...")

    for path in state_files:
        print(f"  {path.name} ...", end=" ", flush=True)
        data = torch.load(path, map_location="cpu")
        key = int(data["id"])
        if key not in label_lookup:
            print("skipped (id not present in results_all.csv)")
            continue
        data["has_hallucination"] = label_lookup[key]
        records = compute_sample_metrics(data)
        if not records:
            raise ValueError(
                f"No generated response tokens found for id={key}. "
                "Check whether extraction saved response_token_positions correctly."
            )
        all_records.extend(records)
        seen_keys.add(key)
        print(f"{len(records)} token(s)")

    missing_states = sorted(set(label_lookup) - seen_keys)
    if missing_states:
        preview = ", ".join(f"id={sid}" for sid in missing_states[:5])
        suffix = "" if len(missing_states) <= 5 else f", ... (+{len(missing_states) - 5} more)"
        raise ValueError(
            "results_all.csv contains labeled rows without matching state files. "
            f"Missing states: {preview}{suffix}"
        )

    return all_records
