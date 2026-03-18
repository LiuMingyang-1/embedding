# extraction/states.py
# 负责从模型中提取 hidden states 和 attentions，并批量处理所有样本。
#
# extract_states:
#   先用 generate() 拿到模型真实输出的 token ids，再对这条完整序列做一次 forward pass
#   (output_hidden_states=True, output_attentions=True)，一次性获取所有层的内部状态。
#   这样可以避免把解码后的字符串重新 tokenizer 一次，导致 token 边界和真实生成不一致。
#
# run_extraction:
#   遍历 dataset.csv，按 batch 做生成，再对每个样本调用 extract_states，
#   将结果保存到 states/ 目录，并生成 results_all.csv。
#   若 results_all.csv 已存在，则按 id 保留已有 manual_has_hallucination / 额外列。

import torch
import pandas as pd

from config import MODEL_NAME, DATASET_CSV, STATES_DIR, RESULTS_CSV
from extraction.model import load_model, generate_answers, _get_model_device
from extraction.prompt import build_prompt
from extraction.correctness import select_answer_tokens, select_response_tokens

EXTRACTION_BATCH_SIZE = 4

DATASET_REQUIRED_COLUMNS = [
    "id",
    "prompt_text",
    "reference_answer",
]

RESULTS_BASE_COLUMNS = [
    "id",
    "prompt_text",
    "reference_answer",
    "model_response",
    "manual_has_hallucination",
    "response_num_tokens",
    "response_token_source",
    "answer_span_text",
    "answer_num_tokens",
    "answer_span_source",
]


def extract_states(model, token_ids):
    device = _get_model_device(model)
    input_ids = token_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise ValueError(
            "Model forward pass did not return attentions. "
            "This analysis pipeline requires per-layer attention weights. "
            "Use an attention implementation that materializes attentions "
            "(for example attn_implementation='eager') and rerun --stage extract."
        )

    # Move to CPU immediately to free GPU memory
    hidden_states = tuple(h[0].cpu().float() for h in outputs.hidden_states)
    attentions = tuple(a[0].cpu().float() for a in outputs.attentions)

    return {
        "hidden_states": hidden_states,  # (num_layers+1, seq_len, hidden_dim)
        "attentions": attentions,         # (num_layers, num_heads, seq_len, seq_len)
    }


def _load_existing_results():
    try:
        existing_df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        return {}, []

    if "id" not in existing_df.columns:
        return {}, []

    if existing_df["id"].duplicated().any():
        print(
            "Existing results_all.csv uses a legacy multi-row-per-id format. "
            "Skipping label preservation for this extract run."
        )
        return {}, []

    existing_rows = {}
    for row in existing_df.to_dict("records"):
        key = int(row["id"])
        existing_rows[key] = row

    extra_columns = [col for col in existing_df.columns if col not in RESULTS_BASE_COLUMNS]
    return existing_rows, extra_columns


def _load_dataset():
    df = pd.read_csv(DATASET_CSV)
    missing = [col for col in DATASET_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{DATASET_CSV} is missing required column(s): {missing_text}. "
            f"Required columns: {', '.join(DATASET_REQUIRED_COLUMNS)}"
        )

    if df["id"].duplicated().any():
        duplicated = df[df["id"].duplicated()]["id"].tolist()
        preview = ", ".join(str(x) for x in duplicated[:5])
        suffix = "" if len(duplicated) <= 5 else f", ... (+{len(duplicated) - 5} more)"
        raise ValueError(f"{DATASET_CSV} contains duplicate id values: {preview}{suffix}")

    return df


def _iter_batches(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


def _save_results_csv(results, extra_columns):
    results_df = pd.DataFrame(results)
    ordered_columns = RESULTS_BASE_COLUMNS + extra_columns
    if results_df.empty:
        results_df = pd.DataFrame(columns=ordered_columns)
    else:
        results_df = results_df[ordered_columns]
    results_df.to_csv(RESULTS_CSV, index=False)
    return results_df


def run_extraction(limit=None):
    model, tokenizer = load_model(MODEL_NAME)
    STATES_DIR.mkdir(exist_ok=True)

    dataset_df = _load_dataset()
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be a positive integer, got {limit}")
        dataset_df = dataset_df.head(limit).copy()
    existing_rows, extra_columns = _load_existing_results()

    results = []
    preserved_manual_labels = 0

    limit_text = f", limit={limit}" if limit is not None else ""
    print(f"\n=== Processing {DATASET_CSV} with batch_size={EXTRACTION_BATCH_SIZE}{limit_text} ===")
    dataset_records = dataset_df.to_dict("records")

    for batch_rows in _iter_batches(dataset_records, EXTRACTION_BATCH_SIZE):
        prompts = [build_prompt(tokenizer, str(row["prompt_text"])) for row in batch_rows]
        generations = generate_answers(model, tokenizer, prompts)

        for row, generation in zip(batch_rows, generations):
            sample_id = int(row["id"])
            prompt_text = str(row["prompt_text"])
            reference_answer = str(row["reference_answer"])

            answer_text = generation["answer_text"]
            response_info = select_response_tokens(
                tokenizer,
                generation["token_ids"],
                generation["prompt_len"],
            )
            answer_info = select_answer_tokens(
                tokenizer,
                generation["token_ids"],
                generation["prompt_len"],
            )

            print(f"  id={sample_id}: {answer_text!r}")

            states = extract_states(model, generation["token_ids"])

            save_data = {
                "id": sample_id,
                "prompt_text": prompt_text,
                "reference_answer": reference_answer,
                "model_response": answer_text,
                "token_ids": generation["token_ids"],
                "generated_token_ids": generation["generated_token_ids"],
                "prompt_len": generation["prompt_len"],
                "response_token_positions": response_info["positions"],
                "response_first_token_pos": response_info["first_position"],
                "response_text": response_info["text"],
                "response_token_source": response_info["source"],
                "answer_token_positions": answer_info["positions"],
                "answer_first_token_pos": answer_info["first_position"],
                "answer_span_text": answer_info["text"],
                "answer_span_source": answer_info["source"],
                **states,
            }

            torch.save(save_data, STATES_DIR / f"sample_{sample_id}.pt")

            existing_row = existing_rows.get(sample_id, {})
            manual_label = existing_row.get("manual_has_hallucination", "")
            if pd.notna(manual_label) and str(manual_label).strip():
                preserved_manual_labels += 1

            result_row = {
                "id": sample_id,
                "prompt_text": prompt_text,
                "reference_answer": reference_answer,
                "model_response": answer_text,
                "manual_has_hallucination": manual_label,
                "response_num_tokens": len(response_info["positions"]),
                "response_token_source": response_info["source"],
                "answer_span_text": answer_info["text"],
                "answer_num_tokens": len(answer_info["positions"]),
                "answer_span_source": answer_info["source"],
            }
            for col in extra_columns:
                result_row[col] = existing_row.get(col, "")
            results.append(result_row)

        results_df = _save_results_csv(results, extra_columns)
        print(f"  Partial results saved: {len(results_df)} sample(s) -> {RESULTS_CSV}")

    results_df = _save_results_csv(results, extra_columns)

    print(f"\nGenerated {len(results_df)} samples.")
    print(f"Results saved to {RESULTS_CSV}")
    if preserved_manual_labels:
        print(f"Preserved manual_has_hallucination for {preserved_manual_labels} sample(s).")
    print("Fill the 'manual_has_hallucination' column in results_all.csv before running --stage analyze.")
