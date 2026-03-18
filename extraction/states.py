# extraction/states.py
# 负责从模型中提取 hidden states 和 attentions，并批量处理所有样本。
#
# extract_states:
#   先用 generate() 拿到模型真实输出的 token ids，再对这条完整序列做一次 forward pass
#   (output_hidden_states=True, output_attentions=True)，一次性获取所有层的内部状态。
#   这样可以避免把解码后的字符串重新 tokenizer 一次，导致 token 边界和真实生成不一致。
#
# run_extraction:
#   遍历 normal.csv 和 induced.csv，对每个样本调用 extract_states，
#   将结果保存到 states/ 目录，并生成 results_all.csv。
#   若 results_all.csv 已存在，则按 (id, type) 保留已有 manual_is_correct / 额外列。

import torch
import pandas as pd

from config import MODEL_NAME, NORMAL_CSV, INDUCED_CSV, STATES_DIR, RESULTS_CSV
from extraction.model import load_model, generate_answer, _get_model_device
from extraction.prompt import build_prompt
from extraction.correctness import select_answer_tokens, select_response_tokens

RESULTS_BASE_COLUMNS = [
    "id",
    "type",
    "question",
    "correct_answer",
    "expected_hallucination",
    "model_response",
    "manual_is_correct",
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

    existing_rows = {}
    for row in existing_df.to_dict("records"):
        key = (int(row["id"]), str(row["type"]))
        existing_rows[key] = row

    extra_columns = [col for col in existing_df.columns if col not in RESULTS_BASE_COLUMNS]
    return existing_rows, extra_columns


def run_extraction():
    model, tokenizer = load_model(MODEL_NAME)
    STATES_DIR.mkdir(exist_ok=True)

    normal_df = pd.read_csv(NORMAL_CSV)
    induced_df = pd.read_csv(INDUCED_CSV)
    existing_rows, extra_columns = _load_existing_results()

    results = []
    preserved_manual_labels = 0

    for csv_type, df in [("normal", normal_df), ("induced", induced_df)]:
        print(f"\n=== Processing {csv_type} ===")
        for _, row in df.iterrows():
            sample_id = int(row["id"])
            misleading_prefix = row.get("misleading_prefix") if csv_type == "induced" else None
            question = row["question"]
            correct_answer = str(row["correct_answer"])
            expected_hallucination = str(row["expected_hallucination"])

            prompt_text = build_prompt(tokenizer, question, misleading_prefix)
            generation = generate_answer(model, tokenizer, prompt_text)
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
                "type": csv_type,
                "question": question,
                "correct_answer": correct_answer,
                "expected_hallucination": expected_hallucination,
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

            torch.save(save_data, STATES_DIR / f"{sample_id}_{csv_type}.pt")

            key = (sample_id, csv_type)
            existing_row = existing_rows.get(key, {})
            manual_label = existing_row.get("manual_is_correct", "")
            if pd.notna(manual_label) and str(manual_label).strip():
                preserved_manual_labels += 1

            result_row = {
                "id": sample_id,
                "type": csv_type,
                "question": question,
                "correct_answer": correct_answer,
                "expected_hallucination": expected_hallucination,
                "model_response": answer_text,
                "manual_is_correct": manual_label,
                "response_num_tokens": len(response_info["positions"]),
                "response_token_source": response_info["source"],
                "answer_span_text": answer_info["text"],
                "answer_num_tokens": len(answer_info["positions"]),
                "answer_span_source": answer_info["source"],
            }
            for col in extra_columns:
                result_row[col] = existing_row.get(col, "")
            results.append(result_row)

    results_df = pd.DataFrame(results)
    ordered_columns = RESULTS_BASE_COLUMNS + extra_columns
    results_df = results_df[ordered_columns]
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"\nGenerated {len(results_df)} samples.")
    print(f"Results saved to {RESULTS_CSV}")
    if preserved_manual_labels:
        print(f"Preserved manual_is_correct for {preserved_manual_labels} sample(s).")
    print("Fill the 'manual_is_correct' column in results_all.csv before running --stage analyze.")
