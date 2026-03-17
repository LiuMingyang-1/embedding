# extraction/states.py
# 负责从模型中提取 hidden states 和 attentions，并批量处理所有样本。
#
# extract_states:
#   先用 generate() 拿到答案，再将 prompt+answer 拼成完整序列做一次 forward pass
#   (output_hidden_states=True, output_attentions=True)，一次性获取所有层的内部状态。
#   原因：generate() 是自回归逐 token 的，收集中间状态很繁琐；拼接后一次 forward
#   结果与逐 token 生成时完全等价（causal attention 保证）。
#
# run_extraction:
#   遍历 normal.csv 和 induced.csv，对每个样本调用 extract_states，
#   将结果保存到 states/ 目录，并生成 results_all.csv。

import torch
import pandas as pd

from config import MODEL_NAME, NORMAL_CSV, INDUCED_CSV, STATES_DIR, RESULTS_CSV
from extraction.model import load_model, generate_answer
from extraction.prompt import build_prompt
from extraction.correctness import is_correct


def extract_states(model, tokenizer, prompt_text, answer_text):
    full_text = prompt_text + answer_text
    inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
    prompt_len = tokenizer([prompt_text], return_tensors="pt").input_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

    # Move to CPU immediately to free GPU memory
    hidden_states = tuple(h[0].cpu().float() for h in outputs.hidden_states)
    attentions = tuple(a[0].cpu().float() for a in outputs.attentions)
    token_ids = inputs.input_ids[0].cpu()

    return {
        "hidden_states": hidden_states,  # (num_layers+1, seq_len, hidden_dim)
        "attentions": attentions,         # (num_layers, num_heads, seq_len, seq_len)
        "token_ids": token_ids,
        "prompt_len": prompt_len,
    }


def run_extraction():
    model, tokenizer = load_model(MODEL_NAME)
    STATES_DIR.mkdir(exist_ok=True)

    normal_df = pd.read_csv(NORMAL_CSV)
    induced_df = pd.read_csv(INDUCED_CSV)

    results = []

    for csv_type, df in [("normal", normal_df), ("induced", induced_df)]:
        print(f"\n=== Processing {csv_type} ===")
        for _, row in df.iterrows():
            sample_id = int(row["id"])
            misleading_prefix = row.get("misleading_prefix") if csv_type == "induced" else None
            question = row["question"]
            correct_answer = str(row["correct_answer"])
            expected_hallucination = str(row["expected_hallucination"])

            prompt_text = build_prompt(tokenizer, question, misleading_prefix)
            answer_text = generate_answer(model, tokenizer, prompt_text)
            correct = is_correct(answer_text, correct_answer, expected_hallucination)

            print(f"  [{'✓' if correct else '✗'}] id={sample_id}: {answer_text!r}")

            states = extract_states(model, tokenizer, prompt_text, answer_text)

            save_data = {
                "id": sample_id,
                "type": csv_type,
                "question": question,
                "correct_answer": correct_answer,
                "expected_hallucination": expected_hallucination,
                "model_response": answer_text,
                "is_correct": correct,
                **states,
            }

            torch.save(save_data, STATES_DIR / f"{sample_id}_{csv_type}.pt")

            results.append({
                "id": sample_id,
                "type": csv_type,
                "question": question,
                "correct_answer": correct_answer,
                "expected_hallucination": expected_hallucination,
                "model_response": answer_text,
                "is_correct": correct,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)

    total = len(results_df)
    correct_count = results_df["is_correct"].sum()
    normal_acc = results_df[results_df["type"] == "normal"]["is_correct"].mean()
    induced_acc = results_df[results_df["type"] == "induced"]["is_correct"].mean()

    print(f"\nOverall accuracy: {correct_count}/{total} = {correct_count/total:.1%}")
    print(f"Normal accuracy:  {normal_acc:.1%}")
    print(f"Induced accuracy: {induced_acc:.1%}")
    print(f"Results saved to {RESULTS_CSV}")
