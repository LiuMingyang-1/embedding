import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "normal.csv"
OUTPUT_FILE = "results_normal.csv"

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question concisely and directly."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def is_correct(response, correct_answer):
    return correct_answer.lower() in response.lower()

def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)

    df = pd.read_csv(DATA_FILE)
    results = []
    correct_count = 0

    for _, row in df.iterrows():
        response = generate_answer(model, tokenizer, row["question"])
        correct = is_correct(response, row["correct_answer"])
        correct_count += int(correct)

        print(f"[{'✓' if correct else '✗'}] Q: {row['question']}")
        print(f"    Expected: {row['correct_answer']}")
        print(f"    Got:      {response}")

        results.append({
            "id": row["id"],
            "domain": row["domain"],
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "model_response": response,
            "correct": correct,
        })

    total = len(df)
    accuracy = correct_count / total
    print(f"\nAccuracy: {correct_count}/{total} = {accuracy:.1%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
