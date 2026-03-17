# extraction/model.py
# 负责模型加载与文本生成。
# load_model: 加载 Qwen2.5-7B-Instruct，float16，device_map="auto"。
# generate_answer: 对给定 prompt 做贪婪解码，返回生成的文本。

import torch
from config import MODEL_NAME


def load_model(model_name=MODEL_NAME):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, prompt_text):
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
