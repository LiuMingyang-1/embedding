# extraction/model.py
# 负责模型加载与文本生成。
# load_model: 加载 Qwen2.5-7B-Instruct，float16，device_map="auto"。
# generate_answer: 对给定 prompt 做贪婪解码，返回生成的文本。

import torch
from config import MODEL_NAME


def _get_model_device(model):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


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
    device = _get_model_device(model)
    prompt_inputs = tokenizer([prompt_text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    prompt_len = prompt_inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    full_token_ids = output_ids[0].detach().cpu()
    generated_token_ids = full_token_ids[prompt_len:].clone()

    return {
        "answer_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip(),
        "prompt_len": prompt_len,
        "token_ids": full_token_ids,
        "generated_token_ids": generated_token_ids,
    }
