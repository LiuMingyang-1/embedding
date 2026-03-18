# extraction/model.py
# 负责模型加载与文本生成。
# load_model: 加载 Qwen2.5-7B-Instruct，float16，device_map="auto"。
# generate_answers: 对一批 prompt 做贪婪解码，返回逐条生成结果。
# generate_answer: 单条封装，复用批处理逻辑。

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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def _trim_generated_token_ids(token_ids, pad_token_id, eos_token_id):
    trimmed = token_ids

    if eos_token_id is not None:
        eos_positions = (trimmed == eos_token_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            trimmed = trimmed[: eos_positions[0].item() + 1]

    if pad_token_id is not None and len(trimmed) > 0 and pad_token_id != eos_token_id:
        non_pad_positions = (trimmed != pad_token_id).nonzero(as_tuple=False)
        if len(non_pad_positions) == 0:
            trimmed = trimmed[:0]
        else:
            trimmed = trimmed[: non_pad_positions[-1].item() + 1]

    return trimmed.clone()


def generate_answers(model, tokenizer, prompt_texts):
    device = _get_model_device(model)
    prompt_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    model_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    padded_prompt_len = prompt_inputs.input_ids.shape[1]
    prompt_lens = prompt_inputs.attention_mask.sum(dim=1).tolist()

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    batch_results = []
    input_ids_cpu = prompt_inputs.input_ids.detach().cpu()
    output_ids_cpu = output_ids.detach().cpu()

    for idx, prompt_len in enumerate(prompt_lens):
        pad_len = padded_prompt_len - prompt_len
        prompt_token_ids = input_ids_cpu[idx][pad_len:]
        generated_token_ids = _trim_generated_token_ids(
            output_ids_cpu[idx][padded_prompt_len:],
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
        )
        full_token_ids = torch.cat([prompt_token_ids, generated_token_ids], dim=0)

        batch_results.append({
            "answer_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip(),
            "prompt_len": prompt_len,
            "token_ids": full_token_ids,
            "generated_token_ids": generated_token_ids,
        })

    return batch_results


def generate_answer(model, tokenizer, prompt_text):
    return generate_answers(model, tokenizer, [prompt_text])[0]
