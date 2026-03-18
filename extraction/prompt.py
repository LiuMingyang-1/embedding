# extraction/prompt.py
# 负责构造输入给模型的 prompt。
# build_prompt: 将单条 prompt_text 组装为 chat template 格式的完整字符串。

SYSTEM_MSG = (
    "You are a helpful assistant. "
    "Answer with only the answer itself, no explanation, no extra words."
)


def build_prompt(tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
