# extraction/prompt.py
# 负责构造输入给模型的 prompt。
# build_prompt: 将 (question, misleading_prefix) 组装为 chat template 格式的完整字符串。

SYSTEM_MSG = (
    "You are a helpful assistant. "
    "Answer with only the answer itself, no explanation, no extra words."
)


def build_prompt(tokenizer, question, misleading_prefix=None):
    content = f"{misleading_prefix} {question}" if misleading_prefix else question
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
