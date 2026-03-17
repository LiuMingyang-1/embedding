# extraction/correctness.py
# 负责判断模型回答是否正确，以及定位生成文本中的关键答案 token。
#
# is_correct:
#   灵活的正误判断，避免 substring match 过严导致的误判（如单复数、缩写、同义表述）。
#   同时检查 expected_hallucination 是否出现（出现则判错）。
#
# identify_answer_tokens:
#   在生成部分的 token 序列中定位关键答案词的 token 位置。
#   正确回答 → 找 correct_answer 对应的 token；
#   错误回答 → 找 expected_hallucination 对应的 token。

import re


def _normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(response, correct_answer, expected_hallucination):
    resp = _normalize(response)
    correct_norm = _normalize(correct_answer)
    halluc_norm = _normalize(expected_hallucination)

    # If expected hallucination appears with high word overlap → wrong
    if halluc_norm and len(halluc_norm) > 2:
        halluc_words = set(halluc_norm.split())
        resp_words = set(resp.split())
        if len(halluc_words & resp_words) / max(len(halluc_words), 1) >= 0.7:
            return False

    # Check correct answer keywords
    correct_words = correct_norm.split()
    if len(correct_words) == 1:
        return correct_norm in resp
    else:
        overlap = sum(1 for w in correct_words if w in resp)
        return overlap / len(correct_words) >= 0.6


def identify_answer_tokens(tokenizer, token_ids, prompt_len,
                           correct_answer, expected_hallucination, sample_is_correct):
    target = correct_answer if sample_is_correct else expected_hallucination
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    generated_ids = token_ids[prompt_len:].tolist()
    positions = []

    # Exact token sequence match (first occurrence)
    for start in range(len(generated_ids) - len(target_ids) + 1):
        if generated_ids[start:start + len(target_ids)] == target_ids:
            positions = [prompt_len + start + offset for offset in range(len(target_ids))]
            break

    # Fallback: individual token text match
    if not positions:
        target_lower = target.lower()
        for i, tid in enumerate(generated_ids):
            tok_text = tokenizer.decode([tid]).strip().lower()
            if tok_text and tok_text in target_lower:
                positions.append(prompt_len + i)

    return positions
