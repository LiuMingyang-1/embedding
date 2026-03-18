# extraction/correctness.py
# 负责在模型真实生成部分中定位 token 位置。
# 当前主分析对象是整段生成内容中的所有内容 token；
# 同时保留答案 span 的启发式，供后续更细粒度分析使用。

import re


def _normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_words(text):
    norm = _normalize(text)
    return norm.split() if norm else []


def _decode_generated_tokens(tokenizer, generated_ids):
    return [tokenizer.decode([tid], skip_special_tokens=True) for tid in generated_ids]


def _is_content_token(token_text):
    return bool(_normalize_words(token_text))


def _has_strong_boundary(token_texts):
    boundary_chars = ".!?;\n"
    for text in token_texts:
        if any(ch in text for ch in boundary_chars):
            return True
    return False


def _get_generated_token_texts(tokenizer, token_ids, prompt_len):
    generated_ids = token_ids[prompt_len:]
    if hasattr(generated_ids, "tolist"):
        generated_ids = generated_ids.tolist()

    token_texts = _decode_generated_tokens(tokenizer, generated_ids)
    return generated_ids, token_texts


def _get_content_positions(token_texts):
    return [i for i, text in enumerate(token_texts) if _is_content_token(text)]


def select_response_tokens(tokenizer, token_ids, prompt_len):
    generated_ids, token_texts = _get_generated_token_texts(tokenizer, token_ids, prompt_len)
    content_positions = _get_content_positions(token_texts)

    if not content_positions:
        return {
            "positions": [],
            "first_position": None,
            "source": "generated_content_tokens",
            "text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        }

    absolute_positions = [prompt_len + pos for pos in content_positions]
    return {
        "positions": absolute_positions,
        "first_position": absolute_positions[0],
        "source": "generated_content_tokens",
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
    }


def _strip_leading_wrapper(tokenizer, generated_ids, positions):
    if not positions:
        return positions

    wrapper_phrases = {
        "the answer is",
        "answer is",
        "it is",
        "it s",
        "this is",
        "the capital is",
        "capital is",
    }

    for k in range(min(4, len(positions)), 0, -1):
        prefix_ids = [generated_ids[pos] for pos in positions[:k]]
        prefix_norm = _normalize(tokenizer.decode(prefix_ids, skip_special_tokens=True))
        if prefix_norm in wrapper_phrases:
            return positions[k:]
    return positions


def _strip_trailing_contrast(positions, token_texts):
    if not positions:
        return positions

    contrast_markers = {"not", "but", "instead", "rather", "however", "although", "except"}
    for idx, pos in enumerate(positions[1:], start=1):
        words = _normalize_words(token_texts[pos])
        if words and any(word in contrast_markers for word in words):
            return positions[:idx]
    return positions


def select_answer_tokens(tokenizer, token_ids, prompt_len):
    generated_ids, token_texts = _get_generated_token_texts(tokenizer, token_ids, prompt_len)
    content_positions = _get_content_positions(token_texts)

    if not content_positions:
        return {
            "positions": [],
            "first_position": None,
            "source": "no_content",
            "text": "",
        }

    segments = []
    current = [content_positions[0]]
    for pos in content_positions[1:]:
        boundary_region = [token_texts[current[-1]], *token_texts[current[-1] + 1:pos]]
        if _has_strong_boundary(boundary_region):
            segments.append(current)
            current = [pos]
        else:
            current.append(pos)
    segments.append(current)

    positions = segments[0]
    positions = _strip_leading_wrapper(tokenizer, generated_ids, positions)
    positions = _strip_trailing_contrast(positions, token_texts)
    if not positions:
        positions = segments[0]

    span_token_ids = [generated_ids[pos] for pos in positions]
    span_text = tokenizer.decode(span_token_ids, skip_special_tokens=True).strip()
    absolute_positions = [prompt_len + pos for pos in positions]
    return {
        "positions": absolute_positions,
        "first_position": absolute_positions[0],
        "source": "generated_first_segment",
        "text": span_text,
    }
