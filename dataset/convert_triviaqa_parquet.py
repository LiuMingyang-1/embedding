import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"<unk>", "nan", "None"} else text


def _extract_reference_answer(answer: Any) -> str:
    if isinstance(answer, str):
        text = answer.strip()
        if not text:
            return ""
        if text.startswith("{") and text.endswith("}"):
            try:
                answer = json.loads(text)
            except json.JSONDecodeError:
                return _clean_text(text)
        else:
            return _clean_text(text)

    if not isinstance(answer, dict):
        return ""

    preferred_keys = ["value", "normalized_value", "matched_wiki_entity_name"]
    for key in preferred_keys:
        candidate = _clean_text(answer.get(key))
        if candidate:
            return candidate

    aliases = answer.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            candidate = _clean_text(alias)
            if candidate:
                return candidate

    normalized_aliases = answer.get("normalized_aliases")
    if isinstance(normalized_aliases, list):
        for alias in normalized_aliases:
            candidate = _clean_text(alias)
            if candidate:
                return candidate

    return ""


def convert_triviaqa_parquet(input_path: Path, output_path: Path, start_id: int = 1) -> pd.DataFrame:
    source_df = pd.read_parquet(input_path)

    if "question" not in source_df.columns:
        raise ValueError("Input parquet must contain a 'question' column.")
    if "answer" not in source_df.columns:
        raise ValueError(
            "Input parquet must contain an 'answer' column so the output matches "
            "dataset_sample.csv with: id,prompt_text,reference_answer."
        )

    records = []
    next_id = start_id

    for _, row in source_df.iterrows():
        prompt_text = str(row["question"]).strip()
        if not prompt_text:
            continue

        reference_answer = _extract_reference_answer(row.get("answer", None))
        records.append(
            {
                "id": next_id,
                "prompt_text": prompt_text,
                "reference_answer": reference_answer,
            }
        )
        next_id += 1

    output_df = pd.DataFrame(records, columns=["id", "prompt_text", "reference_answer"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TriviaQA parquet to dataset.csv format: id,prompt_text,reference_answer"
    )
    parser.add_argument("--input", required=True, help="Path to TriviaQA parquet file")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV path")
    parser.add_argument("--start-id", type=int, default=1, help="Starting id value (default: 1)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = convert_triviaqa_parquet(input_path=input_path, output_path=output_path, start_id=args.start_id)
    empty_answers = int((df["reference_answer"].astype(str).str.strip() == "").sum())

    print(f"Converted rows: {len(df)}")
    print(f"Output: {output_path}")
    print("Output columns: id,prompt_text,reference_answer")
    print(f"Rows with empty reference_answer: {empty_answers}")


if __name__ == "__main__":
    main()
