# main.py
# 入口脚本，通过 --stage 参数控制运行阶段：
#   extract: 云上 GPU 阶段 — 推理 + 提取 hidden states/attentions，保存到 states/
#   analyze: 本地 CPU 阶段 — 加载状态，计算指标，输出统计 + 4 张图
#   all:     两个阶段连续运行
#
# 使用方法：
#   python main.py --stage extract   # 云上跑
#   python main.py --stage analyze   # 本地跑
#   python main.py --stage all

import argparse

from config import MODEL_NAME, RESULTS_CSV
from extraction.states import run_extraction
from analysis.compute import load_all_records
from analysis.statistics import run_statistics
from visualization.single_example import plot_single_example
from visualization.group_average import plot_group_average
from visualization.late_slope import plot_late_slope_distribution
from visualization.correlation import plot_correlation_matrix


def run_analysis():
    import pandas as pd
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    results_df = pd.read_csv(RESULTS_CSV)
    all_records = load_all_records(tokenizer)

    if not all_records:
        print("No answer tokens identified. Cannot proceed.")
        return

    n_correct = sum(1 for r in all_records if r["is_correct"])
    n_wrong = sum(1 for r in all_records if not r["is_correct"])
    print(f"\nTotal answer token records: {len(all_records)}  "
          f"(correct={n_correct}, wrong={n_wrong})")

    run_statistics(all_records)
    plot_single_example(all_records, results_df)
    plot_group_average(all_records)
    plot_late_slope_distribution(all_records)
    plot_correlation_matrix(all_records)

    print("\nAnalysis complete. Figures saved to figures/")


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Demo")
    parser.add_argument(
        "--stage",
        choices=["extract", "analyze", "all"],
        default="all",
        help="extract: run model inference (GPU required); "
             "analyze: compute metrics & plots (CPU only)",
    )
    args = parser.parse_args()

    if args.stage in ("extract", "all"):
        run_extraction()

    if args.stage in ("analyze", "all"):
        run_analysis()


if __name__ == "__main__":
    main()
