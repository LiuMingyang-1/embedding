# main.py
# 入口脚本，通过 --stage 参数控制运行阶段：
#   extract: 云上 GPU 阶段 — 推理 + 提取 hidden states/attentions，保存到 states/
#            同时导出 results_all.csv，等待人工填写 manual_has_hallucination
#   analyze: 本地 CPU 阶段 — 读取人工标注后的 CSV，计算指标，输出统计 + 4 张图
#
# 使用方法：
#   python3 main.py --stage extract   # 云上跑
#   python3 main.py --stage analyze   # 本地跑

import argparse

from config import RESULTS_CSV


def run_analysis():
    import pandas as pd
    from analysis.compute import load_all_records, aggregate_sample_records, build_manual_label_lookup
    from analysis.statistics import run_statistics
    from visualization.single_example import plot_single_example
    from visualization.group_average import plot_group_average
    from visualization.late_slope import plot_late_slope_distribution
    from visualization.correlation import plot_correlation_matrix

    results_df = pd.read_csv(RESULTS_CSV)
    label_lookup = build_manual_label_lookup(results_df)
    token_records = load_all_records(label_lookup)

    if not token_records:
        print("No generated response tokens identified. Cannot proceed.")
        return

    sample_records = aggregate_sample_records(token_records)
    n_non_hallucinated = sum(1 for r in sample_records if not r["has_hallucination"])
    n_hallucinated = sum(1 for r in sample_records if r["has_hallucination"])
    avg_response_len = sum(r["num_response_tokens"] for r in sample_records) / len(sample_records)
    n_multi_token = sum(1 for r in sample_records if r["num_response_tokens"] > 1)
    n_content_tokens = sum(1 for r in sample_records if r["response_token_source"] == "generated_content_tokens")
    print(
        f"\nTotal generated response token records: {len(token_records)}"
        f"\nTotal sample records: {len(sample_records)}"
        f"  (non_hallucination={n_non_hallucinated}, hallucination={n_hallucinated})"
        f"\nAverage generated response length: {avg_response_len:.2f} token(s)"
        f"\nMulti-token responses: {n_multi_token}"
        f"\nContent-token extraction used in {n_content_tokens} sample(s)"
    )

    run_statistics(sample_records)
    plot_single_example(sample_records)
    plot_group_average(sample_records)
    plot_late_slope_distribution(sample_records)
    plot_correlation_matrix(sample_records)

    print("\nAnalysis complete. Figures saved to figures/")


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Demo")
    parser.add_argument(
        "--stage",
        choices=["extract", "analyze"],
        required=True,
        help="extract: run model inference and export a CSV for manual hallucination labeling (GPU required); "
             "analyze: compute metrics & plots from manually labeled CSV (CPU only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only used with --stage extract. Process the first N dataset rows, e.g. --limit 20",
    )
    args = parser.parse_args()

    if args.stage == "extract":
        from extraction.states import run_extraction

        run_extraction(limit=args.limit)
        return

    if args.stage == "analyze":
        run_analysis()


if __name__ == "__main__":
    main()
