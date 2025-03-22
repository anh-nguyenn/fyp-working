#!/usr/bin/env python3

import os
import json
import csv


#############################################
# Count Calculation Function (Item-Level)
#############################################
def calculate_item_metrics(results):
    """
    Computes TP, FP, FN based on `answer` vs `deepseek-answer`.
    Handles both normal list-based answers and Yes/No (boolean) questions.
    """
    overall_counts = {"TP": 0, "FN": 0, "FP": 0}
    type_counts = {}

    total_points = len(results)
    for idx, data_point in enumerate(results, start=1):
        print(f"Processing data point: {idx} of {total_points}")

        qtype = data_point.get("question_type", "unknown")
        correct_answers = data_point.get("answer", [])

        # Ensure correct_answers is a set
        if not isinstance(correct_answers, list):
            print(f"Warning: answer is not a list for data point {idx}, skipping...")
            correct_answers = set()
        else:
            correct_answers = set(correct_answers)

        # Handle Yes/No Questions (ASK Queries)
        if isinstance(data_point.get("deepseek-answer"), bool):
            deepseek_answer = {"yes"} if data_point["deepseek-answer"] else {"no"}
            correct_answers = {"yes"} if "YES" in correct_answers else {"no"}
        else:
            deepseek_answer = set(data_point.get("deepseek-answer", []))

        if qtype not in type_counts:
            type_counts[qtype] = {"TP": 0, "FN": 0, "FP": 0}

        # Debug output
        # print("\n==== DEBUG INFO ====")
        # print(f"Question Type: {qtype}")
        # print(f"Correct Answer List: {list(correct_answers)}")
        # print(f"DeepSeek Answer List: {list(deepseek_answer)}")
        # print("====================\n")

        # Compute True Positives, False Negatives, and False Positives
        tp_items = len(correct_answers & deepseek_answer)
        fn_items = len(correct_answers - deepseek_answer)
        fp_items = len(deepseek_answer - correct_answers)

        # Update overall and per-question-type counts
        overall_counts["TP"] += tp_items
        overall_counts["FN"] += fn_items
        overall_counts["FP"] += fp_items

        type_counts[qtype]["TP"] += tp_items
        type_counts[qtype]["FN"] += fn_items
        type_counts[qtype]["FP"] += fp_items

    return overall_counts, type_counts


#############################################
# Metrics Calculation Function
#############################################
def compute_metrics(counts):
    """
    Compute precision, recall, and F1-score based on TP, FP, FN.
    """
    tp, fn, fp = counts["TP"], counts["FN"], counts["FP"]

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "TP": tp,
        "FN": fn,
        "FP": fp,
    }


#############################################
# Main Function
#############################################
def main():
    # Load the dataset from `deepseek-result.json`
    RESULTS_PATH = os.path.join(
        os.getcwd(), "data", "deepseek-result", "deepseek-result.json"
    )
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            test_results = json.load(f)
            # test_results = all_results[:2]  # Take first 2 data points
    else:
        raise FileNotFoundError(f"Results file {RESULTS_PATH} not found.")

    print(f"Loaded {len(test_results)} data points for evaluation.")

    # Calculate raw counts
    overall_counts, type_counts = calculate_item_metrics(test_results)

    # Compute overall and per-question-type metrics
    overall_metrics = compute_metrics(overall_counts)
    per_type_metrics = {
        qtype: compute_metrics(counts) for qtype, counts in type_counts.items()
    }

    # Write to CSV
    csv_file = os.path.join(os.getcwd(), "deepseek_metrics.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Question Type", "Metric", "Value"])

        # Per Question Type Analysis
        for qtype, metrics in per_type_metrics.items():
            for metric, value in metrics.items():
                writer.writerow(
                    ["Per Question Type Analysis", qtype, metric, f"{value:.4f}"]
                )

        # Overall Analysis
        for metric, value in overall_metrics.items():
            writer.writerow(["Overall Analysis", "Overall", metric, f"{value:.4f}"])

    print(f"Item-Level Analysis exported to {csv_file}")


if __name__ == "__main__":
    main()
