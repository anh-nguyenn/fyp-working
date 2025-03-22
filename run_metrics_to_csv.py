#!/usr/bin/env python3
"""
Before running this script, make sure you have installed the required packages:
    pip install langchain-community langchain-core rdflib
"""

import os
import json
import rdflib
import signal
import csv
from functools import wraps
from typing import List
from langchain_community.graphs import OntotextGraphDBGraph


#############################################
# Timeout Decorator (Unix-like systems only)
#############################################
def time_limit(seconds):
    """
    Decorator to limit the execution time of a function.
    Raises TimeoutError if time exceeds 'seconds'.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Query timed out after {seconds} seconds")

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return wrapper

    return decorator


#############################################
# Custom Ontotext GraphDB Graph Class
#############################################
class CustomOntotextGraphDBGraph(OntotextGraphDBGraph):
    def __init__(self, query_endpoint: str, schema: str) -> None:
        try:
            import rdflib
            from rdflib.plugins.stores import sparqlstore
        except ImportError:
            raise ImportError(
                "Could not import rdflib python package. "
                "Please install it with `pip install rdflib`."
            )
        auth = self._get_auth()
        store = sparqlstore.SPARQLStore(auth=auth)
        store.open(query_endpoint)
        self.graph = rdflib.Graph(store, identifier=None, bind_namespaces="none")
        self._check_connectivity()
        if not os.path.exists(schema):
            raise FileNotFoundError(f"File {schema} does not exist.")
        with open(schema, "r") as file:
            schema_string = file.readlines()
        self.schema = schema_string

    @time_limit(30)  # Set a 30-second timeout for queries
    def safe_query(self, query: str) -> List[rdflib.query.ResultRow]:
        """
        Query the graph with a time limit.
        """
        from rdflib.query import ResultRow

        res = self.graph.query(query)
        if res.type == "ASK":
            return [r for r in res if isinstance(r, bool)]
        return [r for r in res if isinstance(r, ResultRow)]


#############################################
# Count Calculation Function
#############################################
def calculate_counts(results):
    """
    Process the results list to accumulate raw counts for query-level and item-level metrics.
    Returns:
      - overall_query: dict with overall query-level counts
      - overall_item: dict with overall item-level counts
      - type_counts: dict mapping each question_type to its query/item counts
    """
    overall_query = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    overall_item = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    type_counts = {}  # {question_type: {"query_level": counts, "item_level": counts}}

    total_points = len(results)
    for idx, data_point in enumerate(results, start=1):
        print(f"Processing data point: {idx} of {total_points}")
        sparql_response = data_point.get("sparql_response") or data_point.get(
            "generated_sparql"
        )
        sample_query = data_point.get("sparql")
        qtype = data_point.get("question_type", "unknown")

        if qtype not in type_counts:
            type_counts[qtype] = {
                "query_level": {"TP": 0, "FN": 0, "FP": 0, "TN": 0},
                "item_level": {"TP": 0, "FN": 0, "FP": 0, "TN": 0},
            }

        if sparql_response and sample_query:
            try:
                generated_results = set(graph.safe_query(sparql_response))
                sample_results = set(graph.safe_query(sample_query))
                # Query-level evaluation: either the entire query matches or not.
                if generated_results == sample_results:
                    overall_query["TP"] += 1
                    type_counts[qtype]["query_level"]["TP"] += 1
                else:
                    overall_query["FN"] += 1
                    type_counts[qtype]["query_level"]["FN"] += 1
            except Exception as e:
                overall_query["FN"] += 1
                type_counts[qtype]["query_level"]["FN"] += 1

            # Item-level evaluation using set operations.
            tp_items = len(generated_results & sample_results)
            fn_items = len(sample_results - generated_results)
            fp_items = len(generated_results - sample_results)

            overall_item["TP"] += tp_items
            overall_item["FN"] += fn_items
            overall_item["FP"] += fp_items
            type_counts[qtype]["item_level"]["TP"] += tp_items
            type_counts[qtype]["item_level"]["FN"] += fn_items
            type_counts[qtype]["item_level"]["FP"] += fp_items

    return overall_query, overall_item, type_counts


#############################################
# Metrics Calculation Function
#############################################
def compute_metrics(counts):
    """
    Compute a variety of metrics from raw counts.
    Returns a dictionary containing:
      - support: total number of samples
      - accuracy: (TP+TN)/support
      - error_rate: 1 - accuracy
      - precision: TP/(TP+FP)
      - recall: TP/(TP+FN)
      - f1_score: harmonic mean of precision and recall
      - false_negative_rate: FN/(TP+FN)
      - false_positive_rate: FP/(TP+FP)
    """
    support = counts["TP"] + counts["FN"] + counts["FP"] + counts["TN"]
    accuracy = ((counts["TP"] + counts["TN"]) / support) if support else 0
    error_rate = 1 - accuracy if support else 0
    precision = (
        counts["TP"] / (counts["TP"] + counts["FP"])
        if (counts["TP"] + counts["FP"])
        else 0
    )
    recall = (
        counts["TP"] / (counts["TP"] + counts["FN"])
        if (counts["TP"] + counts["FN"])
        else 0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    )
    fnr = (
        counts["FN"] / (counts["TP"] + counts["FN"])
        if (counts["TP"] + counts["FN"])
        else 0
    )
    fpr = (
        counts["FP"] / (counts["TP"] + counts["FP"])
        if (counts["TP"] + counts["FP"])
        else 0
    )
    return {
        "support": support,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }


#############################################
# Main Function
#############################################
def main():
    # Configuration for the Ontotext GraphDB instance
    config = {
        "query_endpoint": "http://localhost:7200/repositories/imkg",
        "schema": "/Users/jerry/Desktop/FYP-working/fine-tune-openai-KGQA/KG/schema.txt",
    }

    global graph  # Make graph accessible to calculate_counts
    graph = CustomOntotextGraphDBGraph(**config)

    # Define the path to the results file (using the smaller dataset in this example)
    RESULTS_PATH = os.path.join(
        os.getcwd(), "data", "llama-result", "llama-result-error.json"
    )
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Processing {len(results)} data points from {RESULTS_PATH}")
    else:
        raise FileNotFoundError(f"Results file {RESULTS_PATH} not found.")

    # Calculate raw counts.
    overall_query_counts, overall_item_counts, type_counts = calculate_counts(results)

    # Compute overall metrics.
    overall_query_metrics = compute_metrics(overall_query_counts)
    overall_item_metrics = compute_metrics(overall_item_counts)

    # Compute average item-level counts per valid query (overall).
    valid_queries_overall = (
        overall_query_counts["TP"]
        + overall_query_counts["FN"]
        + overall_query_counts["FP"]
        + overall_query_counts["TN"]
    )
    if valid_queries_overall:
        avg_item_counts_overall = {
            "TP": overall_item_counts["TP"] / valid_queries_overall,
            "FN": overall_item_counts["FN"] / valid_queries_overall,
            "FP": overall_item_counts["FP"] / valid_queries_overall,
            "TN": overall_item_counts["TN"] / valid_queries_overall,
        }
    else:
        avg_item_counts_overall = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}

    # Compute metrics and average item-level counts for each question type.
    per_type_analysis = {}
    for qtype, counts in type_counts.items():
        query_counts = counts["query_level"]
        item_counts = counts["item_level"]
        query_metrics = compute_metrics(query_counts)
        item_metrics = compute_metrics(item_counts)
        valid_queries = (
            query_counts["TP"]
            + query_counts["FN"]
            + query_counts["FP"]
            + query_counts["TN"]
        )
        if valid_queries:
            avg_item = {
                "TP": item_counts["TP"] / valid_queries,
                "FN": item_counts["FN"] / valid_queries,
                "FP": item_counts["FP"] / valid_queries,
                "TN": item_counts["TN"] / valid_queries,
            }
        else:
            avg_item = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        per_type_analysis[qtype] = {
            "query_level_counts": query_counts,
            "query_level_metrics": query_metrics,
            "item_level_counts": item_counts,
            "item_level_metrics": item_metrics,
            "average_item_counts_per_query": avg_item,
        }

    # Write the analysis report to a CSV file.
    csv_file = os.path.join(os.getcwd(), "llama-new-3.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header row.
        writer.writerow(
            ["Section", "Question Type", "Metric Category", "Metric", "Value"]
        )

        # Write Per Question Type Analysis
        for qtype, analysis in per_type_analysis.items():
            # Query-Level Counts
            for metric, value in analysis["query_level_counts"].items():
                writer.writerow(
                    [
                        "Per Question Type Analysis",
                        qtype,
                        "Query-Level Counts",
                        metric,
                        value,
                    ]
                )
            # Query-Level Metrics
            for metric, value in analysis["query_level_metrics"].items():
                writer.writerow(
                    [
                        "Per Question Type Analysis",
                        qtype,
                        "Query-Level Metrics",
                        metric,
                        f"{value:.4f}",
                    ]
                )
            # Item-Level Counts
            for metric, value in analysis["item_level_counts"].items():
                writer.writerow(
                    [
                        "Per Question Type Analysis",
                        qtype,
                        "Item-Level Counts",
                        metric,
                        value,
                    ]
                )
            # Item-Level Metrics
            for metric, value in analysis["item_level_metrics"].items():
                writer.writerow(
                    [
                        "Per Question Type Analysis",
                        qtype,
                        "Item-Level Metrics",
                        metric,
                        f"{value:.4f}",
                    ]
                )
            # Average Item-Level Counts per Query
            for metric, value in analysis["average_item_counts_per_query"].items():
                writer.writerow(
                    [
                        "Per Question Type Analysis",
                        qtype,
                        "Average Item-Level Counts per Query",
                        metric,
                        f"{value:.4f}",
                    ]
                )

        # Write Overall Analysis Section
        # Overall Query-Level Counts
        for metric, value in overall_query_counts.items():
            writer.writerow(
                [
                    "Overall Analysis",
                    "Overall Query-Level Counts",
                    "Counts",
                    metric,
                    value,
                ]
            )
        # Overall Query-Level Metrics
        for metric, value in overall_query_metrics.items():
            writer.writerow(
                [
                    "Overall Analysis",
                    "Overall Query-Level Metrics",
                    "Metrics",
                    metric,
                    f"{value:.4f}",
                ]
            )
        # Overall Item-Level Counts
        for metric, value in overall_item_counts.items():
            writer.writerow(
                [
                    "Overall Analysis",
                    "Overall Item-Level Counts",
                    "Counts",
                    metric,
                    value,
                ]
            )
        # Overall Item-Level Metrics
        for metric, value in overall_item_metrics.items():
            writer.writerow(
                [
                    "Overall Analysis",
                    "Overall Item-Level Metrics",
                    "Metrics",
                    metric,
                    f"{value:.4f}",
                ]
            )
        # Average Item-Level Counts per Query (Overall)
        for metric, value in avg_item_counts_overall.items():
            writer.writerow(
                [
                    "Overall Analysis",
                    "Overall Average Item-Level Counts per Query",
                    "Counts",
                    metric,
                    f"{value:.4f}",
                ]
            )

    print(f"Analysis metrics written to {csv_file}")


if __name__ == "__main__":
    main()
