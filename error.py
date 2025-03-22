#!/usr/bin/env python3
"""
Before running this script, make sure you have installed the required packages:
    pip install langchain-community langchain-core rdflib
"""

import os
import json
import rdflib
import signal
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
# Export Wrong Item-Level Queries for a Specific Question Type
#############################################
def export_wrong_item_level_director():
    # Configuration for the Ontotext GraphDB instance
    config = {
        "query_endpoint": "http://localhost:7200/repositories/imkg",
        "schema": "/Users/jerry/Desktop/FYP-working/fine-tune-openai-KGQA/KG/schema.txt",
    }

    # Connect to the GraphDB instance.
    global graph  # make graph accessible in this module
    graph = CustomOntotextGraphDBGraph(**config)

    # Define the path to the results file.
    RESULTS_PATH = os.path.join(
        os.getcwd(), "data", "llama-result", "llama-result-error.json"
    )
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results file {RESULTS_PATH} not found.")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} data points from {RESULTS_PATH}")

    wrong_data_points = []
    total_points = len(results)
    for idx, data_point in enumerate(results, start=1):
        qtype = data_point.get("question_type", "unknown")
        # Only process the designated question type.
        if qtype != "actor_to_movie_constraint_year":
            continue

        # Use either "sparql_response" or "generated_sparql" for the generated query.
        sparql_response = data_point.get("sparql_response") or data_point.get(
            "generated_sparql"
        )
        sample_query = data_point.get("sparql")

        # Proceed only if both queries are available.
        if sparql_response and sample_query:
            try:
                generated_results = set(graph.safe_query(sparql_response))
                sample_results = set(graph.safe_query(sample_query))
                # If the item-level sets do not match, export this data point.
                if generated_results != sample_results:
                    wrong_data_points.append(data_point)
            except Exception as e:
                # If an exception occurs (e.g. timeout), log and export the data point.
                print(f"Error processing data point {idx}: {e}")
                wrong_data_points.append(data_point)
        else:
            # If one or both queries are missing, consider this as a wrong query.
            wrong_data_points.append(data_point)

    output_file = os.path.join(
        os.getcwd(), "wrong_item_actor_to_movie_constraint_year.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(wrong_data_points, f, indent=2)
    print(
        f"Exported {len(wrong_data_points)} data points with item-level mismatches for 'actor_to_movie_constraint_year' to {output_file}"
    )


if __name__ == "__main__":
    export_wrong_item_level_director()
