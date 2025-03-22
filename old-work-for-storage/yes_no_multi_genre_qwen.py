#!/usr/bin/env python3
"""
This script processes only questions of type "yesno_actor_genre" from a JSON results file.
It executes the generated and sample SPARQL queries using a GraphDB connection,
collects the query details and results, and writes the output to a file.
Before running, ensure you have installed:
    pip install langchain-community langchain-core rdflib
"""

import os
import json
import rdflib
import signal
from functools import wraps
from typing import List
from langchain_community.graphs import OntotextGraphDBGraph


"""
error at:  {
    "question": "Are there multiple genres that apply to [Head Full of Honey]?",
    "question_type": "yesno_movie_multi_genres",
    "sparql_response": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT DISTINCT ?label ?uri WHERE { ?e1 ?rel ?uri; rdfs:label \"Head Full of Honey\" . ?e3 ?rel ?uri; rdfs:label ?label . FILTER (?label != \"Head Full of Honey\") ?rel rdfs:label \"has_genre\" . }",
    "sparql": "PREFIX wd: <https://www.wikidata.org/entity/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ASK { ?movie wd:P136 ?e1 ; wd:P136 ?e2; rdfs:label \"Head Full of Honey\". FILTER (?e1 != ?e2) }"
  },

"""


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
            raise ImportError("Please install rdflib with `pip install rdflib`.")
        auth = self._get_auth()
        store = sparqlstore.SPARQLStore(auth=auth)
        store.open(query_endpoint)
        self.graph = rdflib.Graph(store, identifier=None, bind_namespaces="none")
        self._check_connectivity()
        if not os.path.exists(schema):
            raise FileNotFoundError(f"Schema file {schema} does not exist.")
        with open(schema, "r") as file:
            self.schema = file.readlines()

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
# Main Function
#############################################
def main():
    # Configuration for the Ontotext GraphDB instance.
    config = {
        "query_endpoint": "http://localhost:7200/repositories/imkg",
        "schema": "/Users/jerry/Desktop/FYP-working/fine-tune-openai-KGQA/KG/schema.txt",  # Update to your actual schema file path.
    }
    graph = CustomOntotextGraphDBGraph(**config)

    # Define the path to the JSON results file.
    results_path = os.path.join(os.getcwd(), "data", "qwen-result.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file {results_path} not found.")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    output_lines = []
    output_lines.append("yesno_movie_multi_genres Query Details")
    output_lines.append("======================================\n")

    # Process only data points with question_type "yesno_actor_genre"
    for idx, data_point in enumerate(results, start=1):
        if data_point.get("question_type", "unknown") == "yesno_movie_multi_genres":
            output_lines.append(f"--- yesno_movie_multi_genres Question {idx} ---")

            sparql_response = data_point.get("sparql_response") or data_point.get(
                "generated_sparql"
            )
            sample_query = data_point.get("sparql")

            output_lines.append("\nGenerated SPARQL Query:")
            output_lines.append(str(sparql_response))

            try:
                generated_results = graph.safe_query(sparql_response)
                output_lines.append("Generated Query Results:")
                output_lines.append(str(generated_results))
            except Exception as e:
                output_lines.append(f"Error executing generated query: {e}")

            output_lines.append("\nSample SPARQL Query:")
            output_lines.append(str(sample_query))
            try:
                sample_results = graph.safe_query(sample_query)
                output_lines.append("Sample Query Results:")
                output_lines.append(str(sample_results))
            except Exception as e:
                output_lines.append(f"Error executing sample query: {e}")
            output_lines.append("=======================================\n")

    # Write the collected output to a file.
    output_file = os.path.join(os.getcwd(), "yesno_movie_multi_genres.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
