import os
import rdflib

from typing import List
from rdflib.query import ResultRow
from rdflib.term import URIRef, BNode, Literal, Identifier
from langchain_community.graphs import OntotextGraphDBGraph


class CustomOntotextGraphDBGraph(OntotextGraphDBGraph):
    def __init__(self, query_endpoint: str, schema: str) -> None:
        try:
            import rdflib
            from rdflib.plugins.stores import sparqlstore
        except ImportError:
            raise ImportError(
                "Could not import rdflib python package. "
                "Please install it with pip install rdflib."
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

    def query(
        self,
        query: str,
    ) -> List[rdflib.query.ResultRow]:
        """
        Query the graph.
        """
        from rdflib.query import ResultRow

        res = self.graph.query(query)
        if res.type == "ASK":
            return [r for r in res if isinstance(r, bool)]
        return [r for r in res if isinstance(r, ResultRow)]


config = {
    "query_endpoint": f"{os.getenv('GRAPH_DB_HOST')}/repositories/imkg",
    "schema": os.getenv("GRAPH_SCHEMA_PATH"),
}
graph = CustomOntotextGraphDBGraph(**config)


def format_node(node: Identifier):
    if isinstance(node, URIRef):
        return f"Resource ({node.toPython()})"
    elif isinstance(node, BNode):
        return "Unnamed Entity (Blank Node)"
    elif isinstance(node, Literal):
        return f"Value: {node.toPython()}"
    else:
        return f"Variable ({node.toPython()})"


def format_row_result(row_result):
    if isinstance(row_result, ResultRow):
        return " - ".join([format_node(item) for item in row_result])
    return row_result


def query_graph_db(query: str):
    results = graph.query(query=query)
    return results
