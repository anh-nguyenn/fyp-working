import os
import requests

from openai import OpenAI
from typing import Callable, Optional

from constants import AVAILABLE_MODELS
from graph import format_row_result, query_graph_db

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


def generate_sparql(question: str, model_name: str):
    messages = [
        {
            "role": "system",
            "content": "You are a useful SparQL assistant. You are tasked to review a question and generate a SparQL to answer the question.\nSparQL Database used is WikiData. [<text>] is topic entity in the question.\nOnly use these two prefixes PREFIX wd: <https://www.wikidata.org/entity/> and PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> if needed.\nDo not use wdt syntax to query WikiData",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    response = requests.post(
        f"{os.getenv('BASE_API_URL')}/query",
        json={"messages": messages, "model_name": model_name},
    )
    response.raise_for_status()
    return response.json()["response"]


def get_natural_response_for_finetunel_model(question: str, queried_results: str):
    messages = [
        {
            "role": "system",
            "content": f"""You are a movie query engine. Answer movie-related questions using only the provided SPARQL queried results.
Do not infer, assume, or add any extra details. Respond strictly with what is asked.
If the queried data is empty, respond naturally in a way that directly addresses the question without providing any additional information.
In your response, do not reveal the results are from the queried data.
SparQL Queried Results for question {question}:
{queried_results}
""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    openai_response = client.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
        temperature=0.1,
        top_p=0.1,
    )

    return openai_response.choices[0].message.content


def get_natural_response_for_base_model(question: str):
    messages = [
        {
            "role": "system",
            "content": f"""You are a movie query engine. Only answer movie-related questions strictly based on what is asked, without adding any extra information.""",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    openai_response = client.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
        temperature=0.1,
        top_p=0.1,
    )

    return openai_response.choices[0].message.content


def get_finetuned_model_response(
    question,
    model_name: str,
    on_generating_query_start: Optional[Callable[[None], None]] = None,
    on_query_generated: Optional[Callable[[str], None]] = None,
    on_end: Optional[Callable[[str], None]] = None,
):
    if model_name not in AVAILABLE_MODELS.values():
        raise ValueError(
            f"Invalid model name. Please only these model name instead. {', '.join(AVAILABLE_MODELS.values())}"
        )

    print("get_finetuned_model_response: Step 1. Generating SparQL query")
    on_generating_query_start and on_generating_query_start()
    generated_sparql_query = generate_sparql(question=question, model_name=model_name)
    print(
        f"get_finetuned_model_response: Generated SparQL query: {generated_sparql_query}"
    )
    on_query_generated and on_query_generated(generated_sparql_query)
    print(
        "get_finetuned_model_response: Step 2. Quering GraphDB with generated SpaQL query"
    )
    queried_results = query_graph_db(query=generated_sparql_query)
    print(
        f"get_finetuned_model_response: Unformatted queried results: {queried_results}"
    )
    formatted_queried_results = "\n\n".join(
        [str(format_row_result(row_result)) for row_result in queried_results]
    )
    print(
        f"get_finetuned_model_response: Formatted queried results: {formatted_queried_results}"
    )

    print("get_finetuned_model_response: Step 3. Getting Natural Response")
    final_response = get_natural_response_for_finetunel_model(
        question=question, queried_results=formatted_queried_results
    )
    print(f"get_finetuned_model_response: Final response: {final_response}")
    on_end and on_end(final_response)
    return final_response


def get_base_model_response(question, on_end: Optional[Callable[[str], None]] = None):
    final_response = get_natural_response_for_base_model(question=question)
    on_end and on_end(final_response)
    return final_response
