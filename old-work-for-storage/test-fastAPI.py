import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "messages": [
            {
                "role": "system",
                "content": "You are a useful SparQL assistant. You are tasked to review a question and generate a SparQL to answer the question.\nSparQL Database used is WikiData. [<text>] is topic entity in the question.\nOnly use these two prefixes PREFIX wd: <https://www.wikidata.org/entity/> and PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> if needed.\nDo not use wdt syntax to query WikiData",
            },
            {
                "role": "user",
                "content": "What [peter jackson] movies were written by [Peter Jackson]?",
            },
        ],
        "model_name": "Llama3.2 3B",
    },
)

response.raise_for_status()
print(response.json())
