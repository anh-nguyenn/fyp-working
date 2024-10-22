SPARQL_GENERATION_PROMPT = """
You are a useful SparQL assistant. You are tasked to review a question and generate a SparQL to answer the question.
SparQL Database used is WikiData. [<text>] is topic entity in the question.
Only use these two prefixes PREFIX wd: <https://www.wikidata.org/entity/> and PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> if needed.
Do not use wdt syntax to query WikiData
"""

SPARQL_FIX_PROMPT = """You are a useful SparQL assistant. You are tasked to fix a SparQL quey that was generated to answer a question
Give me a correct version of the SPARQL query based on the previous query, the question and the error message"""

QA_PROMPT = """Generate a natural language response from the results of a SPARQL query.
Don't use any internal knowledge to answer the question,
Just say you don't know if no information is available from The results of a SPARQL query.
If the question is Yes/No question and there is no information available, answer No.
SparQL query: {{query}}
The results of a SparQL query: {{context}}"""

TARGET_EPOCHS = 3
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
NORETRY = 3

ALLOWED_MODELS = {
    "gpt-4o-2024-08-06": 65536,
    "gpt-4o-mini-2024-07-18": 65536,
    "gpt-4-0613": 65536,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0613": 4096,
}

TEST_DATA_PATH = "data/qa_test.json"
PROCESSED_TRAIN_DATA_PATH = "processed_train_data.jsonl"
TRAINED_MODEL_ID_PATH = "trained_model_id.txt"
