import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def parse_deepseek_response(content):
    # Remove markdown code blocks
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content)
    content = content.strip()

    # Try to parse the content as JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        else:
            # If it's not a list, wrap it in a list
            return [parsed]
    except json.JSONDecodeError:
        # Attempt to extract JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]
            except json.JSONDecodeError:
                pass
    # Return empty list if parsing fails
    return []


def get_deepseek_answer(client, question):
    # Define the role instructions for DeepSeek
    deepseek_role = """You are a movie query engine. Answer movie questions.
Your response must be a valid JSON array [] with no extra text or formatting
For yes/no questions, return a list containing a single boolean value (e.g., [True] or [False]). 
For all other queries, return a list of answer strings ["item_1", "item_2"]. 
Ensure you consider all cases as there may be many valid answers. """

    messages = [
        {"role": "system", "content": deepseek_role},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        response_format={"type": "json_object"},
    )
    response_content = response.choices[0].message.content
    # print(f"Raw response: {response_content}")
    parsed_answer = parse_deepseek_response(response_content)
    return parsed_answer


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found. Please check your .env file.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Load questions from the input JSON file
    with open("./data/qa_test_4124.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Take only the first 10 data points
    data = data[:2]

    output_data = []
    for idx, entry in enumerate(data):
        question = entry.get("question")
        question_type = entry.get("question_type")
        answer = entry.get("answer")
        sparql = entry.get("sparql")
        deepseek_ans = get_deepseek_answer(client, question)

        # Print deepseek answer with the question number
        print(f"Question {idx + 1}: {deepseek_ans}")

        # Construct a new dictionary with the desired keys
        new_entry = {
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "sparql": sparql,
            "deepseek-answer": deepseek_ans,
        }
        output_data.append(new_entry)

    # Write the new data to an output JSON file
    with open(
        "./data/qa_test_4124_deepseek_output.json", "w", encoding="utf-8"
    ) as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
