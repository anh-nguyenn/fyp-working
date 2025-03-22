import json
from collections import defaultdict


def create_small_dataset(input_file: str, output_file: str, n_per_type: int = 100):
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group data points by question_type
    groups = defaultdict(list)
    for item in data:
        # Adjust the key if your data uses a different field name
        qtype = item.get("question_type")
        if qtype is not None:
            groups[qtype].append(item)

    # For each question type, take the first n_per_type data points
    small_data = []
    for qtype, items in groups.items():
        selected_items = items[:n_per_type]
        small_data.extend(selected_items)
        print(f"Selected {len(selected_items)} data points for question type: {qtype}")

    # Write the subset to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(small_data, f, indent=4)
    print(f"Created {output_file} with {len(small_data)} data points.")


if __name__ == "__main__":
    create_small_dataset("llama-result.json", "llama-result-small.json")
