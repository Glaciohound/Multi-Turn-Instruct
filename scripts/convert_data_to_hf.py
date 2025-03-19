import json
import argparse


def convert_format(input_file, output_file):
    """Convert nested dataset format to a list-based format."""
    # Load the original JSON dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to the new format
    converted_data = []
    for category, conversations in data.items():
        for conversation_id, turns in conversations.items():
            converted_data.append({
                "category": category,
                "conversation_id": conversation_id,
                "turns": turns
            })

    # Save the converted dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Conversion completed! Saved to {output_file}")


# Command-line argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset format")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    convert_format(args.input, args.output)
