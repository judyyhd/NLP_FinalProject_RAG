"""
Convert HotpotQA to Self-RAG format.
Reads from original HotpotQA files and outputs in Self-RAG eval_data format.
"""
import json
import argparse
from tqdm import tqdm

def convert_hotpotqa(input_file, output_file):
    """
    Convert HotpotQA format to Self-RAG format.
    
    HotpotQA format:
    {
        "_id": "...",
        "question": "...",
        "answer": "...",
        "context": [["title", ["sent1", "sent2", ...]], ...]
    }
    
    Self-RAG format:
    {
        "id": "...",
        "question": "...",
        "answers": ["..."],
        "ctxs": [{"title": "...", "text": "..."}, ...]
    }
    """
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        hotpot_data = json.load(f)
    
    print(f"Converting {len(hotpot_data)} examples...")
    
    converted = []
    for item in tqdm(hotpot_data):
        # Extract contexts
        ctxs = []
        for title, sentences in item.get('context', []):
            ctxs.append({
                'title': title,
                'text': ' '.join(sentences)  # Join sentences into paragraph
            })
        
        # Create Self-RAG format
        converted_item = {
            'id': item['_id'],
            'question': item['question'],
            'answers': [item['answer']],  # Self-RAG expects list of answers
            'ctxs': ctxs  # Pre-retrieved contexts (from distractor setting)
        }
        
        converted.append(converted_item)
    
    print(f"Writing to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(converted)} examples")
    print(f"   Output: {output_file}")
    
    # Show sample
    print("\nSample converted entry:")
    print(json.dumps(converted[0], indent=2)[:500] + "...")

def main():
    parser = argparse.ArgumentParser(description='Convert HotpotQA to Self-RAG format')
    parser.add_argument('--input', required=True, help='Input HotpotQA JSON file')
    parser.add_argument('--output', required=True, help='Output Self-RAG format JSON file')
    
    args = parser.parse_args()
    convert_hotpotqa(args.input, args.output)

if __name__ == "__main__":
    main()
