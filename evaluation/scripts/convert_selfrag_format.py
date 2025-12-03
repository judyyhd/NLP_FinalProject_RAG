#!/usr/bin/env python3
"""
Convert Self-RAG results to match the JSONL format expected by advanced_evaluation.py.

Self-RAG file has:
  - preds: list of predictions
  - prompts: list of prompts (contains questions)
  - all_results: detailed retrieval info (not needed for evaluation)
  - utility_score, ground_score, etc.: Self-RAG internal metrics (not needed)

We need JSONL format with:
  - id, question, answers (from vanilla_rag as template)
  - output: the Self-RAG prediction
"""

import json
from pathlib import Path

def main():
    # Paths
    selfrag_file = Path("../results/selfrag_devfull_results.json")
    vanilla_file = Path("../results/vanilla_rag_devfull_results.json")
    output_file = Path("../results/selfrag_llama2_devfull_results.json")
    
    print("Converting Self-RAG results to JSONL format...")
    print(f"  Input: {selfrag_file}")
    print(f"  Template: {vanilla_file}")
    print(f"  Output: {output_file}")
    print()
    
    # Load Self-RAG data
    with open(selfrag_file, 'r') as f:
        selfrag_data = json.load(f)
    
    # Load Vanilla RAG data as template (has id, question, answers)
    with open(vanilla_file, 'r') as f:
        vanilla_data = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(selfrag_data['preds'])} Self-RAG predictions")
    print(f"✓ Loaded {len(vanilla_data)} Vanilla RAG examples (template)")
    print()
    
    # Check alignment
    if len(selfrag_data['preds']) != len(vanilla_data):
        print(f"⚠ Warning: Count mismatch!")
        print(f"  Self-RAG: {len(selfrag_data['preds'])} predictions")
        print(f"  Vanilla: {len(vanilla_data)} examples")
        print(f"  Using first {min(len(selfrag_data['preds']), len(vanilla_data))} examples")
        print()
    
    # Convert to JSONL format
    converted_count = 0
    with open(output_file, 'w') as f:
        for i, (pred, vanilla_item) in enumerate(zip(selfrag_data['preds'], vanilla_data)):
            # Create new item with vanilla template + selfrag prediction
            new_item = {
                'id': vanilla_item.get('id', f'unknown_{i}'),
                'question': vanilla_item.get('question', ''),
                'answers': vanilla_item.get('answers', []),
                'output': pred,  # This is what evaluation script looks for
                # Optional: keep ctxs for reference
                'ctxs': vanilla_item.get('ctxs', [])
            }
            
            f.write(json.dumps(new_item) + '\n')
            converted_count += 1
    
    print(f"✓ Converted {converted_count} examples to JSONL format")
    print(f"✓ Saved to: {output_file}")
    print()
    print("Summary of what was kept:")
    print("  ✓ preds → output (model's answer)")
    print("  ✓ id, question, answers (from vanilla template)")
    print("  ✓ ctxs (context passages, optional)")
    print()
    print("What was discarded (not needed for evaluation):")
    print("  ✗ prompts (reconstructed from question)")
    print("  ✗ all_results (retrieval details)")
    print("  ✗ utility_score, ground_score, relevance_score, etc.")
    print("  ✗ metric_results, golds, scores (Self-RAG internal metrics)")
    print()
    print("You can now run the evaluation script!")

if __name__ == '__main__':
    main()
