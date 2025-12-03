#!/usr/bin/env python3
"""
Check which questions are in each model's results and identify missing examples.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_ids_and_questions(filepath):
    """Load question IDs and questions from a results file."""
    ids_questions = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = item.get('id', 'unknown')
                question = item.get('question', '')
                ids_questions[qid] = question
    return ids_questions

def main():
    results_dir = Path("../results")
    
    models = {
        'no_rag': 'no_rag_devfull_results.json',
        'vanilla_rag': 'vanilla_rag_devfull_results.json',
        'instructrag': 'instructrag_llama2_devfull_results.json',
        'selfrag': 'selfrag_llama2_devfull_results.json'
    }
    
    print("Loading question IDs from all models...")
    print("="*70)
    
    all_data = {}
    for model_name, filename in models.items():
        filepath = results_dir / filename
        if filepath.exists():
            data = load_ids_and_questions(filepath)
            all_data[model_name] = data
            print(f"  {model_name:15} {len(data):,} examples")
        else:
            print(f"  {model_name:15} NOT FOUND")
            all_data[model_name] = {}
    
    print("\n" + "="*70)
    print("Analyzing data alignment...")
    print("="*70)
    
    # Get all unique question IDs
    all_ids = set()
    for model_data in all_data.values():
        all_ids.update(model_data.keys())
    
    print(f"\nTotal unique questions across all models: {len(all_ids)}")
    
    # Check which models have which questions
    model_names = list(all_data.keys())
    coverage = defaultdict(list)
    
    for qid in all_ids:
        present_in = [m for m in model_names if qid in all_data[m]]
        coverage[tuple(present_in)].append(qid)
    
    print("\n" + "="*70)
    print("Question Coverage Analysis:")
    print("="*70)
    
    # Sort by coverage pattern
    sorted_coverage = sorted(coverage.items(), key=lambda x: len(x[0]), reverse=True)
    
    for models_tuple, qids in sorted_coverage:
        if len(models_tuple) == len(model_names):
            print(f"\n✓ Present in ALL models: {len(qids)} questions")
        else:
            print(f"\n✗ Present in {models_tuple}: {len(qids)} questions")
            if len(qids) <= 10:
                for qid in qids[:10]:
                    question = all_data[models_tuple[0]][qid]
                    print(f"    {qid}: {question[:80]}...")
            else:
                print(f"    (First 5 of {len(qids)})")
                for qid in qids[:5]:
                    question = all_data[models_tuple[0]][qid]
                    print(f"    {qid}: {question[:80]}...")
    
    # Specific check: What's missing from Self-RAG?
    print("\n" + "="*70)
    print("Self-RAG Missing Questions Analysis:")
    print("="*70)
    
    selfrag_ids = set(all_data['selfrag'].keys())
    other_models_ids = set()
    for model in ['no_rag', 'vanilla_rag', 'instructrag']:
        if model in all_data:
            other_models_ids.update(all_data[model].keys())
    
    missing_from_selfrag = other_models_ids - selfrag_ids
    extra_in_selfrag = selfrag_ids - other_models_ids
    
    print(f"\nQuestions in other models but NOT in Self-RAG: {len(missing_from_selfrag)}")
    if missing_from_selfrag:
        print("\nFirst 10 missing questions:")
        for i, qid in enumerate(list(missing_from_selfrag)[:10], 1):
            # Get question from any other model
            for model in ['no_rag', 'vanilla_rag', 'instructrag']:
                if qid in all_data[model]:
                    question = all_data[model][qid]
                    print(f"  {i}. {qid}: {question[:80]}...")
                    break
    
    if extra_in_selfrag:
        print(f"\nQuestions in Self-RAG but NOT in other models: {len(extra_in_selfrag)}")
        print("First 5:")
        for i, qid in enumerate(list(extra_in_selfrag)[:5], 1):
            question = all_data['selfrag'][qid]
            print(f"  {i}. {qid}: {question[:80]}...")
    
    # Check if IDs are sequential
    print("\n" + "="*70)
    print("Position Analysis:")
    print("="*70)
    
    # Get IDs in order for vanilla (reference)
    vanilla_file = results_dir / 'vanilla_rag_devfull_results.json'
    vanilla_ids_ordered = []
    with open(vanilla_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            vanilla_ids_ordered.append(item['id'])
    
    # Get IDs in order for selfrag
    selfrag_file = results_dir / 'selfrag_llama2_devfull_results.json'
    selfrag_ids_ordered = []
    with open(selfrag_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            selfrag_ids_ordered.append(item['id'])
    
    print(f"\nVanilla RAG has {len(vanilla_ids_ordered)} questions")
    print(f"Self-RAG has {len(selfrag_ids_ordered)} questions")
    print(f"\nFirst 5 IDs in Vanilla RAG: {vanilla_ids_ordered[:5]}")
    print(f"First 5 IDs in Self-RAG:    {selfrag_ids_ordered[:5]}")
    print(f"\nLast 5 IDs in Vanilla RAG: {vanilla_ids_ordered[-5:]}")
    print(f"Last 5 IDs in Self-RAG:    {selfrag_ids_ordered[-5:]}")
    
    # Check if Self-RAG is just a prefix
    is_prefix = all(sid == vid for sid, vid in zip(selfrag_ids_ordered, vanilla_ids_ordered))
    print(f"\nSelf-RAG is a prefix of Vanilla RAG: {is_prefix}")
    
    if is_prefix:
        print("\n✓ Self-RAG contains the FIRST 5,331 examples from the dataset")
        print("  This suggests the generation stopped early or was run on a subset")
    
    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)
    if is_prefix:
        print("\nOption 1: Evaluate all models on the first 5,331 examples")
        print("  - Fair comparison")
        print("  - All models tested on exact same data")
        print("\nOption 2: Get complete Self-RAG results (all 7,405 examples)")
        print("  - Ask teammate to run remaining examples")
        print("  - Or re-run Self-RAG on full dataset")
    else:
        print("\nThe Self-RAG results don't align sequentially with other models.")
        print("Need to investigate why certain questions are missing.")

if __name__ == '__main__':
    main()
