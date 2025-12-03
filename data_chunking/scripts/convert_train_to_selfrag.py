#!/usr/bin/env python3
"""
Convert HotpotQA training data to Self-RAG format and create subsets.

USAGE:
    python convert_train_to_selfrag.py

INPUT:
    - data/train/hotpot_train_v1.1.json (original HotpotQA format)

OUTPUT:
    - data/train/hotpot_train_v1.1_selfrag.json (converted format)
    - data/train/hotpot_train_1000.json (subset)
    - data/train/hotpot_train_5000.json (subset)
    - data/train/hotpot_train_10000.json (subset)

RUNNING IN BACKGROUND:
    sbatch scripts/chunk_training_data.sh  (recommended for large files)

FORMAT CONVERSION:
    HotpotQA format → Self-RAG format
    - "_id" → "id"
    - "answer" → "answers" (list)
    - "context": [[title, [sents]]] → "ctxs": [{title, text}]
"""

import json
from pathlib import Path
from tqdm import tqdm

def convert_hotpot_to_selfrag():
    """Convert downloaded hotpot_train_v1.1.json to Self-RAG format."""
    
    input_file = Path("data/train/hotpot_train_v1.1.json")
    output_file = Path("data/train/hotpot_train_v1.1_selfrag.json")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        return False
    
    print(f"📖 Loading {input_file.name}...")
    with open(input_file, 'r') as f:
        hotpot_data = json.load(f)
    
    print(f"🔄 Converting {len(hotpot_data)} examples to Self-RAG format...")
    
    converted = []
    for i, item in enumerate(tqdm(hotpot_data)):
        try:
            # Extract contexts
            ctxs = []
            for title, sentences in item.get('context', []):
                ctxs.append({
                    'title': title,
                    'text': ' '.join(sentences)
                })
            
            # Create Self-RAG format
            converted_item = {
                'id': item.get('_id', str(i)),
                'question': item.get('question', ''),
                'answers': [item.get('answer', '')],
                'ctxs': ctxs
            }
            converted.append(converted_item)
        except Exception as e:
            print(f"⚠️  Skipping item {i}: {e}")
            continue
    
    print(f"\n💾 Saving {len(converted)} converted examples to {output_file.name}...")
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"   Total examples: {len(converted)}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f}MB")
    
    return True

def create_subsets(input_file, output_dir):
    """Create smaller subsets from converted data."""
    print(f"\n📊 Creating training data subsets...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    subset_sizes = [1000, 5000, 10000]
    
    for size in subset_sizes:
        if size > len(data):
            continue
        
        subset = data[:size]
        output_file = Path(output_dir) / f"hotpot_train_{size}.json"
        
        with open(output_file, 'w') as f:
            json.dump(subset, f, indent=2)
        
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"   ✓ {output_file.name} ({size} examples, {size_mb:.1f}MB)")

if __name__ == "__main__":
    print("=" * 60)
    print("HotpotQA Training Data Conversion")
    print("=" * 60)
    
    # Convert to Self-RAG format
    if convert_hotpot_to_selfrag():
        # Create subsets
        create_subsets("data/train/hotpot_train_v1.1_selfrag.json", "data/train")
        
        print("\n" + "=" * 60)
        print("✅ All done! Your training data is ready.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Use training data for evaluation:")
        print("   cd models/retrieval_lm_core")
        print("   python run_vanilla_rag_hotpot.py \\")
        print("     --input_file ../data/train/hotpot_train_1000.json \\")
        print("     --result_fp results/train_eval.json")
