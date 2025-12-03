import json
import sys
import random
import argparse

def create_subset(input_file, output_file, n, random_sample=False, seed=None):
    """
    Create a subset of data from a JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        n: Number of examples to extract
        random_sample: If True, randomly sample; if False, take first n items
        seed: Random seed for reproducibility (only used if random_sample=True)
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if n > len(data):
        print(f"⚠️  Warning: Requested {n} examples but only {len(data)} available. Using all available.")
        n = len(data)
    
    if random_sample:
        if seed is not None:
            random.seed(seed)
        subset = random.sample(data, n)
        method = f"random (seed={seed})" if seed is not None else "random"
    else:
        subset = data[:n]
        method = "sequential (first n items)"
    
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"✓ Created subset of {n} examples: {output_file}")
    print(f"  Method: {method}")
    print(f"  Total examples in input: {len(data)}")
    print(f"  Output file size: {len(json.dumps(subset, indent=2)) / 1024 / 1024:.1f}MB")

def main():
    parser = argparse.ArgumentParser(
        description="Create a subset of a JSON dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential sampling (first 1000 examples)
  python create_subset.py input.json output.json 1000
  
  # Random sampling (1000 random examples)
  python create_subset.py input.json output.json 1000 --random
  
  # Random sampling with seed for reproducibility
  python create_subset.py input.json output.json 1000 --random --seed 42
        """
    )
    
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_file', help='Path to output JSON file')
    parser.add_argument('n', type=int, help='Number of examples to extract')
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random sampling instead of sequential'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (only with --random)'
    )
    
    args = parser.parse_args()
    
    create_subset(
        args.input_file,
        args.output_file,
        args.n,
        random_sample=args.random,
        seed=args.seed
    )

if __name__ == "__main__":
    # Support both old-style (positional args) and new-style (argparse)
    if len(sys.argv) >= 4 and not any(arg.startswith('-') for arg in sys.argv[1:]):
        # Old-style: python create_subset.py input output n
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        n = int(sys.argv[3])
        create_subset(input_file, output_file, n, random_sample=False, seed=None)
    else:
        # New-style with argparse
        main()
