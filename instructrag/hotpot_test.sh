#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes 1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time 1-01:00:00
#SBATCH --exclude=a100-4029

hf auth login --token "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"

python src/inference.py \
  --dataset_name hotpotqa \
  --devset_name demo_30.json \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --output_dir outputs/tmp \
  --n_docs 10 \
  --max_instances 1000 \
  --max_tokens 100 \
  --temperature 0