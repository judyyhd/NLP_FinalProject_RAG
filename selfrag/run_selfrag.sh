#!/bin/bash
#SBATCH --job-name=selfrag_hotpotqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/selfrag_%j.out
#SBATCH --error=logs/selfrag_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gha2009@nyu.edu

# Load modules
module purge
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11

# Activate conda environment
source ~/.bashrc
conda activate selfrag

# Set environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/scratch/gha2009/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gha2009/.cache/transformers
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Navigate to project directory
cd /home/gha2009/nlp_final

# Run Self-RAG
python SelfRag/retrieval_lm/run_short_form_fixed.py \
    --model_name selfrag/selfrag_llama2_7b \
    --input_file data/dev/hotpotqa_dev.json \
    --mode adaptive_retrieval \
    --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file evaluation/results/selfrag_devfull_results.json \
    --metric match \
    --ndocs 10 \
    --use_groundness \
    --use_utility \
    --use_seqscore \
    --dtype half \
    --w_rel 1.0 \
    --w_sup 1.0 \
    --w_use 0.5