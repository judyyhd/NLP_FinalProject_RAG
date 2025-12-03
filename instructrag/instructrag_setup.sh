#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=setup_instag
#SBATCH --output=setup_instag.log

source ~/.bashrc

conda create -n instrag python=3.10 -y

conda activate instrag

module load cuda/11.8
module load gcc/11.2.0

pip install torch==2.2.1 torchvision==0.17.2 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install flash-attn==2.5.7 --no-build-isolation

pip install vllm==0.4.1

pip install numpy==1.26.4 accelerate packaging

