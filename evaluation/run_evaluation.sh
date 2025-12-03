#!/bin/bash
#SBATCH --job-name=eval_results
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evaluation_%j.log
#SBATCH --error=logs/evaluation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hy1331@nyu.edu

echo "============================================================================"
echo "Running Evaluation Scripts"
echo "============================================================================"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate selfrag

echo "▶ Python: $(which python3)"
echo "▶ Working directory: $(pwd)"
echo ""

# Run advanced evaluation with all metrics
echo "============================================================================"
echo "Running Comprehensive Evaluation"
echo "  - Exact Match (EM)"
echo "  - F1 Score"
echo "  - Accuracy (exact presence)"
echo "  - Semantic similarity"
echo "  - Verbosity analysis"
echo "  - All 3 models: No RAG, Vanilla RAG, InstructRAG"
echo "============================================================================"
cd /home/hy1331/nlp_final
srun python3 evaluation/scripts/advanced_evaluation.py
echo ""

echo "============================================================================"
echo "✅ Evaluation Complete!"
echo "============================================================================"
echo "Check evaluation/outputs/ for all plots and reports"
