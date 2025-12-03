#!/bin/bash
################################################################################
# SIMPLE WRAPPER FOR CHUNKING TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
# Convenience script to submit chunking job or run locally
#
# USAGE:
#   bash chunk_training_data_simple.sh sbatch  # Submit to HPC cluster (recommended)
#   bash chunk_training_data_simple.sh local   # Run on login node (not recommended)
#
# RECOMMENDATION: Use 'sbatch' mode to avoid login node overload
#   bash chunk_training_data_simple.sh sbatch
#
# WHAT IT DOES:
#   1. sbatch mode: Submits chunk_training_data.sh to compute nodes
#   2. local mode: Runs conversion and chunking directly (slower, not recommended)
#
# OUTPUT: Subsets in train_data/ (1K, 5K, 10K examples in Self-RAG format)
################################################################################

set -e

PROJECT_DIR="/home/hy1331/nlp_final"
TRAIN_DATA_DIR="${PROJECT_DIR}/data/train"

# Parse arguments
MODE="${1:-sbatch}"

if [ "$MODE" = "sbatch" ]; then
    echo "📤 Submitting batch job to HPC cluster..."
    cd "${PROJECT_DIR}"
    sbatch chunk_training_data.sh
    echo "✓ Job submitted! Check status with: squeue -u $(whoami)"
    echo "  View logs with: tail -f logs/chunk_data_*.log"

elif [ "$MODE" = "local" ]; then
    echo "⚠️  Running on login node (not recommended for large datasets)"
    echo "   This might be slow. For background processing, use:"
    echo "   bash chunk_training_data_simple.sh sbatch"
    echo ""
    echo "⏳ Starting conversion..."
    cd "${PROJECT_DIR}"
    python3 convert_train_to_selfrag.py
    
    echo ""
    echo "⏳ Creating subsets..."
    for SIZE in 1000 5000 10000; do
        python3 create_subset.py \
            "${TRAIN_DATA_DIR}/hotpot_train_v1.1_selfrag.json" \
            "${TRAIN_DATA_DIR}/hotpot_train_${SIZE}.json" \
            ${SIZE}
    done
    
    echo "✓ Complete!"

else
    echo "Usage: bash chunk_training_data_simple.sh [mode]"
    echo ""
    echo "Modes:"
    echo "  sbatch    Submit as batch job to HPC cluster (recommended)"
    echo "  local     Run directly on login node (not recommended)"
    echo ""
    echo "Examples:"
    echo "  bash chunk_training_data_simple.sh sbatch"
    echo "  bash chunk_training_data_simple.sh local"
    exit 1
fi
