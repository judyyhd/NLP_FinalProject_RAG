#!/bin/bash
#SBATCH --job-name=chunk_hotpot_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/chunk_data_%j.log
#SBATCH --error=logs/chunk_data_%j.err

################################################################################
# CHUNK HOTPOTQA TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
# Converts hotpot_train_v1.1.json to Self-RAG format and creates subsets
# 
# USAGE:
#   sbatch chunk_training_data.sh
#
# CONFIGURATION (edit these before running):
#   - SUBSET_SIZES: Which subset sizes to create (default: 1000 5000 10000)
#   - USE_RANDOM: true for random sampling, false for sequential (default: false)
#   - RANDOM_SEED: Seed for reproducibility when USE_RANDOM=true
#   - SLURM settings: Adjust --cpus-per-task, --mem, --time as needed
#
# MONITORING:
#   squeue -u $(whoami)                  # Check job status
#   tail -f logs/chunk_data_*.log        # Watch real-time progress
#   ls -lh train_data/hotpot_train_*.json # See results
#
# OUTPUT: train_data/hotpot_train_<size>.json files
################################################################################

set -e  # Exit on any error

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/hy1331/nlp_final"
TRAIN_DATA_DIR="${PROJECT_DIR}/data/train"
INPUT_FILE="${TRAIN_DATA_DIR}/hotpot_train_v1.1.json"
CONVERTED_FILE="${TRAIN_DATA_DIR}/hotpot_train_v1.1_selfrag.json"

# Subset sizes to create
SUBSET_SIZES=(1000 5000 10000)

# Random seed for reproducibility (optional - comment out to use sequential)
RANDOM_SEED=42
USE_RANDOM=false  # Set to true for random sampling

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

print_status() {
    echo "▶ $1"
}

print_success() {
    echo "✓ $1"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "✗ Error: File not found: $1"
        exit 1
    fi
}

# ============================================================================
# Main Script
# ============================================================================

print_header "HotpotQA Training Data Chunking"

print_status "Project directory: ${PROJECT_DIR}"
print_status "Train data directory: ${TRAIN_DATA_DIR}"

# Change to project directory for relative paths
cd "${PROJECT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/logs"

# Check if input file exists
print_status "Checking input file..."
check_file "${INPUT_FILE}"
print_success "Input file found: $(du -h ${INPUT_FILE} | cut -f1)"

# Count examples in input file
print_status "Counting examples in input file..."
TOTAL_EXAMPLES=$(python3 << PYEOF
import json
with open('${INPUT_FILE}', 'r') as f:
    data = json.load(f)
print(len(data))
PYEOF
)
print_success "Total examples: ${TOTAL_EXAMPLES}"

# Convert to Self-RAG format if not already done
if [ ! -f "${CONVERTED_FILE}" ]; then
    print_status "Converting to Self-RAG format..."
    python3 "${PROJECT_DIR}/scripts/convert_train_to_selfrag.py"
    print_success "Conversion complete"
else
    print_success "Self-RAG format already exists: $(du -h ${CONVERTED_FILE} | cut -f1)"
fi

# Create subsets
print_header "Creating Data Subsets"

for SIZE in "${SUBSET_SIZES[@]}"; do
    OUTPUT_FILE="${TRAIN_DATA_DIR}/hotpot_train_${SIZE}.json"
    
    print_status "Creating subset: ${SIZE} examples..."
    
    if [ "$USE_RANDOM" = true ]; then
        python3 "${PROJECT_DIR}/scripts/create_subset.py" \
            "${CONVERTED_FILE}" \
            "${OUTPUT_FILE}" \
            ${SIZE} \
            --random \
            --seed ${RANDOM_SEED}
    else
        python3 "${PROJECT_DIR}/scripts/create_subset.py" \
            "${CONVERTED_FILE}" \
            "${OUTPUT_FILE}" \
            ${SIZE}
    fi
    
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    print_success "Created: ${OUTPUT_FILE} (${FILE_SIZE})"
done

# ============================================================================
# Summary
# ============================================================================

print_header "Chunking Complete!"

print_status "Files created in: ${TRAIN_DATA_DIR}"
echo ""
ls -lh "${TRAIN_DATA_DIR}"/hotpot_train_*.json | awk '{print "  " $9 " (" $5 ")"}'

print_status "Subset method: $([ "$USE_RANDOM" = true ] && echo "Random (seed=${RANDOM_SEED})" || echo "Sequential (first N items)")"

echo ""
print_header "Next Steps"
echo ""
echo "1. Use subsets for evaluation:"
echo ""
echo "   cd ${PROJECT_DIR}/models/retrieval_lm_core"
echo ""
echo "   # Test with 1K examples"
echo "   python run_vanilla_rag_hotpot.py \\"
echo "     --model_name \"meta-llama/Llama-2-7b-hf\" \\"
echo "     --input_file \"../../data/train/hotpot_train_1000.json\" \\"
echo "     --result_fp \"../../evaluation/results/train_1k_results.json\" \\"
echo "     --metric \"em\" \\"
echo "     --mode \"retrieval\""
echo ""
echo "2. To create additional custom subsets:"
echo ""
echo "   python ../../scripts/create_subset.py ../../data/train/hotpot_train_v1.1_selfrag.json ../../data/train/hotpot_train_50k.json 50000"
echo ""

print_success "All done!"
