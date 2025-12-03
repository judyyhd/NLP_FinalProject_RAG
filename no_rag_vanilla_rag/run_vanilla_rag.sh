#!/bin/bash
#SBATCH --job-name=vanilla_rag
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/home/hy1331/nlp_final/logs/vanilla_rag_%j.log
#SBATCH --error=/home/hy1331/nlp_final/logs/vanilla_rag_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hy1331@nyu.edu

# Vanilla RAG Evaluation on HotpotQA Data
# This script runs vanilla RAG evaluation WITH retrieval
# Usage: sbatch run_vanilla_rag.sh [DATA_TYPE] [DATA_SIZE]
#   DATA_TYPE: dev, train, or test (default: dev)
#   DATA_SIZE: 10, 100, 500, 1000, 5000, 10000, or full (default: 500)
# Examples:
#   sbatch run_vanilla_rag.sh dev 10
#   sbatch run_vanilla_rag.sh dev 500
#   sbatch run_vanilla_rag.sh train 1000
#   sbatch run_vanilla_rag.sh test full

set -e  # Exit on any error

# Get data type and size from command line arguments
DATA_TYPE=${1:-dev}
DATA_SIZE=${2:-500}

# Get data type and size from command line arguments
DATA_TYPE=${1:-dev}
DATA_SIZE=${2:-500}

# Set resource variables based on data size (for display/batch size calculation)
if [[ "${DATA_SIZE}" == "full" || "${DATA_SIZE}" -ge 10000 ]]; then
    # Large datasets: full train data (~90K examples)
    MEM="64GB"
    TIME="12:00:00"
    GPUS="2"
elif [[ "${DATA_SIZE}" -ge 5000 ]]; then
    # Medium-large datasets: 5K-10K examples
    MEM="48GB"
    TIME="06:00:00"
    GPUS="1"
elif [[ "${DATA_SIZE}" -ge 1000 ]]; then
    # Medium datasets: 1K-5K examples
    MEM="32GB"
    TIME="03:00:00"
    GPUS="1"
else
    # Small datasets: 10-500 examples
    MEM="32GB"
    TIME="01:00:00"
    GPUS="1"
fi

# ============================================================================
# Configuration
# ============================================================================

# Data type and size already set above for SLURM resource allocation

PROJECT_DIR="/home/hy1331/nlp_final"
RETRIEVAL_DIR="${PROJECT_DIR}/models/retrieval_lm_core"
RESULTS_DIR="${PROJECT_DIR}/evaluation/results"

# Model configuration
MODEL_NAME="meta-llama/Llama-2-7b-hf"
MAX_NEW_TOKENS=100

# Adjust batch size based on available resources
if [[ "${GPUS}" == "2" ]]; then
    BATCH_SIZE=10
    WORLD_SIZE=2
else
    BATCH_SIZE=5
    WORLD_SIZE=1
fi

METRIC="em"
MODE="retrieval"  # Options: vanilla, retrieval
PROMPT_NAME="prompt_no_input_retrieval"
CONDA_ENV="selfrag"

# Validate data type
if [[ "${DATA_TYPE}" != "dev" && "${DATA_TYPE}" != "train" && "${DATA_TYPE}" != "test" ]]; then
    echo "Error: Invalid data type '${DATA_TYPE}'"
    echo "Valid options: dev, train, test"
    exit 1
fi

# Set dataset based on type and size parameters
if [ "${DATA_TYPE}" == "dev" ]; then
    DATA_DIR="${PROJECT_DIR}/data/dev"
    case ${DATA_SIZE} in
        10)
            DATASET_FILE="hotpotqa_dev_10.json"
            SUBSET_NAME="dev10"
            ;;
        100)
            DATASET_FILE="hotpotqa_dev_100.json"
            SUBSET_NAME="dev100"
            ;;
        500)
            DATASET_FILE="hotpotqa_dev_500.json"
            SUBSET_NAME="dev500"
            ;;
        full)
            DATASET_FILE="hotpotqa_dev.json"
            SUBSET_NAME="dev_full"
            ;;
        *)
            echo "Error: Invalid dev data size '${DATA_SIZE}'"
            echo "Valid options for dev: 10, 100, 500, full"
            exit 1
            ;;
    esac
elif [ "${DATA_TYPE}" == "train" ]; then
    DATA_DIR="${PROJECT_DIR}/data/train"
    case ${DATA_SIZE} in
        1000)
            DATASET_FILE="hotpot_train_1000.json"
            SUBSET_NAME="train1000"
            ;;
        5000)
            DATASET_FILE="hotpot_train_5000.json"
            SUBSET_NAME="train5000"
            ;;
        10000)
            DATASET_FILE="hotpot_train_10000.json"
            SUBSET_NAME="train10000"
            ;;
        full)
            DATASET_FILE="hotpot_train_v1.1_selfrag.json"
            SUBSET_NAME="train_full"
            ;;
        *)
            echo "Error: Invalid train data size '${DATA_SIZE}'"
            echo "Valid options for train: 1000, 5000, 10000, full"
            exit 1
            ;;
    esac
else  # test
    DATA_DIR="${PROJECT_DIR}/data/test"
    case ${DATA_SIZE} in
        full)
            DATASET_FILE="hotpot_test_fullwiki_v1.json"
            SUBSET_NAME="test_full"
            ;;
        *)
            echo "Error: Invalid test data size '${DATA_SIZE}'"
            echo "Valid options for test: full"
            exit 1
            ;;
    esac
fi

# Data subsets to evaluate
DATASETS=(
    "${DATASET_FILE}:${SUBSET_NAME}"
)

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

run_evaluation() {
    local input_file="$1"
    local subset_name="$2"
    local output_file="$3"
    
    print_status "Running evaluation on ${subset_name} subset..."
    print_status "Input: ${input_file}"
    print_status "Output: ${output_file}"
    
    python run_vanilla_rag_hotpot.py \
        --model_name "${MODEL_NAME}" \
        --input_file "${input_file}" \
        --result_fp "${output_file}" \
        --metric "${METRIC}" \
        --mode "${MODE}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --batch_size ${BATCH_SIZE} \
        --world_size ${WORLD_SIZE} \
        --prompt_name "${PROMPT_NAME}"
    
    print_success "Completed evaluation on ${subset_name}"
    
    # Extract and display detailed metrics
    if [ -f "${output_file}" ]; then
        local total_examples=$(python3 -c "
import json
count = 0
with open('${output_file}', 'r') as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
")
        
        # Extract EM and F1 scores
        local metrics=$(python3 -c "
import json
import numpy as np
data = []
with open('${output_file}', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
em_scores = [item.get('em_score', 0) for item in data]
f1_scores = [item.get('f1_score', 0) for item in data]
primary_scores = [item.get('metric_result', 0) for item in data]
print(f'{np.mean(em_scores):.4f},{np.mean(f1_scores):.4f},{np.mean(primary_scores):.4f}')
")
        
        local em_score=$(echo ${metrics} | cut -d, -f1)
        local f1_score=$(echo ${metrics} | cut -d, -f2) 
        local primary_score=$(echo ${metrics} | cut -d, -f3)
        
        print_success "Results for ${subset_name} (${total_examples} examples):"
        print_success "  - Exact Match (EM): ${em_score}"
        print_success "  - F1 Score: ${f1_score}"
        print_success "  - Primary metric (${METRIC}): ${primary_score}"
    fi
}

# ============================================================================
# Main Script
# ============================================================================

print_header "Vanilla RAG Evaluation on HotpotQA ${DATA_TYPE} Data (${DATA_SIZE} examples)"

# Create descriptive symlinks to log files
if [ -n "${SLURM_JOB_ID}" ]; then
    LOG_DIR="${PROJECT_DIR}/logs"
    ln -sf "vanilla_rag_${SLURM_JOB_ID}.log" "${LOG_DIR}/vanilla_rag_${DATA_TYPE}${DATA_SIZE}_latest.log"
    ln -sf "vanilla_rag_${SLURM_JOB_ID}.err" "${LOG_DIR}/vanilla_rag_${DATA_TYPE}${DATA_SIZE}_latest.err"
    print_status "Log files: vanilla_rag_${DATA_TYPE}${DATA_SIZE}_latest.{log,err}"
fi

# Activate conda environment
print_status "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

print_status "Environment: $(which python)"
print_status "Working directory: ${RETRIEVAL_DIR}"
print_status "Model: ${MODEL_NAME}"
print_status "Mode: ${MODE} (WITH RETRIEVAL)"
print_status "Data type: ${DATA_TYPE}"
print_status "Data size: ${DATA_SIZE}"
print_status "Resources: ${MEM} RAM, ${GPUS} GPU(s), ${TIME} time limit"
print_status "Batch size: ${BATCH_SIZE}"

# Create necessary directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Navigate to retrieval_lm directory
cd "${RETRIEVAL_DIR}"

# Check if data exists
print_status "Checking data availability..."
for dataset_info in "${DATASETS[@]}"; do
    dataset_file=$(echo ${dataset_info} | cut -d: -f1)
    full_path="${DATA_DIR}/${dataset_file}"
    if [ ! -f "${full_path}" ]; then
        echo "✗ Error: Data not found: ${full_path}"
        echo "Please check that the data files are available in ${DATA_DIR}/"
        exit 1
    fi
    dataset_size=$(du -h "${full_path}" | cut -f1)
    print_success "Found: ${dataset_file} (${dataset_size})"
done

# Run evaluations on each subset
print_header "Running Evaluations"

for dataset_info in "${DATASETS[@]}"; do
    dataset_file=$(echo ${dataset_info} | cut -d: -f1)
    subset_name=$(echo ${dataset_info} | cut -d: -f2)
    
    input_file="${DATA_DIR}/${dataset_file}"
    output_file="${RESULTS_DIR}/vanilla_rag_${DATA_TYPE}${DATA_SIZE}_results.json"
    
    echo ""
    run_evaluation "${input_file}" "${subset_name}" "${output_file}"
    
    # Brief pause between evaluations
    sleep 5
done

# ============================================================================
# Summary
# ============================================================================

print_header "Evaluation Complete!"

print_status "Results saved in: ${RESULTS_DIR}"
echo ""
ls -lh "${RESULTS_DIR}"/vanilla_rag_*_results.json | awk '{print "  " $9 " (" $5 ")"}'

print_status "To view detailed results:"
echo ""
echo "  # Summary of all results"
echo "  ls -lh ${RESULTS_DIR}/vanilla_rag_*_results.json"
echo ""
echo "  # View specific result file"
echo "  less ${RESULTS_DIR}/vanilla_rag_dev500_results.json"
echo ""
echo "  # Extract EM and F1 scores quickly"
echo "  python3 -c \""
echo "import json; import numpy as np"
echo "for size in ['dev500']:"
echo "    try:"
echo "        data = []"
echo "        with open('${RESULTS_DIR}/vanilla_rag_' + size + '_results.json', 'r') as f:"
echo "            for line in f:"
echo "                if line.strip():"
echo "                    data.append(json.loads(line))"
echo "        em_scores = [item.get('em_score', 0) for item in data]"
echo "        f1_scores = [item.get('f1_score', 0) for item in data]"
echo "        print(f'{size}: EM={np.mean(em_scores):.4f}, F1={np.mean(f1_scores):.4f} ({len(data)} examples)')"
echo "    except FileNotFoundError:"
echo "        print(f'{size}: Not completed yet')"
echo "\""

# Print summary table of all results
echo ""
print_header "FINAL RESULTS SUMMARY"
echo ""
printf "%-8s %-12s %-12s %-12s\n" "Subset" "EM Score" "F1 Score" "Examples"
printf "%-8s %-12s %-12s %-12s\n" "------" "--------" "--------" "--------"

for dataset_info in "${DATASETS[@]}"; do
    subset_name=$(echo ${dataset_info} | cut -d: -f2)
    output_file="${RESULTS_DIR}/vanilla_rag_${DATA_TYPE}${DATA_SIZE}_results.json"
    
    if [ -f "${output_file}" ]; then
        metrics=$(python3 -c "
import json
import numpy as np
try:
    data = []
    with open('${output_file}', 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    em_scores = [item.get('em_score', 0) for item in data]
    f1_scores = [item.get('f1_score', 0) for item in data]
    print(f'{np.mean(em_scores):.4f},{np.mean(f1_scores):.4f},{len(data)}')
except:
    print('N/A,N/A,0')
")
        em_score=$(echo ${metrics} | cut -d, -f1)
        f1_score=$(echo ${metrics} | cut -d, -f2)
        examples=$(echo ${metrics} | cut -d, -f3)
        printf "%-8s %-12s %-12s %-12s\n" "${subset_name}" "${em_score}" "${f1_score}" "${examples}"
    else
        printf "%-8s %-12s %-12s %-12s\n" "${subset_name}" "N/A" "N/A" "0"
    fi
done

print_success "All evaluations completed successfully!"

# Optional: Send summary to email or notification (if configured)
# echo "Vanilla RAG evaluation completed on $(date)" | mail -s "Job Completed" your.email@domain.com