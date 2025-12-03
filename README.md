# NLP Final Project - RAG Systems Consolidated Repository

**Note: This repository is for record keeping and documentation purposes only.** Due to different path dependencies and configurations across team members' environments, the scripts cannot be directly executed from this repository. Refer to the original working directories for running experiments.

This repository consolidates the work of 5 team members on comparing different RAG (Retrieval-Augmented Generation) systems for question answering using the HotpotQA dataset.

## Repository Structure

```
NLP_FinalProject_RAG/
├── data_chunking/          # Scripts for preprocessing and chunking training data
├── no_rag_vanilla_rag/     # Scripts for running baseline and vanilla RAG experiments
├── evaluation/             # Evaluation scripts and results
├── instructrag/            # InstructRAG implementation (teammate)
├── selfrag/                # Self-RAG implementation (teammate - to be added)
├── requirements.txt        # Python dependencies
└── environment.yml         # Conda environment specification
```

## Components

### 1. Data Chunking (`data_chunking/`)

Scripts for preparing and chunking the HotpotQA training data.

**Main Scripts:**
- `chunk_training_data.sh` - SLURM job script for chunking data on HPC
- `chunk_training_data_simple.sh` - Simple wrapper for local or cluster execution

**Helper Scripts** (`data_chunking/scripts/`):
- `convert_train_to_selfrag.py` - Convert HotpotQA format to Self-RAG format
- `create_subset.py` - Create subsets of different sizes (1K, 5K, 10K examples)
- `convert_hotpotqa_to_selfrag.py` - General conversion utilities

### 2. No RAG & Vanilla RAG (`no_rag_vanilla_rag/`)

Scripts for running baseline (no retrieval) and vanilla RAG experiments.

**Scripts:**
- `run_no_rag.sh` - Run baseline without retrieval
- `run_vanilla_rag.sh` - Run vanilla RAG with retrieval
- `run_vanilla_rag_hotpot.py` - Python implementation for both modes

**Parameters:**
- DATA_TYPE: `dev`, `train`, or `test`
- DATA_SIZE: `10`, `100`, `500`, `1000`, `5000`, `10000`, or `full`

### 3. Evaluation (`evaluation/`)

Comprehensive evaluation scripts and analysis tools.

**Main Script:**
- `run_evaluation.sh` - SLURM job for running evaluations

**Evaluation Scripts** (`evaluation/scripts/`):
- `advanced_evaluation.py` - Comprehensive evaluation with multiple metrics
- `accuracy_eda.py` - Exploratory data analysis on accuracy
- `check_data_alignment.py` - Verify data consistency
- `convert_selfrag_format.py` - Format conversion utilities
- `nyu_colors.py` - Visualization color schemes

**Subdirectories:**
- `evaluation/results/` - JSON output files from experiments (hosted on Google Drive)
- `evaluation/outputs/` - Plots, reports, and analysis

### 4. InstructRAG (`instructrag/`)

InstructRAG implementation - a framework that allows LMs to explicitly denoise retrieved contents by generating rationales for better verifiability and trustworthiness.

**Main Scripts:**
- `train.sh` - Training script for InstructRAG-FT (fine-tuned version)
- `eval.sh` - Evaluation script for both ICL and FT versions
- `generate_rationale.sh` - Generate rationales for retrieved content
- `hotpot_test.sh` - Test on HotpotQA dataset
- `instructrag_setup.sh` - Setup script for InstructRAG environment
- `setup.sh` - General setup with virtual environment

**Source Files** (`src/`):
- `finetune.py` - Fine-tuning implementation
- `inference.py` - Inference and generation
- `data_utils.py` - Data loading and processing utilities
- `metrics.py` - Evaluation metrics
- `common_utils.py` - Common utility functions
- `log_utils.py` - Logging utilities
- `rag.json` - RAG configuration

**Key Features:**
- Self-Synthesis: Leverage instruction-tuned LMs to generate supervision for denoising
- Easy-to-Use: Supports both in-context learning (ICL) and supervised fine-tuning (SFT)
- Effectiveness: Up to 8.3% better results across benchmarks
- Noise Robustness: Robust to increased noise ratios
- Task Transferability: Solves out-of-domain unseen tasks

**Usage:**
```bash
cd instructrag

# Train InstructRAG-FT
conda activate instrag
bash train.sh

# Evaluate
bash eval.sh
```

**Original Repository:** [InstructRAG GitHub](https://github.com/weizhepei/InstructRAG)

### 5. Self-RAG (`selfrag/`)

*To be added by teammate*

## Metrics

We evaluate models using:
- **Exact Match (EM)**: Exact string match with ground truth
- **F1 Score**: Token-level F1 score
- **Accuracy**: Presence of correct answer
- **Semantic Similarity**: Embedding-based similarity
- **Verbosity Analysis**: Output length statistics

## Results

**Result Files:** Due to their large size (2GB total), result JSON files are hosted on Google Drive. See `evaluation/results/README.md` for the download link and file descriptions.

**Visualizations and Reports:** Available in `evaluation/outputs/` including:
- Core metrics plots
- Semantic similarity analysis
- Accuracy agreement matrices
- Response length statistics
- Comprehensive evaluation reports
