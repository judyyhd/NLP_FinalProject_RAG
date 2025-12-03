# Evaluation Results

Due to the large size of the result files (2GB total), they are hosted on Google Drive instead of being tracked in Git.

## Download Results

**Google Drive Link:** [https://drive.google.com/drive/folders/13qShDIY2GVGJ-cy8Yl3wKGMW7gpc_RtZ?usp=sharing]

## Files Available

The following result files are available in the Google Drive folder:

### No RAG Results
- `no_rag_devfull_results.json` (46M)
- `no_rag_train1000_results.json` (6.2M)
- `no_rag_train5000_results.json` (31M)
- `no_rag_train10000_results.json` (62M)
- `no_rag_trainfull_results.json` (561M)

### Vanilla RAG Results
- `vanilla_rag_dev500_results.json` (3.4M)
- `vanilla_rag_1K_results.json` (6.7M)
- `vanilla_rag_5K_results.json` (34M)
- `vanilla_rag_10K_results.json` (67M)
- `vanilla_rag_devfull_results.json` (51M)
- `vanilla_rag_train1000_results.json` (6.8M)
- `vanilla_rag_train5000_results.json` (34M)
- `vanilla_rag_train10000_results.json` (68M)
- `vanilla_rag_trainfull_results.json` (610M)

### Self-RAG Results
- `selfrag_devfull_results.json` (94M)
- `selfrag_llama2_devfull_results.json` (45M)

### InstructRAG Results
- `instructrag_ft_devfull_results.json` (51M)
- `instructrag_llama2_devfull_results.json` (47M)
- `instructrag_llama3_devfull_results.json` (48M)

## File Format

Each result file is in JSON Lines format (`.jsonl`), with one JSON object per line containing:
- `question`: The input question
- `answer`: Model's generated answer
- `ground_truth`: Correct answer
- `em_score`: Exact match score (0 or 1)
- `f1_score`: Token-level F1 score
- Additional metadata depending on the experiment

## Usage

After downloading, place the files in this directory to run analysis scripts locally.
