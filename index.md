---
layout: default
title: Comparing RAG Systems for Multi-Hop Question Answering
---

# Comparing RAG Systems for Multi-Hop Question Answering on HotpotQA

**Course:** Fundamentals of Natural Language Processing  
**Team Members:** [Add names]  
**Date:** December 2025

---

## Abstract

Brief summary of your project (2-3 paragraphs):
- What problem did you tackle?
- What methods did you compare?
- What were your main findings?

---

## Introduction

### Motivation
- Why is RAG important for question answering?
- What challenges exist in multi-hop reasoning?
- Why compare different RAG approaches?

### Research Questions
1. How does retrieval-augmented generation compare to baseline LLMs?
2. What are the tradeoffs between different RAG architectures?
3. How do models handle noisy or irrelevant retrieved information?

---

## Background

### HotpotQA Dataset
- Description of the dataset
- Why it's suitable for multi-hop reasoning
- Dataset statistics and examples

### RAG Approaches
Brief overview of the approaches you compared:
- **Baseline (No RAG):** Direct LLM inference
- **Vanilla RAG:** Standard retrieval + generation
- **Self-RAG:** Self-reflective retrieval-augmented generation
- **InstructRAG:** Instruction-based denoising with rationales

---

## Methodology

### Data Preparation
- How you preprocessed and chunked the HotpotQA data
- Dataset splits and subset sizes used
- Data format conversion for different models

### Model Configurations

#### 1. No RAG (Baseline)
- Model: Llama-2-7b
- Setup and hyperparameters
- Purpose: Establish baseline performance

#### 2. Vanilla RAG
- Model: Llama-2-7b with retrieval
- Retrieval method
- Configuration details

#### 3. Self-RAG
- Architecture and key features
- Training approach
- Special tokens and reflection mechanism

#### 4. InstructRAG
- Architecture and key features
- Rationale generation process
- ICL vs FT variants

### Evaluation Metrics
- **Exact Match (EM):** Binary correctness measure
- **F1 Score:** Token-level overlap
- **Accuracy:** Answer presence detection
- **Semantic Similarity:** Embedding-based similarity
- **Verbosity Analysis:** Response length statistics

---

## Experiments

### Experimental Setup
- Hardware: [GPU type, memory]
- Software: Python version, key libraries
- Training details: epochs, batch size, learning rate
- Evaluation protocol

### Dataset Sizes
Experiments conducted on multiple data scales:
- Small: 10-500 examples (quick validation)
- Medium: 1K-5K examples (development)
- Large: 10K+ examples (comprehensive evaluation)
- Full: Complete train/dev/test sets

---

## Results

### Quantitative Results

#### Overall Performance Comparison
[Table comparing all models across metrics]

| Model | EM | F1 | Accuracy | Semantic Sim |
|-------|-----|-----|----------|--------------|
| No RAG | X.XX | X.XX | X.XX | X.XX |
| Vanilla RAG | X.XX | X.XX | X.XX | X.XX |
| Self-RAG | X.XX | X.XX | X.XX | X.XX |
| InstructRAG | X.XX | X.XX | X.XX | X.XX |

#### Performance by Data Size
[Graph showing how models scale with data]

![Core Metrics Comparison](evaluation/outputs/core_metrics.png)

#### Semantic Similarity Analysis
[Analysis of semantic similarity patterns]

![Semantic Similarity](evaluation/outputs/semantic_similarity.png)

#### Response Length Analysis
[Verbosity comparison across models]

![Response Length](evaluation/outputs/response_length.png)

### Qualitative Analysis

#### Case Studies
Examples showcasing:
1. **Success cases:** Where RAG significantly helps
2. **Failure cases:** Where retrieval introduces noise
3. **Edge cases:** Interesting model behaviors

#### Error Analysis
Common failure patterns:
- Retrieval failures (no relevant docs)
- Reasoning failures (incorrect multi-hop)
- Generation failures (hallucination, format issues)

---

## Discussion

### Key Findings
1. **Finding 1:** [Major insight from your results]
2. **Finding 2:** [Another important observation]
3. **Finding 3:** [Additional insight]

### Comparison of Approaches

#### Vanilla RAG
- Strengths: Simple, interpretable, fast
- Weaknesses: No noise handling, limited reasoning
- Best use cases: Clean retrieval scenarios

#### Self-RAG
- Strengths: Self-correction, adaptive retrieval
- Weaknesses: Complexity, training overhead
- Best use cases: Scenarios requiring verification

#### InstructRAG
- Strengths: Explicit denoising, rationale generation
- Weaknesses: Requires rationale training data
- Best use cases: Noisy retrieval environments

### Limitations
- Computational constraints
- Dataset limitations
- Methodological considerations

### Future Work
- Potential improvements
- Additional experiments to explore
- Open research questions

---

## Conclusion

Summary of:
- What you accomplished
- Key takeaways for practitioners
- Contributions to understanding RAG systems

---

## References

1. HotpotQA paper
2. Self-RAG paper
3. InstructRAG paper
4. Llama 2 paper
5. Other relevant citations

---

## Code and Resources

- **GitHub Repository:** [Link to NLP_FinalProject_RAG]
- **Result Files:** [Google Drive link - 2GB results]
- **Individual Contributions:**
  - Data Chunking & Vanilla RAG: [Name]
  - Self-RAG: [Name]
  - InstructRAG: [Name]
  - Evaluation: [Name]
  - Analysis & Visualization: [Name]

---

## Appendix

### A. Implementation Details
Detailed configuration files, hyperparameters, etc.

### B. Additional Results
Supplementary tables and figures

### C. Example Outputs
Sample model generations with retrieved documents

---

*This project was completed as part of the Fundamentals of Natural Language Processing course at NYU, Fall 2025.*
