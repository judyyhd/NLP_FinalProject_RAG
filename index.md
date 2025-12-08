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
<!--- - What problem did you tackle? --->
<!--- - What methods did you compare? --->
<!--- - What were your main findings? --->

We study how different reasoning-supervision strategies affect Retrieval-Augmented Generation (RAG) on multi-hop QA (HotpotQA). Vanilla RAG conditions a generator on retrieved passages via latent-document marginalization (RAG-Sequence / RAG-Token), combining a DPR-style retriever with a seq2seq LM, but can struggle when retrieval introduces noise and when evidence must be synthesized across hops. We compare: (i) Vanilla RAG; (ii) Self-RAG, which learns to decide when to retrieve and to self-assess relevance/support/utility via reflection tokens; and (iii) InstructRAG, which equips an instruction-tuned LM with self-synthesized denoising rationales used either as in-context demonstrations or for supervised fine-tuning. On our setup (Llama-2-7B backbone, Contriever retriever), we evaluate accuracy, exact match (EM), F1, and semantic similarity on HotpotQA. 

Two findings stood out. First, these methods optimize different parts of the pipeline and therefore produce qualitatively different answers; simple span-matching metrics alone don’t fully capture their strengths or failure modes. Second, there’s a practical trade-off between complexity/compute and perceived “helpfulness.” Self-RAG’s reflection and gated retrieval deliver a structured, reliable boost over vanilla RAG with modest tuning, keeping outputs concise and easier to verify. InstructRAG can edge higher on HotpotQA’s noisy setting, but its longer, rationale-style generations raise verification cost and expose more chances for drift. In our repo runs, we also observed one-shot > zero-shot and Llama-3 > fine-tuned-Llama-3 (authors’ release) > Llama-2—consistent with backbone strength and our harmonized LLaMA-2-centric code path. Takeaway: choose methods by balancing accuracy gains against compute and human verification effort, and evaluate with task-aligned criteria (faithfulness, concision, stability), not just span-matching.

---

## Introduction

### Motivation
- Why is RAG important for question answering?

Many real questions require facts that aren’t inside the model’s parameters. Retrieval-Augmented Generation (RAG) tackles this by pulling evidence from a large corpus at inference time and letting the generator condition on that evidence. This reduces parametric hallucination, keeps answers grounded in sources, and makes systems easier to update (swap or reindex the corpus rather than re-train the LM). In short: RAG is a practical bridge between powerful LMs and ever-changing knowledge needs.

- What challenges exist in multi-hop reasoning?

Multi-hop QA (like HotpotQA) forces a model to locate multiple pieces of evidence and connect them—often across documents—before answering. The dataset’s “distractor” setup mixes gold paragraphs with retrieved noise; it also includes comparison questions and supplies sentence-level supporting facts, so models are tested not only on answer spans but also on whether their reasoning points to the right evidence. Together, this makes retrieval, evidence selection, and explanation supervision core challenges for any RAG system.

- Why compare different RAG approaches?

There are many RAG flavors, but we focus on two that are both technically sound and complementary in how they improve reasoning: Self-RAG, which teaches the model to decide when to retrieve and to self-assess evidence via reflection tokens, and InstructRAG, which leverages instruction-tuned LMs and denoising rationales to make demonstrations more helpful. They target similar goals (better reasoning and grounding) through different mechanisms (learning to control retrieval vs. strengthening instruction/rationale signals), making them directly comparable and, crucially, feasible to implement end-to-end on the same corpus and backbone for a fair study.

### Research Questions
1. How does retrieval-augmented generation compare to baseline LLMs?
2. What are the tradeoffs between different RAG architectures?
3. How do models handle noisy or irrelevant retrieved information?

---

## Background

### HotpotQA Dataset
<!-- - Description of the dataset -->
HotpotQA is a large, Wikipedia-based QA dataset explicitly designed for **multi-hop** reasoning. Each question typically requires pulling facts from **multiple paragraphs** and linking them before answering. The dataset also provides **sentence-level supporting facts**, enabling evaluation of both answers and the evidence path.

**Why it suits multi-hop reasoning**
- Questions are written to **require** evidence from more than one page (not single-span lookups).
- Includes **comparison** and **bridge** questions (e.g., compare attributes across two entities; follow a link from one page to another).
- **Supporting-fact annotations** supervise explainability and allow finer-grained evaluation.

**Noise / distractors**
- In the common *distractor* setting, each example comes with **10 paragraphs**: **2 gold** + **8 distractors** (retrieved but irrelevant). Models must select the right evidence and ignore plausible noise.
- In *full-wiki*, systems retrieve from the entire Wikipedia dump—raising the bar for retrieval and filtering.

**Quick stats**

| Item                   | Value                                    |
|:-----------------------|:-----------------------------------------|
| Total examples         | 112,779                                  |
| Evidence granularity   | Sentence-level supporting facts          |
| Question types         | Bridge, Comparison (plus others)         |
| Context (distractor)   | 2 gold + 8 distractor paragraphs/example |

**Tiny example (illustrative)**
> *Q:* Which author wrote the novel that the film **X** is based on, and where was that author born?  
> *Needs:* Page A (film → novel) + Page B (author → birthplace) → **answer combines both.**

<!-- - Why it's suitable for multi-hop reasoning

- Dataset statistics and examples -->



### RAG Approaches
<!---
Brief overview of the approaches you compared:
- **Baseline (No RAG):** Direct LLM inference
- **Vanilla RAG:** Standard retrieval + generation
- **Self-RAG:** Self-reflective retrieval-augmented generation
- **InstructRAG:** Instruction-based denoising with rationales
--->
- **Baseline (No RAG)** — Direct LLM generation from the question only (no retrieval). Fast and simple, but prone to parametric hallucinations on factual queries.

- **Vanilla RAG** — Retrieve top-k passages (e.g., dense/TF-IDF retriever) and condition the LM on them during generation. Improves grounding when retrieval is accurate, but can suffer when retrieved evidence is noisy.

- **Self-RAG** — Adds *reflection* signals so the model can decide **when** to retrieve, assess **relevance/groundedness/utility**, and re-rank/use evidence accordingly. Aims for more selective retrieval and concise, supported answers.

- **InstructRAG** — Uses instruction-tuned LMs with *denoising rationales* (as demos or fine-tuning signals) to guide reasoning. Tends to produce more detailed, rationale-style outputs—helpful for noisy settings but longer and costlier to verify.

---

## Methodology

### Data Preparation
<!--- 
- How you preprocessed and chunked the HotpotQA data
- Dataset splits and subset sizes used
- Data format conversion for different models
--->
**Source & scope.** : We use the HotpotQA *distractor* split (each example has 2 gold paragraphs + 8 distractors). We evaluate **only on the full converted dev set**. Since we do not train any models in this study, we do not use the train or test splits.

**Unified conversion.** : We converted the original HotpotQA JSON into a **single, unified schema** so all three pipelines consume the **same inputs**:
- Normalize fields (question, answer, titles, supporting facts).
- Package the 10 provided context paragraphs per example into a consistent `passages[]` field.
- Emit a model-agnostic record (JSON/JSONL) used by **Self-RAG**, **Vanilla RAG**, and adapted **InstructRAG**.

**Chunking.** : We keep HotpotQA paragraphs **as-is (paragraph-level)** to preserve sentence and cross-paragraph references. No additional windowing or sub-paragraph chunking is applied.

**Split usage.**
- **Dev set**: used for all evaluations and ablations.
- **Train/Test**: not used (no fine-tuning performed in our experiments).

**Format alignment across models.**
- **Self-RAG / Vanilla RAG**: run directly on the unified records produced by the converter (so they share the exact same inputs).
- **InstructRAG**: lightly adapted to read the same converted records; prompts are constructed from the unified fields so comparisons are apples-to-apples.

**Quality checks.** : After conversion, we run basic validations (field presence, paragraph counts, non-empty questions/answers) to ensure parity across all three systems before evaluation.

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

| Model       | EM    | F1    | Accuracy | Length | Semantic Sim |
| ----------- | ----- | ----- | -------- | ------ | ------------ |
| No RAG      | 6.28  | 13.55 | 15.58    | 6.7    | 32.62        |
| Vanilla RAG | 3.57  | 14.55 | 25.17    | 14.5   | 35.11        |
| Self-RAG    | 14.42 | 30.35 | 48.85    | 11.8   | 48.75        |
| InstructRAG | 3.30  | 11.82 | 61.28    | 53.9   | 39.16        |

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
