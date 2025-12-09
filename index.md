---
layout: default
title: Comparing RAG Systems for Multi-Hop Question Answering
---

# Comparing RAG Systems for Multi-Hop Question Answering on HotpotQA

**Course:** Fundamentals of Natural Language Processing  
**Team Members:** [Ghina Al Shdaifat, Huizhen Jin, Shengduo Li, Yixuan Wang, Judy Yang]  
**Date:** December 2025

---

## Abstract
<!--- - What problem did you tackle? --->
<!--- - What methods did you compare? --->
<!--- - What were your main findings? --->

We study how different reasoning-supervision strategies affect Retrieval-Augmented Generation (RAG) on multi-hop QA (HotpotQA). Vanilla RAG conditions a generator on retrieved passages via latent-document marginalization (RAG-Sequence / RAG-Token), combining a DPR-style retriever with a seq2seq LM, but can struggle when retrieval introduces noise and when evidence must be synthesized across hops. We compare: (i) Vanilla RAG; (ii) Self-RAG, which learns to decide when to retrieve and to self-assess relevance/support/utility via reflection tokens; and (iii) InstructRAG, which equips an instruction-tuned LM with self-synthesized denoising rationales used either as in-context demonstrations or for supervised fine-tuning. On our setup (Llama-2-7B backbone, Contriever retriever), we evaluate accuracy, exact match (EM), F1, and semantic similarity on HotpotQA. 

Two findings stood out. First, these methods optimize different parts of the pipeline and therefore produce qualitatively different answers; simple span-matching metrics alone don’t fully capture their strengths or failure modes. Second, there’s a practical trade-off between complexity/compute and perceived “helpfulness.” Self-RAG’s reflection and gated retrieval deliver a structured, reliable boost over vanilla RAG with modest tuning, keeping outputs concise and easier to verify. InstructRAG can edge higher on HotpotQA’s noisy setting, but its longer, rationale-style generations raise verification cost and expose more chances for drift. In our repo runs, we also observed one-shot > zero-shot and Llama-3 > fine-tuned-Llama-3 (authors’ release) > Llama-2—consistent with backbone strength and our harmonized LLaMA-2-centric code path. Takeaway: choose methods by balancing accuracy gains against compute and human verification effort, and evaluate with task-aligned criteria (faithfulness, concision, stability), not just span-matching.

---

## Introduction

### Motivation
**Why is RAG important for question answering?**

Many real questions require facts that aren’t inside the model’s parameters. Retrieval-Augmented Generation (RAG) tackles this by pulling evidence from a large corpus at inference time and letting the generator condition on that evidence. This reduces parametric hallucination, keeps answers grounded in sources, and makes systems easier to update (swap or reindex the corpus rather than re-train the LM). In short: RAG is a practical bridge between powerful LMs and ever-changing knowledge needs.

**What challenges exist in multi-hop reasoning?**

Multi-hop QA (like HotpotQA) forces a model to locate multiple pieces of evidence and connect them—often across documents—before answering. The dataset’s “distractor” setup mixes gold paragraphs with retrieved noise; it also includes comparison questions and supplies sentence-level supporting facts, so models are tested not only on answer spans but also on whether their reasoning points to the right evidence. Together, this makes retrieval, evidence selection, and explanation supervision core challenges for any RAG system.

**Why compare different RAG approaches?**

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
**Source & scope** : We use the HotpotQA *distractor* split (each example has 2 gold paragraphs + 8 distractors). We evaluate **only on the full converted dev set**. Since we do not train any models in this study, we do not use the train or test splits.

**Unified conversion** : We converted the original HotpotQA JSON into a **single, unified schema** so all three pipelines consume the **same inputs**:
- Normalize fields (question, answer, titles, supporting facts).
- Package the 10 provided context paragraphs per example into a consistent `passages[]` field.
- Emit a model-agnostic record (JSON/JSONL) used by **Self-RAG**, **Vanilla RAG**, and adapted **InstructRAG**.

**Chunking** : We keep HotpotQA paragraphs **as-is (paragraph-level)** to preserve sentence and cross-paragraph references. No additional windowing or sub-paragraph chunking is applied.

**Split usage**
- **Dev set**: used for all evaluations and ablations.
- **Train/Test**: not used (no fine-tuning performed in our experiments).

**Format alignment across models.**
- **Self-RAG / Vanilla RAG**: run directly on the unified records produced by the converter (so they share the exact same inputs).
- **InstructRAG**: lightly adapted to read the same converted records; prompts are constructed from the unified fields so comparisons are apples-to-apples.

**Quality checks** : After conversion, we run basic validations (field presence, paragraph counts, non-empty questions/answers) to ensure parity across all three systems before evaluation.

### Model Configurations
<!--
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
-->
### 1) No RAG (Baseline)
- **Model:** Llama-2-7B (direct generation from the question; no retrieval).
- **Setup:** Uses the shared, converted HotpotQA dev records; executed via the baseline script in the `no_rag_vanilla_rag/` workflow so it shares the same evaluation harness as RAG variants.  
  *Purpose:* establish a grounding baseline before adding retrieval. :contentReference[oaicite:0]{index=0}

### 2) Vanilla RAG
- **Model:** Llama-2-7B **with retrieval** (retrieve-then-generate).
- **Retrieval:** Top-k paragraph retrieval from the provided HotpotQA distractor contexts (2 gold + 8 distractors per example in our converted format).
- **Configuration:** Single-stage generation conditioned on retrieved passages; same data path and evaluation scripts as the baseline to ensure apples-to-apples comparison. :contentReference[oaicite:1]{index=1}

### 3) Self-RAG
- **Architecture & key features:** Adds **adaptive retrieval** (decide when to retrieve) and **self-reflection tokens** to score **relevance**, **support/groundedness**, and **utility**, with optional **critique-aware decoding**. Supports multiple modes (always retrieve / adaptive / no retrieval). :contentReference[oaicite:2]{index=2}
- **Training/usage:** We use the released Self-RAG LM for inference over our unified dev set; experiments are run via the `selfrag/` pipeline in the repo (HPC-ready scripts + environment). :contentReference[oaicite:3]{index=3}
- **Special tokens & mechanism:** Reflection tokens guide retrieval gating and re-ranking/decoding, improving faithfulness and concision compared to vanilla RAG under the same input format. :contentReference[oaicite:4]{index=4}

### 4) InstructRAG
- **Architecture & key features:** Instruction-oriented RAG that **generates denoising rationales** to improve verifiability and robustness. The repo provides scripts for **ICL (zero/one-shot)** and **SFT/FT** variants under a unified runner. :contentReference[oaicite:5]{index=5}
- **Rationale process:** Given retrieved content, the model synthesizes short rationales that filter/clean noisy snippets before answering, yielding rationale-style outputs. :contentReference[oaicite:6]{index=6}
- **ICL vs. FT variants:** We evaluated **zero-shot** and **one-shot** ICL and compared backbones (**Llama-2**, **Llama-3**, and the authors’ **FT Llama-3** release) using the same converted dev set and harmonized prompts for comparability. :contentReference[oaicite:7]{index=7}

### Evaluation Metrics
<!--
- **Exact Match (EM):** Binary correctness measure
- **F1 Score:** Token-level overlap
- **Accuracy:** Answer presence detection
- **Semantic Similarity:** Embedding-based similarity
- **Verbosity Analysis:** Response length statistics
-->
- **Exact Match (EM)** — Strict binary correctness: the predicted string must exactly match a gold alias (after standard normalization). Best for concise, unambiguous answers.

- **F1 (token-level)** — Harmonic mean of precision/recall over tokens after the same normalization. More tolerant than EM, but can reward verbosity that happens to include the gold tokens.

- **Accuracy (answer presence)** — Lenient hit-rate: 1 if any gold alias appears anywhere in the output, else 0. Useful for rationale-heavy methods; insensitive to extra/conflicting text.

- **Semantic Similarity** — Cosine similarity between embeddings of the model output and gold answer (e.g., SBERT `all-MiniLM-L6-v2`). Captures paraphrase-level agreement but may blur fine-grained distinctions.

- **Verbosity (length)** — Summary stats of response length. Frames trade-offs: longer outputs aid explanation but increase verification cost; shorter outputs are easier to audit.

---

## Experiments
<!--
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
-->
### Experimental Setup
- **Hardware.** NYU Greene HPC cluster; single-GPU inference per run on **NVIDIA A100 80GB** (no multi-GPU training).
- **Software.** Python (≥3.10), PyTorch, Hugging Face *transformers*, FAISS (for indexing/retrieval), and *sentence-transformers* (`all-MiniLM-L6-v2`) for semantic similarity. Runs use our repo’s unified data converter and evaluation harness.
- **Training details.** **No fine-tuning** performed.  
  - **Baseline / Vanilla RAG / Self-RAG:** released checkpoints, inference-only.  
  - **InstructRAG:** **ICL** variants (0-shot and 1-shot) only; no additional SFT by us.  
  - Common generation knobs aligned across models (e.g., temperature, max_new_tokens) with fixed seeds for comparability.
- **Evaluation protocol.** All models consume the **same converted HotpotQA distractor dev set**. We compute **EM, F1, Accuracy, Semantic Similarity, and Length**, averaging over the set. Identical preprocessing and prompt construction ensure apples-to-apples comparison.

### Dataset Sizes
- **Reported runs:** **Full HotpotQA dev (distractor)** after conversion/harmonization to our unified schema.  
- (During debugging we occasionally used small slices for quick checks; only full-dev results are reported.)

---

## Results

### Quantitative Results

#### Overall Performance Comparison

| Model       | EM    | F1    | Accuracy | Length | Semantic Sim |
| ----------- | ----- | ----- | -------- | ------ | ------------ |
| No RAG      | 6.28  | 13.55 | 15.58    | 6.7    | 32.62        |
| Vanilla RAG | 3.57  | 14.55 | 25.17    | 14.5   | 35.11        |
| Self-RAG    | 14.42 | 30.35 | 48.85    | 11.8   | 48.75        |
| InstructRAG | 3.30  | 11.82 | 61.28    | 53.9   | 39.16        |

![Core Metrics Comparison](evaluation/outputs/core_metrics.png)

#### Semantic Similarity Analysis

![Semantic Similarity](evaluation/outputs/semantic_similarity.png)

#### Response Length Comparison

![Response Length](evaluation/outputs/response_length.png)

#### Between Metrics Agreement

![Metrics Agreement](evaluation/outputs/accuracy_agreement_matrix.png)

---

## Discussion

### Key Findings
1. **Method choice changes answer *style*, not just scores.** Self-RAG, Vanilla RAG, and InstructRAG optimize different parts of the pipeline, yielding qualitatively different outputs; EM/F1 alone can miss faithfulness and verification cost.
2. **Self-RAG offers the best balance for concise, checkable answers.** With light tuning, it improved EM/F1 and semantic similarity while keeping length moderate—i.e., grounded answers with lower reviewer burden.
3. **InstructRAG can boost hit-rate but at a cost.** It achieved the highest Accuracy on our setup, yet produced much longer, harder-to-audit responses. In our ICL tests, **one-shot > zero-shot** and **Llama-3 > ft-Llama-3 > Llama-2**.

### Comparison of Approaches

#### Vanilla RAG
- **Strengths:** Simple pipeline; easy to reproduce; fast when retrieval is clean.
- **Weaknesses:** Susceptible to distractors; limited mechanisms to filter or verify evidence.
- **Best use cases:** Domains with high-precision indexes or curated corpora where retrieved passages are already reliable.

#### Self-RAG
- **Strengths:** Adaptive retrieval + reflection helps filter noise; good EM/F1 vs. length trade-off; outputs are concise and grounded.
- **Weaknesses:** More moving parts than vanilla (gating/critique); mild sensitivity to threshold/weights; slightly higher runtime than baseline.
- **Best use cases:** Settings requiring verifiable answers and controlled verbosity under imperfect retrieval.

#### InstructRAG
- **Strengths:** Rationale-style outputs that can improve hit-rate under noisy contexts; benefits from stronger backbones and ICL.
- **Weaknesses:** Longer generations increase verification cost; sensitive to demo quality; FT model did not dominate in our harmonized setup.
- **Best use cases:** Tasks where richer explanations are valued and human review is acceptable (e.g., analysis memos, drafting).

### Limitations
- **Scope:** Evaluation on the **converted HotpotQA dev (distractor)** split only; no training by us and no full-wiki runs.
- **Metrics:** Automated metrics may under/over-value verbosity; limited human evaluation of faithfulness.
- **Implementation bias:** Harmonized data/prompt path (LLaMA-2 centric) may favor certain variants; minimal ablations beyond light Self-RAG tuning.
- **Systems cost:** We did not report latency or $/token; real-world deployment should consider throughput and review overhead.

## Future Work

- **Implement training (beyond inference-only).**
  - Fine-tune Self-RAG’s gating/critique on HotpotQA-style data.
  - Explore InstructRAG SFT with higher-quality rationales; compare to pure ICL.
  - Measure training gains vs. added compute/latency.

- **Improve answer extraction & normalization.**
  - Tighten post-processing for short, canonical answers (numbers, names, dates).
  - Add guardrails for InstructRAG to separate *rationales* from the *final answer*.
  - Expand alias lists and unit normalization to reduce EM brittleness.

- **Broaden model/dataset coverage.**
  - Compare additional RAG variants (e.g., adaptive rerankers, multi-step retrievers).
  - Swap backbones (Llama-3.x sizes, Qwen, Mistral) under the same unified data path.
  - Run **full-wiki** HotpotQA and other multi-hop sets (MuSiQue, 2WikiMultihopQA).

- **Richer evaluation.**
  - Add faithfulness/judging with lightweight human audits or LLM-as-judge (with spot-checks).
  - Track cost/latency vs. accuracy to surface practical deployment trade-offs.

- **Ablations & robustness.**
  - Sensitivity to retrieval threshold, top-k, and critique weights (Self-RAG).
  - Noise stress-tests by injecting distractors; measure degradation curves.

---

## Conclusion
<!--
Summary of:
- What you accomplished
- Key takeaways for practitioners
- Contributions to understanding RAG systems
-->
### What we accomplished
We built a unified, apples-to-apples evaluation pipeline for HotpotQA (distractor setting) and ran four systems—No RAG, Vanilla RAG, Self-RAG, and InstructRAG—on the **full converted dev set** using the same inputs and metrics. Self-RAG delivered the strongest balance across EM/F1 and semantic similarity with concise, verifiable answers, while InstructRAG achieved the highest accuracy but produced much longer outputs. 

### Key takeaways for practitioners
- **Retrieval control matters.** Unfiltered retrieval can add noise; reflective control (as in Self-RAG) helps decide *when* to retrieve and *what* to keep, improving faithfulness and precision.  
- **Match method to metric & workflow.** If you need concise, easy-to-verify answers (EM/F1, short outputs), Self-RAG’s calibrated verbosity is a good fit. If you value broader hit-rates/accuracy with richer rationales, InstructRAG can help—at higher verification cost.  
- **Optimize for verification cost.** Longer rationale chains are costlier to check and more prone to drift; structured retrieval + evidence checks reduce review overhead.

### Contributions to understanding RAG systems
- A head-to-head comparison showing how **reasoning-supervision choices** (fixed retrieval, reflective retrieval, rationale-driven prompting) lead to **qualitatively different outputs** and trade-offs across EM/F1/accuracy/semantic similarity.  
- Clear evidence that **ICL design** (one-shot > zero-shot) and **backbone strength** (Llama-3 > ft-Llama-3 > Llama-2) materially affect InstructRAG in our setup.  
- An argument (with code and results) for **task-aligned evaluation** beyond span-matching—tracking **faithfulness, concision, and stability**—so teams can balance accuracy gains against compute and human-verification effort.






---

## References
<!--
1. HotpotQA paper
2. Self-RAG paper
3. InstructRAG paper
4. Llama 2 paper
5. Other relevant citations
-->
1. Amazon Web Services. *[What is Retrieval-Augmented Generation?][aws-rag]* (Accessed: 2024-12-03).

2. Akari Asai, Xinyang Geng, Matthew E. Peters, Eunsol Choi. *[Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection][selfrag]*. arXiv:2310.11511 (2023).

3. Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks][rag-lewis]*. arXiv:2005.11401 (2020).

4. Chaitanya Sharma. *[Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers][rag-survey]*. arXiv:2506.00054 (2025).

5. Hugo Touvron, Louis Martin, Kevin Stone, et al. *[Llama 2: Open Foundation and Fine-Tuned Chat Models][llama2]*. arXiv:2307.09288 (2023).

6. Sentence Transformers. *[all-MiniLM-L6-v2][minilm]* (2021).

7. Zhepei Wei, Wei-Lin Chen, Yu Meng. *[InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales][instructrag]*. arXiv:2406.13629 (2024).

8. Zhilin Yang, Peng Qi, Saizheng Zhang, et al. *[HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering][hotpotqa]*. arXiv:1809.09600 (2018).

<sub>Adapted from our poster reference list.</sub>

[aws-rag]: https://aws.amazon.com/what-is/retrieval-augmented-generation/
[selfrag]: https://arxiv.org/abs/2310.11511
[rag-lewis]: https://arxiv.org/abs/2005.11401
[rag-survey]: https://arxiv.org/abs/2506.00054
[llama2]: https://arxiv.org/abs/2307.09288
[minilm]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[instructrag]: https://arxiv.org/abs/2406.13629
[hotpotqa]: https://arxiv.org/abs/1809.09600

---

## Code and Resources

- [GitHub Repository:] (https://github.com/judyyhd/NLP_FinalProject_RAG.git)
- **Result Files:** [Judy]
- **Individual Contributions:**
  - No RAG & Vanilla RAG: Judy Yang 
  - Data Chunking & Self-RAG: Ghina Al Shdaifat
  - InstructRAG: Huizhen Jin, Shengduo Li, Yixuan Wang
  - Evaluation & Visualization: Judy Yang
  - Analysis : All Team Members

---

## Appendix

### A. Implementation Details
[Judy]


### B. Example Outputs
**Question:** Are Local H and For Against both from the United States?  
**Gold Answer:** `yes`

**No RAG** *(Accuracy = 0, F1 = 0.00)*  
> No, they are not.

**Vanilla RAG** *(Accuracy = 1, F1 = 0.29)*  
> Yes, both are from the United States.

**InstructRAG** *(Accuracy = 1, F1 = 0.01)*  
<details><summary>Show response</summary>

Yes, based on the information provided in the documents, Local H and For Against are both from the United States.

Document 5 (Title: Local H) states that Local H is an American rock band originally formed by guitarist and vocalist Scott Lucas, bassist Matt Garcia, drummer Joe Daniels, and lead guitarist John Sparkman in Zion, Illinois in 1987. This indicates that Local H is from Illinois, which is a state in the United States.

Similarly, Document 7 (Title: For Against) mentions that the band has produced material steadily from its 1984 inception to the present, which suggests that For Against has been active for a long time and is likely from the United States.

There is no information in the provided documents to suggest that Local H or For Against are from any other country. Therefore, based on the available evidence, it can be concluded that both Local H and For Against are from the United States.
</details>

**Self-RAG** *(Accuracy = 1, F1 = 0.17)*  
> Yes, both Local H and For Against are from the United States.


---

*This project was completed as part of the Fundamentals of Natural Language Processing course at NYU, Fall 2025.*
