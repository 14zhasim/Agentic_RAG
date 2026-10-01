# Experiment 0 — LLM Performance (no retrieval)

## Experiment

- **What's being tested:** how well the answer model answers FinanceBench questions on its own, before any retrieval system of ours is involved.
  - Three of FinanceBench's five context conditions, each handing the model a different amount of the filing.
  - No parsing, chunking, indexing or retrieval: the benchmark harness builds the prompt directly from the question and the PDF.
- **Why it is an experiment, not just setup:** it answers the questions every later experiment depends on.
  - What does the model already know without the filing (closed-book)?
  - What is the ceiling if retrieval were perfect (oracle)?
  - How far does a very large context window get on its own (long-context)?
- **Role in the wider project:** Experiment 0 sets the bounds the retrieval experiments are measured against.
  - **Experiment 1** (RAG) targets long-context's accuracy without its context-window limit, and is read against the oracle ceiling.
  - **Experiment 2** (structure-aware RAG) asks whether better retrieval closes the remaining gap to the oracle.
  - Agentic retrieval (formerly Experiment 3) is future work, designed but not built (`Exp 3/Exp 3.md`).
- **Naming:** "Experiment 0" is the dissertation's name for this run. In the code and results folders it is the benchmark's baseline run, labelled `financebench` / `baseline-context-conditions-v1`; the label is kept so the folder still traces to the commit that produced it.

## Data

Identical to Experiments 1 and 2 (full detail in `Exp 1/Exp 1.md` → Data):

- FinanceBench (Islam et al., 2023), 10-K subset: 112 questions across 64 10-K PDFs (`doc_type == "10k"` exactly).
- By generation method: metrics-generated 50, domain-relevant 48, novel-generated 14.
- By cognitive skill (normalised; sums to 128 because 16 questions carry more than one): numerical reasoning 57, information extraction 36, logical reasoning 21, unlabelled 14.
- Every question sits inside one filing, with gold evidence as `(doc_name, evidence_page_num)` pairs (zero-indexed pages).

## Models / architecture

- **Answer model:** `z-ai/glm-5.3-flash` via OpenRouter, reasoning effort `high`, up to 8,192 output tokens, temperature 0. The same model, settings and answer prompt are used in every condition of every experiment, so only the context differs.
- **Judge:** DeepSeek-V4-Flash via Azure Foundry, reference-guided, graded twice with answer order swapped, binary (Zheng et al., 2024). Validated before any benchmark run against 30 published FinanceBench human labels (15 correct, 10 incorrect, 5 refusals): 28/30 agreement (93.3%), above the 27/30 (90%) gate (`benchmark/FinanceBench Implementation Guide.md`).
- **No retrieval system.** The harness itself assembles each prompt, so this experiment measures the model and the context it is given, nothing we built.

## System design — the three conditions

All three use the same answer prompt; the context block uses the same `[Document | Page]` shape as the retrieval experiments, so formatting cannot explain a gap between conditions.

- **Closed-book:** the question alone, no filing. Measures what the model already knows from its training data.
- **Oracle:** the question plus the exact gold evidence pages. Measures the reasoning ceiling: what the model achieves when retrieval is perfect.
- **Long-context:** the question plus the whole filing's text. Tests whether a large context window makes retrieval unnecessary.
  - Prompts are token-counted in full (system message, instructions, question and document) with the model's own tokenizer, and a prompt that does not fit is recorded as `did_not_fit` rather than truncated. None did: the largest prompt was 535,722 tokens of GLM-5.3-flash's 1,048,576.
- The other two FinanceBench conditions, single-store and shared-store, need a retriever: they are Experiment 1.

## Metrics

- **Answer accuracy** from the validated judge, binary. Disagreements between the two judge passes are adjudicated by hand under the rule in `Benchmark.md` (judge section, "adjudication rule"); 16 of the 336 answers needed it.
- **Cost and latency** per answer: requested and returned model, provider, token usage, cost, latency.
- **Segmented** by generation method and cognitive skill, each segment with its own *n*.
- No retrieval metrics: none of the three conditions retrieves.

## Baseline

Experiment 0 is itself the reference point; it has no baseline of its own. Its three conditions bound the later experiments:

- **floor:** closed-book
- **ceiling:** oracle
- **target to match or beat without a context-window limit:** long-context

## Results

Run: `results/20260917-012959--financebench--baseline-context-conditions-v1`, 336 jobs (112 questions × 3 conditions), all successful, none `did_not_fit`, every judge disagreement adjudicated.

**Overall, per condition**

| Condition | Answer accuracy | Cost per answer | Latency | n |
|---|---|---|---|---|
| closed-book | 41.1% (46/112) | $0.0003 | 7.7 s | 112 |
| oracle | 91.1% (102/112) | $0.0003 | 4.2 s | 112 |
| long-context | 85.7% (96/112) | $0.0152 | 8.3 s | 112 |

**By generation method** (answer accuracy)

| Segment | n | closed-book | oracle | long-context |
|---|---|---|---|---|
| metrics-generated | 50 | 28.0% | 96.0% | 94.0% |
| domain-relevant | 48 | 56.3% | 83.3% | 79.2% |
| novel-generated | 14 | 35.7% | 100% | 78.6% |

**By cognitive skill** (answer accuracy)

| Segment | n | closed-book | oracle | long-context |
|---|---|---|---|---|
| Numerical reasoning | 57 | 35.1% | 89.5% | 89.5% |
| Information extraction | 36 | 47.2% | 94.4% | 83.3% |
| Logical reasoning | 21 | 57.1% | 71.4% | 71.4% |
| unlabelled | 14 | 35.7% | 100% | 78.6% |

What this shows:

- **The model needs the filing.** Closed-book reaches 41.1%, and only 28% on metrics-generated questions, which ask for a specific company's figures that the model cannot know. Domain-relevant questions fare better closed-book (56.3%) because many are generic judgements a model can make without the document.
- **With perfect evidence the model is strong.** The oracle reaches 91.1%. The 10 questions it still gets wrong are reasoning or generation failures, not retrieval ones, and they recur in Experiment 1's failure analysis.
- **A huge context window gets most of the way, at a price.** Long-context reaches 85.7%, 5.4 points below the oracle, with every filing fitting. But it costs about 50 times as much per answer as the oracle ($0.0152 against $0.0003), because the whole filing is sent every time, and it only works while a filing fits the window.
- **Logical reasoning is the model's weak spot, whatever the context.** The oracle and long-context both score 71.4% there, so better retrieval cannot lift it; this matters for reading Experiments 1 and 2.

**Limitations of these results**

- One run per condition; the answer model is re-run per job, so a repeat would change some verdicts (Experiment 2 puts run-to-run variation at roughly 18 of 224 answers).
- Human review covers only answers where the two judge passes disagreed.
- FinanceBench's working-capital gold answers use inconsistent definitions (see `Benchmark.md` → adjudication rule).
- Every 10-K fits GLM-5.3-flash's 1M-token window, so long-context never hits its limit here. The case against it on this benchmark is cost and scale, not failure to fit.
