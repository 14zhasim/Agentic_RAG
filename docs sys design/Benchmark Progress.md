# Benchmark Progress — Implemented vs planned

Updated: 7 September 2026

## Implemented or verified

- Cloned and inspected the FinanceBench and LOFin/HiREC reference repositories under `benchmarks/`.
- Read FinanceBench's two source JSONL files, PDF collection, evaluation notebook, and published result files.
- Verified that an exact normalized `doc_type == "10k"` filter selects **112 questions across 64 PDFs**.
- Verified that the 112 selected questions have unique `financebench_id` values and matching local PDFs.
- Verified that FinanceBench evidence is a list and can contain multiple gold pages; its `evidence_page_num` is zero-indexed.
- Verified that the document-information file contains one duplicated annual-report `doc_name` with conflicting periods. It is not referenced by the open-source questions, but the planned loader will still validate metadata before joining.
- Verified the behavior and weaknesses of the original five-condition notebook. It generates answers but does not calculate retrieval metrics or answer accuracy.
- Verified that FinanceBench's published `Correct Answer`/`Incorrect Answer` values are human annotations rather than the output of repository scoring code.
- Inspected HiREC's numeric scorer and GPT-judge implementation as reference material.
- Approved the FinanceBench harness design: OpenRouter/GLM generation, a future Azure Foundry/GPT judge, deterministic 10-K data preparation, five condition builders, page-level metrics, segmented reporting, a named Python package, resumable run artifacts, and minimum correctness checks.
- Recorded the approved design in `docs/superpowers/specs/2026-09-07-financebench-benchmark-harness-design.md` and integrated its actions into `Benchmark.md`.

No benchmark harness code, prepared project-owned dataset, vector store, model integration, metric implementation, or automated test has been implemented yet.

## Planned next

1. Create the `uv` project, pin Python and dependencies, and add safe environment-variable configuration.
2. Implement deterministic FinanceBench preparation and validation while preserving both original JSONL schemas.
3. Implement the shared records/interfaces for condition context, generation, retrieval results, and run outputs.
4. Implement closed-book, oracle, and long-context condition builders.
5. Add the OpenRouter GLM-5.3-Flash generator and resumable per-question output.
6. Implement deterministic page recall, page precision, page MRR, cognitive-skill normalization, and reporting.
7. Add the replaceable retriever interface; implement `single_store` and `shared_store` when the retrieval baseline is selected.
8. Deploy GPT-5.6 Luna as a Direct-from-Azure Microsoft Foundry model under the sponsored subscription, verify that a small request consumes startup credits, and then implement the separate judge stage.
9. Validate the LLM judge against a human-reviewed sample.
10. Add the broader smoke-test and pattern-check pipeline after the harness works end to end.

## Deferred scope

- Selecting and implementing the dissertation retrieval stack, chunking strategy, vector store, hybrid retrieval, and reranker.
- Final generation and judge hyperparameters and prompt wording.
- Optional RAGAS context recall, context precision, faithfulness, and correctness measurements.
- LOFin/HiREC data preparation and multi-document evaluation.

## Reference-repository investigation notes

Findings from actually reading `benchmarks/financebench` and `benchmarks/lofin-hirec`, mapped against the `Overall benchmark actions` checklist in `Benchmark.md`.

Important weaknesses
This is a rough safety mechanism rather than robust context management.

- It checks both tokenizers even though each experiment uses only one provider. An OpenAI run could therefore be shortened because of the Anthropic limit.
- It always uses the tokenizer for gpt-4-1106-preview, even when the selected OpenAI model is GPT-4o or GPT-4.
- It measures the filing alone, not the final prompt containing the question and instructions.
- It does not reserve space for the model’s output.
- It keeps only the beginning of an oversized filing. If the relevant information appears near the end, it is lost.
- The OpenAI branch measures decoded bytes and later slices Python characters; those are not guaranteed to be identical for non-ASCII text.

### "Use FinanceBench dataset + questions - only 10Ks for now"

Real files: `benchmarks/financebench/data/financebench_open_source.jsonl` (150 rows) + `financebench_document_information.jsonl` (document metadata). Resolve metadata by `doc_name`, normalize `doc_type`, and filter `doc_type == "10k"` → 112 rows and 64 PDFs. Fields per question include `question`, `answer`, `justification`, `question_type`, `question_reasoning`, and `evidence[].{evidence_text, evidence_doc_name, evidence_page_num, evidence_text_full_page}`.

⚠️ `evidence_page_num` is zero-indexed (stated in their README) — get this wrong and your oracle condition hands the model the wrong page.

### "Use FinanceBench 5 context conditions for retrieval metrics"

Use `benchmarks/financebench/evaluation_playground.ipynb` as the behavioral baseline for the five conditions (closedBook/oracle/singleStore/sharedStore/inContext), but implement new condition, retrieval, and generation interfaces in our package. The reference repository's `results/` folder contains the paper's outputs for GPT-4, GPT-4-1106, Claude-2, and Llama2 and can be used for format/comparison checks.

### "Use Zheng et al. for LLM-as-judge / final answer accuracy"

This is the one piece nothing hands you — write it yourself. Reference-guided grading (gold answer + justification field into the judge prompt) + position-swap if you do any pairwise comparison.

### "Configure results reporting"

`question_type` and `question_reasoning` are already columns in the JSONL. Preserve both raw values. Because `question_reasoning` contains inconsistent composite alternatives, create a transparent multi-label normalization before cognitive-skill aggregation.

### "Start running closed-book and oracle"

Implement these as first-class conditions in our harness. They can run once the prepared dataset, shared generator, prompt configuration, and OpenRouter key are in place; they do not depend on the future retriever.

### "Testing pipeline — 30 FinanceBench, 20 Hirec subset"

Sample with a fixed seed, stratified by `question_type` so all three categories (metrics-generated/domain-relevant/novel-generated) appear in your dev loop.

Ignore `main_process_iter.py`/`finrag_serving` in the lofin-hirec repo entirely — that's HiREC's own GPU-based system, not the benchmark harness. You only need their dataset + PDFs.

## Does LOFin/HiREC use RAGAS or have an LLM-as-judge to learn from?

**No RAGAS.** Not in `requirements.txt`, not imported anywhere. All homegrown.

**Yes, it has an LLM-as-judge — and it's simple.** `finrag_api/modules/generator.py`, method `calculate_gpt_accuracy` (~line 190):

```python
system_prompt = """You are a financial expert, you are supposed to evaluate if the generated answer is correct based on the given context and question.
You need to respond with a JSON object with the following format:
{"score": 1.0 or 0.0, "reason": "..."}"""

user_prompt = f"""Context: {context}
Question: {question}
Answer: {answer}
Generated Answer: {generated}

Please evaluate if the generated answer is correct based on the given context and question."""
```

Single GPT call, temperature 0.0, binary 1.0/0.0 output with a reason string.

Two things worth learning from, one thing not to copy:

- **Learn from:** they route by question type first — `evaluate_answer` (line 301) checks `answer_type == "pot"` (numeric) and only calls the GPT judge for _text_ answers. Numeric answers skip the judge entirely and go through `calculate_numeric_accuracy` instead — string-to-float, strip `%`/commas, round to the gold answer's own significant-figure count, compare with `np.isclose`. That significant-figure-matching trick is a clean, citable way to implement "allow rounding, truncation" from your own checklist — worth lifting as-is.
- **Don't copy:** the judge prompt itself is bare — no reference-guided grading structure beyond dumping the gold answer in, no position-swap (not applicable here, it's pointwise not pairwise), no chain-of-thought step. This is exactly the gap Zheng's paper is for. Use their routing logic (numeric vs. text) but write your own judge prompt with Zheng's reference-guided framing.

**So:** use their numeric/text split + significant-figure rounding, replace their judge prompt with your own Zheng-informed one.
