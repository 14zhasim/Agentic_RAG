# Benchmark Progress — Repo Investigation Notes

Findings from actually reading `benchmarks/financebench` and `benchmarks/lofin-hirec`, mapped against the `Overall benchmark actions` checklist in `Benchmark.md`.

make sure to pin all obrary versions in a package manager; sort out repo file structure
see if can add in azure or openrouter
dataset contains questions more than one page as rght answer!!!
only gonna do financebench rn, will do Hirec if i have time

Important weaknesses
This is a rough safety mechanism rather than robust context management.

- It checks both tokenizers even though each experiment uses only one provider. An OpenAI run could therefore be shortened because of the Anthropic limit.
- It always uses the tokenizer for gpt-4-1106-preview, even when the selected OpenAI model is GPT-4o or GPT-4.
- It measures the filing alone, not the final prompt containing the question and instructions.
- It does not reserve space for the model’s output.
- It keeps only the beginning of an oversized filing. If the relevant information appears near the end, it is lost.
- The OpenAI branch measures decoded bytes and later slices Python characters; those are not guaranteed to be identical for non-ASCII text.

## "Use FinanceBench dataset + questions - only 10Ks for now"

Real files: `benchmarks/financebench/data/financebench_open_source.jsonl` (150 rows) + `financebench_document_information.jsonl` (doc_type per filing). Join on `doc_name`, filter `doc_type == "10k"` → 112 rows (their own README gives the exact pandas join). Fields per question: `question`, `answer`, `justification`, `question_type`, `question_reasoning`, `evidence[].{evidence_text, doc_name, evidence_page_num, evidence_text_full_page}`.

⚠️ `evidence_page_num` is zero-indexed (stated in their README) — get this wrong and your oracle condition hands the model the wrong page.

## "Use Hirec/Lofin dataset - only >1 pdf, cap ~200 qs"

The git clone's `data/` folder is a dead end — it only has an encryption tool. Their README says as of 2026-05-21 the dataset moved fully open to Hugging Face: `HYdsl/Open-domain_Financial_QA`. No decryption needed, ignore `encrypt_decrypt.py`. PDFs are on a separate Google Drive link in the same README.

Don't hand-filter for ">1 PDF" — LOFin already ships this pre-split. Their `by_data_source` breakdown has a **SEC-QA-derived** subset that IS the multi-document category (per your own Benchmark.md note, ~21% of 1,595 ≈ 335 rows). Pull that split directly off HF, sample down to 200 with a fixed seed.

## "Use FinanceBench 5 context conditions for retrieval metrics"

Don't build this from scratch — `benchmarks/financebench/evaluation_playground.ipynb` already implements all 5 (closedBook/oracle/singleStore/sharedStore/inContext) with LangChain+Chroma. Read it, then swap their retrieval/generation calls for yours. `results/` folder has the original paper's own outputs per condition for GPT-4, GPT-4-1106, Claude-2, Llama2 — use those as your comparison table, don't recompute them.

## "Use Zheng et al. for LLM-as-judge / final answer accuracy"

This is the one piece nothing hands you — write it yourself. Reference-guided grading (gold answer + justification field into the judge prompt) + position-swap if you do any pairwise comparison.

## "Configure results reporting"

`question_type` and `question_reasoning` are already columns in the jsonl — segmenting by them is a groupby, not new code.

## "Start running closed-book and oracle"

Doable today: filtered 112-row set + the notebook's closedBook/oracle cells, no pipeline of your own required yet.

## "Testing pipeline — 30 FinanceBench, 20 Hirec subset"

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
