# Chapter 3 Writing Notes

Reference for writing the dissertation's Chapter 3 (`thesis/Experiment1/experiment1.tex`). Decisions agreed on 3 Oct 2026; structure revised 4 Oct 2026.

## Decisions

- **Numbering:** the thesis's Experiment 1 (Chapter 3) is the repo's Exp 0. Thesis Exp 2 (Ch4) = repo Exp1 hybrid RAG; thesis Exp 3 (Ch5) = repo Exp2 structure-aware RAG; agentic retrieval = future work in the Conclusion.
- **Title:** "Experiment 1 -- Baseline LLM Performance on Long Financial Documents" (fine for now). Running header shortened to "Experiment 1 -- Baseline LLM Performance" with `\chaptermark`.
- **Structure (4 Oct)**, following the viva template (experiment, data, models, results):
  - 3.1 Introduction (written): the experiment, what it tests, the target result
  - 3.2 Evaluation Benchmark Data (written): 3.2.1 FinanceBench, 3.2.2 Question Taxonomy, 3.2.3 Gold Evidence
  - 3.3 Evaluation Metrics (*what* is measured; the benchmark's four analyses plus cost): 3.3.1 Execution Status, 3.3.2 Answer Accuracy, 3.3.3 Retrieval Metrics, 3.3.4 Failure Diagnosis, 3.3.5 Cost, Latency and Segmentation
  - 3.4 Models (the systems under test): 3.4.1 Answer Model and Prompt, 3.4.2 Context Conditions (merged 4 Oct: one subsection per condition would be too short)
  - 3.5 Benchmark Design and Implementation (*how* each metric is obtained, and the code behind it): 3.5.1 Evaluation Flow and Harness, 3.5.2 Data Preparation and Validation, 3.5.3 Answer Judging
  - 3.6 Evaluation Benchmark Results: 3.6.1 Execution, 3.6.2 Overall, 3.6.3 By Segment
  - 3.7 Discussion: 3.7.1 Limitations, 3.7.2 Future Work
- **Why this structure:**
  - Metrics come before design: 3.3 says only what is measured (one-line definitions); 3.5 says how each is obtained. This avoids describing the judge or the page metrics twice.
  - "Models" means what is actually tested: the LLM in three configurations (no context, oracle pages, whole filing), not the LLM alone. The judge is a measuring instrument, so it sits in 3.5.3, not in Models.
  - Design and implementation share one section: each 3.5 subsection explains the design choice, then ends with one short paragraph on the code that implements it, with an appendix reference. The code stays visible and credited, while detail lives in the appendix (built from `docs sys design/benchmark/FinanceBench Implementation Guide.md`, within the 15-page appendix limit shared by all experiments). Keep each implementation paragraph to the module and the guarantee it enforces.
  - No Summary section: the mini-abstract summarises, and Discussion ends the chapter at 3.7 (Batrinca's "Discussion as §X.7").
- **Mini-abstract:** a study abstract in Batrinca's style (what is investigated, motivation, what is analysed, results), plain terminology, no section roadmap. Formatted with the `chapterabstract` environment in `main.tex` (indented both sides, italic, justified), which Chapter 2 also uses.
- **Scope:** Chapter 3 describes the whole benchmark once (data, all five context conditions, judge, every metric including the retrieval metrics), so Chapters 4 and 5 refer back rather than repeat. Retrieval metrics and failure modes are defined here but reported in Chapter 4, since none of this chapter's conditions retrieves.
- **No significance tests** (no McNemar, no bootstrap): n=112 is too small to support claims of significance (`Benchmark.md` → Statistical power). Report the ±9.3-point 95% margin of error at n=112 instead, and say FinanceBench is used for failure diagnosis, not "A beats B" claims.
- **10-K only:** 3.2.1 says only "The other filing types are left to future work." (no further justification).
- **Mini-abstract wording:** "10-K annual reports", not "annual reports", since FinanceBench's separate `10k_annualreport` type is excluded.
- **Overlap rule:** the Introduction says why (conditions as questions, no dataset detail); 3.2 says what the data is; 3.4 says how each condition is built.
- **Writing order:** 3.4 Models and 3.6 Results first (settled facts and tables from `Exp 0.md`), then 3.3 and 3.5, then 3.7. Each section is shown in chat and approved before it is written.

## Where `Benchmark.md` goes

- **3.1 Introduction:** "Purpose and scope" (a reproducible harness reproducing FinanceBench's context conditions); the FinanceBench section's point that the conditions isolate retrieval failures. Still to add: one sentence on the target result.
- **3.2 Evaluation Benchmark Data:** FinanceBench section (150-question release, question types and their definitions); the exact `10k` filter ("Prepare FinanceBench 10-K data"); the `question_reasoning` normalisation rules; gold evidence as (filing, page) pairs.
- **3.3 Evaluation Metrics** (opens with the four analyses, `Benchmark.md` L182-191)
  - 3.3.1 Execution Status: success, terminal `did_not_fit`, retryable `failed`, `missing`; counts overall and by condition.
  - 3.3.2 Answer Accuracy: binary, from the judge, resolved by manual review on disagreement; reported including and excluding `did_not_fit`.
  - 3.3.3 Retrieval Metrics: page recall, precision and MRR on (filing, page) pairs, pre- and post-rerank; filter accuracy for shared-store. Defined here, reported in Chapter 4.
  - 3.3.4 Failure Diagnosis (moved here from 3.5 on 4 Oct): the failure-mode tree at the top of `Benchmark.md` (retrieval, reasoning, generation) and the oracle-comparison classification rules; results in Chapters 4 and 5.
  - 3.3.5 Cost, Latency and Segmentation: per-answer provenance (requested and returned model, provider, tokens, cost, latency); segments by generation method and cognitive skill, each with its own n.
- **3.4 Models**
  - Opening paragraph: FinanceBench's five conditions; three tested here, single-store and shared-store in Chapter 4.
  - 3.4.1 Answer Model and Prompt: GLM-5.3-Flash via OpenRouter; result-affecting settings (high reasoning, temperature 0, 8,192 output tokens); why GLM (the vals.ai note in "Purpose and scope"); the one shared prompt and `[Document | Page]` labels.
  - 3.4.2 Context Conditions: all five. Closed-book, oracle and long-context (what each receives) are run here; single-store and shared-store need a retriever and are evaluated in Chapters 4 and 5, but are introduced here. Ends with long-context's no-truncation rule (count the assembled prompt; `did_not_fit` as a third outcome).
- **3.5 Benchmark Design and Implementation**
  - 3.5.1 Evaluation Flow and Harness: the staged flow chart as a figure (prepare → validate → construct → retrieve → generate → judge → metrics → report) with "Develop testing set pipeline" as its prose; Python 3.12, `uv`, pinned dependencies; per-question checkpointing, `did_not_fit` terminal while API errors retry; experiment and variant run labels; tests with fake clients.
  - 3.5.2 Data Preparation and Validation: metadata join, raising on the conflicting FOOTLOCKER duplicate; idempotent preparation and `dataset-preparation-record.json`; the six validation checks; smoke (10) and pattern (50) development subsets, seed 42, stratified by `question_type`.
  - **Carried from 3.1 (4 Oct):** 3.1 says briefly why the benchmark was rebuilt and that it also serves the retrieval conditions and Chapter 5's scoring; 3.5 gives the detail:
    - Why FinanceBench's released code was not reused (Zubair's note on `Benchmark.md` L48):
      - it depends on outdated LangChain libraries;
      - its basic RAG pipeline uses LangChain rather than LlamaIndex, and a different vector store from Chroma;
      - it lacks the validation stages (PDFs exist, questions have valid answers, JSON parses);
      - this harness builds the context of all five conditions itself;
      - it has no separate LLM judge;
      - this harness computes retrieval metrics on our own chunks.
    - Single-store and shared-store: how the harness hands a retrieval condition to the retrieval system and gets the retrieved chunks back, and how the page metrics and filter accuracy are computed from them.
    - Chapter 5's structure-aware scoring: it runs through the same retrieval call as a configuration (the structure weight), with page metrics computed both before and after reranking.
  - 3.5.3 Answer Judging: DeepSeek-V4-Flash and why; Zheng et al. two-pass design; 30-answer validation with the 27/30 gate (28/30 achieved); the adjudication rule and its working-capital caveat; manual review flow (export → complete → import → rerun report).
- **Verbatim allocation:** `thesis/Zubair Draft/Chapter 3 - Benchmark.md` (Gate 2) holds every `Benchmark.md` block copied under its subsection, with line tags.
- **3.6 Results:** execution status; accuracy, cost and latency overall; by generation method and by cognitive skill.
- **3.7 Discussion**
  - Interpretation: what the three results mean for retrieval.
  - 3.7.1 Limitations: statistical power (±9.3 points at n=112); one run per condition; human review only of judge disagreements; working-capital definitions; long-context never hit its limit; oracle and long-context use different text sources (dataset page text vs PyMuPDF extraction).
  - 3.7.2 Future Work: HiREC/LOFin; other filing types; LLM-judged retrieval metrics.
- **Not in the thesis:** working notes ("clone both repos, `cat`/`head` the files").

## Appendix to-do

- State that the Experiment 1 results are in `results/20260917-012959--financebench--baseline-context-conditions-v1` (Zubair's note on `Benchmark.md` L1).
- Benchmark harness section built from `docs sys design/benchmark/FinanceBench Implementation Guide.md` (within the 15-page appendix limit shared by all experiments).

## Facts checked (4 Oct)

- The 3M 2018 10-K has 160 pages.
- All 112 questions name their company; eight do so in a short form (AMEX, Coca Cola, JnJ, JPM).
- Generation method by cognitive skill (the table in 3.2.2):
  - all 14 novel-generated questions are unlabelled, so the novel-generated and unlabelled segments are the same questions;
  - all 16 multi-skill questions are domain-relevant;
  - metrics-generated questions have no logical-reasoning label.

## Open questions

- Were the hand-labelled failure sub-types ever done? If not, Chapter 4 states it as a limitation.
- Interval columns in the 3.6 tables (optional; arithmetic on reported counts, no code).
- Should the mini-abstract's motivation sentence cite `li2024ragorlc`?
- `Benchmark.md` line 170: no LLM-judged retrieval metrics, so state the limitation (automated page metrics over-report false negatives) in 3.7.1 or Chapter 4.
- Facts from the code, not the docs, to confirm while drafting 3.4: provider pinned to Z.ai with fallbacks off; the oracle uses FinanceBench's `evidence_text_full_page`, long-context uses PyMuPDF text.
