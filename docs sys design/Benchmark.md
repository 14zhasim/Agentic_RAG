## Pipeline failure modes

Create an evaluation pipeline where every benchmark question can be inspected:
Question
│
├── Retrieved chunks
│
├── Retrieval scores
│
├── Reranked chunks
│
├── Final context
│
├── Generated answer
│
├── Expected answer
│
└── Error classification

Then classify failures:

Retrieval failure
├── wrong document
├── wrong section
├── wrong chunk
├── missed table
└── insufficient recall

Reasoning failure
├── arithmetic
├── comparison
├── multi-hop reasoning
└── temporal reasoning

Generation failure
├── unsupported claim
├── wrong number
├── hallucination
└── formatting


(sources for benchmark based off resources in lit review folder: pdf of the benchmark papers, info in matrix, their github repo - summarised under 'benchmarking.md')

## Purpose and scope

Build a reproducible, plain-Python evaluation harness for the open-source FinanceBench questions whose documents are classified as 10-Ks. The harness will reproduce FinanceBench's five context conditions, support replacement retrieval pipelines, calculate deterministic page-retrieval metrics, generate answers with GLM-5.3-Flash, and later judge those answers with GPT-5.6 Luna.

For the current MVP, use the cloned FinanceBench repository as read-only source
material. Deterministically prepare:
- a project-owned, generated 10-K subset
- while preserving both original JSONL schemas from dataset 
- The prepared subset and PDFs remain out
of Git and can be recreated with a command.

Answer generation uses `z-ai/glm-5.3-flash` through OpenRouter. Final-answer judging will use `gpt-5.6-luna` through a Direct-from-Azure Microsoft Foundry deployment. Due to accuracy/cost efficiency of both models, as per https://www.vals.ai/benchmarks/fab v2

The initial dataset contains 112 questions across 64 PDFs. FinanceBench 10-Qs, 8-Ks, earnings documents.


## Overall benchmark actions

The staged flow is:

```text
prepare data
    ↓
validate 112 questions and 64 PDFs
    ↓
construct condition inputs for 5 testing conditions
    ↓
retrieve where applicable
    ↓
generate and checkpoint answers
    ↓
judge saved answers separately
    ↓
calculate and aggregate metrics
    ↓
write reports
```


Clone both repos, `cat`/`head` the actual data files.

Use Python 3.12 and `uv`. Pin direct dependency versions in `pyproject.toml` and
commit the complete `uv.lock`. 

Extract questions + corresponding PDFs to meet this criteria:

- Use FinanceBench dataset + questions - only 10Ks for now, add in 10Qs, 8Ks etc. later (150 qs -> 112 qs across 64 PDFs)
- Use Hirec/Lofin dataset + questions - only for those where answer is > 1 pdf, cap say at 200 qs
- Include full Hirec dataset to test for statistical significance

Prepare FinanceBench 10-K data:

- Read the two source files `financebench_open_source.jsonl` and `financebench_document_information.jsonl`.
- Join document metadata to questions by `doc_name` in memory.
- Filter on `doc_type == "10k"` exactly — the value is lowercase, and a separate `10k_annualreport` type exists (6 docs). A `startswith`/case-insensitive match silently admits annual reports and breaks the 112.
- The metadata file has 361 rows but only 360 unique `doc_name`s: `FOOTLOCKER_2023_annualreport` appears twice with conflicting `doc_period` (2023 vs 2022). A naive `{doc_name: row}` join silently keeps the last row — the join must raise on a conflicting duplicate instead.
- Make preparation idempotent, so running it repeatedly produces the same prepared dataset without duplicate records.
- Create a manifest that records:
  - source;
  - filter (i.e just 10-K filings);
  - expected and observed counts;
  - selected PDF filenames;
  - file hashes.

Validate the question set
_- `financebench_id` is unique._
_- Every selected question resolves to unambiguous document metadata._
_- Every selected PDF exists._
_- Evidence document names and zero-indexed page numbers are usable._
_- The complete evidence list is retained for every question._
_- Final counts are 112 questions and 64 PDFs._

Gold evidence should still be keyed as `(doc_name, page_num)` pairs rather than bare page numbers.

Develop testing set pipeline:

- use questions mentioned above + context from PDF - but include checks to ensure fits in cotext window. Dont truncate prompts or only beginning of filing' if prompt cannot fit, it fails
  - count tokens over the **assembled prompt** (system message + instructions + question + document text), not the filing alone, using the tokenizer of the model actually being called, with a configured reserve for the completion
  - a prompt that does not fit is recorded as a third outcome, `did_not_fit` — not correct, not incorrect, not an error. This keeps _n_ constant across conditions (so the paired McNemar comparison still works) and keeps the evidence for the claim that long-context is limited by context window size. Report accuracy both including and excluding these rows.
  - `did_not_fit` is terminal and must not be retried on resume; API errors are transient and must be retried
- integrate FinanceBench 5 context conditions (read section below) into our pipeline, for the retrieval metrics below, using placeholders for our RAG pipeline
- fix result-affecting settings before paid benchmark runs: use `high` GLM reasoning effort, reserve up to 8,192 output tokens for reasoning plus the visible answer, and retrieve the top 10 chunks for `single_store` and `shared_store`
- label every run with an explicit experiment and variant; retain both labels in its directory name, configuration snapshot and reports
- configure results reporting:
  - for FinanceBench: report results segmented across generation method + cognitive skill + condition, with sample count for each segment
    - `question_type` is clean and used as-is: metrics-generated (50), domain-relevant (48), novel-generated (14)
    - `question_reasoning` is **not** clean. Normalise to the paper's 3 skills: split case-insensitively on a standalone `OR`, discard only the known trailing empty part, casefold-compare complete labels, fold `Logical reasoning (based on numerical reasoning)` into Logical reasoning only, deduplicate skills within a question, keep `None` visible as an `unlabelled` segment, and raise on anything unmapped so the taxonomy can't silently grow.
    - after normalising: Numerical reasoning 57, Information extraction 36, Logical reasoning 21, unlabelled 14 — summing to 128, since 16 questions carry >1 skill. Segment counts exceeding the sample size is expected and is why each segment reports its own _n_.
  - with this segmentation, report: page recall, page precision, page MRR (count number of unique pages retrieved, MRR is first chunk from golden page)
  - for each answer, record:
    - requested model;
    - returned model;
    - serving provider;
    - token usage;
    - latency;
    - cost.
- use Zheng et al. to create a two-pass LLM-as-a-judge for the final answer accuracy metric
  - use DeepSeek-V4-Flash through Azure Foundry
  - provide it: question, reference answer + evidence + human labeller's justification, candidate answer
  - report final answer accuracy (allow rounding, truncation, but binary correct/incorrect)
- Zheng et al. (2024) - gpt judge agrees almost as much as human, but use different model from judging to generating, give judge correct reference answer BEFORE it grades, grade twice with answer order swapped and only trust verdict both times agreed on. have it putput a 1 or 0 for correct or not
  - before the full benchmark, validate the judge against 30 published FinanceBench human-labelled answers: 15 correct, 10 incorrect and 5 refusals, using 30 unique questions, all three `question_type` categories, multiple published model/condition files and a fixed seed
  - require at least 27/30 (90%) agreement with the human labels; count a two-pass disagreement as a failed match, report the three label groups separately and manually inspect every mismatch
  - reproduce the validation source by cloning the official FinanceBench repository at commit `cc39aeb4afdf33909ee1412188bf89035950c2eb`, while keeping the clone ignored by this repository
  - consider doing again using LLM as judge to calculate retrievla metrics: Context Recall, Context Precision, Faithfulness, Correctness. If not, add as a methodology limitation on accuracy of automated retrieval metrics calculations (higher reported false negatives than in reality)
- start running benchmark for closed-book and oracle stages

Develop a validation set pipeline (for purpose of debugging pipeline works):

- smoke test (5-10 questions): check pipeline works
- pattern check (50 questions): identify patterns across segmented question types e.g. does chunking table work? on maybe subset of 30 questions FinanceBench, 20 questions Hirec? the purpose is run as a test suite for development

## NEED TO ADD: Statistical power — what our sample sizes can actually support

**Formula that matters:** margin of error on a proportion (95% CI) ≈ `1.96 × sqrt(p(1−p)/n)`. To detect a real difference `d` between two systems at 80% power (independent samples): `n ≈ 7.84 × [p1(1−p1) + p2(1−p2)] / d²` per arm.

**What our current sample sizes allow:**

| n                               | Margin of error | Reliably detects                                                            |
| ------------------------------- | --------------- | --------------------------------------------------------------------------- |
| 112 (FinanceBench, 10-K only)   | ±9.3pp          | Only large/dramatic effects (~20pp+ gap, or a heavily skewed McNemar split) |
| 150 (FinanceBench, full subset) | ±8pp            | Same — marginally better, not meaningfully                                  |
| 1,389 (LOFin-1.4k)              | ±2.6pp          | Realistic 5pp improvements                                                  |

A 45% vs. 53% comparison at n=112–150 will likely have overlapping CIs — not statistically defensible even if the number looks better.

**What size is ideal, and for which test:**

- **Detecting a 5–10pp improvement, independent-samples comparison:** ~390–1,550 questions per arm. Neither FinanceBench subset gets there; LOFin-1.4k does for the 5pp case.
- **Paired comparison (McNemar's, same questions through both pipeline variants — what we're actually doing):** power depends on _discordant_ pairs, not total n. At ~20–30% discordance, n=112–150 yields only ~25–45 discordant pairs — need ~200 for a modest (60/40) real effect. Only a dramatic (~75/25) split on disagreements would show up reliably.

**Practical split:** use FinanceBench (112/150) for **qualitative failure-mode diagnosis** — which taxonomy category breaks, does the pattern make sense — not for confident "system A beats system B" claims. Use **LOFin for any quantitative significance claim.** Report bootstrap CIs alongside point accuracy everywhere, not accuracy alone.

---

## FinanceBench `[VERIFIED] - Islam et al. (2023)`

**Data / what it assesses / how:**

150 question subset with human-graded answers (full dataset 10,231 questions) over 10-K/10-Q/8-K filings (>110 are 10K)

**RAG pipeline failures (specifically, retrieval failures)** are isolated by running the same question under 5 context conditions:

- **closed-book**: no context given - accounts for knowledge in model's parameters
- **oracle**: exact correct pages handed to model - reasoning ceiling if retrieval is perfect
- **long-context**: full document(s) get stuffed directly into the model's context window: tests RAG vs long-context question directly
- **REQUIRE RETRIEVER:**
- **vector-store**: tests retrieval from vector store, if the only stored filing is the one containing the answer (errors here due to bad chunking - wrong or insufficient recall)
- **shared-store**: vector store contains chunks from ALL companies' filings (error here from finding correct document)

**Question types** (§3.1, p.3–4. Note - % given from original 10,000 qs, not the 150q subset):

These are the skills these questions test; however, they are WITHIN a single document, on one or more pages
But, tests open-domain (have to navigate across all filings)
Each question has the below labels in dataset.

- **By generation method**
  - domain-relevant (9%): generic finance questions applicable to many 10-Ks.
  - novel-generated (12.9%): annotator-written, company/report/industry-specific questions requiring close reading of the filing
  - metrics-generated (78%): retrieving 'typical' metrics from filing eg revenue, ebitda etc. (18 different metrics in total)
- **By cognitive skills (can have >1 label)**:
  - **Information extraction** (28%): "extracting specific data or textual content"
  - **Numerical reasoning** (66%) — "performing mathematical calculations or comparing numerical data"
  - **Logical reasoning** (6%) — "using logical deductions to evaluate, contrast, or make judgments", still from one filing (even if many pages)

**Question metadata:**
See github README.md - very exhaustive

- gold answer, human-evaluator explanation
- gold answer pdf + page number

**Dataset metrics**:
So, we can label each chunk with its pdf + page number metadata, and compare with gold answer metadata (doc_name, evidence_page_num) to compute:

- **MRR, Recall, Precision**
  - page recall: retrieved gold pages / all gold pages
  - page precision: retrieved gold pages / all unique retrieved pages
  - page MRR: 1 / rank of first chunk containing a gold page
  - we can _segment across both question types_, as long as we say how many questions belong to each
  - given it provides 'evidence_text' - we could even give to **LLM as judge + RAGAS library** to computer these metrics as well e.g. retrieves from different page, but still correct
    - can compute Context Recall, Context Precision, Faithfulness, Correctness (looks at each retrieved chunk vs evidence text, see if it is relevant or not)
- **overall answer accuracy**
  - can give gold answer + human-evaluator explanation to LLM-as-judge to compute
- **Token cost**: also wanna compute this

**Limitations**:

- restricted to SEC filings in dataset, may have more issues from having same company filings over last 10 years that we have not accounted for
- However, we do give option for metadata filtering for company during search, so is somewhat mitigated

**Reproducibility:** Public GitHub repo includes data + 5-condition harness. No LLM as judge. https://github.com/patronus-ai/financebench

---

## HiREC / LOFin `[VERIFIED]`

MAIN TAKEAWAY: _filter questions and filings for:_

- after 2009\_ due to the XBRL thing
- only the multi-document/SEC-QA questions (can filter by the answer/evidence field referencing more than one unique PDF)
- restricted to maybe 200 questions across numeric (table), numeric (text), textual

**Data / what it assesses / how:**
LOFin benchmark — **145,897 SEC filings** (10-K/10-Q/8-K, Oct 2001–Apr 2025 (p.16665)), **516 S&P 500 companies**, **1,595 open-domain QA pairs** (main experiments use LOFin-1.4k = 1,389 pairs).

Sourced from three places (Table 9, p.16674):

- single-document questions:
  - **FinQA-derived (~70%)**: single-document. Designed closed-domain — assuming the correct filing is already in hand — but LOFin repurposes those same questions to be open-domain (within the benchmark's filings, not open-domain)
  - **FinanceBench (~9%)**.: see features named above - not explicity however
- multi-document questions:
  - **SEC-QA-derived (~21%)**: explicitly multi-document/multi-page templates — dividend-over-years, cross-company percentage-difference, multi-year revenue-growth, superlative-among-companies (Appendix B.4, p.16675).

**Question types — only three formally-reported categories, confirmed as the complete taxonomy** (p.16667–16668):

- 'Open-domain' to the extent of only SEC filings in the dataset, not unbounded web search
- **Numeric (Table)**: "the answer is a number from a table or can be calculated from numbers in tables"
- **Numeric (Text)**: "a number derived by extracting and combining numerical information from text instead of a table"
- **Textual**: "a textual explanation"

**Question metadata:**

- question
- gold answer
- pdf source + page number(s), >1 of each where evidence spans more than one page

**Dataset metrics** (Table 2, p.16668):

- **Page Recall**
- **Page Precision**
  - both done deterministically
- **Answer Accuracy**
  - with gold-evidence oracle (perfect retrieval), answer accuracy capped at 65.9%
  - done with LLM as judge - prompts given e.g. allow rounding, truncation, but binary correct/incorrect
- **_k_** (average passages used)

**Reproducibility:** https://github.com/JaeyoungChoe/LOFin-bench-HiREC — dataset also on HuggingFace.
                                           
