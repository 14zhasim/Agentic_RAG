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

Might want to include MRR metric, and also token cost per query

(sources for benchmark based off resources in lit review folder: pdf of the benchmark papers, info in matrix, their github repo - summarised under 'benchmarking.md')

## Overall benchmark actions

Clone both repos, `cat`/`head` the actual data files.

Extract questions + corresponding PDFs to meet this criteria:

- Use FinanceBench dataset + questions - only 10Ks for now, add in 10Qs, 8Ks etc. later
- Use Hirec/Lofin dataset + questions - only for those where answer is > 1 pdf, cap say at 200 qs

Develop an all question pipeline:

- use questions mentioned above
- use FinanceBench 5 context conditions below for the retrieval metrics below
- use Zheng et al. to create a proper LLM-as-a-judge/RAGAS for the final answer accuracy metric
- configure results reporting:
  - for FinanceBench: report results segmented across generation method + cognitive skill
  - report page recall, page precision, page MRR. consider doing again using LLM as judge to calculate Context Recall, Context Precision, Faithfulness, Correctness
  - report final answer accuracy (allow rounding, truncation, but binary correct/incorrect)
- start running benchmark for closed-book and oracle stages

Develop a testing pipeline:

- test on maybe subset of 30 questions FinanceBench, 20 questions Hirec? the purpose is run as a test suite for development

## FinanceBench `[VERIFIED] - Islam et al. (2023)`

**Data / what it assesses / how:**

150 question subset with human-graded answers (full dataset 10,231 questions) over 10-K/10-Q/8-K filings (>110 are 10K)

**RAG pipeline failures (specifically, retrieval failures)** are isolated by running the same question under 5 context conditions:

- **closed-book**: no context given - accounts for knowledge in model's parameters
- **oracle**: exact correct page handed to model - reasoning ceiling if retrieval is perfect
- **vector-store**: tests retrieval from vector store, if the only stored filing is the one containing the answer (errors here due to bad chunking - wrong or insufficient recall)
- **shared-store**: vector store contains chunks from ALL companies' filings (error here from finding correct document)
- **long-context**: full document(s) get stuffed directly into the model's context window: tests RAG vs long-context question directly

**Question types** (§3.1, p.3–4. Note - % given from original 10,000 qs, not the 150q subset):

These are the skills these questions test; however, they are WITHIN a single document, on ONE page
But, tests open-domain (have to navigate across all filings)
Each question has the below labels in dataset.

- **By generation method**
  - domain-relevant (9%): generic finance questions applicable to many 10-Ks.
  - novel-generated (12.9%): annotator-written, company/report/industry-specific questions requiring close reading of the filing
  - metrics-generated (78%): retrieving 'typical' metrics from filing eg revenue, ebitda etc. (18 different metrics in total)
- **By cognitive skill**:
  - **Information extraction** (28%): "extracting specific data or textual content"
  - **Numerical reasoning** (66%) — "performing mathematical calculations or comparing numerical data"
  - **Logical reasoning** (6%) — "using logical deductions to evaluate, contrast, or make judgments", still from one filing (even if many pages)

**Question metadata:**
See github README.md - very exhaustive

- gold answer, human-evaluator explanation
- gold answer pdf + page number

**Dataset metrics**:
So, we can label each chunk with its pdf + page number metadata, and compare with gold answer metadata to compute:

- **MRR, Recall, Precision**
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

MAIN TAKEAWAY: _filter questions and filings for:_

- after 2009\_ due to the XBRL thing
- only the multi-document/SEC-QA questions (can filter byt answer/evidence field referencing more than one unique PDF)
- restricted to maybe 200 questions across numeric (table), numeric (text), textual

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
