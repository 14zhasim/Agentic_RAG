# Systems Design Draft Notes

\--

Retrieving files directly

<https://www.sec.gov/answers/reada10k.htm>

also check downloaded pdf file

Yes. If your goal is **SEC filings and learning how companies actually
tag financial statements**, I'd focus on **XBRL US + SEC + XBRL
International**, in that order.

This site (<https://data.sec.gov>) provides public access to the U.S.
Securities and Exchange Commission\'s EDGAR data via Application
Programming Interfaces (API). Automated access must comply with
SEC.gov\'s Privacy and Security Policy.

Please visit <https://www.sec.gov/edgar/sec-api-documentation> for
documentation. More developer resources and Fair Access guidelines can
be found at <https://www.sec.gov/developer>

<https://www.sec.gov/search-filings/edgar-application-programming-interfaces>

\-\--

**[Retrieving Filings and their Metadata]**

**Identifiers:**

- Accession number: identifies one specific filing (e.g.
  0000320193-23-000106), dashes stripped for the URL

**CIK → accession number (and other important metadata)):**

1.  CIK: SEC\'s company ID, zero-padded to 10 digits (Apple: 0000320193)

2.  Ticker → CIK via sec.gov/files/company_tickers.json

3.  CIK → accession number via
    data.sec.gov/submissions/CIK##########.json. It returns v useful
    fields such as
    1.  **cik** --- the company identifier, stable for the company\'s
        life

    2.  **accessionNumber** --- unique ID for this specific filing

    3.  **formType/form** --- \"10-K\", \"10-Q\", \"8-K\", etc.
        (amendments get a /A suffix, e.g. \"10-K/A\")

    4.  filter parallel arrays for form == \"10-K\", read
        accessionNumber + primaryDocument at that index

    5.  **reportDate** --- the actual fiscal period end this filing
        covers (e.g. a 10-K filed in November 2023 might have reportDate
        of 2023-09-30, the fiscal year-end, not the filing date). **This
        is the field you actually want when matching a question like
        \"what was revenue in FY2023,\" not filingDate.**

    6.  **primaryDocument** --- the filename you use to construct the
        actual document URL

    7.  **isXBRL** --- boolean, whether this filing has inline XBRL

**HTML, not PDF:**

- 10-Ks are HTML, tables have real \<table\>/\<tr\>/\<td\> structure
  already, unlike PDF

- Since 2019, XBRL is embedded directly in that same HTML as inline
  XBRL (iXBRL)

**XBRL:** tags specific numbers with standardized concepts (e.g.
us-gaap:Revenues) so they\'re comparable across companies/filings -- all
companies mandates from 2009-11

**Table formatting:**

Table

↓

row labels

↓

column headers

↓

footnote

↓

preceding paragraph

**Pipeline pattern to adopt:** fetch → cache to disk → transform/embed →
store

**Parsing methods investigated:**

- sec-parser: unmaintained, 10-K secondary. Not recommended.

- Unstructured.io: PDF-oriented, dependency friction.

- BeautifulSoup classifier: table/text fine, heading detection
  unresolved (bold via inline CSS, not \<b\>/\<strong\>).

May just opt for parsing pdfs instead of html, given that the former is
way more applicable to industry practice than working with html
reporting e.g. financial reports of private companies.

\-\--

**The big one: every 10-K follows the exact same fixed structure, by
law**

Every 10-K, from any company, is organized into the same four Parts with
the same numbered Items in the same order, because the SEC mandates this
template. You can build heading-detection logic around matching these
specific Item numbers/names as known anchors, rather than relying purely
on styling heuristics (bold, font-size) that you\'ve seen already vary
unpredictably between filers.

**The full structure, with what\'s actually useful vs. low-value:**

**Part I** --- mostly qualitative, describes the business

- Item 1, Business --- company overview

- Item 1A, Risk Factors --- high value for your project; explicitly
  the risk-factor content your Phase 3 list flagged

- Item 1B, Unresolved Staff Comments --- almost always
  boilerplate/empty

- Item 1C, Cybersecurity --- newer item, company-specific

- Item 2, Properties --- usually low information density

- Item 3, Legal Proceedings --- can matter for qualitative questions

- Item 4, Mine Safety Disclosures --- near-universally irrelevant
  unless mining company

**Part II** --- this is where your actual numbers live

- Item 5, Market for Registrant\'s Common Equity --- stock/equity
  data, moderate value

- **Item 6 is explicitly \"\[Reserved\]\"** --- literally nothing
  here, don\'t waste parser effort looking for content

- **Item 7, MD&A** --- your highest-value section. This is where
  management explains _why_ the numbers changed, exactly the
  qualitative reasoning layer that sits on top of the raw financial
  statements

- Item 7A, Quantitative and Qualitative Disclosures About Market Risk
  --- often short, sometimes skipped entirely (see exemption note
  below)

- **Item 8, Financial Statements and Supplementary Data** --- the
  actual financial statements (income statement, balance sheet, cash
  flow) plus their footnotes. This is your other highest-value
  section, and it\'s exactly where the table→footnote relationship
  problem lives

- Item 9 / 9A / 9B / 9C --- almost entirely boilerplate/compliance
  disclosures, low retrieval value

**Part III** --- often not actually present in the document

- Items 10--14 (directors, executive compensation, ownership,
  related-party transactions, accountant fees - respectively).
  **Important gotcha**: the form explicitly allows companies to
  incorporate this whole Part by reference from their proxy statement
  (DEF 14A) instead of writing it into the 10-K itself.

**Part IV** --- pure filing mechanics

- Item 15, Exhibits and Financial Statement Schedules --- mostly a
  list of attached exhibit files

- Item 16, Form 10-K Summary --- optional, most companies skip it
  entirely

- Signatures --- boilerplate, ignorable for retrieval purposes

**Other structural quirks worth knowing:**

- **The cover page is a wall of yes/no checkboxes** (large accelerated
  filer? shell company? etc.), filer-classification metadata:
  low-priority noise in your parser

- **\"Smaller reporting companies\" get exemptions**: they can skip
  Item 7A (market risk) and Item 1A (risk factors) entirely.

- **Item 6 being \"\[Reserved\]\"** is a leftover from an old
  requirement (previously \"Selected Financial Data\") that got
  removed but the number was kept empty

**Practically, for your parser:** the single highest-leverage move here
is building a lookup list of the known Item headings (\"Item 1.\",
\"Item 1A.\", \... \"Item 16.\") and using them as reliable
section-boundary anchors, on top of or instead of your current
bold-styling heuristic. Since this structure is legally mandated and
identical across every filer, it\'s a far more robust signal than
inferring headings from inconsistent CSS styling

---

Check if benchmark accounts for failure points in RAG research doc.
Then start building system design:

- basic chatgpt notes + Claude chats
- boot.dev
- codebase retrieval parsing md
- (and maybeeee lit review stuff + claude convos)

Progress

- worry about agentic tooling later: research papers (+ Claude Opus chat), bootdev agent course
- not including LoFin benchmark yet

Then, identify failure points in architecture and if benchmark can help with this.

---

Try record benchmark results for each change. LlamaIndex to orchestrate pipeline

Ingest files

- Load doc
- Parse document: extract text, identify structural elements (sections, titles, tables, text, figures), preserve information like table structure and data
  - output as .md - like Kim et al., (2025) found improvements on - or json
  - Jimeno Yepes (2024) used basic VLM to identify text, titles, tables and chunk along those boundaries (+LLM generated summary to embed)
    - Rationale:
    - helps LLM read non-plaintext file formats to LLM-interpretable representations, like tables.
    - Also, just using LLM misses out metadata (year, file type, ticker) and structural content (e.g. subheading of each chunk)
  - Preserve metadata in the doc i.e. pg numbers, denoting (sub)headings, tables etc.
  - Parsing Options
    - "VLM-agentic" parsers (read the page like a person, with a correction/verification pass). E.g. Llamaparse, reducto ($$$), Azure Document Intelligence ($100 for ~80 docs around 12,000 pages, 82.7% on RD-TableBench, use base Layout), GPT-5.6 sol. **Good if endless edge cases exist in human writing, just use VLM**
    - "layout engine" parsers (specialized detection models + rules, no LLM-in-the-loop by default, self-hostable and free): PyMuPDF / pymupdf4llm, good for machine-generated docs like statements - apparently not that good for tables

- Chunk (+structure parsing) + save metadata for each chunk (SEC-filing-type + company ticker + financial year + page number), for filtering chunks. _View chunks manually, writing has infinite edge cases_
  - table-aware chunking
  - fixed-size chunking
    - 1,024 tokens, 30 overlap, via Langchain's RecursiveCharacterTextSplitter - HiRec (Choe et al., (2025)). Can supplement with semantic chunking (split around sentences, not arbitrary words, via regex)
    - ideally, attach metadata + pg no. to each chunk
  - experiment 2:
    - FinSTAR: keep track of document structure to attach as metadata to a chunk (HiChunk).
      - is there way to preserve boundaries for (sections, titles, tables, text, figures) like Jimeno Yepes, to chunk along? like HiChunk's chunk-point predictor — fine-tuned Qwen3-4B
      - is there way to extract the structure/hierarchy of document, so we know the headings of each chunk? Can HiChunk help here?
      - What we could do: use PageIndex, https://docs.pageindex.ai/sdk/documents#read-a-document. Use get_tree() to pull structure (using pure parsing of e.g. contents page, section headings!) - section headings and starting page for each one, and nesting to indicate if it is a sub-heading. Can use 'local mode' to be quick, tho need to check pdfs are machine-readable, not image-only pdfs (for both datasets!). Also, only shows starting, not ending page; can compute with nesting and starting page of subsequent sections. then, construct data structure storing concatenated headings (each one has its own embedding score), with its corresponding page ranges, then for own parsed pdf, compare chunk's page with this to know which heading belongs to it
      - FinSTAR: get SLM to generate another section header based on the section's content specifically (does HiChunk have similar implementation to use here?)
      - Concatenate the chunk's heading + SLM-generated secion heading, save as metadata to chunk
      - FinSTAR method: For chunk ci, and hierarchical path hi, use SLM, Gθ, to generate virtual node vi = Gθ(ci, hi), where enriched path Pi = hi⊕vi. To ensure robustness, Pi is constrained by (1) depth control, enforcing Depth(Pi) <= 5 and (2) discriminativeness, requiring vi to capture essential semantics absent from hi.

- Pre-process for keyword search (/Users/zubairasim/rag-search-engine/course_notes/module-01-preprocessing.md, /Users/zubairasim/rag-search-engine/cli/lib/keyword_search.py)
  - what normal keyword search does is, for both the query AND the document store:
    -> lowercase
    -> remove punctuation
    -> split into tokens
    -> remove empty tokens
    -> remove stop words
    -> stem tokens
    -> compare tokens
    - im not sure, from questions, which preprocessing to do. remove punctuation - worried about removing currency info. lower-case, not usre if stuff like EBITDA or FY matters - maybe not? similarly, stemming may be an issue, but maybe not if query also stemmed? would need agent to look at the questions to determine what is suitable

  - then, store the following
    - map tokens to each (chunk id, pg no, doc id)
    - map chunk ID to chunk object itself
    - map chunk-id to map of words and each of their counts
    - map each chunk to its chunk length
  - need a way of storing/loading these files, either here for now, or in database like in ElasticSearch; can also cache the files e.g. with .pkl - ask AI whether to worry about this now or later

- Embed chunk (using same chunk as before)
  - cheapest embedder with >80% accuracy is voyage-4-lite: http://mteb-leaderboard.hf.space/benchmark/RTEB(fin%2C%20beta)
  - also has first 200m tokens free allowance, then $0.02/1m tokens via their api on voyageai.com
  - cache/store embedding (alongisde each chunk's metadata like year, ticker, report-type, text - so can filter by all or none)
  - Experiment 2:
  - embed each chunk's concatenated heading
  - store that embedding as well (in metadata store? is that possible?)

Retrieve - Elastic search

- LLM query enhancement - to what extent? check lesson 7 + claude link on prompting: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- Metadata filtering: let agent filter by ticker, doc-type, year. Let it know whether its desired file exists + was retrieved. Like Claude Code, maybe naively give prompt all available metadata / files they can search from.
- Hybrid search
  - BM25 (keyword search) - like FinCARDS (2026), use k1-1.5, b=0.75
    - calculate BM25(term, doc) = bm25_tf \* bm25_idf
      - BM25 TF: (tf _ (k1 + 1)) / (tf + k1 _ length_norm)
        - length norm = 1 - b + b \* (doc_length / avg_doc_length), where b=0.75
      - BM25 TF: (tf \* (k1 + 1)) / (tf + k1), k1 = 1.5
  - Semantic search (searching 'concepts')
    - search: embed user query, cosine similarity query with each document, rank by top-k scores (if too many embeddings, find nearest using database)
    - Experiment 2:
      - Do cosine-similarity with the query, concatenated with itself, and the chunk's embedded concatenated with its section-header's embedding!
      - Structural injection is computed as follows: given a chunk embedding `H_ci = Embed(c_i) ∈ R^d` and its hierarchical path levels `p_1, ..., p_n` (ancestor headers plus virtual node), each level is independently embedded as `V_j = Embed(p_j)`; relevance scores are computed via dot product `s_j = H_ci · V_j`, then normalised with standard `√d`-scaled softmax, `a_j = exp(s_j/√d) / Σ_k exp(s_k/√d)`, giving `a_j ≥ 0` and `Σ_j a_j = 1`; these weights are used to aggregate the level embeddings into a structural context vector `H_si = Σ_j a_j·V_j`, equivalent to `Attn(Q=H_ci, K=V={V_1,...,V_n})` with `W_Q=W_K=W_V=I` (no learned projections); this is then concatenated — not summed — with the original chunk embedding to form the final index vector `H+_ci = [H_si ; H_ci] ∈ R^{2d}`, preserving the chunk embedding unmodified in its own sub-space; at retrieval time the query embedding `q = Embed(query)` is duplicated to match dimensionality, `q+ = [q ; q] ∈ R^{2d}`, so that the retrieval score decomposes additively as `q+ · H+_ci = (q · H_si) + (q · H_ci)` — a structural-relevance term plus a content-relevance term, computed independently. All embeddings are produced by the same off-the-shelf embedding model with no access to model internals or hidden states required, the vector store collection dimension is set to `2d`, and no training loop, labelled data, or loss function is needed.
  - rrf (say k=60?). issue is are we doing BM25 for chunks too? usually for docs - but need score for individual chunks now, hopefully not too complicated
  - Like Hi-Chunk, shall i use their rule for retriving parent chunk i.e. auto-merge?
  - tune top-k: claude contextual-retrieval uses top-20 w. 800 token chunks, 50 token instructions, 100 tokens context, FinCARDS uses top-10
- use reranker on topk-k to retrieve top-n (query-document)
  - BAAI/bge-reranker-v2-gemma via the FlagEmbedding library's FlagLLMReranker: FinSage (2025)
- feed LLM top-n results: but include metadata and heading as well

Generate answer

- decide model: GLM-5.3-flash with openrouter, determined with https://www.vals.ai/benchmarks/fabv2 (which we cant use as doesnt score retrieval)

LLM as judge

- for retrieval metrics, Zhou et al., (2026) FinCARDA use MRR@10, maybe @10 is good
- Azure
- Zheng et al. (2024) - gpt judge agrees almost as much as human, but use different model from judging to generating, give judge correct reference answer BEFORE it grades, grade twice with answer order swapped and only trust verdict both times agreed on. have it putput a 1 or 0 for correct or not

Experiment 3 - Agents: LLMs autonomously using tools in a loop
https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
this makes case for letting agents intelligently navigate files, THEN load its contents into context.

- FinSTAR: decompose query into 'atomic' subqueries, use symbolic logic topology of (∩ / \ / aggregation), to figure out how they will lead to answer, before conductin retrieval and compare retrieved info against plan to adjust as go along
- tools:
  - choose metadata filters before doing search
  - BM25 or semantic search tools + specific queries to use for each
  - recursive rag - using results search to inform next search. use thinking-mode for feedback loop (ReAct, HiRec)

Later

- OP-RAG (orig 19) — retrieves normally, then reorders the retrieved chunks by their original document position before prompting

- Agents (see multi-hyde notes for algorithm - copy their notation)
  - verification agent (compare retrieved info vs generated answer - Chain of Verification paper)
  - query enhancement - rewrite question so makes sense reading on itself without needing prior context (FinSTAR)
  - query enhancement (+ HyDe - fake answer to use for semantic search)
  - calculator tool
  - let agent navigate document's structure for info akin to PageIndex (like PDFTriage, 2023)

IMPLEMENTATION HELP

### Lu et al. 2026 — HiChunk (`github.com/TencentCloudADP/hichunk`)

**Verified, matches paper exactly** — the cleanest, most reproducible repo of the whole batch.

- Chunk-point predictor: fine-tuned `Qwen/Qwen3-4B` (confirmed in `HiChunk_train_config.yaml`, LLaMA-Factory full-parameter SFT), inference via vLLM over 16384-token sliding windows.
- Auto-Merge (`retrieval_algo.py::tree_chunk_retrieval_auto_merge`): three AND-gated conditions (coverage threshold scaled by remaining token budget; ≥2 same-parent siblings retrieved; remaining budget fits the whole parent) — matches the paper's coherence/substantiality/feasibility triad conceptually, though the code doesn't use those literal names.
- Leaf chunk size: confirmed **200 tokens** via `argparse` default and README usage examples.
- **Benchmark:** HiCBench (their own, T1/T2 variants) plus LongBench, Qasper, GutenQA, OHRBench(T0) as external comparisons. Metrics: Evidence Recall (ERec), ROUGE, F1.

### Zhou et al. 2026 — FinCARDS, useful for BM25 (`github.com/XanderZhou2022/FINCARDS`)

**Verified real** (was flagged "unverified" in the sheet — it checks out).

- Three-stage cascade fully implemented and parameterized: Stage 1 BM25 (k1=1.5, b=0.75, dynamic candidate count 60–150); Stage 2 card screening (`gpt-5-mini-2025-08-07`, group size 25, keeps 8–15/group, final top 50); Stage 3 bootstrap Borda (up to 5 rounds, Jaccard convergence 0.9).
- **No chunking code anywhere.** All three stages load a pre-built `unique_chunks.json` from an external, unreleased "FinBenchQA" data-prep step — chunk size cannot be determined from this codebase.
- Hard dependency on `OPENAI_API_KEY` for 3 of 4 stages; only BM25 is dependency-free.
- **Benchmark:** FinAgentBench — intra-document evidence reranking over U.S. SEC 10-K/10-Q filings, one document per query. Metrics: nDCG@10, MAP@10, MRR@10, Rank Variance.

TECH STACK:

add instructions to create and active venv after creating uv project;include instructions for this inreadme.md

TOML is a configuration/data format. pyproject.toml is a standardized Python project configuration file written in TOML. It can declare metadata, dependencies, build requirements, and tool configuration. Package-management/build tools such as uv, pip, Poetry, etc. read this information and perform actions such as resolving and installing dependencies and building the project. The resulting packages live in an environment, usually a virtual environment. The Python source files then import and use those installed packages.

                         pyproject.toml
                               │
       ┌───────────────┬───────┼────────┬───────────────┐
       ↓               ↓       ↓        ↓               ↓

Project Metadata Runtime Build Dev Tool Config
Deps System Deps
│ │ │ │ │
│ │ │ │ ┌────┼────┐
│ │ │ │ ↓ ↓ ↓
│ │ │ │ pytest ruff mypy
│ │ │ │ │ │ │
↓ ↓ ↓ ↓ ↓ ↓ ↓
name/version pandas uv_build pytest Testing Linting
description openai │ │ Type-checking
authors pymupdf │ │
etc. │ │ │
↓ ↓ ↓
Run the app Build app Develop/test it

Llamaindex for orchestration and specific functions e.g. baseline chunking methods

OpenAI sdk - idk how to use concurrently with llamaindex. need to also check if comppatible for non-openAI models for the toolings i want to use

Elastic search as compatible with all my retrieval methods + uses HNSW
need to consider others in bootdev doc. can also visit (/Users/zubairasim/rag-search-engine/course_notes/module-4b-temp.md fo rmore info)
