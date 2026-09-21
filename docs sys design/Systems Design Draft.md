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

- [x] CHECK FIRST (both can change the plan): does `BM25Retriever`'s `filters` argument actually filter on our data? Build a retriever over ~5 chunks with different `doc_name` metadata, retrieve with a filter, confirm only matching chunks come back. If not, fall back to building the retriever from an already-filtered node list
  - ANSWERED at `llama-index-retrievers-bm25==0.8.0`: yes — the filter becomes a `corpus_weight_mask` applied before scoring, so filter-then-rank holds and the fallback is not needed. Kept as `tests/test_bm25_metadata_filters.py`
  - CAVEAT that changes retrieval code: filtered chunks are not dropped, they are returned with `score == 0.0` as padding when `similarity_top_k` exceeds the surviving count. Retrieval MUST discard zero-score hits, or out-of-scope chunks reach the context block. A filter matching nothing raises `ValueError` rather than returning the whole corpus
- [ ] CHECK FIRST: is Azure Document Intelligence's section nesting good enough for Exp2's heading path? Run `prebuilt-layout` on 2-3 10-Ks, open the JSON, and look at `sections` (do sections nest, and do Item headings sit at the top level?) and at `paragraphs` with role `title` / `sectionHeading` (right text, right page?). If nesting is flat or wrong, use PageIndex instead
- Edit the existing file:
results/20260917-012959--financebench--baseline-context-conditions-v1/manual_review.csv
  -  Then import it:
    uv run sec-rag-benchmark import-manual-review \
      --run-dir results/20260917-012959--financebench--baseline-context-conditions-v1
  -  Regenerate both reports:
    uv run sec-rag-benchmark report \
      --run-dir results/20260917-012959--financebench--baseline-context-conditions-v1


Deferred
- 3 experiments writeup (make sure to specify baseline/ablation)
- worry about agentic tooling later: research papers (+ Claude Opus chat), bootdev agent course
- not including LoFin benchmark yet
- Exp3: query decomposition into sub-questions + FinSTAR symbolic logic topology (∩ / \ / aggregation). Pays off on multi-document questions = LOFin, which is deferred; every FinanceBench question sits inside one filing
- Exp3: "fetch whole page" tool (PDFTriage pattern) — pull the full page, or the next one, once a promising chunk is found. Targets tables split across pages

---

Try record benchmark results for each change. LlamaIndex to orchestrate pipeline

Tech stack

- LlamaIndex — orchestration + components: chunking and markdown parsing (heading paths), BM25 retrieval (wraps `bm25s`) with metadata filters, semantic retrieval with metadata filters, RRF fusion (`QueryFusionRetriever(mode="reciprocal_rerank")`, `num_queries=1` to stop it inventing extra queries), reranking (Voyage post-processor), and the Exp3 agent loop (`max_iterations` + `early_stopping_method="generate"`, so a capped run still answers instead of erroring)
- we write ourselves: Exp2's scorer (~20-line `BaseRetriever` subclass — a weighted sum of two similarity scores per chunk, which RRF can't express because it fuses ranked lists), the Exp3 tool functions, trace logging, and the glue into the benchmark harness
  - trace logging is ours, NOT LlamaIndex's event system: each tool appends what it did + what it returned to a per-question list (~3 lines per tool, and it doesn't break when the framework changes its events)
  - remaining glue for Exp3: turn the framework's "cap reached" signal into our fourth outcome
- Chroma — vector store + chunk metadata + `where` filtering before search, so we don't hand-roll save/load. For Exp2, pull the embeddings out and score in numpy
- `bm25s` (via LlamaIndex) — keyword index. `BM25Retriever.from_defaults(filters=MetadataFilters(...))` filters before searching; this was a recent addition, so PIN A CURRENT VERSION and test it on our data. Fallback: build the retriever from an already-filtered node list (2 lines)
  - filtering applies to the chunks searched; BM25 word statistics still come from the whole corpus (same as Elasticsearch, so results stay comparable if we switch)
- Azure Document Intelligence — parsing. OPEN: PageIndex is a live fallback if Azure's section nesting is poor
- Voyage API — voyage-4-lite embeddings + `rerank-3-lite` reranker. Same account and same `VOYAGE_API_KEY` for both; two endpoints
  - DECIDED (Stage 0.4): call Voyage **directly** via the `voyageai` client (`vo.embed(...)`, `vo.rerank(query, documents, model, top_k)`), for both embeddings and reranking — not through `llama-index-postprocessor-voyageai-rerank` / `llama-index-embeddings-voyageai`
    - two fewer packages, and we assemble the candidate list ourselves anyway (Exp2's custom retriever must), so the LlamaIndex wrappers buy no consistency we actually use
    - it keeps `input_type` explicit at the call site — `document` for chunks, `query` for questions. Mixing them degrades retrieval silently rather than erroring, so a wrapper default is the wrong place for it
  - reranker call limits: ≤1,000 documents per call; query + any single document ≤32,000 tokens; (query tokens × documents) + all document tokens ≤600,000 per call. At ~50 chunks × ~1k tokens we use ~60k, so no batching needed
- OpenAI SDK via OpenRouter — answer generation (already pinned, with provider routing). LlamaIndex components don't call the model, so nothing clashes
- pandas + pytest — reporting and tests (already in place)
- LATER (document as future work): Elasticsearch as a second retriever behind the same interface — keyword, vectors (HNSW) and filters in one query. At ~10k chunks exact search is faster to build and more accurate, so it buys skills, not results
- new dependencies, installed and exact-pinned at Stage 0.4: `llama-index-core==0.14.24`, `llama-index-retrievers-bm25==0.8.0`, `llama-index-vector-stores-chroma==0.6.0`, `chromadb==1.5.9`, `azure-ai-documentintelligence==1.0.2`, `voyageai==0.5.0`, `numpy==2.5.3` (`bm25s==0.3.11` comes in transitively)
  - the LlamaIndex Voyage wrappers are deliberately NOT installed (see the Voyage decision above)
- credentials needed in `.env` / `.env.example`: `OPENROUTER_API_KEY` (generation), `AZURE_DEEPSEEK_API_KEY` + `AZURE_DEEPSEEK_ENDPOINT` (judge), `VOYAGE_API_KEY` (embeddings + reranker), `AZURE_DOCUMENT_INTELLIGENCE_KEY` + `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` (parsing)

Ingest files

- Load doc
- Parser output decisions - goal is it should identify different media (tables etc.) and nested subheading structure accurately!
  - output as .md - like Kim et al., (2025) found improvements on - or json
  - Jimeno Yepes (2024) used basic VLM to identify text, titles, tables and chunk along those boundaries (+LLM generated summary to embed)
    - Rationale:
    - helps LLM read non-plaintext file formats to LLM-interpretable representations, like tables.
    - Also, just using LLM misses out metadata (year, file type, ticker) and structural content (e.g. subheading of each chunk)
  - Preserve metadata in the doc i.e. pg numbers, denoting (sub)headings, tables etc.
  - Parsing Options
    - "VLM-agentic" parsers (read the page like a person, with a correction/verification pass). E.g. Llamaparse, reducto ($$$), Azure Document Intelligence ($100 for ~80 docs around 12,000 pages, 82.7% on RD-TableBench, use base Layout), GPT-5.6 sol. **Good if endless edge cases exist in human writing, just use VLM**
    - "layout engine" parsers (specialized detection models + rules, no LLM-in-the-loop by default, self-hostable and free): PyMuPDF / pymupdf4llm, good for machine-generated docs like statements - apparently not that good for tables
    - verdict: use Azure Document Intelligence as i have startup credits (unless it has complicated setup/infra?)
      - setup is light: create resource → endpoint + key → `azure-ai-documentintelligence` SDK, `prebuilt-layout` model, markdown output. Paid tier only (free tier reads first 2 pages). ~$10 / 1,000 pages
  - Parser output decisions: goal is to identify subheading structure accurately!
    - parse once, cache forever: save Azure's raw JSON per PDF (out of Git, with manifest like data prep), never re-parse when chunking changes
      - store derived parser artefacts per filing:
        `data/financebench/parsed/<doc_name>/azure-layout.json` plus an optional reproducible
        `structure.txt`; do not copy the source PDF out of the prepared dataset's `pdfs/` directory
      - keep `data/financebench/parsed/manifest.json` separate from the prepared-dataset manifest.
        It records each PDF/JSON hash, page count, model/output settings and SDK provenance,
        completion time and whether a missing entry was repaired from an already-valid cache
      - the unflagged `sec-rag parse --config configs/sec_rag.toml` command is read-only and checks
        every PDF unless `--documents` narrows it. `--execute-paid` recreates any missing manifest
        entry from a valid cache, then submits only genuinely missing filings, serially
      - before a cache can suppress another paid call, require readable JSON, non-empty content and
        an Azure page count matching PyMuPDF's PDF page count. Detailed page-number and span
        validation belongs to 1.2-1.3, where those fields are used
      - write the JSON and manifest atomically after each filing, stop on the first failure, and
        resume by skipping completed documents
    - raw JSON is the source of truth; markdown is one field inside it (`content`), other fields (`pages`, `paragraphs`, `tables`, `sections`) point into that markdown by character position, which is how each piece maps back to a page
      - save `result.as_dict()` verbatim, preserving all returned fields including figures and
        `styles`. Section 1.1 only acquires and validates raw input; stripping noise, page-index
        conversion, metadata attachment, heading repair and chunking happen in sections 1.2-1.3
      - attribution is therefore by character offset, not by page range (e.g. 200,000 characters in a filing, it gives character 'width' of each page, paragraph, table etc.): every paragraph, table, section and page carries spans (offset + length), so text is attached to its heading exactly rather than approximately
    - strip page headers/footers/page numbers (Azure labels these) before chunking
      - they appear inline in content as \<!-- PageHeader="..." --> and \<!-- PageNumber="13" -->;
    - document-level metadata (company, doc_type, doc_period) comes from `financebench_document_information_10k.jsonl`, not from parsing. No ticker field in FinanceBench
    - page numbers: FinanceBench `evidence_page_num` is zero-indexed; Azure `pageNumber` is 1-indexed → `evidence_page_num = pageNumber - 1`. Never use the footer's printed page number. Add a test

  - Fix the heading list before chunking 
    - an LLM pass takes the heading rows (text, level, page) plus the file's metadata and returns the same rows corrected
      - repairs split-word typos — Busines s., ESTIMA TES, Equit y (16 of 295 headings)
      - drops headings that only restate the file metadata, e.g. UNITED STATES SECURITIES AND EXCHANGE COMMISSION..., which is otherwise the root ancestor of ~106 pages. This can be derived from the filename
      - splits absorbed headings — Azure glued PART I onto the end of the previous title
      - re-levels based of common sense e.g. heading called 'part 2' should not be different level to 'part 1' - using the generic rule that a numbered series (Item 7, Chapter 3, Article II) are siblings
    - cached per filing like the parse, so everything downstream is deterministic and identical across every condition and experiment
    - validated structurally, with no domain knowledge: every output heading traces to an input heading (whitespace repair and splits only), levels form a valid tree with no jumps greater than one, numbered series share a level, pages unchanged. On failure, keep Azure's raw levels for that filing and log it
    - accepted limitation: genuinely missed sub-headings (underlined/italic ones Azure never marked) cannot be recovered. Not too much impact, because Exp 2 weights headings by similarity rather than by depth

- Chunk (+structure parsing) + save metadata for each chunk (SEC-filing-type + company ticker + financial year + page number), for filtering chunks. _View chunks manually, writing has infinite edge cases_

  Backwards compatible as long as chunks + embeddings are saved as plain files and search sits behind the existing retriever interface; ES then becomes one more loader + retriever

  - hard rule: no chunk crosses a page boundary ie one page per chunk. This is what keeps page recall/precision/MRR exact against evidence_page_num. A chunk spanning 3 pages would get three chances to contain the gold page, making the metric's bias vary with chunk size
  - table-aware chunking: doable, as each item e.g. 'table' has its own character offset location
    - after removing tables and noise, join the page's remaining prose into one stream before cutting.
  - fixed-size chunking
    - 512 tokens (whatever is the median of structural parsing chunking, this size also still backed by literature)
      - 30 overlap that is allowed to come from previous page, via Langchain's RecursiveCharacterTextSplitter - HiRec (Choe et al., (2025)). 
      - Can supplement with semantic chunking (split around sentences, not arbitrary words, via regex)
      - include headers in the text to chunk
    - ideally, attach metadata + pg no. to each chunk
  - chunking decisions
    - use LlamaIndex (not LangChain) for splitting, consistent with BM25/hybrid/routing choice
    - inspection script: "show all chunks for doc X, page Y"; run on a question's gold page to spot missed-table failures
  - experiment 2 (2 versions - one with chunking but dont use heading for similarity score, one where you do):
    - cut at section headings instead, with a floor of ~250 tokens and a ceiling of 1,024
    - accept a heading cut only if the piece before it and the remainder both clear ~250 tokens; otherwise skip that cut
    - anything still over 1,024 tokens is halved at the nearest sentence boundary to its midpoint, recursively. Halving something over 1,024 always leaves both sides over 512, so the floor can't be violated
    - on 3M 2018 this gives 244 text chunks + 119 table chunks, median 512 tokens, 89% between 250 and 1,024
    - 30-token overlap, split around sentences, allowed to come from the previous page, but page metric still only 1 page
      - if a heading cut is rejected for being too small, keep the page whole and merge the heading paths: shared ancestors once, distinct tails joined
      - PART I > [Item 2. Properties | Item 3. Legal Proceedings]
      - this works with Exp 2 unchanged, because the structure vector is a softmax-weighted average over a set of headings, not a single path — the weights favour whichever tail matches the chunk's content
    -  if a subheading exceeds the depth limit of 5, ignore it and pretend the text sits under the nearest surviving 
    - keep track of the following:
      - its merged heading_path, derived from character offsets
      - its page_num — exactly one page, never a set, because of the hard page boundary
      - doc_name and the file metadata from the JSONL
    - FinSTAR: keep track of document structure to attach as metadata to a chunk (HiChunk).
      - is there way to preserve boundaries for (sections, titles, tables, text, figures) like Jimeno Yepes, to chunk along? e.g. more than one chunk same page
      - is there way to extract the structure/hierarchy of document, so we know the headings of each chunk? 
      - What we could do: use Azure doc intelligence PageIndex - just need heading, nesting level, and its start page. PageIndex https://docs.pageindex.ai/sdk/documents#read-a-document. Use get_tree() to pull structure (using pure parsing of e.g. contents page, section headings!) - section headings and starting page for each one, and nesting to indicate if it is a sub-heading. Can use 'local mode' to be quick, tho need to check pdfs are machine-readable, not image-only pdfs (for both datasets!). Also, only shows starting, not ending page; can compute with nesting and starting page of subsequent sections. then, construct data structure storing concatenated headings (each one has its own embedding score), with its corresponding page ranges, then for own parsed pdf, compare chunk's page with this to know which heading belongs to it
      - (DEFERRED, add later) FinSTAR: get SLM to generate another section header based on the section's content specifically
      - Save the chunk's heading path (list of headings, top level first) as metadata to chunk. Each heading level is embedded separately and combined at retrieval (see retrieval → Experiment 2 theory), not joined into one string
      - attribute headings to chunks: from each heading's start page, compute its page range (ends where the next heading at the same or higher level starts); a chunk gets the headings whose range covers its page. If two headings start on the same page, attribute the one covering more of that page
      - FinSTAR method: For chunk ci, and hierarchical path hi, use SLM, Gθ, to generate virtual node vi = Gθ(ci, hi), where enriched path Pi = hi⊕vi. To ensure robustness, Pi is constrained by (1) depth control, enforcing Depth(Pi) <= 5 and (2) discriminativeness, requiring vi to capture essential semantics absent from hi.

- Pre-process for keyword search (/Users/zubairasim/rag-search-engine/course_notes/module-01-preprocessing.md, /Users/zubairasim/rag-search-engine/cli/lib/keyword_search.py)
  - what normal keyword search does is, for both the query AND the document store:
    -> lowercase
    -> remove punctuation
    -> split into tokens
    -> remove empty tokens
    -> remove stop words
    -> stem tokens
    -> split letters from digits ()`FY2018` → `fy` + `2018`)
    -> compare tokens
    - use `bm25s` via LlamaIndex's `BM25Retriever` instead of hand-built maps (same structures, faster)
  - then, store the following
    - map tokens to each chunk id
    - map chunk ID to chunk object itself (which has metadata on page and doc too)
    - map chunk-id to map of words and each of their counts
    - map each chunk to its chunk length
  - need a way of storing/loading these files, either here for now, or in database like in ElasticSearch; can also cache the files e.g. with .pkl - ask AI whether to worry about this now or later

- Embed chunk (using same chunk as before)
  - cheapest embedder with >80% accuracy is voyage-4-lite: http://mteb-leaderboard.hf.space/benchmark/RTEB(fin%2C%20beta)
  - also has first 200m tokens free allowance, then $0.02/1m tokens via their api on voyageai.com
  - cache/store embedding (alongisde each chunk's metadata like year, ticker, report-type, text - so can filter by all or none)
  - Experiment 2:
  - embed each heading level separately (each unique heading embedded once, reused across chunks)
  - store heading embeddings separately from chunk embeddings, linked by chunk ID via the chunk's heading path, not in metadata.
  - embedding decisions
    - cost: ~10k chunks × ~1k tokens ≈ 10M tokens, inside the 200M free allowance even with several re-chunks
    - label inputs: embed chunks with Voyage `input_type="document"`, questions with `input_type="query"`
    - cache each embedding keyed by model name + hash of chunk text, so re-chunking only re-embeds changed chunks
    - HARNESS (build with Exp1): prediction row records embedding + rerank cost too, not just the answer model's


Retrieve - Elastic search? can think about tech stack later

- LLM query enhancement - to what extent? check lesson 7 + claude link on prompting: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
  - ONE LLM call returning structured JSON: company, year(s), doc_type, keyword query (rephrase shorthand expanded - PPNE → property, plant and equipment; COGS, DPO, FCF, capex), semantic query.
  - Metadata filtering: filenames are `COMPANY_YEAR_TYPE.pdf`, so the LLM supplies all three and we select the filing by filename (not from metadata table columns). Parse from the RIGHT — `JOHNSON_JOHNSON_2022_10K` has an underscore in the company name. Simplest: give it the filename list and have it return a filename. Filter chunks BEFORE search. Let it know whether its desired file exists + was retrieved. Like Claude Code, maybe naively give prompt all available metadata / files they can search from.

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
      - normalise both vectors to length 1 before scoring; embed each unique heading once, chunks with no heading - use fallback as chunk score alone. Implementation: maybe subclass Llamaindex's baseretriever
      - Theory explanation:
      - Structural injection is computed as follows: given a chunk embedding `H_ci = Embed(c_i) ∈ R^d` and its hierarchical path levels `p_1, ..., p_n` (ancestor headers plus virtual node), each level is independently embedded as `V_j = Embed(p_j)`; relevance scores are computed via dot product `s_j = H_ci · V_j`, then normalised with standard `√d`-scaled softmax, `a_j = exp(s_j/√d) / Σ_k exp(s_k/√d)`, giving `a_j ≥ 0` and `Σ_j a_j = 1`; these weights are used to aggregate the level embeddings into a structural context vector `H_si = Σ_j a_j·V_j`, equivalent to `Attn(Q=H_ci, K=V={V_1,...,V_n})` with `W_Q=W_K=W_V=I` (no learned projections); this is then concatenated — not summed — with the original chunk embedding to form the final index vector `H+_ci = [H_si ; H_ci] ∈ R^{2d}`, preserving the chunk embedding unmodified in its own sub-space; at retrieval time the query embedding `q = Embed(query)` is duplicated to match dimensionality, `q+ = [q ; q] ∈ R^{2d}`, so that the retrieval score decomposes additively as `q+ · H+_ci = (q · H_si) + (q · H_ci)` — a structural-relevance term plus a content-relevance term, computed independently. All embeddings are produced by the same off-the-shelf embedding model with no access to model internals or hidden states required, the vector store collection dimension is set to `2d`, and no training loop, labelled data, or loss function is needed.
      - Exp2 decisions (from review; confirm or strike)
        - Exp2 = Exp1 pipeline with ONE change: the dense score. Parsing, chunks, query enhancement, BM25, RRF, reranker, generation all identical
        - structure source: OPEN — Azure Document Intelligence or PageIndex. Either works with the same method, as long as it gives each heading, its nesting level and its start page (see chunking → experiment 2 attribution rule). Check both on 2-3 filings
        - SLM virtual node DEFERRED (add later)
        - Fin-STAR constraints: depth ≤ 5 applies now — if the heading path has more than 5 levels, drop the excess (deepest) headings. Discriminativeness only applies to the virtual node, so it comes back when that does
        - softmax divisor (temperature) is a CONFIG SETTING, starting value 0.05. Try 1 / √d only if time allows
          - why: the √d in the theory text (√1024 = 32) squashes heading-to-chunk similarities (which sit between -1 and 1) towards 0, so every heading level ends up weighted equally and the structure vector becomes a plain average of the heading embeddings
          - how much it matters, measured on synthetic vectors (top 10 of 500 chunks, vs the equal-weights version): divisor 1 returns ~9.8/10 of the same chunks (effectively identical), divisor 0.05 returns ~7.3/10 (a genuinely different system)
          - write-up: state that the divisor departs from the √d in the source theory, and say why
        - no 2d vector store needed: the concatenated score equals (q · H_si) + (q · H_ci), so compute the two scores separately and add them. Normalise H_si to length 1 first (an average of length-1 vectors is shorter than 1)
        - framing: training-free approximation of Fin-STAR, not a replication 
        - build order: (A) Exp1 → (B) heading path, no virtual node → (C) path + virtual node (deferred). A vs B alone is a complete Exp2 result
        - report page metrics before reranking too: the reranker only sees chunk text and could wash out a structural gain
  - rrf (say k=60?). issue is are we doing BM25 for chunks too? usually for docs - but need score for individual chunks now, hopefully not too complicated
  - Like Hi-Chunk, shall i use their rule for retriving parent chunk i.e. auto-merge?
  -  top-k: FinCARDS uses top-10.  Align `retrieval_depth` in config (currently 5)
- use reranker on topk-k to retrieve top-n (query-document)
  - BAAI/bge-reranker-v2-gemma via the FlagEmbedding library's FlagLLMReranker: FinSage (2025). Use Voyage reranker API instead
  - reranker model: `rerank-3-lite` ($0.02/1M tokens, 200M free). NOT the 2.5 line — `rerank-2.5` and `rerank-2.5-lite` have ZERO free allowance; the 3 line is newer and Voyage state it is strictly better on quality, context length, latency and throughput
    - usage is tiny: reranking bills (query tokens × number of docs) + all doc tokens ≈ 6M tokens for a full 112-question run at ~50 candidates × ~1k tokens, i.e. ~3% of the free allowance
    - `rerank-3` (the full model) is also free at our volume — move up only if the dev-subset pre/post-rerank metrics say reranking is the bottleneck. One string change
    - no reliable public reranker leaderboard exists (unlike RTEB for embedders); published comparisons put the leading rerankers within 1-3 NDCG points, and the loudest claims are vendor self-citations. So decide it on OUR pre/post-rerank page metrics, not a leaderboard
    - write-up: FinSage used bge-reranker-v2-gemma; it needs GPU inference, so a hosted reranker of comparable class was substituted
  - HARNESS (build with Exp1): compute page metrics both before and after reranking (call `page_metrics()` twice, two sets of fields on the prediction row), so the reranker's effect is visible
- feed LLM top-n results: but include metadata and heading as well
- retrieval decisions (from review; confirm or strike)
  - if the filter matches no filing, search unfiltered rather than returning nothing
  - HARNESS (build with Exp1): record filter accuracy — chosen filename == gold `doc_name`. Splits "wrong document" from "wrong chunk" in the failure tree. Only meaningful in shared_store (single_store already scopes to the question's filing)
  - HARNESS (build with Exp1): make the oracle and retrieval context blocks identical, `[Document | Page | Section]` (Section once Exp2's heading path exists), so prompt shape can't explain a results gap
  - Exp1 retrieval must be ONE function taking arguments — query, retrieval method (BM25 / semantic / hybrid), metadata filters, top_k — not a hardcoded pipeline. Exp3's search tool is this same function, with the agent choosing those arguments at runtime; Exp1 passes fixed ones. Getting this wrong means writing retrieval twice
  - the query-enhancement prompt lives in one place, called by both: Exp1 calls it once up front, Exp3's agent does the same job through steps 1-2. Otherwise the two experiments quietly diverge

Generate answer

- decide model: GLM-5.3-flash with openrouter, determined with https://www.vals.ai/benchmarks/fabv2 (which we cant use as doesnt score retrieval)
  - same model, settings and answer prompt across every condition and experiment, so only retrieval differs
  - model-facing behaviour belongs to `sec_rag`, while the benchmark owns what each evaluation
    condition is allowed to supply
    - closed-book, oracle and long-context give the direct-answer pipeline no context, gold-page
      context or full-filing context
    - single-store and shared-store give a RAG pipeline one-document or all-document scope
    - `sec_rag` receives supplied context or document scope, never FinanceBench condition names
    - one shared formatter produces `[Document | Page | Section]` blocks for supplied pages and
      retrieved chunks, then one shared prompt/generator answers
- HARNESS config changes — do these BEFORE the next paid run (they invalidate answers generated at the old settings):
  - `reasoning_effort` low → high (GLM-5.3-Flash exposes `low`, `high`, `max` only — there is no `medium`)
  - `retrieval_depth` 5 → 10
  - `max_output_tokens` 2048 → 8192: reasoning tokens count as output, and an agent spends them every turn. Headroom is fine (largest prompt 535,722 of 1,048,576)
  - merge the judge branch (`feature/azure-ragas-judge`: judge, `did_not_fit`, accuracy reporting) — not on `main` yet
  - fix `metrics.cognitive_skills()`: it substring-matches and returns "unspecified" instead of the rule in Benchmark.md, so segment counts (57/36/21/14) won't reproduce
  - label each run with its experiment + ablation variant (e.g. `exp2-B-heading-path`, `exp3-A-single-pass`): set in config, snapshotted into the run's `config.toml`, and carried into `summary.json` / `summary.csv` so every results row says which system produced it. Both Exp2 and Exp3 have A/B/C ladders to compare, and retrofitting labels onto finished runs is painful

Experiment 3 - Agents: LLMs autonomously using tools in a loop
https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
this makes case for letting agents intelligently navigate files, THEN load its contents into context.


- Exp3 decisions (from review; confirm or strike)
  - FINAL TOOL SET: fixed one-off steps first, loop after
    1. (one-off) give the agent the list of available filings → it chooses metadata filters
    2. (one-off) first search, choosing retrieval method (BM25 / semantic / hybrid) + its own queries for each of them
    3. (loop) repeat search: new query, method or filters, informed by what came back. Recursive RAG - use thinking-mode for feedback loop (ReAct, HiRec). Let it also know whether its file was retrieved and retry
    4. (loop) calculator
    5. answer
  - deferred tools: fetch whole page (see Deferred section), decompose query: FinSTAR: decompose query into 'atomic' subqueries, use symbolic logic topology of (∩ / \ / aggregation), to figure out how they will lead to answer, before conductin retrieval and compare retrieved info against plan to adjust as go along
  - HARNESS (build at Stage 2.0, before Exp1 results): `execution/job.py` currently runs a fixed sequence (build context → one model call → save). An agent calls the model before and between retrievals, so job.py must hand the condition inputs to a selected pipeline instead: the plug becomes "pipeline returns answer + final chunks + retrieval stages + trace + usage/provenance", not "retriever returns chunks". Exp1/Exp2 then become one-round pipelines behind the same plug. Largest single item (~a day)
    - decompose the benchmark's `_build_retrieval_context()` rather than moving it: condition code
      keeps scope selection; `sec_rag` owns retrieval and context formatting
    - move `pipeline/generation.py`'s shared prompt, context-limit check and OpenRouter generation to
      `sec_rag` at the same time
    - keep metric calculation, prediction checkpointing, judging and reporting in the benchmark
    - update the FinanceBench implementation guide, README commands and fake-pipeline tests over
      all five conditions in the same migration slice
    - page recall/precision computed on the FINAL retrieval only (comparable with Exp1/Exp2). NOT tracking "all chunks seen" as a separate scored set — too much plumbing for the value; the trace already shows what was searched
    - report average k (passages used), since it now varies per question — as LOFin does
    - save a trace per question (its own file next to `predictions.jsonl`): each iteration's filters, query, tool, results. This is what makes the failure-mode tree usable for Exp3
  - loop caps: max iterations (start at 5) + max tool calls; hitting the cap is a FOURTH outcome alongside success / error / `did_not_fit` — skipped on resume, excluded from the accuracy denominator, counted in reports, so n stays constant across experiments. Checkpoint after each question
    - tell the model its budget in the prompt, and how many iterations remain each turn, so it can answer with what it has instead of being cut off mid-search
  - reasoning effort: raise from `low` to `high` for ALL conditions and experiments. GLM-5.3-Flash exposes `low`, `high`, `max` only — there is no `medium`. If Exp3 uses `max`, that is a deviation to justify in the write-up, or run it as an ablation
  - ablation ladder: (A) Exp1 single pass → (B) + retry when retrieval is empty/wrong filing → (C) + calculator/verification. A vs B alone is a result


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
