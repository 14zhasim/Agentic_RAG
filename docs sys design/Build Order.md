# Build Order

What gets built, in what order, with the requirements for each stage carried over from
`Systems Design Draft.md` so a stage can be built without switching files.

- `Benchmark.md` — benchmark requirements
- `Systems Design Draft.md` — the pipeline: what and how
- `Exp 1/`, `Exp 2/`, `Exp 3/` — each experiment's description and architecture
- this file — what and how, **in order**, with what to stub and what to cut

## The rule that keeps this cheap

Build the **data shape** for all three experiments now; build the **behaviour** when its
experiment arrives.

- carrying an unused field costs nothing; changing a field's shape after 64 filings are parsed and
  ~10k chunks embedded costs money and a day
  - parsing keeps section headings and nesting, even though Exp1 ignores them
  - every chunk carries a `heading_path` field, empty or not
  - the embedding cache is keyed by content hash, so heading embeddings later reuse it unchanged
  - retrieval is ONE function taking query, method, filters, top_k and a structure weight that
    defaults to zero — with the weight at zero it *is* Exp1
  - every run carries an experiment + ablation variant label
- the one piece of **behaviour** worth building early is the harness plug (pipeline returns answer
  + final chunks + trace + usage), because retrofitting it touches every module
  - so it is built at the START of Stage 2 (section 2.0), before Exp1's prediction rows and reports
    are written against the old fixed sequence. Stage 4 then only adds the loop behind a plug that
    already exists

## Timeline reality check

Code completion ~18 Sep, first draft 19 Sep, final 21 Sep (from `AGENTS.md`). Stages 0-4 total
roughly four days of building before any writing. Durations below are rough scope estimates, not
measurements: they assume working with Claude Code, and they exclude API run time and
first-contact debugging of unfamiliar libraries.

- Stages 0-2 are non-negotiable: they produce Exp1, the dissertation's spine
- Stage 3 (Exp2) is the most likely novel contribution — protect it
- Stage 4 (Exp3) is the largest build; expect rungs A and B only, and say so
- every stage has a **minimum result**: hitting the minimum on all three beats finishing one

---

# Stage 0 — Unblock (half a day)

**Goal:** answer the two questions that can change the design, and fix everything that would
corrupt or invalidate a paid run.

## 0.1 Spikes — do these first

- does `BM25Retriever`'s `filters` argument actually filter on our data?
  - build a retriever over ~5 chunks with different `doc_name` metadata, retrieve with a filter,
    confirm only matching chunks come back
  - real chunks don't exist until Stage 1, so hand-write ~5 short fixture chunks with overlapping
    words; this runs before ingestion on purpose, because the answer changes Stage 2's design
  - `bm25s` via LlamaIndex: `BM25Retriever.from_defaults(filters=MetadataFilters(...))` filters
    before searching; this was a recent addition, so PIN A CURRENT VERSION and test it on our data
  - if not: fall back to building the retriever from an already-filtered node list (2 lines)
  - keep it as a test so a version bump can't silently break it
  - **ANSWERED (`llama-index-retrievers-bm25==0.8.0`): yes, with one caveat.** Kept as
    `tests/test_bm25_metadata_filters.py`; the fallback is not needed
    - the filter is applied BEFORE scoring, as the design assumes: internally it becomes a
      `corpus_weight_mask` handed to `bm25s`, so out-of-scope chunks never compete for rank
    - BUT filtered chunks are not removed from the result list — they are returned with
      `score == 0.0` as padding whenever `similarity_top_k` exceeds the number of surviving
      chunks. **Retrieval must discard zero-score hits**, otherwise chunks from filings the
      question never asked about reach the context block
    - a filter matching nothing raises `ValueError` rather than silently returning the whole
      corpus, which is the behaviour we want
- is Azure Document Intelligence's section nesting good enough for Exp2's heading path?
  - run `prebuilt-layout` on 2-3 10-Ks, open the JSON, and look at `sections` (do sections nest,
    and do Item headings sit at the top level?) and at `paragraphs` with role `title` /
    `sectionHeading` (right text, right page?)
  - if nesting is flat or wrong, use PageIndex instead
  - structure source is OPEN either way: both work with the same method, as long as it gives each
    heading, its nesting level and its start page
- if the fallback is needed, PageIndex's how-to (from the draft, so a failed spike isn't a dead end)
  - use `get_tree()` to pull structure (pure parsing of e.g. the contents page, section headings) —
    section headings and the starting page for each one, and nesting to indicate a sub-heading
  - can use 'local mode' to be quick, **but first check the PDFs are machine-readable, not
    image-only (for both datasets!)** — this precheck must happen before committing to PageIndex
  - it only shows the starting page, not the ending page; compute the end from nesting plus the
    starting page of subsequent sections
  - then construct a data structure storing the headings (each with its own embedding) and their
    page ranges; for our own parsed PDF, compare a chunk's page against this to know which heading
    belongs to it

## 0.2 Config changes — BEFORE the next paid run

These invalidate answers generated at the old settings, so they come before spending anything.

- `reasoning_effort` low → high
  - GLM-5.3-Flash exposes `low`, `high`, `max` only — there is no `medium`
  - same model, settings and answer prompt across every condition and experiment, so only
    retrieval differs
  - if Exp3 uses `max`, that is a deviation to justify in the write-up, or run it as an ablation
- `retrieval_depth` 5 → 10 (FinCARDS uses top-10)
- `max_output_tokens` 2048 → 8192
  - reasoning tokens count as output, and an agent spends them every turn
  - headroom is fine: largest prompt 535,722 of 1,048,576

## 0.3 Correctness fixes

- merge the judge branch (`feature/azure-ragas-judge`: judge, `did_not_fit`, accuracy reporting) —
  not on `main` yet
- fix `metrics.cognitive_skills()`
  - it substring-matches and returns "unspecified" instead of the rule in `Benchmark.md`, so
    segment counts (57/36/21/14) won't reproduce
- label each run with its experiment + ablation variant (e.g. `exp2-B-heading-path`,
  `exp3-A-single-pass`)
  - set in config, snapshotted into the run's `config.toml`, carried into `summary.json` /
    `summary.csv`, so every results row says which system produced it
  - both Exp2 and Exp3 have A/B/C ladders to compare, and retrofitting labels onto finished runs is
    painful

## 0.4 Tech stack set-up

- new dependencies, added and exact-pinned to match the existing house style (DONE):
  `llama-index-core==0.14.24`, `llama-index-retrievers-bm25==0.8.0`,
  `llama-index-vector-stores-chroma==0.6.0`, `chromadb==1.5.9`,
  `azure-ai-documentintelligence==1.0.2`, `voyageai==0.5.0`, `numpy==2.5.3`
  - `bm25s==0.3.11` arrives transitively via the BM25 retriever
  - the two LlamaIndex Voyage wrappers are NOT installed — see the Voyage decision below
- credentials to add to `.env` and `.env.example`: `VOYAGE_API_KEY`,
  `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` (alongside the existing
  `OPENROUTER_API_KEY`, `AZURE_DEEPSEEK_API_KEY`, `AZURE_DEEPSEEK_ENDPOINT`)
- add instructions to create and activate the venv after creating the uv project; include these in
  `README.md`
- what each tool is for
  - LlamaIndex — orchestration + components: chunking and markdown parsing (heading paths), BM25
    retrieval (wraps `bm25s`) with metadata filters, semantic retrieval with metadata filters, RRF
    fusion (`QueryFusionRetriever(mode="reciprocal_rerank")`, `num_queries=1` to stop it inventing
    extra queries), reranking (Voyage post-processor), and the Exp3 agent loop (`max_iterations` +
    `early_stopping_method="generate"`)
  - Chroma — vector store + chunk metadata + `where` filtering before search, so we don't hand-roll
    save/load. For Exp2, pull the embeddings out and score in numpy
  - Azure Document Intelligence — parsing (PageIndex is the live fallback)
  - Voyage API — voyage-4-lite embeddings + `rerank-3-lite` reranker; one account, one
    `VOYAGE_API_KEY`, two endpoints
    - DECIDED: call Voyage **directly** through the `voyageai` client, for both embeddings and
      reranking. Two fewer packages, and we assemble the candidate list ourselves anyway, so the
      LlamaIndex wrappers buy nothing. It also keeps `input_type` (`document` for chunks, `query`
      for questions) explicit at the call site rather than hidden in a wrapper's default — getting
      it wrong degrades retrieval quietly instead of erroring
  - OpenAI SDK via OpenRouter — answer generation (already pinned, with provider routing).
    LlamaIndex components don't call the model, so nothing clashes
  - pandas + pytest — reporting and tests (already in place)
- we write ourselves: Exp2's scorer (~20-line `BaseRetriever` subclass — a weighted sum of two
  similarity scores per chunk, which RRF can't express because it fuses ranked lists), the Exp3 tool
  functions, trace logging, and the glue into the benchmark harness
- storage constraint that keeps Elasticsearch cheap later: chunks + embeddings are saved as plain
  files, and search sits behind the existing retriever interface; ES then becomes one more loader +
  retriever
- BM25 filtering behaviour worth knowing: filtering applies to the chunks searched; BM25 word
  statistics still come from the whole corpus (same as Elasticsearch, so results stay comparable if
  we switch)
- LATER (document as future work): Elasticsearch as a second retriever behind the same interface —
  keyword, vectors (HNSW) and filters in one query
  - at ~10k chunks exact search is faster to build and more accurate, so it buys skills, not results

**Done when:** both spikes answered, `uv run pytest` green, and a 5-question **oracle** smoke run at
high effort shows sane cost and latency (no retrieval pipeline exists yet, so closed-book or
oracle are the only conditions available).

**Minimum result:** the config and correctness fixes. Spikes can be answered during Stage 1 if
pressed.

---

# Stage 1 — Ingestion (about a day)

**Goal:** parse once, chunk, index, embed — with the Exp2 hooks built in, so Exp2 is "populate a
field and turn a weight on" rather than a refactor.

## 1.1 Parse

- parse document: extract text, identify structural elements (sections, titles, tables, text,
  figures), preserve information like table structure and data
  - output as `.md` — like Kim et al. (2025) found improvements on — or JSON
  - Jimeno Yepes (2024) used a basic VLM to identify text, titles, tables and chunk along those
    boundaries (+ an LLM-generated summary to embed — that summary step is DEFERRED, add later)
  - figures: no 10-K figure is expected to carry answer content, so nothing is built for them, but
    they are not stripped from the parser output either
  - rationale: helps the LLM read non-plaintext formats, like tables, as interpretable
    representations
  - just using an LLM misses metadata (year, file type, company) and structural content (e.g. the
    subheading of each chunk)
  - preserve metadata in the doc i.e. page numbers, denoting (sub)headings, tables etc.
- parsing options considered (keep for the write-up justification)
  - "VLM-agentic" parsers read the page like a person, with a correction/verification pass: e.g.
    LlamaParse, Reducto ($$$), Azure Document Intelligence ($100 for ~80 docs around 12,000 pages,
    82.7% on RD-TableBench, **use base Layout**), GPT-5.6. Good if endless edge cases exist in human
    writing — just use a VLM
  - "layout engine" parsers use specialised detection models + rules, no LLM in the loop by default,
    self-hostable and free: PyMuPDF / pymupdf4llm, good for machine-generated docs like statements,
    apparently not that good for tables
  - verdict: use Azure Document Intelligence as I have startup credits
- Azure Document Intelligence, `prebuilt-layout` (base Layout), markdown output
  - setup is light: create resource → endpoint + key → `azure-ai-documentintelligence` SDK
  - paid tier only (free tier reads first 2 pages), ~$10 / 1,000 pages → ~$100 budget for this
    corpus
- parser output decisions
  - parse once, cache forever: save Azure's raw JSON per PDF (out of Git, with a manifest like data
    prep), never re-parse when chunking changes
  - raw JSON is the source of truth; markdown is one field inside it (`content`), other fields
    (`pages`, `paragraphs`, `tables`, `sections`) point into that markdown by character position,
    which is how each piece maps back to a page
  - strip page headers/footers/page numbers (Azure labels these) before chunking
  - figures are out of scope

## 1.2 Page mapping — write the test

- page numbers: FinanceBench `evidence_page_num` is zero-indexed; Azure `pageNumber` is 1-indexed →
  `evidence_page_num = pageNumber - 1`
  - never use the footer's printed page number
  - **add a test** — every page metric in the harness depends on this

## 1.3 Chunk

- chunk (+ structure parsing) + save metadata for each chunk, for filtering chunks
- chunking decisions
  - use LlamaIndex (not LangChain) for splitting, consistent with the BM25/hybrid/routing choice
  - chunk within page boundaries: each chunk covers exactly one page, so page metrics stay exact.
    Overlap applies within a page only
    - HARNESS: assert one page per chunk — a chunk with two pages is a chunking bug
  - tables: each table is its own chunk; an oversized table is split by rows, repeating the header
    row. Everything else → the 1,024-token splitter (count tokens, not characters)
  - 1,024 tokens, 30 overlap — HiREC (Choe et al., 2025), who used LangChain's
    `RecursiveCharacterTextSplitter`; use LlamaIndex's equivalent sentence/token splitter instead,
    configured to count tokens
  - dropped "semantic chunking via regex": the recursive/sentence splitter already prefers
    paragraph then sentence breaks
- metadata per chunk: SEC filing type + company + financial year + page number
  - document-level metadata (company, doc_type, doc_period) comes from
    `financebench_document_information.jsonl`, **not from parsing** — this keeps the duplicate
    `doc_name` conflict guard in `Benchmark.md` meaningful
  - there is no ticker field in FinanceBench, so company stands in for ticker throughout
  - the filename rule (`COMPANY_YEAR_TYPE.pdf`, parsed from the RIGHT) is a QUERY-TIME concern —
    how the LLM selects a filing in Stage 2.1 — not the source of chunk metadata
- **Exp2 hook, built now:** every chunk carries a `heading_path` field
  - populate it in this stage if the Azure spike passed, rather than re-parsing later
  - attribute headings to chunks: from each heading's start page, compute its page range (ends where
    the next heading at the same or higher level starts); a chunk gets the headings whose range
    covers its page
    - if two headings start on the same page, attribute the one covering more of that page
  - Fin-STAR constraint: depth ≤ 5 — if the heading path has more than 5 levels, drop the excess
    (deepest) headings
  - save the path as a list of headings, top level first. Each level is embedded separately at Stage
    3.2, not joined into one string
  - leave the field empty only if the spike failed and PageIndex has not yet been wired in

## 1.4 Inspect the chunks

- _view chunks manually, writing has infinite edge cases_
- inspection script: "show all chunks for doc X, page Y"
  - run it on a question's gold page to spot missed-table failures

## 1.5 Pre-process for keyword search

- what normal keyword search does, for both the query AND the document store:
  - lowercase → remove punctuation → split into tokens → remove empty tokens → remove stop words →
    stem tokens → **split letters from digits** (`FY2018` → `fy` + `2018`) → compare tokens
  - the letter/digit split matters: 98 of 112 questions use FY-style years, filings write "2018" or
    "fiscal 2018", so `fy2018` as one token matches nothing. Expand 2-digit years (`FY22` → `2022`)
  - checked against all 112 questions: punctuation removal is safe (no question uses `$`; they write
    "USD"), lowercasing is safe, stemming is safe because both sides are stemmed
- reference implementations to reuse: `/Users/zubairasim/rag-search-engine/course_notes/module-01-preprocessing.md`
  and `/Users/zubairasim/rag-search-engine/cli/lib/keyword_search.py`
- the four structures a BM25 index needs (what `bm25s` builds for us)
  - map tokens to each chunk id
  - map chunk ID to the chunk object itself (which carries page and doc metadata too)
  - map chunk id to a map of words and each of their counts
  - map each chunk to its chunk length
- use `bm25s` via LlamaIndex's `BM25Retriever` instead of hand-built maps (same structures, faster)
  - the token map only needs chunk IDs; page and doc name live in the chunk's metadata
- persist the index: save and reload it from disk (LlamaIndex's `persist` / `from_persist_dir`), so
  it is not rebuilt on every run
  - plain files now; Elasticsearch later becomes one more loader behind the same interface

## 1.6 Embed

- embed chunk (using the same chunks as above) with voyage-4-lite
  - cheapest embedder above 80% on the RTEB(fin) leaderboard
  - first 200M tokens free, then $0.02/1M
- embedding decisions
  - cost: ~10k chunks × ~1k tokens ≈ 10M tokens, inside the free allowance even with several
    re-chunks
  - label inputs: chunks `input_type="document"`, questions `input_type="query"`
  - cache each embedding keyed by model name + hash of chunk text, so re-chunking only re-embeds
    changed chunks
- store in Chroma alongside each chunk's metadata (year, company, report type, text) so we can
  filter by all or none

**Done when:** chunks exist for all 64 filings, the page-conversion test passes, and the gold page
of 3 sample questions visibly contains the answer.

**Minimum result:** chunks with correct page numbers. Heading paths may be empty.

---

# Stage 2 — Experiment 1 (about a day, plus run time)

**Goal:** the baseline architecture, end to end, with the metrics that every later experiment
reports.

## 2.0 Harness plug FIRST (moved up from Exp3 — see the rule at the top)

- `execution/job.py` currently runs a fixed sequence: build context → one model call → save
- change it now so it hands the question to a pipeline that returns answer + final chunks + trace +
  usage
  - Exp1 and Exp2 are then one-round pipelines behind that plug; Stage 4 adds the loop without
    rewriting the interface underneath finished runs
  - doing this after Stage 2/3 would mean re-verifying every prediction row and report built against
    the old sequence

## 2.1 LLM query enhancement

- to what extent? check boot.dev lesson 7 + the Claude prompting guide:
  https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- ONE LLM call returning structured JSON: company, year(s), doc_type, keyword query (shorthand
  expanded — PPNE → property, plant and equipment; COGS, DPO, FCF, capex), semantic query
- let it know whether its desired file exists and was retrieved
  - like Claude Code, naively give the prompt all available metadata / files it can search from
  - in Exp1 this is one-shot feedback in the prompt; retrying on that feedback is Exp3's loop
- metadata filtering: filenames are `COMPANY_YEAR_TYPE.pdf`, so the LLM supplies all three and we
  select the filing by filename (not from metadata table columns)
  - simplest: give it the filename list and have it return a filename — a closed list means
    "3M" vs "3M Company" can't mismatch
  - filter chunks BEFORE search
  - if the filter matches no filing, search unfiltered rather than returning nothing
- the query-enhancement prompt lives in ONE place, called by both Exp1 and Exp3
  - Exp1 calls it once up front; Exp3's agent does the same job through its first two steps
  - otherwise the two experiments quietly diverge

## 2.2 Retrieval — one function, taking arguments

- Exp1 retrieval must be ONE function taking arguments — query, retrieval method (BM25 / semantic /
  hybrid), metadata filters, top_k — not a hardcoded pipeline
  - Exp3's search tool is this same function, with the agent choosing those arguments at runtime;
    Exp1 passes fixed ones
  - getting this wrong means writing retrieval twice
  - **Exp2 hook:** the same function takes a structure weight, defaulting to zero
- hybrid search
  - BM25 (keyword), like FinCARDS (2026): k1=1.5, b=0.75
    - BM25(term, doc) = bm25_tf × bm25_idf
    - BM25 TF: (tf × (k1 + 1)) / (tf + k1 × length_norm), length_norm = 1 − b + b × (doc_length /
      avg_doc_length)
    - "document" here means whatever unit is indexed — we index chunks, so we get chunk scores
  - semantic search: embed the user query, cosine similarity against each chunk, rank by top-k
    (if too many embeddings to scan, find nearest using the database — Chroma handles this)
- scope filtering: keep only chunks whose `doc_name` is in scope FIRST, then rank each list within
  scope, then fuse
  - RRF uses ranks, so ranking globally then filtering would leave gaps in ranks and change fused
    scores
- RRF, k=60
  - fuse longer lists (e.g. top-50 from each search), then cut to top-k after fusion
- no auto-merge for Exp1: page-bounded ~1,024-token chunks are already ~one page, so parent ≈ chunk

## 2.3 Rerank

- use a reranker on the top-k to retrieve top-n (query-document)
  - FinSage (2025) used BAAI/bge-reranker-v2-gemma via FlagEmbedding — that needs a GPU, so use the
    Voyage reranker API instead (same key/account as embeddings)
  - model: `rerank-3-lite` — $0.02/1M tokens with a 200M free allowance
    - NOT `rerank-2.5` / `rerank-2.5-lite`: those have ZERO free allowance, and the 3 line is newer
      (Voyage state it is strictly better on quality, context length, latency and throughput)
    - our volume is ~6M tokens per full run (query tokens × candidates + all candidate tokens), so
      ~3% of the free allowance — `rerank-3` is affordable too, it is one string away
    - there is no trustworthy public reranker leaderboard, and published comparisons sit within 1-3
      NDCG points of each other, so settle it with OUR pre/post-rerank page metrics on the
      50-question pattern check rather than a citation

## 2.4 Feed the LLM

- answer model: GLM-5.3-flash via OpenRouter
  - chosen on accuracy/cost from https://www.vals.ai/benchmarks/fab v2 (which we can't use as a
    benchmark itself, as it doesn't score retrieval)
  - same model, settings and answer prompt across every condition and experiment, so only retrieval
    differs
- feed the LLM top-n results, including metadata and heading as well
- context format: make the oracle and retrieval context blocks identical, `[Document | Page |
  Section]` (Section once Exp2's heading path exists), so prompt shape can't explain a results gap

## 2.5 Metrics built here

- HARNESS: compute page metrics both before and after reranking (call `page_metrics()` twice, two
  sets of fields on the prediction row), so the reranker's effect is visible
- HARNESS: record filter accuracy — chosen filename == gold `doc_name`
  - splits "wrong document" from "wrong chunk" in the failure tree
  - only meaningful in shared_store (single_store already scopes to the question's filing)
- HARNESS: the prediction row records embedding + rerank cost too, not just the answer model's

## 2.6 Run and report

- dev loop before the full run
  - smoke test (5-10 questions): check the pipeline works
  - pattern check (50 questions): identify patterns across segmented question types, e.g. does
    chunking tables work? Sample with a fixed seed, stratified by `question_type`. Run it as a test
    suite during development
- run single-store and shared-store; re-run closed-book, oracle and long-context at the new
  settings
- judge the saved answers, then report
- record a benchmark run for every pipeline change

**Done when:** a segmented report exists for all five conditions, with answer accuracy and
retrieval metrics.

**Minimum result:** Exp1 vs the three baselines, judged and reported. **If everything else fails,
this is still a dissertation.**

---

# Stage 3 — Experiment 2 (half a day, plus run time)

**Goal:** Exp1 with ONE change — the dense score. Parsing, chunks, query enhancement, BM25, RRF,
reranker and generation stay identical, so any difference is attributable to structure.

## 3.1 Heading path per chunk

- save the chunk's heading path (list of headings, top level first) as metadata on the chunk
  - each heading level is embedded separately and combined at retrieval, not joined into one string
- attribute headings to chunks: from each heading's start page, compute its page range (ends where
  the next heading at the same or higher level starts); a chunk gets the headings whose range
  covers its page
  - if two headings start on the same page, attribute the one covering more of that page
- Fin-STAR constraint: depth ≤ 5 applies now — if the heading path has more than 5 levels, drop the
  excess (deepest) headings
  - discriminativeness only applies to the virtual node, so it comes back when that does
- SLM virtual node DEFERRED — the method, for when it returns
  - Fin-STAR: for chunk `ci` and hierarchical path `hi`, use an SLM `Gθ` to generate a virtual node
    `vi = Gθ(ci, hi)`, giving the enriched path `Pi = hi ⊕ vi`
  - constrained by (1) depth control, `Depth(Pi) ≤ 5`, and (2) discriminativeness, requiring `vi` to
    capture essential semantics absent from `hi`
  - depth control already applies now (Stage 1.3); discriminativeness only applies to the virtual
    node, so it returns with it
- no 2d vector store is needed, and here is why the shortcut is valid
  - the theory concatenates into `H+_ci = [H_si ; H_ci]` and duplicates the query into
    `q+ = [q ; q]`, so the score is `q+ · H+_ci = (q · H_si) + (q · H_ci)`
  - a dot product of concatenated vectors is the sum of the halves' dot products, so computing the
    two scores separately and adding them is identical — with no training loop, labelled data, or
    access to model internals

## 3.2 Heading embeddings

- embed each heading level separately; each unique heading embedded once, reused across chunks
- store heading embeddings separately from chunk embeddings, linked by chunk ID via the chunk's
  heading path, not in metadata

## 3.3 The scorer

- combine the heading embeddings into one structure vector using softmax weights
  - **the weights come from CHUNK-to-heading similarity, not query-to-heading** — this is the part
    that is easy to get wrong
  - given a chunk embedding `H_ci` and its path levels embedded separately as `V_j`: score each
    level by dot product `s_j = H_ci · V_j`, turn those into weights with a softmax
    `a_j = exp(s_j/T) / Σ_k exp(s_k/T)`, then aggregate: `H_si = Σ_j a_j · V_j`
  - this is exactly attention with the chunk as query and the headings as keys and values, with no
    learned projections (`W_Q = W_K = W_V = I`)
  - all weights are ≥ 0 and sum to 1, so `H_si` is a weighted average of the heading embeddings
  - then normalise `H_si` to length 1 (an average of length-1 vectors is shorter than 1)
  - normalise the chunk and query vectors to length 1 as well, so all three scores share a scale
- score = content similarity + structure similarity
  - no 2d vector store needed: the concatenated score equals (q · H_si) + (q · H_ci), so compute
    the two scores separately and add them
  - chunks with no heading fall back to the chunk score alone
  - implementation: subclass LlamaIndex's `BaseRetriever` (~20 lines) — RRF can't express this,
    because it fuses ranked lists rather than scoring chunks
- softmax divisor (temperature `T`) is a CONFIG SETTING, starting value 0.05; try the theory's
  √d-scaling only if time allows
  - the √d in the theory text (√1024 = 32) squashes heading-to-chunk similarities (between -1 and 1)
    towards 0, so every level ends up weighted equally and the structure vector becomes a plain
    average
  - measured on synthetic vectors (top 10 of 500 chunks, vs the equal-weights version): divisor 1
    returns ~9.8/10 of the same chunks (effectively identical), divisor 0.05 returns ~7.3/10 (a
    genuinely different system)
  - write-up: state that the divisor departs from the √d in the source theory, and say why

## 3.4 Verify, then run

- **check the weight-zero case reproduces Exp1 exactly** before trusting any gain
- report page metrics before reranking too: the reranker only sees chunk text and could wash out a
  structural gain
- build order: (A) Exp1 → (B) heading path, no virtual node → (C) path + virtual node (deferred)
- framing: training-free approximation of Fin-STAR, not a replication

**Done when:** A and B runs exist under their labels, with pre-rerank metrics for both.

**Minimum result:** A vs B on page metrics alone, even if the judge pass is thin.

---

# Stage 4 — Experiment 3 (a day or more)

**Goal:** the agent loop, on top of Exp1's retrieval, so the difference is attributable to the loop.

- the motivating argument: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  makes the case for letting agents intelligently navigate files, THEN load the contents into context
- prep if time allows: research papers (+ Claude Opus chat) and the boot.dev agent course
- the plug is already built (Stage 2.0), so this stage only adds tools, the loop and the trace

## 4.1 What the plug now has to carry

- the plug itself was built in Stage 2.0: "pipeline returns answer + final chunks + trace + usage",
  not "retriever returns chunks"
  - an agent calls the model before and between retrievals, which is why the fixed sequence (build
    context → one model call → save) could not stay
- page recall/precision computed on the FINAL retrieval only (comparable with Exp1/Exp2)
  - NOT tracking "all chunks seen" as a separate scored set — too much plumbing for the value; the
    trace already shows what was searched
- report average k (passages used), since it now varies per question — as LOFin does
- save a trace per question (its own file next to `predictions.jsonl`): each iteration's filters,
  query, tool, results
  - this is what makes the failure-mode tree usable for Exp3
  - trace logging is ours, not LlamaIndex's event system: each tool appends what it did and returned
    to a per-question list (~3 lines per tool, and it doesn't break when the framework changes its
    events)

## 4.2 Tool set — fixed one-off steps first, loop after

1. (one-off) give the agent the list of available filings → it chooses metadata filters
2. (one-off) first search, choosing retrieval method (BM25 / semantic / hybrid) + its own queries
   for each of them
3. (loop) repeat search: new query, method or filters, informed by what came back. Recursive RAG —
   use thinking-mode for the feedback loop (ReAct, HiREC). Let it also know whether its file was
   retrieved and retry
4. (loop) calculator
5. answer

- deferred tools, with the mechanism kept so they can be resurrected cheaply
  - **fetch whole page** (PDFTriage pattern): pull the full page, or the next one, once a promising
    chunk is found. Targets tables split across pages
  - **decompose query** (Fin-STAR): break the query into 'atomic' sub-queries, use a symbolic logic
    topology of (∩ / \ / aggregation) to figure out how they lead to the answer before conducting
    retrieval, then compare retrieved info against the plan and adjust as you go
    - deferred because it pays off on multi-document questions = LOFin, which is deferred; every
      FinanceBench question sits inside one filing
- verify the model emits well-formed tool calls on ~5 questions BEFORE building the loop
  - if GLM-5.3-flash can't, a stronger model for the agent breaks "same model everywhere" and must
    be stated

## 4.3 Loop control

- loop caps: max iterations (start at 5) + max tool calls
  - max tool calls has no agreed value yet — OPEN. A safe starting point is 3 × max iterations (15),
    high enough not to bite in normal runs but low enough to stop a runaway
  - hitting the cap is a FOURTH outcome alongside success / error / `did_not_fit` — skipped on
    resume, excluded from the accuracy denominator, counted in reports, so n stays constant across
    experiments
  - checkpoint after each question
- tell the model its budget in the prompt, and how many iterations remain each turn, so it can
  answer with what it has instead of being cut off mid-search
  - this is a deliberate, stated deviation from "the answer prompt is identical everywhere"
- use LlamaIndex's agent for the loop: `max_iterations` + `early_stopping_method="generate"`, so a
  capped run still answers instead of erroring
  - remaining glue: turn the framework's "cap reached" signal into our fourth outcome

## 4.4 Run

- ablation ladder: (A) Exp1 single pass → (B) + retry when retrieval is empty/wrong filing → (C) +
  calculator/verification
- A vs B alone is a result

**Done when:** rung B completes all 112 questions without a stuck loop, with traces saved.

**Minimum result:** rung B on a subset with traces, written up honestly as a partial run.

---

# Failure analysis (runs after any stage that produces results)

Automatic first pass from saved results; hand-label only a sample.

- oracle correct, retrieval condition wrong → retrieval failure
  - page recall = 0 in shared_store but > 0 in single_store → wrong document
  - right document but page recall = 0 → wrong section/chunk
  - page recall > 0 but < 1 → insufficient recall
- oracle wrong → reasoning or generation failure (retrieval is not the cause)
- `did_not_fit` → context-limit outcome, kept separate from all three
- sub-types (arithmetic vs hallucination, missed table) → hand-label a small sample only, using the
  judge's reason plus the question's cognitive skill label

---

# Cut list, in the order to cut

Cut from the bottom up. None of these invalidate the work; each becomes a stated limitation.

1. Exp3 rung C (calculator and verification)
2. The softmax divisor comparison in Exp2
3. Exp3 entirely — describe the design, report Exp1 and Exp2
4. Judged retrieval metrics (context recall/precision/faithfulness) — already written up as a
   methodology limitation
5. Bootstrap confidence intervals — but then make no "A beats B" claim

Already deferred and staying deferred: LOFin, the SLM virtual node, HiChunk, query decomposition,
fetch-whole-page, Elasticsearch, OP-RAG.

# Write-up runs alongside, not after

- write each experiment's description before building it — they are a page each and they expose
  gaps while there is still time to act
  - each description must state its baseline AND its ablations explicitly
- after each stage, paste the numbers into that experiment's Results section the same day
  - results written up the day they are produced take minutes; a week later they take hours of
    re-deriving what a run's settings were
