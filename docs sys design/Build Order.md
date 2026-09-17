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

## Timeline reality check

Code completion ~18 Sep, first draft 19 Sep, final 21 Sep. Stages 0-4 total roughly four days of
building before any writing.

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
  - `bm25s` via LlamaIndex: `BM25Retriever.from_defaults(filters=MetadataFilters(...))` filters
    before searching; this was a recent addition, so PIN A CURRENT VERSION and test it on our data
  - if not: fall back to building the retriever from an already-filtered node list (2 lines)
  - keep it as a test so a version bump can't silently break it
- is Azure Document Intelligence's section nesting good enough for Exp2's heading path?
  - run `prebuilt-layout` on 2-3 10-Ks, open the JSON, and look at `sections` (do sections nest,
    and do Item headings sit at the top level?) and at `paragraphs` with role `title` /
    `sectionHeading` (right text, right page?)
  - if nesting is flat or wrong, use PageIndex instead
  - structure source is OPEN either way: both work with the same method, as long as it gives each
    heading, its nesting level and its start page

## 0.2 Config changes — BEFORE the next paid run

These invalidate answers generated at the old settings, so they come before spending anything.

- `reasoning_effort` low → medium
  - same model, settings and answer prompt across every condition and experiment, so only
    retrieval differs
  - if Exp3 uses `high`, that is a deviation to justify in the write-up, or run it as an ablation
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

- new dependencies: `llama-index-core`, `llama-index-retrievers-bm25`, `chromadb`,
  `azure-ai-documentintelligence`, `voyageai`, `numpy`
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
  - Voyage API — voyage-4-lite embeddings + reranker
  - OpenAI SDK via OpenRouter — answer generation (already pinned, with provider routing).
    LlamaIndex components don't call the model, so nothing clashes
  - pandas + pytest — reporting and tests (already in place)
- we write ourselves: Exp2's scorer, the Exp3 tool functions, trace logging, and the glue into the
  benchmark harness
- LATER (document as future work): Elasticsearch as a second retriever behind the same interface

**Done when:** both spikes answered, `uv run pytest` green, and a 5-question smoke run at medium
effort shows sane cost and latency.

**Minimum result:** the config and correctness fixes. Spikes can be answered during Stage 1 if
pressed.

---

# Stage 1 — Ingestion (about a day)

**Goal:** parse once, chunk, index, embed — with the Exp2 hooks built in, so Exp2 is "populate a
field and turn a weight on" rather than a refactor.

## 1.1 Parse

- parse document: extract text, identify structural elements (sections, titles, tables, text),
  preserve information like table structure and data
  - rationale: helps the LLM read non-plaintext formats, like tables, as interpretable
    representations
  - just using an LLM misses metadata (year, file type, company) and structural content (e.g. the
    subheading of each chunk)
  - preserve metadata in the doc i.e. page numbers, denoting (sub)headings, tables etc.
- Azure Document Intelligence, `prebuilt-layout`, markdown output
  - setup is light: create resource → endpoint + key → `azure-ai-documentintelligence` SDK
  - paid tier only (free tier reads first 2 pages), ~$10 / 1,000 pages
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
  - 1,024 tokens, 30 overlap — HiREC (Choe et al., 2025)
  - dropped "semantic chunking via regex": the recursive/sentence splitter already prefers
    paragraph then sentence breaks
- metadata per chunk: SEC filing type + company + financial year + page number
  - document-level metadata comes from the filename (`COMPANY_YEAR_TYPE.pdf`), parsed from the
    RIGHT — `JOHNSON_JOHNSON_2022_10K` has an underscore in the company name
  - there is no ticker field in FinanceBench
- **Exp2 hook, built now:** every chunk carries a `heading_path` field (populated if the Azure
  spike passed, empty otherwise)

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
- use `bm25s` via LlamaIndex's `BM25Retriever` instead of hand-built maps (same structures, faster)
  - the token map only needs chunk IDs; page and doc name live in the chunk's metadata

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

## 2.1 LLM query enhancement

- ONE LLM call returning structured JSON: company, year(s), doc_type, keyword query (shorthand
  expanded — PPNE → property, plant and equipment; COGS, DPO, FCF, capex), semantic query
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

## 2.4 Feed the LLM

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
- SLM virtual node DEFERRED

## 3.2 Heading embeddings

- embed each heading level separately; each unique heading embedded once, reused across chunks
- store heading embeddings separately from chunk embeddings, linked by chunk ID via the chunk's
  heading path, not in metadata

## 3.3 The scorer

- combine the heading embeddings into one structure vector using softmax weights, then normalise it
  to length 1 (an average of length-1 vectors is shorter than 1)
- score = content similarity + structure similarity
  - no 2d vector store needed: the concatenated score equals (q · H_si) + (q · H_ci), so compute
    the two scores separately and add them
  - chunks with no heading fall back to the chunk score alone
  - implementation: subclass LlamaIndex's `BaseRetriever` (~20 lines) — RRF can't express this,
    because it fuses ranked lists rather than scoring chunks
- softmax divisor (temperature) is a CONFIG SETTING, starting value 0.05; try 1 or √d only if time
  allows
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

## 4.1 Harness plug — the largest single item

- `execution/job.py` currently runs a fixed sequence (build context → one model call → save). An
  agent calls the model before and between retrievals, so job.py must hand the whole question over
  instead
  - the plug becomes "pipeline returns answer + final chunks + trace + usage", not "retriever
    returns chunks"
  - Exp1/Exp2 then become one-round pipelines behind the same plug
- page recall/precision computed on the FINAL retrieval only (comparable with Exp1/Exp2)
  - NOT tracking "all chunks seen" as a separate scored set — too much plumbing for the value; the
    trace already shows what was searched
- report average k (passages used), since it now varies per question — as LOFin does
- save a trace per question (its own file next to `predictions.jsonl`): each iteration's filters,
  query, tool, results
  - this is what makes the failure-mode tree usable for Exp3
  - trace logging is ours, not LlamaIndex's event system: each tool appends what it did and returned
    to a per-question list

## 4.2 Tool set — fixed one-off steps first, loop after

1. (one-off) give the agent the list of available filings → it chooses metadata filters
2. (one-off) first search, choosing retrieval method (BM25 / semantic / hybrid) + its own queries
   for each of them
3. (loop) repeat search: new query, method or filters, informed by what came back. Recursive RAG —
   use thinking-mode for the feedback loop (ReAct, HiREC). Let it also know whether its file was
   retrieved and retry
4. (loop) calculator
5. answer

- deferred tools: fetch whole page; decompose query into atomic sub-queries with Fin-STAR's
  symbolic logic topology
- verify the model emits well-formed tool calls on ~5 questions BEFORE building the loop
  - if GLM-5.3-flash can't, a stronger model for the agent breaks "same model everywhere" and must
    be stated

## 4.3 Loop control

- loop caps: max iterations (start at 5) + max tool calls
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
- after each stage, paste the numbers into that experiment's Results section the same day
  - results written up the day they are produced take minutes; a week later they take hours of
    re-deriving what a run's settings were
