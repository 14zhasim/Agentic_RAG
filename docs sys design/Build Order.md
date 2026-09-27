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

Revised 18 Sep 2026 after a two-week extension. The time is **not** two clear weeks — the job takes
the weekdays, so it is two four-day blocks plus the days between them.

| Block | Dates | What must be true at the end |
|---|---|---|
| Build | Fri 18 – Mon 21 Sep | Stages 0-3 done: Exp1 and Exp2 built, run, and their results written into `Exp 1.md` / `Exp 2.md`. Stage 4 (Exp3) **not expected** |
| Work | Tue 22 – Thu 24 Sep | Nothing owed. At most, tidy notes taken during the build |
| Write | Fri 25 – Sun 27 Sep | A complete first draft covering Exp1 and Exp2 end to end, results included |
| Work | Mon 28 Sep – Thu 1 Oct | Nothing owed |
| Finish | Fri 2 – Sun 4 Oct | Draft finished. Only then, in this order, if time genuinely remains: (1) debug and improve the existing experiments, (2) Exp3, (3) deferred items |
| Submit | Mon 5 Oct | Final submission — CONFIRM against the formal extension |

Durations in the stages below are rough scope estimates, not measurements: they assume working
with Claude Code, and they exclude API run time and first-contact debugging of unfamiliar
libraries.

- Stages 0-2 are non-negotiable: they produce Exp1, the dissertation's spine
- Stage 3 (Exp2) is the most likely novel contribution — protect it
- Stage 4 (Exp3) now sits **below writing** in priority. A designed-but-not-run Exp3 chapter is an
  acceptable outcome and the cut list already anticipates it. An unfinished dissertation is not
- every stage has a **minimum result**: hitting the minimum on all three beats finishing one

### Scope freeze

The extension buys **depth on work already scoped**. It does not buy more scope.

- nothing moves back up the cut list and nothing deferred becomes live again — in particular LoFin,
  Exp2's rung C virtual node, and Exp3's query decomposition stay out
- a new idea goes into Deferred as future work, not into the build
- the extra time goes on understanding what is being built, testing it, and failure analysis — the
  things the original three-day plan was going to skip
- if a block finishes early the next block starts early; it does not expand to fill the time

### Guarding against drift

The old plan had a deadline close enough to feel. This one does not, so the pressure has to come
from the blocks above rather than from the date.

- close a numbered stage each session; a session ending mid-stage records exactly what remains
- a stage is not done until its result is in the relevant `Exp N.md` — see the write-up rule at the
  bottom of this file
- if a block's "what must be true" is not true by the end of it, cut scope rather than borrow time
  from the next block

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
  - **ANSWERED by the Stage 0 spike on 3M 2018: use Azure; PageIndex not needed.**
    - heading TEXT detection is reliable — all 21 Items found, sections nest to depth 8, spans
      populated 296/296
    - heading LEVELS are not usable raw — the same 21 Items sit across four levels, leaving 106 of
      160 pages with no Item-level ancestor, including 75 of the 76 financial-statement pages
    - so the answer is "yes, after a fixing pass": see the heading-fix step at the start of 1.3.
      Rejecting PageIndex because the text is already right — only the levels need work, and a
      re-parse would cost the money again
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
- keep the benchmark harness and the system being evaluated as separate top-level packages (DONE)
  - `src/sec_rag_benchmark/` owns FinanceBench conditions, execution, evaluation and reporting
  - `src/sec_rag/` owns parsing, chunking, indexing, retrieval, reranking, generation and the
    experiment pipelines
  - a separate `sec-rag` CLI for building and inspecting the RAG system; `sec-rag-benchmark` stays
    for benchmark runs and reports
  - one Python project and lockfile, packaging both top-level modules with `uv_build`; a workspace
    would add administration without changing the experiment
  - RAG settings in `configs/sec_rag.toml`, separate from FinanceBench's evaluation and run
    settings in `configs/financebench.toml`
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

**Goal:** all 64 filings parsed by Azure exactly once, saved, checked, and readable as clean pages.
The reasoning behind each decision is in `Systems Design Draft.md` → Ingest files; it is repeated
here so the stage can be built without switching files.

- **What the parser has to give us.** Text, tables and (sub)headings, each tied to the page it sits on, so every chunk can later carry its page number and heading path. Azure Document Intelligence `prebuilt-layout`, markdown output
  - Why not hand the PDF straight to an LLM: it loses the metadata (year, filing type, company) and each chunk's subheading, and it reads tables badly (Jimeno Yepes, 2024)
  - Figures: no 10-K figure is expected to carry an answer, so nothing is built for them at parse time — but they aren't removed from Azure's output either. The chunker (1.3) gives each figure its own chunk
  - Jimeno Yepes also embeds an LLM-written summary of each chunk. That step is deferred
  - Cost: ~$10 per 1,000 pages, paid tier only (the free tier reads just the first 2 pages). The full corpus is 10,757 pages ≈ $108
- **Output as markdown AND JSON.** Kim et al. (2025) found markdown helps, and Azure gives us both: the markdown is the `content` field inside its raw JSON
  - The raw JSON is the source of truth. Its other fields (`pages`, `paragraphs`, `tables`, `sections`) point into that markdown by character position, which is how every piece maps back to a page
  - We save `result.as_dict()` exactly as Azure returns it, keeping every field
- **Preserve the structure: page numbers, (sub)headings, tables.**
  - Every paragraph, table, section and page carries a span (start position + length) in the markdown. E.g. in 3M 2018, page 13 owns characters 54,574-58,758, so a heading at character 54,629 starts on page 13
  - So text is attached to its page and heading exactly, by character position, rather than guessed from page ranges
- **How the parse run works.** One command loops over the 64 PDFs, sends each to Azure and saves the JSON it returns as `data/financebench/parsed/<doc_name>.json` (kept out of Git). Each filing costs ~$1.70 and takes a few minutes, ~$108 for all 64, so the run must never pay twice or leave a broken file. Each rule below guards against one thing going wrong:
  - **Skip filings that are already saved.** Why: re-running the command, or re-running after it stopped halfway, would otherwise pay again for everything. What it does: before calling Azure for a PDF, it checks whether that PDF's JSON is already on disk and moves on if so. This is what "parse once, never re-parse" means in code
  - **Nothing is spent unless you add `--execute-paid`.** Why: running the command by accident, or just wanting to see where things stand. What it does: plain `sec-rag parse` only reports, e.g. "1 done, 63 to do", and spends nothing. You add `--execute-paid` when you mean it (project rule: paid commands must be obviously paid)
  - **Pick specific filings with `--documents`.** Why: parse 3 filings, look at the output, and only then spend the other ~$100. What it does: `--documents 3M_2018_10K PEPSICO_2022_10K` parses just those two
  - **One filing at a time, stop at the first error.** Why: a wrong key or Azure refusing a request shouldn't fail 64 times. What it does: the loop stops straight away; everything saved so far stays saved, so re-running picks up where it stopped
  - **Write to a temporary file, then rename it.** Why: if the laptop dies or you press Ctrl-C mid-write, a half-written 21 MB file would sit on disk, the skip rule would see it and skip that filing forever. What it does: writes `<doc_name>.json.partial` first and renames it to `<doc_name>.json` only once complete. Renaming is instant, so the file is either fully there or not there at all
  - **Check a saved file before trusting it.** Why: Azure could return something truncated or empty, or the saved file could be corrupt. What it does: before a saved file counts as "done", it checks the JSON opens, the text isn't empty, and Azure's page count equals the PDF's real page count (counted with PyMuPDF, the PDF library already installed). If a file fails, stop and tell me — don't overwrite it or pay again automatically
  - **Eyeball the structure with `sec-rag inspect-parse`.** Why: to see whether Azure found the Items and headings before trusting it. What it does: writes `<doc_name>.structure.txt`, a readable list of the headings and each page's heading path. Inspection only; nothing depends on it
- **The heading list: `extract_headings`.** Reads the saved JSON into one row per heading — raw offset in `content`, page, level (Azure's sections tree) and text. Used by the inspection report now and by the heading-fix pass and chunker in 1.3. The offset never changes when a heading is later renamed or re-levelled, so it is also the heading's ID
- **Reading the parsed JSON: `load_pages(doc_name)`.** Turns one saved JSON into clean pages ready for chunking. Runs locally in about a second, so it's free. It never edits the saved JSON: every span is a character position in `content`, so deleting any text would shift every position after it and break the page/heading mapping. Each page it returns has:
  - **Page headers, footers and page numbers blanked out.** Azure labels these, and they appear inline in `content` as `<!-- PageHeader="..." -->` and `<!-- PageNumber="13" -->` (4% of paragraphs on 3M 2018). Each marker is replaced by the same number of spaces, so the page's text stays exactly as long as its slice of `content`
  - **The page's start position in `content`** (`start_offset`), so position `i` in the page's text is raw position `start_offset + i`. Chunks get their heading path by matching these positions with the heading list's offsets (1.3)
  - **The filing's company name, year and filing type**, split from the PDF's filename (`COMPANY_YEAR_TYPE`), not from parsing. Read from the right, because one company name itself contains an underscore: `JOHNSON_JOHNSON_2022_10K` → `JOHNSON_JOHNSON` / `2022` / `10K`. The filename is also what the model picks from when choosing which filing to search, so the chunk metadata and the model's choice use the same name. Checked against FinanceBench's own metadata: the years match for all 64 filings and the company names match apart from punctuation (`COCACOLA` vs "Coca-Cola")
  - **The FinanceBench page number.** FinanceBench's `evidence_page_num` counts from 0; Azure's `pageNumber` counts from 1, so `evidence_page_num = pageNumber - 1`. Never use the page number printed in the footer. **Add a test**: every page metric in the harness depends on this
- **What the Stage 0 spike on 3M 2018 already confirmed** (160 pages, saved JSON is 21.5 MB)
  - Azure's `sections` really do carry spans on real filings (296/296), unlike Microsoft's own sample where they're empty
  - No table crosses a page (119/119 sit on one page), so "each table is its own chunk" stays unambiguous
- **Font styling add-on: not bought.** Why it has to be decided now: add-ons are chosen at parse time, so adding one later means paying for all 64 filings again. Why not: without it `styles` only says whether text is handwritten; bold/italic/underline would only give us extra data to post-process, not change which lines Azure itself marks as headings
- **Where the code lives**
  - `src/sec_rag/ingestion/parse.py` — the parse run: send each PDF to Azure, check it, save its JSON. The only code that calls Azure
  - `src/sec_rag/ingestion/inspect_parse.py` — `sec-rag inspect-parse`'s file handling: pick filings, read and check each saved JSON, write `<doc_name>.structure.txt`
  - `src/sec_rag/ingestion/structure_report.py` — the analysis behind that report, with no files involved: one loaded JSON in, report text out
  - `src/sec_rag/ingestion/headings.py` — `extract_headings`: the heading list, shared by the report and 1.3
  - `src/sec_rag/ingestion/load_pages.py` — `load_pages`: load and prepare each page's content for chunking. Its own module because it's a different job from parsing — free, offline, and run as often as chunking needs
- **Order to do it in**
  1. Make the code match this section: flat `parsed/<doc_name>.json` files, no parsing record. Do this first — the 3M parse already on disk is in that flat layout, so otherwise it gets paid for again
  2. Parse 2-3 named filings with `--documents`, run `inspect-parse`, and read their `structure.txt`
  3. Parse the rest
  4. Build `load_pages` with its page-number and positions tests

**Done when:** all 64 JSON files are saved and pass the check, and the `load_pages` page-number test passes.

## 1.2 Page mapping — moved into 1.1

The page-number rule and its test are now part of `load_pages` in 1.1. The number is kept so later
sections keep their numbers.

## 1.3 Chunk

### Before chunking: fix the heading list — moved to 3.0

The heading-fix LLM pass now runs at the start of Stage 3, after Exp1 has results. Exp1 still chunks
along the document structure, because cutting needs only where each heading **is**, not its level:
the chunker cuts at Azure's heading **positions** as it returned them, before the fix. Positions are
already reliable (all 21 Items found on 3M 2018); only the levels are wrong, and cutting doesn't use
levels.

- the fix corrects levels and text, which only `heading_path` uses — and `heading_path` is Exp2's
- chunk boundaries are fixed once made: the fix at 3.0 relabels `heading_path` and **never moves a
  cut**, so Exp1 and Exp2 search exactly the same chunks. Its effect on cut positions would be
  negligible anyway — re-levelling and typo repair move no offsets, a split absorbed heading would
  create a piece far under the floor, and dropping cover-page boilerplate removes a cut or two on
  page 1
  - measured on all 64 filings by simulating the cut rule: of 20,789 headings only 3,327 produce a
    cut (the floor rejects the rest). Of 2,553 suspicious-looking headings (pages 1-2, or the same
    text on 3+ pages of a filing, e.g. Adobe's running page header), 185 produce a cut, and those are
    real headings the fix would keep (`BRAZIL`, `DENMARK` in an exhibit, `Year 2017 results:`).
    Running headers sit at the top of a page, so the floor always rejects a cut there
  - headings that restate the filing's metadata (cover-page boilerplate, "Form 10-K", "Securities
    and Exchange Commission"): 224 across the 64 filings, 1 produces a cut — and that one is a real
    heading (`ITEM 16. FORM 10-K SUMMARY`). They sit on cover pages, where the floor rejects the cut
  - other false headings are not dropped by the fix as designed, so running it early would not
    remove them; and a false cut still leaves two pieces over the floor, split at a line break — no
    worse than a halving cut
  - missing headings are not a reason to fix first either: the fix cannot add headings (every output
    heading must trace to an input one), so a section Azure never marked stays unmarked whenever the
    fix runs
- No re-parse is needed at Stage 3: the headings come from the cached JSON, and each chunk stores
  its raw offsets so `heading_path` can be attributed without re-chunking

### Chunking decisions

- **hard rule: no chunk crosses a page boundary.** This is what keeps page recall/precision/MRR
  exact against `evidence_page_num` — a chunk spanning three pages gets three chances to contain the
  gold page, so the metric's bias would vary with chunk size
  - HARNESS: assert one page per chunk — a chunk with two pages is a chunking bug
- use LlamaIndex (not LangChain), consistent with the BM25/hybrid/routing choice — for its token
  counter and sentence detector, both bundled offline in `llama-index-core`. The cutting rules
  below are our own code: LlamaIndex's `SentenceSplitter` packs sentences up to the size limit and
  leaves the remainder as a small last piece, which would break the ~250-token floor
- tables: each table is its own chunk, never split
  - measured on 48 filings: 5,044 tables, none crosses a page, largest 3,863 tokens (median 346) —
    far inside the 32,000-token input limit of both the Voyage embedder and reranker, so the
    earlier "split an oversized table by rows" rule was dropped as unneeded
- figures: each figure is its own chunk too, cut out exactly like a table — Azure's text for it
  (axis labels, legend, numbers) stays together rather than being mixed into the prose
  - measured on all 64 filings: 266 figures in 52 filings, none crosses a page or overlaps a table,
    median 38 tokens, largest 336 — mostly stock-performance graphs
  - a figure with no text (39 of 266, likely logos) is not a chunk: there is nothing to search
- after removing tables and figures (headers/footers/page numbers are already blanked —
  `load_pages`, 1.1), **join the page's remaining prose into one stream before cutting**. Cutting a
  table out of the middle leaves disconnected slivers; chunking those separately produced 50% of
  chunks under 100 tokens in the spike
- **Structure-based chunking — used by Exp1, and unchanged by Exp2** (Exp2 is Exp1 with only the
  dense score changed, so it searches the same chunks): cut at section headings, floor ~250 tokens,
  ceiling 1,024
  - accept a heading cut only if the piece before it AND the remainder both clear ~250 tokens;
    otherwise skip that cut
  - anything still over 1,024 tokens is halved at the nearest sentence boundary to its midpoint,
    recursively. Halving something over 1,024 always leaves both sides over 512, so the floor cannot
    be violated
  - measured on 3M 2018: 244 text chunks + 119 table chunks, median 512 tokens, 89% between 250 and
    1,024
  - if a heading cut is rejected for being too small, keep the page whole and **merge the heading
    paths: shared ancestors once, distinct tails joined** —
    `PART I > [Item 2. Properties | Item 3. Legal Proceedings]`
    - this needs no change to Exp2: the structure vector is a softmax-weighted average over a SET of
      headings, not a single path, so the weights favour whichever tail matches the chunk
    - the merge happens when `heading_path` is attributed at Stage 3, from the chunk's raw offsets
  - re-measure on all 64 filings when the chunker is built: the 3M figures above used characters ÷
    4 as a token proxy
- **DEFERRED, only if time remains — fixed-size chunking ablation for Exp1:** a 512-token
  recursive/sentence splitter within the page, no heading cuts, compared against structure-based
  chunking
  - 512 to match the structural chunks' median, so the comparison tests structure rather than size
  - HiREC (Choe et al., 2025) used 1,024 via LangChain's `RecursiveCharacterTextSplitter`; use
    LlamaIndex's equivalent sentence/token splitter, configured to count tokens
  - needs its own embeddings and BM25 index, so it is a full extra run, not a flag
- **overlap at every cut that is not a heading: carry the last sentence** — at a page break or a
  halving cut, the new chunk starts with the last sentence before the cut, complete or broken off,
  capped at ~100 tokens. Nothing is carried at a heading cut: a new section doesn't need the previous
  section's last sentence
  - why page breaks: the page rule forces a cut at every page break, and on all 64 filings 18% of
    page breaks (1,789 of 10,194) split a sentence — e.g. 3M 2018 page ends "...investment losses on
    plan", next page starts "assets, and relevant legislative...". Carrying the last sentence
    restores it; when the page ended cleanly, it still carries the lead-in ("This increase was
    driven by..." needs the sentence before it)
  - why halving cuts: they fall between sentences, so no sentence breaks, but the link between them
    does — the same problem as a clean page break
  - one rule for both, rather than a page-break-only fragment rule: simpler to code, test and
    explain. This replaces the earlier blind 30-token overlap
  - the carried sentence lives in the chunk's TEXT only; `page_num` and the chunk's raw offsets stay
    those of its own piece, so page metrics and Stage 3's heading attribution are untouched
  - a chunk can exceed the 1,024 ceiling by up to the ~100-token cap; the floor is unaffected and
    the embedder's limit is 32,000
- dropped "semantic chunking via regex": the recursive/sentence splitter already prefers
  paragraph then sentence breaks
- a page whose whole content is under the floor is still its own chunk — the floor governs whether
  to SPLIT, not a minimum chunk size
- metadata per chunk: SEC filing type + company + financial year + page number
  - each chunk inherits company, year and filing type from its page, which `load_pages` (1.1)
    split from the filename (`COMPANY_YEAR_TYPE`, read from the right), **not from parsing**
  - the same filenames are what the LLM picks from when selecting a filing in Stage 2.1, so a
    chunk's metadata and the model's choice always use the same name
- **Exp2 hook, built now:** every chunk carries a `heading_path` field
  - left empty in Stage 1.3; populated at Stage 3 once the heading fix (3.0) has corrected the
    levels. Populating it now from Azure's raw levels would give 106 of 160 pages on 3M 2018 no
    Item-level ancestor
  - how it is populated at Stage 3 — attribute headings to chunks **by character offset**, which
    the cached JSON makes exact: a heading holds from its own offset until the next heading at the
    same or higher level starts, so a chunk gets the headings whose offset range covers it
    - the earlier page-range rule was the weaker fallback for a world without offsets; offsets are
      present, so use them
    - a chunk containing text under two headings merges them (see the merge rule above), rather than
      picking one
  - Fin-STAR constraint: depth ≤ 5 — if the heading path has more than 5 levels, drop the excess
    (deepest) headings, keeping the outermost five
    - verified safe on 3M 2018: 52 of 160 pages have paths deeper than five, and keeping the first
      five loses the Item-level anchor on ZERO of them
  - save the path as a list of headings, top level first. Each level is embedded separately at Stage
    3.2, not joined into one string

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
- keep FinanceBench's five condition definitions in `sec_rag_benchmark`; they are evaluation
  protocols, not RAG-system behaviour
  - closed-book, oracle and long-context select no pages, gold pages or all filing pages,
    respectively, then pass those supplied context items to `sec_rag`'s direct-answer pipeline
  - single-store and shared-store select one-filing or all-filing document scope, respectively,
    then pass the question and scope to the selected RAG pipeline
  - `sec_rag` must not receive FinanceBench condition names. It receives either supplied context
    or retrieval scope, formats the common `[Document | Page]` context, and owns every
    model-facing operation
- remove the benchmark's current retriever callback by decomposing it rather than moving it
  wholesale
  - `conditions.py` keeps condition selection and document-scope construction, but no longer calls
    retrieval or formats retrieved chunks
  - move shared generation, answer-prompt construction and context formatting from
    `sec_rag_benchmark/pipeline/generation.py` into `sec_rag`
  - the direct, one-round RAG and future agent pipelines all return the common answer + final chunks
    + retrieval stages + trace + usage/provenance result
  - the benchmark alone calculates metrics, checkpoints predictions, judges answers and reports
- when implementing this migration, update
  `docs sys design/benchmark/FinanceBench Implementation Guide.md` in the same slice so its data
  shapes, code-reading order, pseudocode, tests and terminal commands describe the code actually
  present
- verify the migration with fake-pipeline tests over all five conditions, the complete no-spend
  test suite, and the README's smoke and pattern dry-run commands before any paid benchmark smoke
  run

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
- no auto-merge for Exp1: chunks are page-bounded and at most 1,024 tokens, and a typical page's
  prose is ~630 tokens, so a chunk is already most of a page — parent ≈ chunk

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
- feed the LLM top-n results, each labelled with its document and page
- context format: make the oracle and retrieval context blocks identical, `[Document | Page]`, so
  prompt shape can't explain a results gap
  - DECIDED: no Section label, in any experiment. Exp2's heading path feeds only the scorer; if Exp2
    also showed it to the answer model, Exp2 would change both ranking and what the model reads, and
    a gain couldn't be attributed. It also keeps the heading fix off Exp1's critical path, and
    matches the label the finished baseline runs already used (`conditions.py`)

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

## 3.0 Fix the heading list (moved from 1.3)

Moved here because Exp1 ignores headings: building this before Exp1 has results would put a new
LLM pass in front of the dissertation's spine. It must run before 3.1, which attributes the fixed
headings to chunks. It never changes the chunks themselves: Exp1's chunker already cut at Azure's
raw heading offsets (1.3).

Azure detects heading **text** reliably but its **levels** are not usable as they come. On 3M 2018
all 21 Items were found, but spread across four levels, leaving 106 of 160 pages with no
Item-level ancestor — including 75 of the 76 financial-statement pages, where most FinanceBench
answers live.

- an LLM pass takes the heading rows `(text, level, page)` plus the file's metadata and returns the
  same rows corrected
  - repairs split-word typos — `Busines s.`, `ESTIMA TES`, `Equit y` (16 of 295 headings)
  - drops headings that only restate the file metadata, e.g. the SEC cover boilerplate, otherwise
    the root ancestor of ~106 pages
  - splits absorbed headings — Azure glued `PART I` onto the end of the previous title
  - re-levels using the generic rule that a numbered series (`Item 7`, `Chapter 3`, `Article II`)
    are siblings
- deliberately NOT SEC-specific regex: matching `^Item \d+` overfits to 10-Ks and would not
  transfer to other document types
- cached per filing like the parse, so everything downstream is deterministic and identical across
  every condition and experiment
- validated structurally, with no domain knowledge: every output heading traces to an input heading
  (whitespace repair and splits only), levels form a valid tree with no jumps greater than one,
  numbered series share a level, pages unchanged. On failure, keep Azure's raw levels for that
  filing and log it
- accepted limitation: sub-headings Azure never marked (underlined/italic) cannot be recovered.
  Bounded, because Exp2 weights headings by similarity rather than by depth

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
  - depth control already applies now (see the `heading_path` hook in 1.3); discriminativeness
    only applies to the virtual node, so it returns with it
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
fetch-whole-page, Elasticsearch, OP-RAG, and the fixed-size chunking ablation for Exp1 (only if time
remains, see 1.3).

# Write-up runs alongside, not after

- write each experiment's description before building it — they are a page each and they expose
  gaps while there is still time to act
  - each description must state its baseline AND its ablations explicitly
- after each stage, paste the numbers into that experiment's Results section the same day
  - results written up the day they are produced take minutes; a week later they take hours of
    re-deriving what a run's settings were
