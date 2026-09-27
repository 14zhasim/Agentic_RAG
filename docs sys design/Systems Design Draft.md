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
  - SECOND CAVEAT (found building Stage 1.5, 27 Sep 2026): the filter is read only when the retriever is created; changing `retriever.filters` afterwards is silently ignored. Since each question has its own filter, retrieval loads the saved index once and creates a new `BM25Retriever` per search with that search's filter (~0.01 s, nothing is re-indexed)
- [x] CHECK FIRST: is Azure Document Intelligence's section nesting good enough for Exp2's heading path? Run `prebuilt-layout` on 2-3 10-Ks, open the JSON, and look at `sections` (do sections nest, and do Item headings sit at the top level?) and at `paragraphs` with role `title` / `sectionHeading` (right text, right page?). If nesting is flat or wrong, use PageIndex instead
  - ANSWERED by the spike on 3M 2018: use Azure. It finds the heading text reliably (all 21 Items), but its heading levels can't be used as they come, so they're fixed first — see "Fix the heading list before chunking"
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
  - the filter is fixed when a `BM25Retriever` is created, so each search creates its own from the once-loaded index, passing that search's filter (setting it later is silently ignored)
- Azure Document Intelligence — parsing, including the heading structure (confirmed by the Stage 0 spike on 3M 2018)
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
- Parser requirements - goal is it should identify different media (tables etc.) and nested subheading structure accurately!
  - Jimeno Yepes (2024) used basic VLM to identify text, titles, tables and chunk along those boundaries (+LLM generated summary to embed)
    - Rationale:
    - helps LLM read non-plaintext file formats to LLM-interpretable representations, like tables.
    - Also, just using LLM misses out metadata (year, file type, ticker) and structural content (e.g. subheading of each chunk)
  - Parsing Options
    - "VLM-agentic" parsers (read the page like a person, with a correction/verification pass). E.g. Llamaparse, reducto ($$$), Azure Document Intelligence ($100 for ~80 docs around 12,000 pages, 82.7% on RD-TableBench, use base Layout), GPT-5.6 sol. **Good if endless edge cases exist in human writing, just use VLM**
    - "layout engine" parsers (specialized detection models + rules, no LLM-in-the-loop by default, self-hostable and free): PyMuPDF / pymupdf4llm, good for machine-generated docs like statements - apparently not that good for tables
    - verdict: use Azure Document Intelligence as i have startup credits (unless it has complicated setup/infra?)
      - setup is light: create resource → endpoint + key → `azure-ai-documentintelligence` SDK, `prebuilt-layout` model, markdown output. Paid tier only (free tier reads first 2 pages). ~$10 / 1,000 pages
  - Parser output decisions
    - **Output as markdown AND JSON.** Kim et al. (2025) found markdown helps, and Azure gives us both: the markdown is the `content` field inside its raw JSON
      - The raw JSON is the source of truth. Its other fields (`pages`, `paragraphs`, `tables`, `sections`) point into that markdown by character position, which is how every piece maps back to a page
      - We save `result.as_dict()` exactly as Azure returns it, keeping every field
    - **Preserve the structure: page numbers, (sub)headings, tables.**
      - Every paragraph, table, section and page carries a span (start position + length) in the markdown. E.g. in 3M 2018, page 13 owns characters 54,574-58,758, so a heading at character 54,629 starts on page 13
      - So text is attached to its page and heading exactly, by character position, rather than guessed from page ranges
      - Confirmed on 3M 2018: every section carries its character positions (296/296), and no table crosses a page (119/119 sit on one page), so each table can be its own chunk without breaking the one-page-per-chunk rule
    - **How the parse run works.** One command loops over the 64 PDFs, sends each to Azure and saves the JSON it returns as `data/financebench/parsed/<doc_name>.json` (kept out of Git). Each filing costs ~$1.70 and takes a few minutes, ~$108 for all 64, so the run must never pay twice or leave a broken file. Each rule below guards against one thing going wrong:
      - **Skip filings that are already saved.** Why: re-running the command, or re-running after it stopped halfway, would otherwise pay again for everything. What it does: before calling Azure for a PDF, it checks whether that PDF's JSON is already on disk and moves on if so. This is what "parse once, never re-parse" means in code
      - **Nothing is spent unless you add `--execute-paid`.** Why: running the command by accident, or just wanting to see where things stand. What it does: plain `sec-rag parse` only reports, e.g. "1 done, 63 to do", and spends nothing. You add `--execute-paid` when you mean it (project rule: paid commands must be obviously paid)
      - **Pick specific filings with `--documents`.** Why: parse 3 filings, look at the output, and only then spend the other ~$100. What it does: `--documents 3M_2018_10K PEPSICO_2022_10K` parses just those two
      - **One filing at a time, stop at the first error.** Why: a wrong key or Azure refusing a request shouldn't fail 64 times. What it does: the loop stops straight away; everything saved so far stays saved, so re-running picks up where it stopped
      - **Write to a temporary file, then rename it.** Why: if the laptop dies or you press Ctrl-C mid-write, a half-written 21 MB file would sit on disk, the skip rule would see it and skip that filing forever. What it does: writes `<doc_name>.json.partial` first and renames it to `<doc_name>.json` only once complete. Renaming is instant, so the file is either fully there or not there at all
      - **Check a saved file before trusting it.** Why: Azure could return something truncated or empty, or the saved file could be corrupt. What it does: before a saved file counts as "done", it checks the JSON opens, the text isn't empty, and Azure's page count equals the PDF's real page count (counted with PyMuPDF, the PDF library already installed). If a file fails, stop and tell me — don't overwrite it or pay again automatically
      - **Eyeball the structure with `sec-rag inspect-parse`.** Why: to see whether Azure found the Items and headings before trusting it. What it does: writes `<doc_name>.structure.txt`, a readable list of the headings and each page's heading path. Inspection only; nothing depends on it. While inspecting the first 2-3 filings, also check whether Azure returned any figures (`<figure>` tags in `content`) — the 3M parse has none
    - **The heading list: `extract_headings`.** Why: the inspection report now, and the heading-fix pass and chunker later, all need the same list of headings, each with a fixed position to match chunks against. What it does: reads the saved JSON into one row per heading — raw offset in `content` (from the paragraph's span), page, level (from Azure's sections tree) and text. The offset never changes, even when a later pass renames or re-levels a heading, so it doubles as the heading's permanent ID
    - **Reading the parsed JSON: `load_pages(doc_name)`.** Turns one saved JSON into clean pages ready for chunking. Runs locally in about a second, so it's free. It never edits the saved JSON: every span is a character position in `content`, so deleting any text would shift every position after it and break the page/heading mapping. Each page it returns has:
      - **Page headers, footers and page numbers blanked out.** Azure labels these, and they appear inline in `content` as \<!-- PageHeader="..." --> and \<!-- PageNumber="13" -->. Each marker is replaced by the same number of spaces rather than cut out, so the page's text stays exactly as long as its slice of `content`
      - **The page's start position in `content`** (`start_offset`). Why: chunks get their heading path by matching positions with the heading list, so every piece of text must keep its raw position. What it does: position `i` in a page's text is raw position `start_offset + i` — e.g. 3M page 13 starts at 54,574 and its `PART II` heading at 54,616 is at position 42. The leftover spaces are tidied when each chunk's final text is made
      - **Figure text kept in the page text.** Whatever Azure reads from a chart — captions, axis labels, numbers — stays in the page's text, so a number that only appears in a chart can still be found. The chunker then cuts each figure out as its own chunk (see Chunk)
      - **The filing's company name, year and filing type**, split from the PDF's filename (`COMPANY_YEAR_TYPE`), not from parsing. Read from the right, because one company name itself contains an underscore: `JOHNSON_JOHNSON_2022_10K` → `JOHNSON_JOHNSON` / `2022` / `10K`. The filename is also what the model picks from when choosing which filing to search, so the chunk metadata and the model's choice use the same name. Checked against FinanceBench's own metadata: the years match for all 64 filings and the company names match apart from punctuation (`COCACOLA` vs "Coca-Cola")
      - **The FinanceBench page number.** FinanceBench's `evidence_page_num` counts from 0; Azure's `pageNumber` counts from 1, so `evidence_page_num = pageNumber - 1`. Never use the page number printed in the footer. Add a test: every page metric depends on this

  - Fix the heading list (built at Build Order Stage 3.0, after Exp1 has results — Exp1 cuts chunks at the raw heading offsets and doesn't need the corrected levels)
    - Why: on 3M 2018 Azure found all 21 Items, but put them across four different heading levels. As a result 106 of 160 pages have no Item heading above them, including 75 of the 76 financial-statement pages, so their heading paths would be wrong
    - an LLM pass takes the heading rows (offset, text, level, page) plus the file's metadata and returns the same rows corrected, keyed by offset
      - repairs split-word typos — Busines s., ESTIMA TES, Equit y (16 of 295 headings)
      - drops headings that only restate the file metadata, e.g. UNITED STATES SECURITIES AND EXCHANGE COMMISSION..., which is otherwise the root ancestor of ~106 pages. This can be derived from the filename
      - splits absorbed headings — Azure glued PART I onto the end of the previous title
      - re-levels based of common sense e.g. heading called 'part 2' should not be different level to 'part 1' - using the generic rule that a numbered series (Item 7, Chapter 3, Article II) are siblings
    - cached per filing like the parse, so everything downstream is deterministic and identical across every condition and experiment
    - validated structurally, with no domain knowledge: every output heading traces to an input heading (whitespace repair and splits only), levels form a valid tree with no jumps greater than one, numbered series share a level, pages unchanged. On failure, keep Azure's raw levels for that filing and log it
    - accepted limitation: genuinely missed sub-headings (underlined/italic ones Azure never marked) cannot be recovered. Not too much impact, because Exp 2 weights headings by similarity rather than by depth

- Chunk (+structure parsing) + save metadata for each chunk (SEC-filing-type + company ticker + financial year + page number), for filtering chunks. _View chunks manually, writing has infinite edge cases_

  Backwards compatible as long as chunks + embeddings are saved as plain files and search sits behind the existing retriever interface; Elastic Search then becomes one more loader + retriever

  - **Which experiments use this.** Exp1 builds these chunks; Exp2 searches the identical chunks and changes only the dense score. The rules below are applied in this order, one page at a time
  - **1. Hard rule: no chunk crosses a page.** Why: this is what keeps page recall/precision/MRR exact against `evidence_page_num` — a chunk spanning 3 pages would get three chances to contain the gold page, making the metric's bias vary with chunk size. What it does: every chunk has exactly one `page_num`, and the harness asserts it
  - **2. Tables and figures: each is its own chunk, never split.** Doable because Azure gives every table and figure its own character position (the element boundaries Jimeno Yepes (2024) chunked along: sections, titles, tables, text, figures — so a page can hold several chunks)
    - tables: on 48 filings none of 5,044 tables crosses a page and the largest is 3,863 tokens, far inside Voyage's 32,000-token embed/rerank limit, so no row-splitting is needed
    - **a table chunk carries the text just above it** (up to ~100 tokens). Why: cutting a table out separates it from its title — found in the corpus run, where 742 of the 811 prose chunks under 50 tokens sat on a table page, many of them a statement's title alone (3M's "Consolidated Balance Sheet" in one chunk, the balance sheet's rows in the next, which never says "balance sheet"). What it does: the visible text between the previous table or figure on the page (or the page's top) and this table is carried into the table chunk's text, keeping its end if over the cap, since the title sits right above the table. Same mechanism as rule 4: text only, offsets and page stay the table's own, and the title's prose chunk is left as it is (small, harmless noise)
      - rejected: one chunk per table page holding the whole page. 1,496 of the 4,305 table pages hold two or more tables (e.g. income statement + comprehensive income), which would share one blurred embedding; the median table page is 1,101 tokens and the largest 3,870, so top-10 context fills with unrelated rows
      - rejected: Azure's own table captions — only 89 of 6,441 tables have one
      - a table at the very top of a page (often a table continued from the previous page) has nothing above it and carries nothing
      - caveat for Exp2: a table chunk now contains its nearby heading as words, which is some of the structure signal Exp2 adds to the dense score. Defensible — it is text printed on the same page, not a heading path from elsewhere in the filing — but it can narrow Exp2's gain on table questions, and the write-up must say so
    - figures: cut out like a table, so a chart's labels and numbers stay together instead of mixing into the prose. On all 64 filings: 266 figures, none crosses a page or overlaps a table, median 38 tokens
    - a figure with no text (37 of 266, likely logos) is not a chunk: there is nothing to search. A captioned figure is always kept, since its caption sits inside its text
  - **3. Prose: join the page, then cut at headings (structure-based chunking).**
    - after removing tables, figures and noise, join the page's remaining prose into one stream before cutting (chunking the slivers between tables separately made 50% of chunks under 100 tokens in the spike)
    - cut at section headings, with a floor of ~250 tokens and a ceiling of 1,024. Accept a heading cut only if the piece before it and the remainder both clear ~250 tokens; otherwise skip that cut. Several headings on one page are tried in order, left to right

    - cut using Azure's heading positions as it returned them, before the heading fix. Every heading has a position (where it starts in the text, e.g. 3M's `Item 5` at character 54,629 — Azure gets these right) and a level (how deep it sits in the outline — Azure gets these wrong)
    - anything still over 1,024 tokens is halved at the nearest sentence boundary to its midpoint, recursively. Halving something over 1,024 always leaves both sides over 512, so the floor can't be violated
      - the chunker only needs positions: "is there a heading here, and would cutting leave both pieces over ~250 tokens?" doesn't depend on the level, so Exp1 doesn't wait for the heading fix
      - the fix changes only levels and wording (`Item 5` level 3 → 2, `Busines s.` → `Business.`), never positions, so every cut stays where it was and the chunks don't change
      - what the fix does change is the heading path, built from levels — which only Exp2 uses. So Exp1 (run before the fix) and Exp2 (after) search identical chunks, and the heading score is the only difference
    - a page whose whole prose is under the floor is still its own chunk: the floor governs whether to split, not a minimum chunk size
    - heading lines stay in the chunk's text (as markdown), so a chunk's own words include its heading
  - **4. Overlap (up to ~100 tokens) when a chunk starts at a page break or a halving cut, not at the start of a new section.** What it does: the new chunk starts with the last sentence before the cut, complete or broken off, capped at ~100 tokens so it almost always carries the whole sentence (a 10-K sentence is often 25-40 tokens). Nothing at a heading cut — a new section doesn't need the previous section's last sentence
    - why page breaks: 18% of page breaks across the 64 filings split a sentence, and the page rule forces a cut there; when a page ended cleanly, the carried sentence still gives the lead-in ("This increase was driven by..." needs the sentence before it)
    - why halving cuts: they fall between sentences, so no sentence is broken, but the link between sentences is — the same problem as a clean page break. One rule covers both, which is simpler to code, test and explain
    - the carried sentence is in the chunk's text only — page_num and raw offsets stay its own piece's, so the page metric still counts 1 page
    - a chunk can exceed the 1,024 ceiling by up to the ~100-token cap; the floor is unaffected
    - tables reuse the same carried-text field for the text just above them (rule 2), under the same ~100-token cap
  - **5. What each chunk records.**
    - its page_num — exactly one page, never a set, because of the hard page boundary
    - doc_name and the file metadata (company, year, filing type), split from the filename by `load_pages`
    - its raw start/end offsets in Azure's `content`, so heading_path can be attributed at Stage 3 without re-chunking
    - its heading_path — present on every chunk, empty until Stage 3
  - **6. The heading path (filled in at Build Order Stage 3, for Exp2).** Like FinSTAR, keep the document structure as metadata on each chunk; Azure's sections give the hierarchy, corrected by the heading-fix pass above
    - attribute headings to chunks by character offset: a heading holds from its own offset until the next heading at the same or higher level starts, and a chunk gets the headings whose range covers it. This replaces the earlier page-range rule (a heading's page range, ties on a page going to the heading covering more of it), which was the fallback for a world without offsets
    - a chunk covering text under two headings (e.g. a heading cut rejected for being too small) merges the heading paths: shared ancestors once, distinct tails joined — PART I > [Item 2. Properties | Item 3. Legal Proceedings]
      - this works with Exp 2 unchanged, because the structure vector is a softmax-weighted average over a set of headings, not a single path — the weights favour whichever tail matches the chunk's content
    - depth limit of 5 (Fin-STAR): if a path has more than 5 levels, drop the deepest, so the text is treated as sitting under the deepest surviving heading
    - save the path as a list of headings, top level first. Each heading level is embedded separately and combined at retrieval (see retrieval → Experiment 2 theory), not joined into one string
  - **7. Tools.**
    - LlamaIndex (not LangChain), consistent with the BM25/hybrid/routing choice — for its token counter and sentence detector, both bundled offline in `llama-index-core`. The cutting rules above are our own code: LlamaIndex's `SentenceSplitter` packs sentences up to the size limit and leaves the remainder as a small last piece, breaking the floor
    - inspection script: "show all chunks for doc X, page Y"; run on a question's gold page to spot missed-table failures
  - **Deferred**
    - fixed-size chunking — only if time remains: an Exp1 ablation against the structure-based chunking above
      - 512 tokens (the median of structure-based chunking, so the comparison tests structure rather than size; this size also still backed by literature)
      - 30 overlap that is allowed to come from previous page, via Langchain's RecursiveCharacterTextSplitter - HiRec (Choe et al., (2025)) — use LlamaIndex's equivalent
      - split around sentences, not arbitrary words
      - include headers in the text to chunk
      - needs its own embeddings and BM25 index, so it is a full extra run
    - FinSTAR virtual node (Exp2 rung C): get an SLM to generate another section header based on the section's content specifically
      - method: for chunk ci and hierarchical path hi, use SLM Gθ to generate virtual node vi = Gθ(ci, hi), where enriched path Pi = hi⊕vi. To ensure robustness, Pi is constrained by (1) depth control, enforcing Depth(Pi) <= 5 and (2) discriminativeness, requiring vi to capture essential semantics absent from hi
    - HiChunk's learned chunk-point predictor: cut, it needs a GPU

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
    - for chunks only, first remove HTML table tags (`td`, `tr`, `th`, ...): they are 12% of the
      corpus's words and would make tables look long, which BM25 scores down. Only BM25's copy
      is stripped; the answer model still sees the tables' HTML (decided 27 Sep 2026)
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
    - cost: measured at 10.1M tokens over 21,039 chunks (Build Order 1.3), inside the 200M free allowance even with several re-chunks
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
        - structure source: Azure Document Intelligence, after the heading-fix pass (answered by the Stage 0 spike). It gives each heading, its nesting level and its start page, which is all the method needs (see chunking → experiment 2 attribution rule)
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
- feed LLM top-n results, labelled with document and page. DECIDED: no section/heading label in any experiment — Exp2's heading path feeds only the scorer, so Exp2 changes ranking alone rather than ranking plus what the answer model reads
- retrieval decisions (from review; confirm or strike)
  - if the filter matches no filing, search unfiltered rather than returning nothing
  - HARNESS (build with Exp1): record filter accuracy — chosen filename == gold `doc_name`. Splits "wrong document" from "wrong chunk" in the failure tree. Only meaningful in shared_store (single_store already scopes to the question's filing)
  - HARNESS (build with Exp1): make the oracle and retrieval context blocks identical, `[Document | Page]`, so prompt shape can't explain a results gap
  - Exp1 retrieval must be ONE function taking arguments — query, retrieval method (BM25 / semantic / hybrid), metadata filters, top_k — not a hardcoded pipeline. Exp3's search tool is this same function, with the agent choosing those arguments at runtime; Exp1 passes fixed ones. Getting this wrong means writing retrieval twice
    - inside it, the BM25 side creates a fresh `BM25Retriever` from the once-loaded index on every call, with that call's filters: the retriever only reads its filter when created (Tech stack → `bm25s`)
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
    - one shared formatter produces `[Document | Page]` blocks for supplied pages and
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
