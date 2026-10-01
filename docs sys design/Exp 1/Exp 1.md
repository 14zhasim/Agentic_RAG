# Experiment 1 — Baseline Retrieval Architecture

## Experiment

- **Build:** a retrieval pipeline over SEC filings.
  - Indexing: structure-aware parsing, page-bounded chunking, metadata filtering.
  - Retrieval: BM25 hybrid search, re-ranking.
- **Input:** a natural-language finance question about a US public company plus access to its filings.
- **Output:** a free-text answer, scored for accuracy.
- **What it must do:** retrieve the pages that actually contain the answer.
  - Out of a store that may hold one filing or every filing in the benchmark.
  - Hand a generation model only that retrieved context — not the whole filing.
- **Measured against the Exp 0 conditions** (`Exp 0/Exp 0.md`): closed-book (the floor), oracle (the ceiling) and long-context, a direct LLM API call with the whole filing and no RAG.
- **Target:** match or beat Exp 0's long-context performance while not being limited by context window size.
- **Role in the wider project:** Experiment 1 is the shared spine.
  - Experiment 2 changes only the dense-retrieval score (structure-aware embeddings).
  - Agentic retrieval (Experiment 3, future work: `Exp 3/Exp 3.md`) would change only the retrieval step into an agent loop (query planning, retries, a calculator tool).
  - Everything else — parsing, chunking, query enhancement, BM25, fusion, reranking, generation — stays identical, so that any difference in results can be attributed to the one thing that changed.

## Data

FinanceBench (Islam et al., 2023), restricted to the 10-K subset:

- 112 questions across 64 10-K PDFs.
  - The full FinanceBench open-source set is 150 questions across >110 10-K/10-Q/8-K filings.
  - This project filters to `doc_type == "10k"` exactly, which is what produces 112/64.
- **By generation method:**
  - metrics-generated: 50 — retrieving "typical" metrics from a filing, e.g. revenue, EBITDA (18 distinct metrics across the dataset).
  - domain-relevant: 48 — generic finance questions applicable to many 10-Ks.
  - novel-generated: 14 — annotator-written, company/report/industry-specific questions requiring close reading of the filing.
- **By cognitive skill** (a question can carry more than one label, so these sum to 128, not 112):
  - Numerical reasoning: 57 — performing calculations or comparing numerical data.
  - Information extraction: 36 — extracting specific data or textual content.
  - Logical reasoning: 21 — using logical deductions to evaluate, contrast, or make judgments.
  - unlabelled: 14
    - the raw `question_reasoning` field is not clean and must be normalised to these three skills:
      - split on a standalone `OR`
      - casefold-compare
      - fold the parenthetical variant into logical reasoning
      - keep `None` visible as its own segment
      - raise on anything unmapped so the taxonomy can't silently grow
- Every question sits inside a single filing.
  - Even the "open-domain" framing only means the retriever has to find the right filing among the ones it is given, not search the open web.
- Each question carries:
  - a gold answer
  - a human-evaluator explanation
  - gold evidence as `(doc_name, evidence_page_num)` pairs — page numbers are zero-indexed in FinanceBench
- **Scoring:** answer accuracy is judged by an LLM-as-judge, using the Zheng et al. (2024) method.
  - Judge model: **DeepSeek-V4-Flash, via Azure Foundry** — deliberately a different model from the generation model (GLM-5.3-flash).
  - Given to the judge: the question, the reference answer, the evidence, and the human labeller's justification, plus the candidate answer.
  - Output: a binary correct/incorrect verdict (rounding and truncation in the candidate answer are allowed, but the verdict itself is binary).
  - Process: the judge is given the correct reference answer before grading, and is asked to grade twice with answer order swapped, keeping only the verdict when both agree.

## Models / architecture

- **Exp 0 conditions:** a direct LLM API call with no retrieval: closed-book, oracle, and long-context (the whole filing, or as much as fits).
  - Long-context is what Experiment 1's target is measured against; the oracle is its ceiling.
- **Pipeline tiers** (Experiment 1's architecture):
  1. **Ingestion**
     - parse each filing PDF
     - chunk it, page-bounded
     - attach metadata to every chunk
  2. **Storage**
     - a keyword index (BM25)
     - a vector store (chunk embeddings)
     - both filterable by document metadata before search
  3. **Retrieval**
     - one LLM call to enhance the query into structured filters and search strings
     - hybrid search over both indexes
     - reciprocal-rank fusion of the two ranked lists
     - then a reranker over the fused top-k
  4. **Generation** — the reranked top-n chunks, with their metadata and page numbers, are handed to GLM-5.3-flash (via OpenRouter) to produce the free-text answer.
- The same model, settings, and answer prompt are used across every condition and every experiment, so that only retrieval differs between runs.

## System design

**Ingestion** — parse each filing, map its pages back correctly, chunk it within page boundaries, and attach filtering metadata to every chunk.

- **Parse:** each 10-K PDF is parsed with Azure Document Intelligence's `prebuilt-layout` model, producing markdown output plus structural JSON (pages, paragraphs, tables, sections).
  - The raw JSON is the source of truth. The markdown is one field (`content`) inside that JSON; other fields point into it by character position, which is how each piece of content maps back to a page.
  - The raw JSON is cached to disk once, out of Git — chunking never re-parses.
  - Page headers, footers, and printed page numbers are stripped before chunking (Azure labels these explicitly).
- **Page mapping:** Azure's `pageNumber` is 1-indexed; FinanceBench's `evidence_page_num` is zero-indexed.
  - Convert by subtracting 1 — never use the printed footer page number.
- **Chunk:** chunking is page-bounded — every chunk covers exactly one page.
  - A chunking bug is defined as any chunk belonging to two pages, and the harness asserts against this — this is what keeps page-based retrieval metrics exact.
  - Overlap: a chunk that starts at a page break or a halving cut begins with the last sentence before the cut (capped at ~100 tokens), because 18% of page breaks split a sentence. That sentence is part of the chunk's text only; the chunk's page number is still its own page. Nothing is carried at a heading cut.
  - Tables are their own chunks: each table becomes one chunk, never split — on 48 filings none of 5,044 tables crosses a page and the largest is 3,863 tokens, far inside the embedder's and reranker's 32,000-token limit. Each table chunk also carries the text just above it on its page (up to ~100 tokens, usually the statement's title), because cutting a table out otherwise separates it from its title: 3M's balance-sheet table never said "balance sheet".
  - Figures are their own chunks too, cut out like tables, so a chart's labels and numbers stay together; a figure with no text (e.g. a logo) is not a chunk.
  - The remaining prose is chunked along the document structure: cut at section headings, with a floor of ~250 tokens and a ceiling of 1,024 (token count, not character count). A heading cut is taken only if both pieces clear the floor; anything over the ceiling is halved at the sentence boundary nearest its midpoint.
    - Cuts use Azure's raw heading positions. The later heading-fix pass (Experiment 2) corrects heading levels for the heading path but never moves a cut, so Experiments 1 and 2 search identical chunks.
    - A fixed-size (512-token) chunking ablation is deferred, run only if time remains.
- **Metadata:**
  - Document-level metadata (company, doc_type, doc_period) is read from
    `financebench_document_information_10k.jsonl`, not inferred from parsing — FinanceBench has no
    ticker field.
  - Each chunk carries metadata for filtering: SEC filing type, company ticker, financial year, and page number.
- **Structure source:** the Stage 0 spike selected Azure and rejected PageIndex.
  - Heading text detection is reliable and section spans are populated, but Azure's raw levels are
    not usable directly. Experiment 1 needs only the headings' positions, to cut chunks; the levels
    are corrected in the separate heading-fix pass at Stage 3.0, for Experiment 2's heading path.
    This changes neither Experiment 1's parse nor its chunks.

**Storage** — two indexes are built over the chunks, each filterable by metadata before search runs.

- **Vector store:** chunk embeddings via `voyage-4-lite`.
  - Chosen as the cheapest embedder scoring above 80% on the relevant MTEB/RTEB (finance) leaderboard segment, with a 200M-token free allowance that comfortably covers the corpus (measured at 10.1M tokens over 21,039 chunks, even across several re-chunks).
  - Chunks are embedded with `input_type="document"`, questions with `input_type="query"`.
  - Each embedding is cached, keyed by model name plus a hash of the chunk text, so re-chunking only re-embeds what changed.
- **Keyword index:** `bm25s`, via LlamaIndex's `BM25Retriever`.
  - Standard BM25 parameters (k1=1.5, b=0.75).
  - Words are split into letters and digits (`FY2018` → `fy` + `2018`), and tables are counted without their HTML tags, which would otherwise be 12% of the corpus's words and push tables down; the answer model still sees the tags.
  - Filtering narrows which chunks are searched, but BM25's word statistics (term frequency, document frequency) are still computed over the whole corpus — the same behaviour as Elasticsearch, kept deliberately so results stay comparable if the keyword backend is swapped later.
- **What's stored per chunk:** the chunk text, its embedding, and its filtering metadata (ticker, filing type, year, page) together, so retrieval can filter and score in the same place.
- **Filtering support:** both indexes support metadata filtering before search.
  - `where`-style filtering on the vector store.
  - `MetadataFilters` on the BM25 retriever, if it actually filters correctly — see Design decisions.

**Retrieval** — query enhancement narrows and expands the question, metadata filtering narrows the search pool, BM25 and semantic search run over that pool, and the two ranked lists are fused before reranking.

1. **Query enhancement:** one LLM call returns structured JSON.
   - Fields: a filename chosen from the list of searchable filings (or none, if unsure), a keyword query (financial shorthand expanded out, e.g. PP&E → property, plant and equipment; COGS, DPO, FCF, capex), and a semantic query. The filename already encodes company, year and filing type.
   - Model: GLM-5.3-flash with reasoning effort `low` — choosing from a list and rewriting a question is a light task, and the answer model's `high` setting is unchanged.
   - The same call runs in both retrieval conditions; only the list differs. In single-store the list holds just the in-scope filing, so the choice is trivial; in shared-store it holds all 64, and the choice is scored as filter accuracy.
   - This same prompt is used by both Experiment 1 (called once, up front) and Experiment 3 (called by the agent through its own first two steps), so the two experiments can't quietly diverge on this piece.
2. **Metadata filtering:** filenames follow the pattern `COMPANY_YEAR_TYPE.pdf`.
   - The enhancement step chooses a filename directly (parsed from the right, since company names themselves can contain underscores) rather than matching separate metadata table columns.
   - Filtering happens on chunks *before* search.
   - If no valid filename comes back, the system searches the whole scope unfiltered rather than returning nothing, and records a filter miss. The filter only narrows within the scope the benchmark condition supplies.
3. **BM25 + semantic search:** the keyword query goes to BM25 and the semantic query to the vector store (embedded with `input_type="query"`), each over the (possibly filtered) chunk set, each returning its top 50. BM25 results scoring zero share no word with the query and are dropped.
4. **RRF fusion:** the two ranked lists are combined with reciprocal rank fusion.
   - RRF, k=60: each chunk scores 1/(60 + rank) in each list it appears in, summed, so a chunk both searches rank highly beats one only a single search likes.
   - Implemented directly (about ten lines) rather than with LlamaIndex's `QueryFusionRetriever`, which sends one query string to every retriever, whereas BM25 and the vector store here receive different queries.
5. **Reranking:** the fused top 50 is reranked by the Voyage reranker API (`rerank-3-lite`), and its top 10 becomes the final context.
   - Reranking 50 rather than 10 lets the reranker rescue a relevant chunk that fusion placed 11th-50th.
   - The reranker scores chunks against the original question rather than the rewritten semantic query, so a poor rewrite cannot mislead it and its input is the same in every ablation.
   - Page metrics are taken on the fused top 10 (pre-rerank) and the reranked top 10 (post-rerank), so both are scored at the same depth.
   - `rerank-3-lite` over the 2.5 line because the 3 models carry a 200M free-token allowance while 2.5 has none, and Voyage state the newer line is strictly better on quality, context length, latency and throughput.
   - a full 112-question run costs roughly 6M reranking tokens, about 3% of that allowance, so `rerank-3` is equally affordable and one string away if needed.
   - which model to keep is decided on our own pre/post-rerank page metrics, since no trustworthy public reranker leaderboard exists and published comparisons sit within 1-3 NDCG points of each other.

**Generation** — the reranked context is formatted consistently and handed to the answer model.

- **Context format:** the reranked chunks, each labelled with document and page, are assembled into a `[Document | Page]` context block.
  - No section label, in this or any experiment: Experiment 2's heading path feeds only its retrieval score, so that Experiment 2 changes ranking alone and not also what the answer model reads.
  - The oracle condition and the retrieval-condition context blocks are built in the identical shape, so prompt formatting itself can't explain a results gap between conditions.
- **Answer model:** GLM-5.3-flash, via OpenRouter, produces the free-text answer.
  - The same model, settings, and prompt are used across every condition and every experiment, so retrieval is isolated as the only variable.

```mermaid
flowchart TB
    subgraph ING["INGESTION"]
        A1[Parse PDF<br/>Azure Document Intelligence<br/>prebuilt-layout]
        A2[Cache raw JSON<br/>content + pages/paragraphs/tables/sections]
        A3[Strip headers/footers/page numbers]
        A4[Page-bounded chunking<br/>no chunk crosses a page<br/>prose cut at headings, 250-1,024 tokens]
        A5[Tables and figures -&gt; own chunk<br/>never split]
        A6[Attach chunk metadata<br/>ticker, filing type, year, page]
        A1 --> A2 --> A3 --> A4
        A4 --> A5
        A4 --> A6
    end

    subgraph STO["STORAGE"]
        B1[(BM25 keyword index<br/>bm25s)]
        B2[Embed chunks<br/>voyage-4-lite, input_type=document]
        B3[(Vector store<br/>chunk embeddings + metadata)]
        B2 --> B3
    end

    subgraph RET["RETRIEVAL"]
        C1[Query enhancement LLM call<br/>-> filename from list,<br/>keyword query, semantic query]
        C2[Metadata filter<br/>chosen COMPANY_YEAR_TYPE filename<br/>fallback: whole scope if none]
        C3[BM25 search]
        C4[Semantic search]
        C5[RRF fusion<br/>k=60]
        C6[Voyage reranker<br/>fused top 50 -> top 10]
        C1 --> C2 --> C3
        C2 --> C4
        C3 --> C5
        C4 --> C5
        C5 --> C6
    end

    subgraph GEN["GENERATION"]
        D1[Assemble context<br/>Document | Page]
        D2[GLM-5.3-flash via OpenRouter]
        D3[Free-text answer]
        D1 --> D2 --> D3
    end

    A6 --> B1
    A6 --> B2
    B1 --> C3
    B3 --> C4
    C6 --> D1
```

## Design decisions

| Decision | Why |
|---|---|
| Azure Document Intelligence for parsing (paid tier, `prebuilt-layout`, markdown output) | "VLM-agentic" parsers read a page the way a person would, with a correction/verification pass — good when endless edge cases exist in human-authored layout (tables spanning pages, inconsistent styling). Chosen over `sec-parser` (unmaintained), Unstructured.io (PDF-oriented but dependency friction), and a hand-built BeautifulSoup classifier (heading detection unresolved, since bold is often applied via inline CSS rather than `<b>`/`<strong>`). Startup credits made cost a non-issue; setup is light (endpoint + key, one SDK). Still open: whether its section nesting is reliable enough for Experiment 2's heading path — PageIndex is the live fallback. |
| Parse once, cache forever | Azure's raw JSON is saved per PDF as `parsed/<doc_name>.json`, out of Git; a filing counts as parsed when that file exists and passes the page-count check. Re-chunking should never require re-parsing. |
| Page-bounded chunking (every chunk belongs to exactly one page; a carried-over sentence is text only) | Keeps page-based retrieval metrics (recall, precision, MRR) exact — a chunk can't straddle a page boundary and blur which page it "belongs" to. Enforced with a harness assertion. |
| Tables and figures as their own chunks | Table structure and the footnote/table relationship is exactly where naive text chunking breaks; giving each table its own chunk preserves it. No table is large enough to need splitting (largest 3,863 tokens against a 32,000-token limit). Figures get the same treatment so a chart's labels and numbers stay together. |
| Structure-based chunking for prose (cut at headings, ~250-1,024 tokens) | Chunks follow the document's own sections, so each chunk mostly sits under one heading — which also gives Experiment 2 clean heading paths on the same chunks. Cuts use Azure's raw heading positions, so Experiment 1 doesn't wait for the heading-fix pass. A fixed-size baseline is deferred as an ablation. |
| LlamaIndex for orchestration | Provides chunking, BM25 retrieval (wrapping `bm25s`) with metadata filters, and semantic retrieval with metadata filters as ready components (fusion and the reranker call are short enough to write directly), kept consistent with the same framework across chunking, hybrid retrieval, and later the Experiment 3 agent loop, rather than mixing frameworks (e.g. LangChain for splitting). |
| `bm25s` via LlamaIndex's `BM25Retriever` | Faster than a hand-built keyword index using the same underlying data structures; must be checked (pinned to a current version) that its `filters` argument actually filters on this project's data, since that support was only recently added — fallback is building the retriever from an already-filtered node list. |
| voyage-4-lite for embeddings | Cheapest embedder scoring above 80% on the relevant MTEB/RTEB (finance) leaderboard; the 200M free-token allowance covers the full corpus (~10M tokens) even across several re-chunks. |
| Filter chunks before search, then RRF-fuse BM25 and semantic results | Filtering narrows the candidate set to the right filing(s) before either retrieval method runs, so fusion combines two rankings over the same (correct) pool rather than fusing then filtering. |
| RRF with k=60 | Standard reciprocal-rank-fusion constant for combining ranked lists from heterogeneous scoring systems (BM25 scores and cosine similarities aren't on the same scale, so rank position rather than raw score is fused). |
| 50 candidates per search and into the reranker; 10 chunks to the answer model | Fusing and reranking longer lists lets a relevant chunk ranked below 10 by either search still reach the context. The final depth of 10 matches FinCARDS' top-10. |
| Query enhancement at reasoning effort `low` | Choosing a filename from a closed list and rewriting a question is a light task; `low` keeps the extra call per question fast. The answer model's settings are unchanged. |
| Voyage reranker instead of a GPU reranker (e.g. `BAAI/bge-reranker-v2-gemma`) | Avoids standing up local GPU infrastructure for reranking; used as an API call instead, at the cost of a per-call dependency on Voyage's service. |
| Filenames (not a metadata table) as the filter target | Filenames already encode `COMPANY_YEAR_TYPE`, so the query-enhancement LLM call can select a filing directly by returning a filename (parsed from the right, since company names can contain underscores), rather than joining against separate metadata columns. |
| Fall back to unfiltered search when the chosen filter matches no filing | Prevents a wrong or overly narrow filter from silently returning nothing — the system still attempts to answer rather than failing closed. |
| Retrieval is one function taking arguments (query, retrieval method, metadata filters, top_k) rather than a hardcoded pipeline | Experiment 1 calls it once with fixed arguments; Experiment 3's agent search tool is the *same* function, with the agent choosing those arguments at runtime. Writing retrieval as a hardcoded pipeline would mean writing it twice. |
| Query-enhancement prompt lives in one place, called by both Experiment 1 and Experiment 3 | Experiment 1 calls it once up front; Experiment 3's agent does the same job across its own first two steps. Keeping it in one place stops the two experiments quietly diverging on this piece. |
| Same generation model, settings, and prompt across every condition and experiment | Isolates retrieval as the only variable — GLM-5.3-flash via OpenRouter is used identically in the Exp 0 conditions, Experiment 1 and Experiment 2, and would be in the future-work Experiment 3. |

## Metrics

- **Page-level retrieval metrics** — computed for every retrieval condition.
  - Page recall — retrieved gold pages ÷ all gold pages.
  - Page precision — retrieved gold pages ÷ all unique retrieved pages.
  - Page MRR — 1 ÷ rank of the first chunk that contains a gold page (document-aware: computed against `(doc_name, page_num)` pairs, not bare page numbers, so a page number matching in the wrong document doesn't count).
  - All three are computed **both before and after reranking** (`page_metrics()` called twice, recorded as two sets of fields on the prediction row), so the reranker's effect on retrieval quality is visible on its own rather than only showing up in final answer accuracy.
- **Filter accuracy** — chosen filename == gold `doc_name`.
  - This splits "wrong document" from "wrong chunk" in the failure-mode tree.
  - It is only meaningful in the shared-store condition, since single-store already scopes the search to the question's own filing.
- **Answer accuracy** — via the LLM-as-judge described under Data, binary correct/incorrect.
- **Token cost** and **latency** — recorded per answer.
  - Fields: requested model, returned model, serving provider, token usage, latency, and cost.
  - This also covers embedding and reranking cost, not just the generation model's.
- **FinanceBench's five context conditions** — each isolates a different point of possible failure. The first three are the Exp 0 conditions (`Exp 0/Exp 0.md`); Experiment 1 runs the last two:
  - **closed-book** — no context given; measures what the model already knows from its own parameters, with no filing at all.
  - **oracle** — the exact correct pages are handed to the model directly; measures the reasoning ceiling if retrieval were perfect.
  - **long-context** — the full filing(s) are stuffed directly into the model's context window; tests RAG against simply using a large context window.
  - **single-store** — the vector store contains only the filing that holds the answer; errors here come from bad chunking (wrong or insufficient recall within the right document).
  - **shared-store** — the vector store contains chunks from every company's filings; errors here can additionally come from finding the wrong document entirely, which is what filter accuracy is designed to catch.

## Baseline and ablations

- **Baseline: the Exp 0 conditions** — closed-book, oracle and long-context, the direct LLM API call with no RAG (`Exp 0/Exp 0.md`). Long-context is the target; the oracle is the ceiling.
- **Ablations on the Experiment 1 pipeline** (each removes or swaps exactly one component so its individual contribution can be measured):
  - BM25 only — drop semantic search and fusion, keep query enhancement and reranking.
  - Semantic only — drop BM25 and fusion, keep query enhancement and reranking.
  - No reranker — take the RRF-fused top 10 straight to generation, skipping the Voyage reranking step.
  - No query enhancement — search with the raw user question, no metadata filter, no expanded keyword/semantic query pair.
  - DEFERRED, only if time remains: fixed-size chunking — 512-token chunks within each page instead of heading cuts, to measure what structure-based chunking contributes.

## Results

Runs (each folder's `summary.xlsx` is committed):

- Experiment 1: `results/20260928-202531--exp1--full`, 224 jobs (112 questions × single-store and shared-store), all successful.
- Exp 0 conditions (closed-book, oracle, long-context): `results/20260917-012959--financebench--baseline-context-conditions-v1`, 336 jobs (`Exp 0/Exp 0.md`).
- Judging: 209 of Experiment 1's 224 answers were settled by the judge (both passes agreed); the 15 where the passes disagreed were adjudicated by hand under the rule in `Benchmark.md`.

**Overall (Experiment 1 vs. the Exp 0 conditions)**

Page metrics are after reranking unless marked pre-rerank. Cost is per answer: generation, plus retrieval (query enhancement, embedding and reranking) for the two retrieval conditions. Latency is the answer call, plus retrieval where there is one.

| Condition | Page recall (pre-rerank) | Page precision (pre-rerank) | Page MRR (pre-rerank) | Page MRR (post-rerank) | Filter accuracy | Answer accuracy | Cost per answer | Latency | n |
|---|---|---|---|---|---|---|---|---|---|
| closed-book (Exp 0) | — | — | — | — | — | 41.1% (46/112) | $0.0003 | 7.7 s | 112 |
| oracle (Exp 0) | — | — | — | — | — | 91.1% (102/112) | $0.0003 | 4.2 s | 112 |
| long-context (Exp 0, target) | — | — | — | — | — | 85.7% (96/112) | $0.0152 | 8.3 s | 112 |
| single-store | 0.957 (0.854) | 0.144 (0.128) | 0.537 | 0.783 | — | 90.2% (101/112) | $0.0022 | 9.1 s + 3.8 s | 112 |
| shared-store | 0.942 (0.793) | 0.142 (0.122) | 0.533 | 0.766 | 98.2% (110/112) | 86.6% (97/112) | $0.0015 | 9.5 s + 3.8 s | 112 |

What this shows:

- **The target is met.** Both retrieval conditions match or beat Exp 0's long-context condition (90.2% and 86.6% against 85.7%) while sending the model 10 chunks instead of the whole filing, at about a seventh to a tenth of the cost per answer. Single-store comes within one question of the oracle ceiling (101 against 102).
- **Retrieval nearly always finds the evidence.** After reranking, a gold page is among the 10 chunks for 95.7% of single-store and 94.2% of shared-store questions.
- **The reranker does most of the ranking work.** It lifts recall from 0.854 to 0.957 (single-store) and 0.793 to 0.942 (shared-store), and MRR from about 0.53 to about 0.78: the gold page moves from typically second or third to typically first.
- **Precision is low by construction.** Most questions have one gold page, and the answer model always receives 10 chunks, so about 0.1-0.2 is the ceiling rather than a weakness.
- **Finding the right filing is nearly solved.** Query enhancement chose the gold filing for 110 of 112 shared-store questions. The two misses (`financebench_id_02981`, `financebench_id_06247`) are the only shared-store failures caused by the wrong document.
- **Retrieval is slower.** The query-enhancement call accounts for most of the 3.8 s retrieval time.

**Where the wrong answers come from** (failure analysis against the oracle run)

| | single-store | shared-store |
|---|---|---|
| Wrong answers | 11 | 15 |
| Oracle also wrong (the model fails even with the right pages) | 7 | 7 |
| Gold pages retrieved, answer still wrong | 4 | 6 |
| Wrong filing chosen | 0 | 2 |
| Correct answers with no gold page retrieved | 4 | 4 |

Most errors are not retrieval errors: in 7 of 11 single-store and 7 of 15 shared-store failures, the model is also wrong when handed the exact gold pages. Four answers per condition are correct without any gold page, from the model's own knowledge or from a page the annotators did not mark.

**Segmented by generation method** (single-store / shared-store; page metrics after reranking)

| Segment | n | Page recall | Page precision | Page MRR | Answer accuracy |
|---|---|---|---|---|---|
| metrics-generated | 50 | 1.000 / 0.960 | 0.171 / 0.163 | 0.937 / 0.897 | 94.0% / 86.0% |
| domain-relevant | 48 | 0.910 / 0.917 | 0.124 / 0.127 | 0.606 / 0.606 | 83.3% / 85.4% |
| novel-generated | 14 | 0.964 / 0.964 | 0.117 / 0.120 | 0.845 / 0.845 | 100% / 92.9% |

Domain-relevant questions are the hardest to retrieve for: their gold page ranks first much less often (MRR 0.61 against 0.94 for metrics-generated). They are generic questions ("is this company capital-intensive?") whose wording matches many sections.

**Segmented by cognitive skill** (single-store / shared-store; page metrics after reranking; segments sum to 128, not 112, since 16 questions carry more than one skill)

| Segment | n | Page recall | Page precision | Page MRR | Answer accuracy |
|---|---|---|---|---|---|
| Numerical reasoning | 57 | 1.000 / 0.965 | 0.169 / 0.163 | 0.836 / 0.801 | 91.2% / 86.0% |
| Information extraction | 36 | 0.944 / 0.944 | 0.117 / 0.117 | 0.729 / 0.729 | 91.7% / 91.7% |
| Logical reasoning | 21 | 0.841 / 0.857 | 0.126 / 0.133 | 0.556 / 0.556 | 71.4% / 76.2% |
| unlabelled | 14 | 0.964 / 0.964 | 0.117 / 0.120 | 0.845 / 0.845 | 100% / 92.9% |

Logical reasoning is the weakest segment, but mostly because of the model rather than retrieval: the oracle, with perfect pages, also scores 71.4% (15/21) there.

**Ablations** — not run. They were the first cut in the revised timeline (`Build Order.md` → Timeline reality check), so the contribution of each component is not measured separately. The pre- and post-rerank columns above give the reranker's effect on retrieval, but not on answer accuracy.

**Limitations of these results**

- One run per condition. The answer model is re-run for every job, so a repeat run would change some answers even with identical retrieval; Experiment 2's comparison puts this at roughly 18 of 224 answers changing verdict (see `Exp 2.md` → Results).
- Human review covers only the answers where the two judge passes disagreed. An answer both passes got wrong, or both got right, is never checked by hand.
- FinanceBench's working-capital gold answers use inconsistent definitions: Corning's uses operating items, while American Water Works' uses total current items, although its justification lists operating items. The adjudication rule accepts any standard, stated definition that reaches the gold's conclusion.
