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
- **Baseline it is tested against:** a direct LLM API call, full context window, no RAG.
- **Target:** match or beat LLM-in-context-window performance while not being limited by context window size.
- **Role in the wider project:** Experiment 1 is the shared spine.
  - Experiment 2 changes only the dense-retrieval score (structure-aware embeddings).
  - Experiment 3 changes only the retrieval step into an agent loop (query planning, retries, a calculator tool).
  - Everything else — parsing, chunking, query enhancement, BM25, fusion, reranking, generation — stays identical across all three, so that any difference in results can be attributed to the one thing that changed.

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

- **Baseline:** a direct LLM API call, full context window (the whole filing, or as much as fits), no retrieval.
  - This is the FinanceBench "long-context" condition and is what Experiment 1's target is measured against.
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
  - Any overlap applies only within a page. A chunking bug is defined as any chunk spanning two pages, and the harness asserts against this — this is what keeps page-based retrieval metrics exact.
  - Tables are their own chunks: each table becomes a chunk on its own; an oversized table is split by rows, repeating the header row in each split.
  - Everything that isn't a table goes to a 1,024-token recursive/sentence splitter (token count, not character count).
- **Metadata:**
  - Document-level metadata (company, doc_type, doc_period) is read from `financebench_document_information.jsonl`, not inferred from parsing — FinanceBench has no ticker field.
  - Each chunk carries metadata for filtering: SEC filing type, company ticker, financial year, and page number.
- **This is currently open:** whether Azure Document Intelligence's section nesting is good enough to also carry heading paths (needed for Experiment 2) is unverified.
  - The fallback is PageIndex if Azure's `sections` don't nest correctly or Item headings aren't at the top level.

**Storage** — two indexes are built over the chunks, each filterable by metadata before search runs.

- **Vector store:** chunk embeddings via `voyage-4-lite`.
  - Chosen as the cheapest embedder scoring above 80% on the relevant MTEB/RTEB (finance) leaderboard segment, with a 200M-token free allowance that comfortably covers the corpus (~10k chunks × ~1k tokens ≈ 10M tokens, even across several re-chunks).
  - Chunks are embedded with `input_type="document"`, questions with `input_type="query"`.
  - Each embedding is cached, keyed by model name plus a hash of the chunk text, so re-chunking only re-embeds what changed.
- **Keyword index:** `bm25s`, via LlamaIndex's `BM25Retriever`.
  - Standard BM25 parameters (k1=1.5, b=0.75).
  - Filtering narrows which chunks are searched, but BM25's word statistics (term frequency, document frequency) are still computed over the whole corpus — the same behaviour as Elasticsearch, kept deliberately so results stay comparable if the keyword backend is swapped later.
- **What's stored per chunk:** the chunk text, its embedding, and its filtering metadata (ticker, filing type, year, page) together, so retrieval can filter and score in the same place.
- **Filtering support:** both indexes support metadata filtering before search.
  - `where`-style filtering on the vector store.
  - `MetadataFilters` on the BM25 retriever, if it actually filters correctly — see Design decisions.

**Retrieval** — query enhancement narrows and expands the question, metadata filtering narrows the search pool, BM25 and semantic search run over that pool, and the two ranked lists are fused before reranking.

1. **Query enhancement:** one LLM call returns structured JSON.
   - Fields: company, year(s), doc_type, a keyword query (financial shorthand expanded out, e.g. PP&E → property, plant and equipment; COGS, DPO, FCF, capex), and a semantic query.
   - This same prompt is used by both Experiment 1 (called once, up front) and Experiment 3 (called by the agent through its own first two steps), so the two experiments can't quietly diverge on this piece.
2. **Metadata filtering:** filenames follow the pattern `COMPANY_YEAR_TYPE.pdf`.
   - The enhancement step chooses a filename directly (parsed from the right, since company names themselves can contain underscores) rather than matching separate metadata table columns.
   - Filtering happens on chunks *before* search.
   - If the filter matches no filing, the system falls back to searching unfiltered rather than returning nothing.
3. **BM25 + semantic search:** BM25 keyword search and semantic (embedding cosine-similarity) search run over the (possibly filtered) chunk set, in parallel.
4. **RRF fusion:** the two ranked lists are combined with reciprocal rank fusion.
   - RRF, k=60, via LlamaIndex's `QueryFusionRetriever` in `reciprocal_rerank` mode.
   - `num_queries=1` so it doesn't invent extra queries.
5. **Reranking:** the fused top-k (retrieval depth 10) is reranked by the Voyage reranker API to produce the final top-n context.

**Generation** — the reranked context is formatted consistently and handed to the answer model.

- **Context format:** the reranked chunks, each labelled with document, page, and (once Experiment 2 exists) section, are assembled into a `[Document | Page | Section]` context block.
  - The oracle condition and the retrieval-condition context blocks are built in the identical shape, so prompt formatting itself can't explain a results gap between conditions.
- **Answer model:** GLM-5.3-flash, via OpenRouter, produces the free-text answer.
  - The same model, settings, and prompt are used across every condition and every experiment, so retrieval is isolated as the only variable.

```mermaid
flowchart TB
    subgraph ING["INGESTION"]
        A1[Parse PDF<br/>Azure Document Intelligence<br/>prebuilt-layout]
        A2[Cache raw JSON<br/>content + pages/paragraphs/tables/sections]
        A3[Strip headers/footers/page numbers]
        A4[Page-bounded chunking<br/>1 chunk = 1 page]
        A5[Tables -&gt; own chunk<br/>split by rows if oversized]
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
        C1[Query enhancement LLM call<br/>-> company, year, doc_type,<br/>keyword query, semantic query]
        C2[Metadata filter<br/>select filename by COMPANY_YEAR_TYPE<br/>fallback: unfiltered if no match]
        C3[BM25 search]
        C4[Semantic search]
        C5[RRF fusion<br/>k=60]
        C6[Voyage reranker<br/>top-k -> top-n]
        C1 --> C2 --> C3
        C2 --> C4
        C3 --> C5
        C4 --> C5
        C5 --> C6
    end

    subgraph GEN["GENERATION"]
        D1[Assemble context<br/>Document | Page | Section]
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
| Parse once, cache forever | Azure's raw JSON is saved per PDF, out of Git, with a manifest (as in the benchmark data prep). Re-chunking should never require re-parsing. |
| Page-bounded chunking (one chunk = one page, overlap only within a page) | Keeps page-based retrieval metrics (recall, precision, MRR) exact — a chunk can't straddle a page boundary and blur which page it "belongs" to. Enforced with a harness assertion. |
| Tables as their own chunks | Table structure and the footnote/table relationship is exactly where naive text chunking breaks; giving each table its own chunk (splitting oversized ones by row, repeating the header) preserves it. |
| LlamaIndex for orchestration | Provides chunking, BM25 retrieval (wrapping `bm25s`) with metadata filters, semantic retrieval with metadata filters, RRF fusion, and a Voyage reranker postprocessor as ready components, kept consistent with the same framework across chunking, hybrid retrieval, and later the Experiment 3 agent loop, rather than mixing frameworks (e.g. LangChain for splitting). |
| `bm25s` via LlamaIndex's `BM25Retriever` | Faster than a hand-built keyword index using the same underlying data structures; must be checked (pinned to a current version) that its `filters` argument actually filters on this project's data, since that support was only recently added — fallback is building the retriever from an already-filtered node list. |
| voyage-4-lite for embeddings | Cheapest embedder scoring above 80% on the relevant MTEB/RTEB (finance) leaderboard; the 200M free-token allowance covers the full corpus (~10M tokens) even across several re-chunks. |
| Filter chunks before search, then RRF-fuse BM25 and semantic results | Filtering narrows the candidate set to the right filing(s) before either retrieval method runs, so fusion combines two rankings over the same (correct) pool rather than fusing then filtering. |
| RRF with k=60 | Standard reciprocal-rank-fusion constant for combining ranked lists from heterogeneous scoring systems (BM25 scores and cosine similarities aren't on the same scale, so rank position rather than raw score is fused). |
| Retrieval depth (top-k) 10 before reranking | Matches FinCARDS' use of top-10 as the candidate pool size handed to the reranker. |
| Voyage reranker instead of a GPU reranker (e.g. `BAAI/bge-reranker-v2-gemma`) | Avoids standing up local GPU infrastructure for reranking; used as an API call instead, at the cost of a per-call dependency on Voyage's service. |
| Filenames (not a metadata table) as the filter target | Filenames already encode `COMPANY_YEAR_TYPE`, so the query-enhancement LLM call can select a filing directly by returning a filename (parsed from the right, since company names can contain underscores), rather than joining against separate metadata columns. |
| Fall back to unfiltered search when the chosen filter matches no filing | Prevents a wrong or overly narrow filter from silently returning nothing — the system still attempts to answer rather than failing closed. |
| Retrieval is one function taking arguments (query, retrieval method, metadata filters, top_k) rather than a hardcoded pipeline | Experiment 1 calls it once with fixed arguments; Experiment 3's agent search tool is the *same* function, with the agent choosing those arguments at runtime. Writing retrieval as a hardcoded pipeline would mean writing it twice. |
| Query-enhancement prompt lives in one place, called by both Experiment 1 and Experiment 3 | Experiment 1 calls it once up front; Experiment 3's agent does the same job across its own first two steps. Keeping it in one place stops the two experiments quietly diverging on this piece. |
| Same generation model, settings, and prompt across every condition and experiment | Isolates retrieval as the only variable — GLM-5.3-flash via OpenRouter is used identically in the baseline, Experiment 1, Experiment 2, and Experiment 3. |

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
- **FinanceBench's five context conditions** — each isolates a different point of possible failure:
  - **closed-book** — no context given; measures what the model already knows from its own parameters, with no filing at all.
  - **oracle** — the exact correct pages are handed to the model directly; measures the reasoning ceiling if retrieval were perfect.
  - **long-context** — the full filing(s) are stuffed directly into the model's context window; tests RAG against simply using a large context window.
  - **single-store** — the vector store contains only the filing that holds the answer; errors here come from bad chunking (wrong or insufficient recall within the right document).
  - **shared-store** — the vector store contains chunks from every company's filings; errors here can additionally come from finding the wrong document entirely, which is what filter accuracy is designed to catch.

## Baseline and ablations

- **Baseline:** direct LLM API call, full context window, no RAG (the long-context condition above, run without any retrieval step).
- **Ablations on the Experiment 1 pipeline** (each removes or swaps exactly one component so its individual contribution can be measured):
  - BM25 only — drop semantic search and fusion, keep query enhancement and reranking.
  - Semantic only — drop BM25 and fusion, keep query enhancement and reranking.
  - No reranker — take the RRF-fused top-k straight to generation, skipping the Voyage reranking step.
  - No query enhancement — search with the raw user question, no metadata filter, no expanded keyword/semantic query pair.

## Results

*Placeholder — to be filled in once benchmark runs are complete. Do not populate with invented numbers.*

**Overall (Experiment 1 vs. baseline, per condition)**

| Condition | Page recall | Page precision | Page MRR (pre-rerank) | Page MRR (post-rerank) | Filter accuracy | Answer accuracy | Token cost | Latency | n |
|---|---|---|---|---|---|---|---|---|---|
| closed-book | — | — | — | — | — | | | | |
| oracle | — | — | — | — | — | | | | |
| long-context (baseline) | — | — | — | — | — | | | | |
| single-store | | | | | — | | | | |
| shared-store | | | | | | | | | |

**Segmented by generation method** (n per segment: metrics-generated 50, domain-relevant 48, novel-generated 14)

| Segment | n | Page recall | Page precision | Page MRR | Answer accuracy |
|---|---|---|---|---|---|
| metrics-generated | 50 | | | | |
| domain-relevant | 48 | | | | |
| novel-generated | 14 | | | | |

**Segmented by cognitive skill** (n per segment: numerical reasoning 57, information extraction 36, logical reasoning 21, unlabelled 14 — segments sum to 128, not 112, since 16 questions carry more than one skill)

| Segment | n | Page recall | Page precision | Page MRR | Answer accuracy |
|---|---|---|---|---|---|
| Numerical reasoning | 57 | | | | |
| Information extraction | 36 | | | | |
| Logical reasoning | 21 | | | | |
| unlabelled | 14 | | | | |

**Ablations** (shared-store condition, full pipeline vs. each ablation)

| Variant | Page recall | Page precision | Page MRR | Filter accuracy | Answer accuracy |
|---|---|---|---|---|---|
| Full pipeline | | | | | |
| BM25 only | | | | — | |
| Semantic only | | | | — | |
| No reranker | | | | | |
| No query enhancement | | | | | |
