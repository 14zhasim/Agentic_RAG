---
name: library-docs
description: Local offline documentation for the project's tech-stack libraries — LlamaIndex, Chroma, Voyage AI, Azure Document Intelligence, OpenRouter, bm25s. Use when implementing or debugging against any of these, instead of fetching pages from the web.
---

# Library documentation (local)

Offline docs for the libraries in `docs sys design/Systems Design Draft.md` → Tech stack.
Not in Git (see `.gitignore`). Re-fetch or refresh with:

```bash
bash scripts/fetch_library_docs.sh
```

**Read the specific file for the task, not a whole `llms-full.txt`** — the Chroma and OpenRouter
full bundles are hundreds of KB to several MB. Grep them instead:

```bash
grep -n -i "where filter" docs/libraries/chroma/llms-full.txt | head
```

## Which file answers which question

### LlamaIndex — orchestration and components

| File | Use it for |
|---|---|
| `llamaindex/bm25_retriever.md` | `BM25Retriever.from_defaults`, the `filters` / `MetadataFilters` argument, stemmer and stopword settings, persist and reload |
| `llamaindex/rrf_fusion.md` | `QueryFusionRetriever` in `reciprocal_rerank` mode, combining BM25 with a vector retriever, `num_queries=1` |
| `llamaindex/retrievers.md` | The retriever interface, and subclassing `BaseRetriever` — needed for Exp2's scorer |
| `llamaindex/node_parsers.md` | Splitters and the markdown node parser that records heading paths |
| `llamaindex/node_postprocessors.md` | Where reranking plugs in |
| `llamaindex/vector_stores.md` | Vector-store integrations, including Chroma |
| `llamaindex/metadata_filtering.md` | Metadata filters against Chroma specifically |
| `llamaindex/voyage_embeddings.md` | Voyage embeddings through LlamaIndex, including `input_type` |
| `llamaindex/voyage_rerank.md` | The Voyage reranker post-processor |
| `llamaindex/agents.md`, `llamaindex/agent_tutorial.md` | `FunctionAgent`, tools, `max_iterations`, `early_stopping_method` — Exp3's loop |
| `llamaindex/llms.txt` | Index of all their docs, plus their search/grep/read API endpoints if something is missing here |

Their site serves raw markdown for any page by appending `index.md` to the page URL, so adding a
new page to the fetch script is a one-line change.

### Chroma — vector store

| File | Use it for |
|---|---|
| `chroma/llms.txt` | Map of their docs |
| `chroma/llms-full.txt` | Everything: collections, `add`/`query`, `where` metadata filters, persistence. **Grep, don't read** (~800KB) |

### Voyage AI — embeddings and reranking

| File | Use it for |
|---|---|
| `voyage/embeddings.md` | `voyage-4-lite`, `input_type="document"` vs `"query"`, batching limits |
| `voyage/reranker.md` | Reranker models and call shape |
| `voyage/rate-limits.md` | RPM/TPM per tier — check before embedding 10k chunks |
| `voyage/pricing.md` | Cost per million tokens and the free allowance |
| `voyage/tokenization.md` | Counting tokens before a call |
| `voyage/batch-inference.md` | Cheaper bulk embedding, if the corpus grows |
| `voyage/flexible-dimensions-and-quantization.md` | Output dimensions, if we ever shrink vectors |
| `voyage/error-codes.md`, `voyage/api-key-and-installation.md`, `voyage/quickstart-tutorial.md` | Setup and failure handling |

### Azure Document Intelligence — parsing

| File | Use it for |
|---|---|
| `azure-di/layout-model.md` | `prebuilt-layout`: what it returns — `pages`, `paragraphs` and their roles, `tables`, `sections`, spans into `content` |
| `azure-di/markdown-output.md` | The markdown output format, including page-break and page-number markers |
| `azure-di/sdk-python-quickstart.md` | `azure-ai-documentintelligence` client setup and the analyse call |
| `azure-di/service-limits.md` | Page and file-size limits, tier differences (the free tier reads only 2 pages) |

### OpenRouter — answer generation

| File | Use it for |
|---|---|
| `openrouter/llms.txt` | Map of their docs |
| `openrouter/llms-full.txt` | Provider routing, reasoning-effort parameters, usage and cost fields, error handling. **Grep, don't read** (~4MB) |

### bm25s — keyword index internals

| File | Use it for |
|---|---|
| `bm25s/README.md` | What LlamaIndex's `BM25Retriever` wraps: k1/b defaults, stemming, saving and loading an index |

## Version notes

Record the installed version next to anything version-sensitive, so a future session knows whether
these docs still match the code.

- `BM25Retriever`'s `filters` argument is a recent addition — pin a current
  `llama-index-retrievers-bm25` and keep the Stage 0 spike test that proves filtering works
- fill in the rest after `uv add`:
  - `llama-index-core`: _TBC_
  - `llama-index-retrievers-bm25`: _TBC_
  - `chromadb`: _TBC_
  - `voyageai`: _TBC_
  - `azure-ai-documentintelligence`: _TBC_
  - `openai`: 3.8.0 (already pinned)
