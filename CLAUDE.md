# Prompts for Research Papers

## [Master Prompt]

### Agentic AI for Financial Statement Analysis (UCL MSc Dissertation)

#### What I'm doing

My UCL MSc Computer Science dissertation (supervisor: Prof. Philip Treleaven) builds and evaluates AI agents that automate financial statement analysis: extracting information from long, unstructured financial documents and, time permitting (lower priority), using it to populate/audit spreadsheet-based financial models.

**Skill level:**

I'm a hands-on novice at LLM/agent/RAG/pipeline/Python tooling — I've completed boot.dev's "Build an AI Agent" and "RAG" courses, but haven't yet built a production pipeline from scratch. Don't assume familiarity with agent frameworks, vector stores, or Python tooling conventions beyond what those courses cover; explain implementation choices rather than assuming I already know the trade-offs.

**Timeline:**

Code completion ~18 September; dissertation writing starts after; first draft submission 19 September 2026. Note I plan to continue coding etc. and iterating on report until final submission on 21st September. When helping me sequence work, prioritize accordingly — flag anything that may not fit before 18 September to be scoped down or cut, not just noted as a nice-to-have.

#### Problem + contribution (brief)

**Problem:** Real financial work is messy and long-horizon — documents are unstructured and inconsistent, spreadsheets have complex cross-references, and LLMs are unreliable at self-correcting without external feedback. Current benchmarks show large capability gaps — this project measures against FinanceBench (112 questions over 64 10-K filings).

**Contribution:**

- A retrieval pipeline built in three stages — plain RAG → structure-aware RAG → agentic RAG — for extracting qualitative/numerical information from long financial documents (10-K/10-Q filings), evaluated against a direct-LLM-API baseline.

#### Current build-phase status

Need to setup benchmark for my testing, as per instructions in `docs sys design/Benchmark.md`.

Lit review and system design are done. The tool stack is now SETTLED (see `docs sys design/Systems Design Draft.md` → Tech stack): LlamaIndex for orchestration and components, Chroma for vectors, `bm25s` for keyword, Azure Document Intelligence for parsing (PageIndex is the live fallback), Voyage for embeddings and reranking, OpenAI SDK via OpenRouter for generation. Elasticsearch is documented as future work, not a live option. Do not reopen these unless something fails on contact with the data.

**Read these first, in this order:**

- `docs sys design/Build Order.md` — what to build, in what order, with each stage's requirements. START HERE.
- `docs sys design/Systems Design Draft.md` — the end-to-end pipeline: what and how, plus tech stack and all design decisions
- `docs sys design/Benchmark.md` — benchmark requirements and metrics
- `docs sys design/benchmark/FinanceBench Implementation Guide.md` — the harness as built
- `docs sys design/Exp 1/Exp 1.md`, `Exp 2/Exp 2.md`, `Exp 3/Exp 3.md` — one per experiment: description, architecture, decisions, ablations, results


#### Experiments — what I'm building, how I benchmark, and details

**Domain:** Corporate finance / equity research — SEC filings (10-K, 10-Q), financial statements.

**Benchmark** (same for all three): FinanceBench, 112 questions over 64 10-K PDFs, five conditions — closed-book, oracle, long-context, single-store, shared-store. See `docs sys design/Benchmark.md` and `docs sys design/benchmark/FinanceBench Implementation Guide.md`.

**Invariants — these make differences attributable, so don't break them:**

- each experiment is **Exp 1 with exactly one thing changed** (Exp 2: the dense score; Exp 3: who calls retrieval and how often)
- retrieval is **ONE function** `(query, method, filters, top_k, structure_weight=0)` — Exp 1 passes fixed arguments, Exp 3's search tool passes agent-chosen ones
- **ONE** shared query-enhancement prompt, used by Exp 1 up front and Exp 3's first two tools
- identical model, settings and answer prompt everywhere: `z-ai/glm-5.3-flash` via OpenRouter, reasoning effort `medium`, `[Document | Page | Section]` context blocks. Sole exception: Exp 3's budget / remaining-turns line
- every run labelled with its experiment and ablation variant, in config, the run's `config.toml` and its reports

##### Experiment 1 — Baseline retrieval architecture (the spine)

Fixed single-pass hybrid RAG over SEC 10-Ks. Not agentic (that's Exp 3), not structure-aware (that's Exp 2); both are built on it, so it ships first and complete.

- **Implementing:** Azure DI `prebuilt-layout`, parse once and cache the raw JSON → page-bounded chunks (one chunk = one page, tables their own chunk, 1,024-token splitter for prose), metadata from the prepared FinanceBench document-information JSONL → BM25 (`bm25s` via LlamaIndex, k1=1.5, b=0.75, letter/digit token split so `FY2018` matches "2018") + dense (`voyage-4-lite` in Chroma) → one query-enhancement call returning filters, keyword query and semantic query → filter to scope FIRST, then rank → RRF k=60 → top-10 → `rerank-3-lite` → GLM answers.
- **In / out:** finance question about a US public company + its filings → free-text answer, scored for accuracy.
- **Baseline:** direct LLM call, full context, no RAG. **Target:** match or beat it without being context-window-limited.
- **Ablations:** BM25 only; semantic only; no reranker; no query enhancement.
- **Metrics (master set, inherited by Exp 2 and 3):** page recall / precision / MRR on `(doc_name, page_num)` pairs, **pre- and post-rerank**; filter accuracy (chosen filing == gold, shared-store only); answer accuracy from the judge (DeepSeek-V4-Flash on Azure Foundry, reference-guided, double-graded with order swapped, binary); token cost incl. embedding and rerank; latency. Segmented by generation method and cognitive skill.
- **Open:** the two Stage 0 spikes — does `BM25Retriever`'s `filters` argument really filter, and is Azure's section nesting good enough for Exp 2.
- **Docs:** `Exp 1/Exp 1.md` → `Systems Design Draft.md` (Ingest files, Retrieve, Generate answer) → `Build Order.md` Stages 1-2.

##### Experiment 2 — Structure-aware retrieval

Exp 1 with ONE change: the dense score gains a structure term from each chunk's heading path. A training-free approximation of Fin-STAR, **not** a replication.

- **Implementing:** heading path per chunk (each heading's page range runs to the next heading at the same or higher level; ties on a page go to the heading covering more of it; depth capped at 5 by dropping the deepest) → each level embedded **separately**, each unique heading once, stored apart from chunk embeddings and keyed by chunk ID → structure vector = softmax-weighted average of that chunk's heading embeddings, weights from **chunk**-to-heading similarity (not query-to-heading), divisor from config starting at 0.05, normalised to length 1 → score = cosine(query, chunk) + cosine(query, structure), in a ~20-line custom `BaseRetriever`. **No double-length vector store** — the theory's concatenation decomposes into those two added scores.
- **Baseline:** Exp 1, which *is* this system with the structure weight at zero — verify that equivalence before trusting any gain.
- **Ablations:** (A) Exp 1 → (B) heading path, no virtual node → (C) + virtual node (deferred). **A vs B alone is a complete result.** Optional: the divisor comparison.
- **Metrics:** Exp 1's, plus pre-rerank page metrics **required** — the reranker sees only chunk text, so it can wash out a structural gain and hide the finding.
- **Open:** structure source, Azure `sections` vs PageIndex. **Deferred:** the SLM virtual node. **Cut:** HiChunk (its chunk-point predictor needs a GPU).
- **Docs:** `Exp 2/Exp 2.md` → `Systems Design Draft.md` (chunking → experiment 2; Semantic search → Experiment 2) → `Build Order.md` Stage 3. Idea originates in the lit review (Fin-STAR row of `CS_Research_Matrix.xlsx`) — that's its source, not its spec.

##### Experiment 3 — Agentic retrieval

Exp 1 with ONE change: the model uses tools in a loop instead of one fixed retrieve-then-answer pass, choosing its own filters, method and queries, seeing results, and searching again.

- **Tools, fixed steps then loop:** (1) one-off — given the filing list, choose metadata filters; (2) one-off — first search, choosing method (BM25 / semantic / hybrid) and query; (3) loop — search again, told whether its chosen filing was retrieved so it can retry; (4) loop — calculator; (5) answer.
- **Implementing:** LlamaIndex's agent with `max_iterations` and `early_stopping_method="generate"` so a capped run still answers; trace written by **our** tool functions (~3 lines each), not LlamaIndex's event system; loop caps (iterations start at 5, plus max tool calls) with **cap-reached as a fourth outcome** beside success / error / `did_not_fit` — skipped on resume, excluded from the accuracy denominator, counted in reports, so *n* stays constant; budget and remaining turns stated in the prompt each iteration; `max_output_tokens` 8192, since reasoning tokens count as output and are spent every turn.
- **Harness:** `execution/job.py` hands the whole question to a pipeline returning answer + final chunks + trace + usage, instead of running a fixed sequence. Exp 1 and 2 become one-round pipelines behind the same plug. Built at `Build Order.md` Stage 2.0 — retrofitting it later touches every module.
- **Baseline:** Exp 1's single pass. **Ablations:** (A) single pass → (B) + retry on empty/wrong-filing retrieval → (C) + calculator and verification. **A vs B alone is a result**, and it separates retry-on-failure from calculation.
- **Metrics:** Exp 1's, plus average *k* (passages used, now varying per question, as LOFin reports), iterations used, cap-reached counts. Page metrics on the **final retrieval only**, to stay comparable; "all chunks seen" deliberately not scored, since the trace covers it.
- **Verify first:** that GLM-5.3-flash emits well-formed tool calls (~5 questions) before building the loop. If not, a stronger model for the agent breaks "same model everywhere" and must be stated.
- **Deferred:** query decomposition with Fin-STAR's symbolic logic topology (pays off on multi-document questions = LOFin, itself deferred; every FinanceBench question sits in one filing); fetch-whole-page (PDFTriage pattern, for tables split across pages).
- **Docs:** `Exp 3/Exp 3.md` → `Systems Design Draft.md` (Experiment 3 — Agents) → `Build Order.md` Stage 4.

## Project constraints

**Python:** 3.12 (`.python-version`, `requires-python = ">=3.12"`). Managed with `uv` — never `pip install` into the venv; use `uv add` so `pyproject.toml` and `uv.lock` stay authoritative. Every direct dependency is pinned to an exact version; keep it that way and commit the updated `uv.lock`.

**Dependencies — currently installed:** `jinja2`, `openai`, `pandas`, `pymupdf`, `transformers`, `xlsxwriter`; `pytest` (dev). **Approved to add when its stage arrives:** `llama-index-core`, `llama-index-retrievers-bm25`, `llama-index-vector-stores-chroma`, `chromadb`, `azure-ai-documentintelligence`, `voyageai`, `numpy`, and — only if Voyage goes through LlamaIndex — `llama-index-postprocessor-voyageai-rerank` and `llama-index-embeddings-voyageai`. **Do not add anything else** without agreeing it first: no LangChain, no vector database beyond Chroma, no Elasticsearch (documented as future work), no RAGAS.

**File structure:**
- `src/sec_rag_benchmark/` — the benchmark package, with `cli.py` and `config.py` at the root
- `src/sec_rag_benchmark/dataset/` — FinanceBench preparation and development subsets
- `src/sec_rag_benchmark/pipeline/` — context construction and answer generation
- `src/sec_rag_benchmark/execution/` — preflight, one-job execution and resumable run orchestration
- `src/sec_rag_benchmark/evaluation/` — retrieval metrics, answer judging and manual adjudication
- `src/sec_rag_benchmark/reporting/` — aggregation, failure diagnosis and workbook rendering
- `tests/` — pytest, one `test_<module>.py` per module, fixtures in `conftest.py`. Tests must make no paid API calls
- `configs/financebench.toml` — public, editable run settings. Anything that can change results lives here, not in code
- `scripts/` — standalone utilities that are not part of the package
- `docs sys design/` — requirements and design (the living sources of truth)
- `docs/libraries/` — offline library docs, gitignored except `SKILL.md`
- Generated and never committed: `data/`, `benchmarks/`, `results/`, `.venv/`, `.env`
- New code goes in the folder that matches its responsibility. A new module needs a reason it is not an existing one.

**Credentials:** never hard-code or print a key. Read from the environment, named in `.env.example`: `OPENROUTER_API_KEY`, `AZURE_DEEPSEEK_API_KEY`, `AZURE_DEEPSEEK_ENDPOINT`, `VOYAGE_API_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`. Load lazily, so no-spend commands and tests run without them.

**Spending:** any command that calls a paid API must be obviously paid, and must have a no-spend equivalent (`--dry-run`, or a fake client in tests). Checkpoint after every question so a crash never loses completed work.

## Workflow

Treat the user as the learner-owner of the system, not merely an approver. Keep README files navigational; requirements and design belong in the system-design documents, while implementation detail belongs in the relevant experiment's implementation guide.

### 1. Brainstorm

- Start with the relevant stage in `docs sys design/Build Order.md`: it states what to build next, the minimum result and what can be cut.
- Read the corresponding parts of `docs sys design/Systems Design Draft.md`: it is the comprehensive source for requirements, architecture, rationale, literature grounding and rejected alternatives.
- Also read `docs sys design/Benchmark.md` when the work affects benchmarking.
- Use the brainstorming skill to surface missing requirements, unclear assumptions and genuine alternatives. State uncertainty rather than choosing silently.
- Integrate approved requirement or design changes into both `Systems Design Draft.md` and the corresponding part of `Build Order.md`, preserving their existing wording and logical flow. Keep the relevant `Exp N/Exp N.md` consistent when a decision affects how the experiment should be described.
- Show the unstaged documentation diff for review before planning.

### 2. Plan

- After the brainstorming changes are approved, use the writing-plans skill to create or update one `docs sys design/Exp N/Implementation Guide.md` per experiment.
- Base the guide on the approved requirements in `Systems Design Draft.md` and `Build Order.md`.
- Work through three approval gates: architecture/libraries; files/data/interfaces; pseudocode/vertical slices/tests. Brainstorm normal, edge and failure cases with the user.
- Keep the plan limited to requested functionality. Record agreed planning changes in the same guide rather than relying on conversation history or creating another specification.
- Organise the guide so it can later be read alongside the finished code in a main-function-first order.

### 3. Execute

- Use the executing-plans or subagent-driven-development skill and implement one approved vertical slice at a time.
- Write tests before or alongside the implementation, then run the focused tests and relevant regression suite.
- Update the same implementation guide with the actual files, functions, data flow, tests and any plan/code discrepancies.
- Show the unstaged Git diff and as-built explanation, then draft the commit for the user to execute.
- Write-up runs alongside development: keep the relevant `Exp N.md` description consistent, and add results to its Results section on the day they are produced.

#### Coding practices

- Keep code simple and traceable: one clear responsibility per module and one complete operation per public function.
- Add helpers only when they make a real step clearer. Prefer explicit loops, branches and named values over compressed expressions or speculative abstractions.
- Apply DRY to business rules and knowledge, not superficial syntax. Small readable repetition is better than an abstraction that hides the flow.
- Keep CLI code to argument parsing and delegation. Comments should explain purpose and data movement, not ordinary Python syntax.
- Modify only what the approved slice requires. Mention unrelated problems without fixing or refactoring them.
- Before presenting code, ensure the implementation guide, comments and tests follow the same main-function-first flow.

### 4. Verify

- Before claiming a slice is complete, run Ruff's formatting check and linter, mypy over `src/`, the complete pytest suite, the lockfile check and `git diff --check`.
- Fix warnings rather than hiding them unless a project-level exclusion has been agreed. Apply formatting deliberately; do not run broad lint autofixes without inspecting the findings.
- Automated verification must use fake clients and make no paid API calls.
- Present the verification results with the unstaged diff.

### Git hygiene

A diff shows *what* changed; the commit message explains *why*. British English throughout.

- **Commit authority:** Keep all changes unstaged and uncommitted for review. Never run `git add`, `git commit`, `git push`, merge branches, or create a pull request. When I ask to "commit," interpret that as: run verification, show the uncommitted diff, list the files that belong in the commit, and draft the commit message and exact Git commands for me to run myself.
- **Subject line: imperative mood, capitalised, no trailing period, ≤ 50 chars.** Test: "If applied, this commit will [subject]". Write `Add heading-path attribution`, not `Added heading-path attribution`.
- **Standard formatting:** blank line between subject and body, body wrapped at 72 characters.
- **Use the body for what and why, not how.** The diff shows how. The body carries what the diff cannot: the problem being solved, why this approach over the alternatives, the trade-offs accepted, and any number or finding that justified the choice.
- **Small, focused commits — one logical change each.** Never bundle "fix bug + add feature + rename file": bundled commits break `git bisect` when locating a regression.
- **Separate code from documentation** where practical, and never mix an experiment's results into a code commit.
- **Commit on a branch, not `main`,** for anything larger than a doc fix; merge when its tests pass.
- **Never commit** credentials, generated data, parsed output, embeddings, or run results.
- **Do not cite AI-generated artefacts** in commit messages or in the dissertation. Reference the design documents, the benchmark, or the papers.
- **Record every result-affecting change as its own commit,** so a run's numbers can be traced to the exact tree that produced them.
