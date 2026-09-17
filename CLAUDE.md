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

**Problem:** Real financial work is messy and long-horizon — documents are unstructured and inconsistent, spreadsheets have complex cross-references, and LLMs are unreliable at self-correcting without external feedback. Current benchmarks show large capability gaps, e.g. FAB v1.1.

**Contribution:**

- An agentic retrieval pipeline (RAG → Graph RAG) for extracting qualitative/numerical information from long financial documents (10-K/10-Q filings), evaluated against a direct-LLM-API baseline.
- I'm currently reconsidering Experiment 3 (single-operation Excel agent) — I want to replace/extend it with **more document/RAG-centric architectures** rather than pure spreadsheet manipulation, to keep the thesis's core contribution concentrated on the retrieval/reasoning side rather than generic Excel tooling. Flag relevant RAG/agent architectures from literature that could substitute here.

#### Current build-phase status

Need to setup benchmark for my testing, as per instructions in `docs sys design/Benchmark.md`.

Lit review and system design are done. The tool stack is now SETTLED (see `docs sys design/Systems Design Draft.md` → Tech stack): LlamaIndex for orchestration and components, Chroma for vectors, `bm25s` for keyword, Azure Document Intelligence for parsing (PageIndex is the live fallback), Voyage for embeddings and reranking, OpenAI SDK via OpenRouter for generation. Elasticsearch is documented as future work, not a live option. Do not reopen these unless something fails on contact with the data.

**Read these first, in this order:**

- `docs sys design/Build Order.md` — what to build, in what order, with each stage's requirements. START HERE.
- `docs sys design/Systems Design Draft.md` — the end-to-end pipeline: what and how, plus tech stack and all design decisions
- `docs sys design/Benchmark.md` — benchmark requirements and metrics
- `docs sys design/benchmark/FinanceBench Implementation Guide.md` — the harness as built
- `docs sys design/Exp 1/Exp 1.md`, `Exp 2/Exp 2.md`, `Exp 3/Exp 3.md` — one per experiment: description, architecture, decisions, ablations, results

Plan for sys design:
**Start building system design – using RAG research doc, boot.dev notes, codebase_retrieval_parsing md (and maybeeee lit review stuff + claude chat to query all my notes on them). Then, identify failure points in architecture and if benchmark can help with this. References for these resources**:

- boot.dev notes: look at the md files /Users/zubairasim/rag-search-engine/course_notes
  `docs sys design/Benchmark.md`
- RAG research doc: /Users/zubairasim/Downloads/RAG\ research.docx
  Design\ Draft.md
- retrieval parsing: <../../../Library/CloudStorage/OneDrive-Personal/Documents/Masters/MSc Computer Science Project/Implementation Notes/Lit review/CODEBASE_RETRIEVAL_PARSING_ANALYSIS.md
- lit review stuff: <../../../Library/CloudStorage/OneDrive-Personal/Documents/Masters/MSc Computer Science Project/Implementation Notes/Lit review/CS_Research_Matrix.xlsx

#### Experiments — what I'm building, how I benchmark, and details

**Domain:** Corporate finance / equity research — SEC filings (10-K, 10-Q), financial statements, spreadsheets/financial models. Please read these files to understand my progress:
- `docs sys design/benchmark/Systems Design Draft.md`
**Benchmarking** consistent across all 3 experiments, visit `docs sys design/Benchmark.md`, `docs sys design/benchmark/FinanceBench Implementation Guide.md`

##### Experiment 1 — Agentic RAG from long financial documents

- **Build:** Retrieval pipeline over SEC filings. Full systems design will be done, still iterating as per 'Systems Design Draft.md'. Indexing: structure-aware parsing, structure-aware chunking, metadata filtering. Retrieval: BM25 hybrid search, re-ranking.
- **Input:** Natural-language finance question about a US public company + access to its filings.
- **Output:** Free-text answer, scored for accuracy.
- **Baseline:** Direct LLM API call, full context window, no RAG.
- **Benchmark, Metrics** See `docs sys design/Benchmark.md`
- **Target:** Match or beat LLM-in-context-window performance while not being limited by context window size.

##### Experiment 2 — Structured embedding and retrieval

- **Build:** Visit '/Users/zubairasim/Library/CloudStorage/OneDrive-Personal/Documents/Masters/MSc Computer Science Project/Implementation Notes/Lit review'

See the CS_Research_Matrix.xlsx + RAG research doc, basically want to index the chunk's subheading heading within the document, embed, concatenate with chunk's embedding vector, then do cosine similarity against query embedding vector concatenated with itself

- **Input/Output/Benchmark/Metric:** Same as Exp 1 (FinanceBench: 112 questions, 64 10-K PDFs), same baseline. Full design in `docs sys design/Exp 2/Exp 2.md`.

Reference deck: "Zubair_Research_Planning_Condense_v2" (converted to Zubair_Research_Planning_Condense_v2.md) contains the full slide-by-slide detail — benchmark leaderboards, task taxonomies, and source quotations — behind everything above, including the appendix material on Experiments 4 and 5.

##### Experiment 3 — agentic tooling, verification, looping

## Understanding-first development workflow

- Treat the user as the learner-owner of the system, not merely an approver.
- Edit tracked user-authored requirements documents directly as living sources of truth, preserving their wording and logical flow; establish a clean or explicitly staged baseline first, then use Git diff/history for review and recovery instead of duplicate specifications.
- Use a separate implementation guide for technical detail. Apply three explicit gates: architecture/libraries; files/data/interfaces; pseudocode/slices/tests.
- Implement and test one vertical slice at a time. Show its uncommitted Git diff and as-built explanation, then wait for user approval before committing.
- Keep README files navigational and status-oriented. Requirements and design belong in the living source; implementation detail belongs in the guide.
- Write-up runs alongside development, not after it.
  - Write each experiment's description before building it — they are a page each and they expose gaps while there is still time to act.
  - After each stage, paste the numbers into that experiment's Results section the same day. Results written up the day they are produced take minutes; a week later they take hours of re-deriving what a run's settings were.
- Avoid over-engineering. Only make changes that are directly requested or clearly
necessary. 
- Keep code simple and traceable: one clear responsibility per module, one complete operation per public function, and helpers only when they clarify a real step. Prefer explicit loops, branches and named values over clever expressions or unnecessary abstractions. Keep CLI code to parsing and delegation. Before presenting code, ensure the implementation guide, comments and tests follow the same main-function-first flow.
