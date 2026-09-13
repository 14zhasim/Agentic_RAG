# Prompts for Research Papers

## [Master Prompt]

### Agentic AI for Financial Statement Analysis (UCL MSc Dissertation)

#### What I'm doing

My UCL MSc Computer Science dissertation (supervisor: Prof. Philip Treleaven) builds and evaluates AI agents that automate financial statement analysis: extracting information from long, unstructured financial documents and, time permitting (lower priority), using it to populate/audit spreadsheet-based financial models.

**Skill level:**

I'm a hands-on novice at LLM/agent/RAG/pipeline/Python tooling — I've completed boot.dev's "Build an AI Agent" and "RAG" courses, but haven't yet built a production pipeline from scratch. Don't assume familiarity with agent frameworks, vector stores, or Python tooling conventions beyond what those courses cover; explain implementation choices rather than assuming I already know the trade-offs.

**Timeline:**

Code completion ~11 September; dissertation writing starts after; first draft submission 13 September 2026. Note I plan to continue coding etc. and iterating on report until final submission on 21st September. When helping me sequence work, prioritize accordingly — flag anything that may not fit before 30 Aug to be scoped down or cut, not just noted as a nice-to-have.

#### Problem + contribution (brief)

**Problem:** Real financial work is messy and long-horizon — documents are unstructured and inconsistent, spreadsheets have complex cross-references, and LLMs are unreliable at self-correcting without external feedback. Current benchmarks show large capability gaps, e.g. FAB v1.1.

**Contribution:**

- An agentic retrieval pipeline (RAG → Graph RAG) for extracting qualitative/numerical information from long financial documents (10-K/10-Q filings), evaluated against a direct-LLM-API baseline.
- I'm currently reconsidering Experiment 3 (single-operation Excel agent) — I want to replace/extend it with **more document/RAG-centric architectures** rather than pure spreadsheet manipulation, to keep the thesis's core contribution concentrated on the retrieval/reasoning side rather than generic Excel tooling. Flag relevant RAG/agent architectures from literature that could substitute here.

#### Current build-phase status

Need to setup benchmark for my testing, as per instructions in `docs sys design/Benchmark.md`.

I have done lit review and have all the contents needed to make sys design - just need to assemble it into proper design. I also haven't finalized the specific tool stack yet (e.g. LangChain vs. alternatives, Qdrant vs ElasticSearch vs. alternatives for the vector store, etc.) — treat tool-stack questions as open, not settled.

Plan for sys design:
**Start building system design – using RAG research doc, boot.dev notes, codebase_retrieval_parsing md (and maybeeee lit review stuff + claude chat to query all my notes on them). Then, identify failure points in architecture and if benchmark can help with this. References for these resources**:

- boot.dev notes: look at the md files /Users/zubairasim/rag-search-engine/course_notes
  `docs sys design/Benchmark.md`
- RAG research doc: /Users/zubairasim/Downloads/RAG\ research.docx
  Design\ Draft.md
- retrieval parsing: <../../../Library/CloudStorage/OneDrive-Personal/Documents/Masters/MSc Computer Science Project/Implementation Notes/Lit review/CODEBASE_RETRIEVAL_PARSING_ANALYSIS.md
- lit review stuff: <../../../Library/CloudStorage/OneDrive-Personal/Documents/Masters/MSc Computer Science Project/Implementation Notes/Lit review/CS_Research_Matrix.xlsx

#### Experiments — what I'm building, how I benchmark, and details

**Domain:** Corporate finance / equity research — SEC filings (10-K, 10-Q), financial statements, spreadsheets/financial models.
**Benchmarking** consistent across all 3 experiments, visit `docs sys design/Benchmark.md`

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

- **Input/Output/Benchmark/Metric:** Same as Exp 1 (FAB v1.1), same baseline.

Reference deck: "Zubair_Research_Planning_Condense_v2" (converted to Zubair_Research_Planning_Condense_v2.md) contains the full slide-by-slide detail — benchmark leaderboards, task taxonomies, and source quotations — behind everything above, including the appendix material on Experiments 4 and 5.

##### Experiment 3 — agentic tooling, verification, looping

## Understanding-first development workflow

- Treat the user as the learner-owner of the system, not merely an approver.
- Edit tracked user-authored requirements documents directly as living sources of truth, preserving their wording and logical flow; establish a clean or explicitly staged baseline first, then use Git diff/history for review and recovery instead of duplicate specifications.
- Use a separate implementation guide for technical detail. Apply three explicit gates: architecture/libraries; files/data/interfaces; pseudocode/slices/tests.
- Prefer the simplest architecture satisfying current requirements and identify deferred work explicitly.
- Implement and test one vertical slice at a time. Show its uncommitted Git diff and as-built explanation, then wait for user approval before committing.
- Explain libraries, file structure, data flow, persisted artifacts, public functions, and main-function-first helper call order at the user's current Python level.
- Keep README files navigational and status-oriented. Requirements and design belong in the living source; implementation detail belongs in the guide.
- Before coding, use the implementation guide to explain and approve the proposed solution from the top down: requirements traceability, system design and libraries, file/data/interface structure, then pseudocode, vertical slices, and tests.
- After each implemented slice, update that same guide with the exact files and important functions, actual call/data flow and artifacts, corresponding tests, material plan/code discrepancies and their rationale, updated diagrams where needed, and a main-function-first order for reading the guide alongside the code. Integrate those file/function/test pointers into the topical sections they explain and order the sections to match the code-reading path; if a file must be previewed early or revisited later, say so in that section rather than creating a separate lookup table. Do not leave superseded planned behavior presented as current implementation.
