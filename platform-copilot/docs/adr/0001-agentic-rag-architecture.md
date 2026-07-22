# ADR 0001 — Agentic RAG architecture for the Platform Copilot

- **Status:** Accepted
- **Date:** 2026-07-22
- **Deciders:** Platform team

## Context

We want a service that answers operational questions about the AI/ML platform, grounded in its
own documents (ADRs, runbooks, model cards, postmortems) with citations. It must be cheap to run
locally, honest about what it knows, and observable enough to reason about quality, latency, and
cost. The corpus is modest (hundreds to low-thousands of documents) but heterogeneous
(Markdown, HTML, PDF) and full of exact tokens that matter — error strings, service names, CVE
IDs, config keys.

## Decision

Build a **retrieval-first, agent-wrapped** RAG system:

1. **PostgreSQL is the source of truth** for document metadata; **OpenSearch is the index.**
   Documents are parsed once (Docling), stored in Postgres, then indexed into OpenSearch.
2. **Retrieval is hybrid** — BM25 (keyword) and dense kNN vectors, fused with **Reciprocal Rank
   Fusion** — served from a **single OpenSearch cluster** that does both.
3. **The LLM is local (Ollama)** behind a thin, provider-swappable interface.
4. **An agent (LangGraph) wraps retrieval** only where it earns its keep: a guardrail node, a
   relevance-grading node, and a query-rewrite/retry loop.
5. **Every request is traced and costed (Langfuse)** and **cacheable (Redis)**.

## Alternatives considered

### Retrieval: pure vector search vs. hybrid
Pure vector search loses on exact-token queries (`drift_detector`, a CVE ID, a config key) that
dominate ops questions. We start with a **BM25 baseline**, measure it, then add vectors and fuse
with RRF — adopting semantic search only where the eval shows it helps. This mirrors how mature
teams actually build search.

### Index: OpenSearch (BM25 + kNN) vs. pgvector vs. a dedicated vector DB
`pgvector` keeps everything in Postgres but has weaker lexical search and filtering ergonomics.
A dedicated vector DB (e.g., a hosted service) adds an operational dependency and cost for a
corpus this size. **OpenSearch does BM25 and kNN in one engine**, so hybrid retrieval and rich
filtering live in a single store we already understand.

### LLM: local Ollama vs. a hosted API
Local-first keeps the platform's internal docs on-prem, makes the demo zero-cost and reproducible,
and forces us to engineer prompt/latency budgets honestly. The LLM sits behind an interface, so
swapping in a hosted model later is a config change, not a rewrite.

### Orchestration: a plain single-shot RAG chain vs. an agent
A single retrieve-then-generate chain is simpler and is the right default. We add an agent **only**
for behaviors that a static chain can't express well: refusing out-of-scope questions (guardrail),
detecting low-relevance retrievals and **rewriting the query**, and retrying. The agent is scoped
to those nodes — not "agent for its own sake."

## Consequences

**Positive**
- Clear separation: Postgres owns truth, OpenSearch owns retrieval, the agent owns control flow.
- Retrieval quality is *measured* (eval harness) before and after each enhancement.
- Cost and latency are observable per request from day one.

**Negative / risks**
- Two stores to keep consistent (mitigated: OpenSearch is a derived index, rebuildable from Postgres).
- Running OpenSearch + Ollama locally needs ~8 GB RAM.
- The agent adds latency; it must be justified per-node by evals, and short-circuited on the happy path.

## Follow-ups

- ADR 0002 — chunking strategy and embedding model choice (M3)
- ADR 0003 — agent state machine and guardrail policy (M6)
- ADR 0004 — caching keys and invalidation (M5)
