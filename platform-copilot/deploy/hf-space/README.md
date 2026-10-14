---
title: Platform Copilot
emoji: 🛠️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Agentic RAG over platform runbooks & ADRs, with citations
---

# Platform Copilot — demo

A self-contained demo of an **agentic RAG** service that answers operational questions
about an AI/ML platform — grounded in its runbooks, ADRs and incident postmortems, with
citations.

This Space runs the real [`platform_copilot`](https://github.com/hastikacheddy/ai-workload-platform/tree/main/platform-copilot)
package, wired for a single CPU container: in-memory hybrid retrieval (keyword +
`all-MiniLM-L6-v2` vectors, fused with Reciprocal Rank Fusion) and a small local model
(`Qwen/Qwen2.5-0.5B-Instruct`). No external services, no API key.

**Heads up:** the first reply is slow while the model loads, and a 0.5B CPU model gives
terse answers. The production service uses OpenSearch + Ollama and a LangGraph agent
(guardrail → grade → rewrite → retry) — see the repo for the full write-up, the 40-test
suite, and screenshots of the real deployment.

**Source:** [github.com/hastikacheddy/ai-workload-platform › platform-copilot](https://github.com/hastikacheddy/ai-workload-platform/tree/main/platform-copilot)
