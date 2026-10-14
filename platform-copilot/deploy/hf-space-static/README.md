---
title: Platform Copilot
emoji: 🛩️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Grounded, cited copilot over platform ops docs
---

# Platform Copilot — showcase

A static showcase of **Platform Copilot**, an agentic-RAG assistant that answers
operational questions about an AI/ML platform, grounded in its own runbooks, ADRs, and
design docs, with citations.

The full interactive service (hybrid OpenSearch retrieval + a LangGraph agent + a served
LLM) lives in the repository. Hosting it live as a Gradio Space needs an HF PRO account, so
this free static page shows **illustrative example answers** — grounded in the document
corpus and cited to it — rather than live inference.

The interactive Gradio build is in the repo at `platform-copilot/deploy/hf-space/`
(self-contained; `python deploy.py` publishes it once PRO is enabled).

**Code & corpus:** https://github.com/hastikacheddy/ai-workload-platform/tree/main/platform-copilot
