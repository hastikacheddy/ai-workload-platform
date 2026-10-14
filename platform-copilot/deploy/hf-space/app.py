"""Platform Copilot — self-contained Hugging Face Space demo.

Runs the real ``platform_copilot`` retrieval + RAG pipeline, but wired for a single
CPU container: an in-memory index (seeded from ``corpus/`` at startup) instead of
OpenSearch, and a small local model via Transformers instead of Ollama. No external
services, no API key. The production service uses OpenSearch + Ollama and a LangGraph
agent — see github.com/hastikacheddy/ai-workload-platform.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from platform_copilot.services.chunking import chunk_document
from platform_copilot.services.embeddings.sentence_transformer import SentenceTransformerEmbedder
from platform_copilot.services.ingestion.markdown import parse_markdown
from platform_copilot.services.rag.pipeline import RagPipeline
from platform_copilot.services.rag.render import render_answer
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever

CORPUS = Path(__file__).parent / "corpus"
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_pipeline: RagPipeline | None = None


def _build_pipeline() -> RagPipeline:
    from platform_copilot.services.llm.transformers_llm import TransformersLLM

    embedder = SentenceTransformerEmbedder()
    chunks = []
    for path in sorted(CORPUS.glob("*.md")):
        document = parse_markdown(
            path.read_text(encoding="utf-8"), source_type="doc", source_ref=path.name
        )
        chunks.extend(chunk_document(document))
    retriever = HybridRetriever(InMemoryBackend(chunks, embedder), embedder)
    return RagPipeline(retriever, TransformersLLM(MODEL))


def pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        print(f"Loading corpus + model ({MODEL})...")
        _pipeline = _build_pipeline()
        print("Ready.")
    return _pipeline


def respond(message: str, history: list) -> str:
    if not message.strip():
        return "Ask a question about the platform's runbooks, ADRs, or incidents."
    try:
        return render_answer(pipeline().answer(message, k=4))
    except Exception as exc:  # surface errors in the UI rather than failing silently
        return f"⚠️ Error: {type(exc).__name__}: {exc}"


demo = gr.ChatInterface(
    fn=respond,
    title="Platform Copilot",
    description=(
        "Agentic RAG over an AI/ML platform's runbooks, ADRs and incident postmortems — "
        "**grounded answers with citations**. This is a self-contained demo (in-memory "
        "hybrid retrieval + a small CPU model), so the first reply is slow while the model "
        "loads and answers are terse. The production service uses OpenSearch + Ollama and a "
        "LangGraph agent. Source + full README: "
        "[github.com/hastikacheddy/ai-workload-platform]"
        "(https://github.com/hastikacheddy/ai-workload-platform/tree/main/platform-copilot)."
    ),
    examples=[
        "What are the first steps for a demand-forecaster drift alert?",
        "Why does the model promotion gate keep rejecting candidates?",
        "What caused the serving latency spike and how was it fixed?",
    ],
)

# Warm the model at startup so the first visitor doesn't wait for a cold load,
# unless told to skip (used by local UI checks that don't need the model).
if not os.environ.get("SKIP_MODEL_LOAD"):
    pipeline()

if __name__ == "__main__":
    demo.launch()
