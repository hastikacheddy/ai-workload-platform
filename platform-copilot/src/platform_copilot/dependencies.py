"""FastAPI dependency providers.

Heavy clients (OpenSearch, Ollama, Jina) are imported lazily inside the providers so
importing the app stays light and the offline test suite never needs them. Tests
override these providers with in-memory fakes.
"""

from __future__ import annotations

from functools import lru_cache

from platform_copilot.config import get_settings
from platform_copilot.services.embeddings.base import Embedder
from platform_copilot.services.observability.base import TraceSink
from platform_copilot.services.rag.pipeline import AnswerEngine, RagPipeline
from platform_copilot.services.retrieval.retriever import HybridRetriever


def _build_embedder() -> Embedder:
    settings = get_settings()
    if settings.jina_api_key:
        from platform_copilot.services.embeddings.jina import JinaEmbedder

        return JinaEmbedder(settings.jina_api_key)
    # Offline / no key: deterministic placeholder so the app still runs end to end.
    from platform_copilot.services.embeddings.fake import HashEmbedder

    return HashEmbedder()


@lru_cache
def get_retriever() -> HybridRetriever:
    from opensearchpy import OpenSearch

    from platform_copilot.services.opensearch.backend import OpenSearchBackend

    settings = get_settings()
    client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}]
    )
    backend = OpenSearchBackend(client, settings.opensearch_index)
    return HybridRetriever(backend, _build_embedder())


def _build_trace_sink() -> TraceSink:
    settings = get_settings()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from platform_copilot.services.observability.langfuse_sink import LangfuseSink

        return LangfuseSink(
            settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host
        )
    from platform_copilot.services.observability.memory import NullSink

    return NullSink()


@lru_cache
def get_pipeline() -> AnswerEngine:
    from platform_copilot.services.cache.redis_cache import RedisCache
    from platform_copilot.services.llm.ollama import OllamaLLM
    from platform_copilot.services.rag.cached import CachedPipeline
    from platform_copilot.services.rag.observed import ObservedPipeline

    settings = get_settings()
    base = RagPipeline(
        get_retriever(), OllamaLLM(settings.ollama_base_url, settings.ollama_model)
    )
    cached = CachedPipeline(base, RedisCache(settings.redis_url))
    return ObservedPipeline(cached, _build_trace_sink())
