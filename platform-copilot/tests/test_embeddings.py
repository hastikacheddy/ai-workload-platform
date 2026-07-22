import math

from platform_copilot.services.embeddings.base import Embedder
from platform_copilot.services.embeddings.fake import HashEmbedder


def test_deterministic_and_unit_normalized() -> None:
    embedder = HashEmbedder(dim=64)
    first = embedder.embed(["drift alert psi"])[0]
    second = embedder.embed(["drift alert psi"])[0]

    assert first == second
    assert len(first) == 64
    assert math.isclose(sum(x * x for x in first), 1.0, abs_tol=1e-9)


def test_different_text_gives_different_vector() -> None:
    embedder = HashEmbedder(dim=64)
    assert embedder.embed(["drift alert"])[0] != embedder.embed(["deploy rollback"])[0]


def test_satisfies_embedder_protocol() -> None:
    embedder: Embedder = HashEmbedder()
    assert embedder.dim == 256
    assert len(embedder.embed(["one", "two"])) == 2
