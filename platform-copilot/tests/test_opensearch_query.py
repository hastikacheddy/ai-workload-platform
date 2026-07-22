from platform_copilot.services.opensearch.index import index_settings
from platform_copilot.services.opensearch.query import bm25_query, knn_query


def test_bm25_query_structure_and_filters() -> None:
    query = bm25_query("drift alert", filters={"source_type": "runbook"}, size=5)
    assert query["size"] == 5
    match = query["query"]["bool"]["must"][0]["multi_match"]
    assert match["query"] == "drift alert"
    assert {"term": {"source_type": "runbook"}} in query["query"]["bool"]["filter"]


def test_knn_query_carries_vector_and_k() -> None:
    query = knn_query([0.1, 0.2, 0.3], size=3)
    assert query["query"]["knn"]["embedding"]["k"] == 3
    assert query["query"]["knn"]["embedding"]["vector"] == [0.1, 0.2, 0.3]


def test_index_mapping_defines_both_retrieval_arms() -> None:
    mapping = index_settings(embedding_dim=768)
    props = mapping["mappings"]["properties"]
    assert props["text"]["type"] == "text"
    assert props["embedding"]["type"] == "knn_vector"
    assert props["embedding"]["dimension"] == 768
    assert mapping["settings"]["index"]["knn"] is True
