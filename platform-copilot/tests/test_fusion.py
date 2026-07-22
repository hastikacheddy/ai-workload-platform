from platform_copilot.services.retrieval.fusion import reciprocal_rank_fusion


def test_shared_top_result_wins_and_all_ids_kept() -> None:
    keyword = ["x", "y", "z"]
    vector = ["x", "z", "w", "y"]
    fused = reciprocal_rank_fusion([keyword, vector])
    ids = [doc_id for doc_id, _ in fused]

    assert ids[0] == "x"  # ranked first by both arms
    assert ids.index("z") < ids.index("y")  # z ranks higher across the two lists
    assert "w" in ids  # appears in only one list, still retained


def test_scores_are_sorted_descending() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
