from platform_copilot.services.eval.metrics import (
    EvalCase,
    evaluate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_precision() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3  # found b, d out of 3 relevant
    assert precision_at_k(ranked, relevant, k=2) == 0.5  # a, b -> 1 hit of 2


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_evaluate_aggregates_over_cases() -> None:
    cases = [EvalCase("q1", {"a"}), EvalCase("q2", {"x"})]
    results = {"q1": ["a", "b"], "q2": ["y", "x"]}
    out = evaluate(results, cases, k=2)
    assert out["recall_at_k"] == 1.0  # both relevant docs within top-2
    assert out["mrr"] == (1.0 + 0.5) / 2
