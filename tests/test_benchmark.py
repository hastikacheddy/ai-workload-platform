"""
Tests for the inference benchmark harness.

Uses the MockBackend so it runs anywhere (no GPU, no model server). Proves the
*measurement* is correct: percentiles are ordered, throughput scales with
concurrency, GPU sampling degrades honestly to N/A, and the compare() table
renders.
"""
import math

from benchmarks.harness import benchmark, compare, _percentile, GpuSampler
from benchmarks.backends import build_backend, MockBackend


def test_percentile_nearest_rank():
    vals = [float(i) for i in range(1, 101)]  # 1..100 sorted
    assert _percentile(vals, 50) == 50
    assert _percentile(vals, 99) == 99
    assert _percentile(vals, 100) == 100
    assert math.isnan(_percentile([], 50))


def test_benchmark_reports_ordered_percentiles():
    backend = MockBackend(base_latency_ms=20, max_concurrency=8)
    res = benchmark(backend, {"max_tokens": 16}, label="unit",
                    n_requests=60, concurrency=8, warmup=3)
    lat = res.latency_ms
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]
    assert res.errors == 0
    assert res.n_requests == 60
    assert res.throughput_rps > 0


def test_higher_concurrency_raises_throughput():
    # A batching backend (high concurrency ceiling) should serve more rps when
    # driven with more concurrency — the core inference-optimization signal.
    backend = MockBackend(base_latency_ms=40, max_concurrency=16)
    low = benchmark(backend, {}, label="c2", n_requests=40, concurrency=2, warmup=2)
    high = benchmark(backend, {}, label="c16", n_requests=40, concurrency=16, warmup=2)
    assert high.throughput_rps > low.throughput_rps


def test_gpu_sampler_honest_when_absent():
    s = GpuSampler()
    summ = s.summary()
    # On a CPU-only box this must be a clean N/A, never a fabricated number.
    assert "available" in summ
    if not summ["available"]:
        assert summ["util_mean_pct"] is None


def test_compare_table_and_delta():
    b = MockBackend(base_latency_ms=120, max_concurrency=2)
    a = MockBackend(base_latency_ms=40, max_concurrency=16)
    rb = benchmark(b, {}, label="Vanilla", n_requests=30, concurrency=8, warmup=2)
    ra = benchmark(a, {}, label="vLLM", n_requests=30, concurrency=8, warmup=2)
    table = compare([rb, ra])
    assert "p50 (ms)" in table and "Throughput (rps)" in table
    assert "Vanilla" in table and "vLLM" in table
    assert "higher throughput" in table  # delta line rendered


def test_build_backend_factory():
    assert isinstance(build_backend("mock"), MockBackend)
    oai = build_backend("vllm", base_url="http://x:8000/v1", model="m")
    assert oai.base_url == "http://x:8000/v1"
