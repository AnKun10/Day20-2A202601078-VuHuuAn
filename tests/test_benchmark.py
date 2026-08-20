"""Benchmark metric computations (offline)."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    aggregate_metrics,
    citation_coverage,
    run_benchmark,
    total_cost_usd,
)


def _state_with_answer(answer: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="Doc A", url="https://a.example", snippet="s"),
        SourceDocument(title="Doc B", url="https://b.example", snippet="s"),
    ]
    state.final_answer = answer
    return state


def test_citation_coverage_counts_index_title_and_url() -> None:
    state = _state_with_answer("Claim [1]. More text without the second source.")
    assert citation_coverage(state) == 0.5
    state = _state_with_answer("Doc A says X, see https://b.example too.")
    assert citation_coverage(state) == 1.0


def test_citation_coverage_none_without_sources() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.final_answer = "answer"
    assert citation_coverage(state) is None


def test_run_benchmark_measures_latency_and_survives_failures() -> None:
    ok_state = _state_with_answer("Claim [1] and [2].")
    state, metrics = run_benchmark("ok", "some query here", lambda _q: ok_state)
    assert state is ok_state
    assert metrics.failure_rate == 0.0
    assert metrics.citation_coverage == 1.0

    def boom(_query: str) -> ResearchState:
        raise RuntimeError("provider down")

    state, metrics = run_benchmark("fail", "some query here", boom)
    assert state is None
    assert metrics.failure_rate == 1.0
    assert "provider down" in metrics.notes


def test_aggregate_metrics_averages_and_counts_failures() -> None:
    _, ok = run_benchmark("run", "some query here", lambda _q: _state_with_answer("x [1] [2]"))
    _, fail = run_benchmark("run", "some query here", _raise)
    combined = aggregate_metrics("run (mean)", [ok, fail])
    assert combined.failure_rate == 0.5


def _raise(_query: str) -> ResearchState:
    raise RuntimeError("boom")


def test_total_cost_sums_agent_results() -> None:
    state = _state_with_answer("x")
    assert total_cost_usd(state) is None
