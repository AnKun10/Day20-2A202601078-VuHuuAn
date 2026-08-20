"""Benchmark for single-agent vs multi-agent runs.

Measures latency, estimated token cost, citation coverage, LLM-judged quality, and
failure rate over one query or a suite of queries.
"""

import logging
import re
from collections.abc import Callable
from statistics import mean
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]
QualityJudge = Callable[[ResearchState], float | None]

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Score the answer to the research question on a 0-10 "
    "rubric: relevance and completeness (0-4), factual grounding and citations (0-3), "
    "structure and clarity (0-3). Respond with ONLY the numeric score."
)


def total_cost_usd(state: ResearchState) -> float | None:
    """Sum estimated cost over all recorded agent results (None when nothing was recorded)."""

    costs = [
        float(cost)
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    return sum(costs) if costs else None


def citation_coverage(state: ResearchState) -> float | None:
    """Fraction of sources actually referenced in the final answer.

    A source counts as cited when its [n] index, title, or URL appears in the answer.
    Returns None when there are no sources to cite (e.g. the single-agent baseline).
    """

    if not state.sources or not state.final_answer:
        return None
    answer = state.final_answer
    cited = 0
    for index, source in enumerate(state.sources, start=1):
        if (
            f"[{index}]" in answer
            or source.title in answer
            or (source.url is not None and source.url in answer)
        ):
            cited += 1
    return cited / len(state.sources)


def make_llm_judge() -> QualityJudge:
    """Build an LLM-as-judge quality scorer (imported lazily to keep tests offline)."""

    from multi_agent_research_lab.services.llm_client import LLMClient

    llm = LLMClient()

    def judge(state: ResearchState) -> float | None:
        if not state.final_answer:
            return None
        response = llm.complete(
            _JUDGE_SYSTEM_PROMPT,
            f"Question: {state.request.query}\n\nAnswer:\n{state.final_answer}",
        )
        match = re.search(r"\d+(?:\.\d+)?", response.content)
        if match is None:
            logger.warning("judge returned no numeric score: %r", response.content)
            return None
        return min(10.0, max(0.0, float(match.group())))

    return judge


def run_benchmark(
    run_name: str, query: str, runner: Runner, judge: QualityJudge | None = None
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run one query through a runner and measure it; failures become metrics, not crashes."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a benchmark must survive runner failures
        latency = perf_counter() - started
        logger.exception("benchmark run %s failed", run_name)
        return None, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"failed: {exc}",
        )
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost_usd(state),
        quality_score=judge(state) if judge else None,
        citation_coverage=citation_coverage(state),
        failure_rate=0.0,
        notes=f"routes: {' -> '.join(state.route_history) or 'n/a'}",
    )
    return state, metrics


def aggregate_metrics(run_name: str, per_query: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Average per-query metrics into one row (failure rate = failed runs / total runs)."""

    def avg(values: list[float]) -> float | None:
        return mean(values) if values else None

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=mean(m.latency_seconds for m in per_query),
        estimated_cost_usd=avg(
            [m.estimated_cost_usd for m in per_query if m.estimated_cost_usd is not None]
        ),
        quality_score=avg([m.quality_score for m in per_query if m.quality_score is not None]),
        citation_coverage=avg(
            [m.citation_coverage for m in per_query if m.citation_coverage is not None]
        ),
        failure_rate=sum(1 for m in per_query if m.failure_rate == 1.0) / len(per_query),
        notes=f"mean over {len(per_query)} queries",
    )
