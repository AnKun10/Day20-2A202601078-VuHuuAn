"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _table(metrics: list[BenchmarkMetrics]) -> list[str]:
    lines = [
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    return lines


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    per_query: list[BenchmarkMetrics] | None = None,
    analysis: str = "",
) -> str:
    """Render benchmark metrics to markdown (aggregate table, per-query detail, analysis)."""

    lines = ["# Benchmark Report", "", "## Aggregate", ""]
    lines += _table(metrics)
    if per_query:
        lines += ["", "## Per-query detail", ""]
        lines += _table(per_query)
    if analysis:
        lines += ["", "## Analysis", "", analysis]
    lines += [
        "",
        "## How metrics are computed",
        "",
        "- **Latency**: wall-clock seconds per query (`time.perf_counter`).",
        "- **Cost**: estimated from token usage recorded by `LLMClient` per agent step.",
        "- **Quality**: LLM-as-judge rubric 0-10 (relevance, grounding, clarity).",
        "- **Citation coverage**: sources referenced in the final answer / total sources.",
        "- **Failure rate**: failed runs / total runs (exceptions caught by the benchmark).",
    ]
    return "\n".join(lines) + "\n"
