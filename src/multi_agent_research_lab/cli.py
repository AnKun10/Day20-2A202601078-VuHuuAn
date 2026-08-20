"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.agents.single import SingleAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    aggregate_metrics,
    make_llm_judge,
    run_benchmark,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import init_tracing, traced

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_DEFAULT_BENCHMARK_QUERIES = [
    "Research GraphRAG state-of-the-art and write a 500-word summary",
    "Compare LoRA and full fine-tuning for adapting LLMs to a new domain",
    "What are the main approaches to evaluating multi-agent LLM systems?",
]


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_single(query: str) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    return SingleAgent().run(state)


def _run_multi(query: str) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    return MultiAgentWorkflow().run(state)


def _print_run_summary(state: ResearchState) -> None:
    tokens = sum(
        (result.metadata.get("input_tokens") or 0) + (result.metadata.get("output_tokens") or 0)
        for result in state.agent_results
    )
    costs = [
        float(cost)
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    console.print(f"[dim]routes: {' -> '.join(state.route_history)}[/dim]")
    console.print(f"[dim]tokens: {tokens} | est. cost: ${sum(costs):.6f}[/dim]" if costs else "")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline: one LLM call answers everything."""

    _init()
    state, metrics = run_benchmark("baseline", query, traced("baseline")(_run_single))
    if state is None or state.final_answer is None:
        console.print(Panel.fit(metrics.notes, title="Baseline failed", style="red"))
        raise typer.Exit(code=1)
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    console.print(f"[dim]latency: {metrics.latency_seconds:.2f}s[/dim]")
    _print_run_summary(state)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer -> Critic."""

    _init()
    state, metrics = run_benchmark("multi-agent", query, traced("multi-agent")(_run_multi))
    if state is None or state.final_answer is None:
        console.print(Panel.fit(metrics.notes, title="Multi-agent failed", style="red"))
        raise typer.Exit(code=1)
    console.print(Panel.fit(state.final_answer, title="Multi-Agent Answer"))
    console.print(f"[dim]latency: {metrics.latency_seconds:.2f}s[/dim]")
    _print_run_summary(state)


@app.command()
def benchmark(
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Query to benchmark (repeatable)"),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Markdown report path")] = Path(
        "reports/benchmark_report.md"
    ),
    with_judge: Annotated[bool, typer.Option(help="Score quality with an LLM judge")] = True,
) -> None:
    """Benchmark single-agent vs multi-agent over a set of queries and write the report."""

    _init()
    queries = query or _DEFAULT_BENCHMARK_QUERIES
    judge = make_llm_judge() if with_judge else None

    per_query: list[BenchmarkMetrics] = []
    aggregates: list[BenchmarkMetrics] = []
    for run_name, runner in (("baseline", _run_single), ("multi-agent", _run_multi)):
        rows = []
        for item in queries:
            console.print(f"[cyan]{run_name}[/cyan] :: {item[:70]}")
            _, metrics = run_benchmark(run_name, item, traced(run_name)(runner), judge)
            rows.append(metrics)
            per_query.append(
                metrics.model_copy(update={"run_name": f"{run_name} :: {item[:40]}..."})
            )
        aggregates.append(aggregate_metrics(f"{run_name} (mean)", rows))

    report = render_markdown_report(aggregates, per_query=per_query)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(Panel.fit(f"Report written to {output}", title="Benchmark done"))

    table = Table(title="Aggregate results")
    for column in ("Run", "Latency (s)", "Cost (USD)", "Quality", "Citation", "Failure"):
        table.add_column(column)
    for m in aggregates:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            "" if m.estimated_cost_usd is None else f"{m.estimated_cost_usd:.5f}",
            "" if m.quality_score is None else f"{m.quality_score:.1f}",
            "" if m.citation_coverage is None else f"{m.citation_coverage:.0%}",
            "" if m.failure_rate is None else f"{m.failure_rate:.0%}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
