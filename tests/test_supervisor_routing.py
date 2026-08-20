"""Unit tests for the supervisor routing policy (replaces the skeleton guard test)."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.state import ResearchState

_SOURCE = SourceDocument(title="Doc", url="https://example.com", snippet="snippet")


def _supervisor() -> SupervisorAgent:
    return SupervisorAgent(max_iterations=6)


def test_routes_to_researcher_when_no_sources(state: ResearchState) -> None:
    assert _supervisor().decide(state) == "researcher"


def test_routes_to_analyst_when_sources_but_no_analysis(state: ResearchState) -> None:
    state.sources = [_SOURCE]
    assert _supervisor().decide(state) == "analyst"


def test_routes_to_writer_when_analysis_ready(state: ResearchState) -> None:
    state.sources = [_SOURCE]
    state.analysis_notes = "analysis"
    assert _supervisor().decide(state) == "writer"


def test_routes_to_critic_once_after_final_answer(state: ResearchState) -> None:
    state.final_answer = "answer"
    assert _supervisor().decide(state) == "critic"
    state.route_history = ["researcher", "analyst", "writer", "critic"]
    assert _supervisor().decide(state) == "done"


def test_stops_at_max_iterations(state: ResearchState) -> None:
    supervisor = SupervisorAgent(max_iterations=3)
    state.iteration = 3
    assert supervisor.decide(state) == "done"


def test_fallback_skips_agent_after_repeated_failures(state: ResearchState) -> None:
    state.errors = ["researcher: boom", "researcher: boom again"]
    assert _supervisor().decide(state) == "analyst"


def test_run_records_route_and_increments_iteration(state: ResearchState) -> None:
    result = _supervisor().run(state)
    assert result.route_history == ["researcher"]
    assert result.iteration == 1
