"""End-to-end workflow test with offline fakes: the whole graph must converge."""

from conftest import FakeLLMClient

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_workflow_runs_end_to_end_with_fakes() -> None:
    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(max_iterations=6),
        researcher=ResearcherAgent(
            llm=FakeLLMClient("research notes"), search=SearchClient(api_key=None, use_mock=True)
        ),
        analyst=AnalystAgent(llm=FakeLLMClient("analysis notes")),
        writer=WriterAgent(llm=FakeLLMClient("Final answer citing [1] and [2].")),
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    assert result.final_answer == "Final answer citing [1] and [2]."
    assert result.route_history == ["researcher", "analyst", "writer", "critic", "done"]
    assert result.sources
    assert result.research_notes == "research notes"
    assert result.analysis_notes == "analysis notes"
    # Critic recorded its citation check.
    critic_results = [r for r in result.agent_results if r.agent == "critic"]
    assert critic_results and critic_results[0].metadata["citation_coverage"] is not None


def test_workflow_stops_on_iteration_budget_even_if_workers_fail() -> None:
    class ExplodingSearch(SearchClient):
        def __init__(self) -> None:
            self._use_mock = False

        def search(self, query: str, max_results: int = 5):  # type: ignore[override]
            raise RuntimeError("search down")

    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(max_iterations=4),
        researcher=ResearcherAgent(llm=FakeLLMClient(), search=ExplodingSearch()),
        analyst=AnalystAgent(llm=FakeLLMClient("analysis")),
        writer=WriterAgent(llm=FakeLLMClient("answer")),
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    # Researcher failed twice, supervisor fell back to analyst -> writer instead of looping.
    assert result.errors
    # +1 because recording the final "done" route also increments the counter.
    assert result.iteration <= 5
    assert "done" in result.route_history or result.final_answer is not None
