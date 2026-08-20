"""Worker agents populate their slice of the shared state (offline fakes)."""

from conftest import FakeLLMClient

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, SingleAgent, WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


def test_researcher_populates_sources_and_notes(
    state: ResearchState, fake_llm: FakeLLMClient, mock_search: SearchClient
) -> None:
    result = ResearcherAgent(llm=fake_llm, search=mock_search).run(state)
    assert result.sources
    assert result.research_notes == "fake response"
    assert result.agent_results[-1].agent == "researcher"
    assert result.agent_results[-1].metadata["cost_usd"] is not None


def test_analyst_populates_analysis_notes(state: ResearchState, fake_llm: FakeLLMClient) -> None:
    state.research_notes = "notes"
    result = AnalystAgent(llm=fake_llm).run(state)
    assert result.analysis_notes == "fake response"


def test_writer_populates_final_answer(state: ResearchState) -> None:
    llm = FakeLLMClient(content="Answer with citation [1].")
    state.analysis_notes = "analysis"
    result = WriterAgent(llm=llm).run(state)
    assert result.final_answer == "Answer with citation [1]."


def test_single_agent_answers_in_one_step(state: ResearchState, fake_llm: FakeLLMClient) -> None:
    result = SingleAgent(llm=fake_llm).run(state)
    assert result.final_answer == "fake response"
    assert result.route_history == ["single"]
