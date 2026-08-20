"""Analyst agent: turns research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent, record_llm_result
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a critical analyst. Given research notes and their sources, produce a "
    "structured analysis: (1) key claims with the [n] source index that supports each, "
    "(2) points where sources agree or disagree, (3) claims backed by weak or single "
    "evidence, flagged explicitly, (4) open questions. Do not write the final answer."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        llm = self._llm or LLMClient()
        with trace_span("analyst", {"query": state.request.query}) as span:
            response = llm.complete(
                _SYSTEM_PROMPT,
                (
                    f"Query: {state.request.query}\n\n"
                    f"Sources:\n{format_sources(state)}\n\n"
                    f"Research notes:\n{state.research_notes or '(none)'}"
                ),
            )
        state.analysis_notes = response.content
        record_llm_result(state, AgentName.ANALYST, response, span["duration_seconds"])
        state.add_trace_event("analyst.done", {"duration_seconds": span["duration_seconds"]})
        return state
