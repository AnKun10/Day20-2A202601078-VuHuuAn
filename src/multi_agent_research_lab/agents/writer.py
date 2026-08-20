"""Writer agent: produces the final answer from research and analysis notes."""

from multi_agent_research_lab.agents.base import BaseAgent, record_llm_result
from multi_agent_research_lab.agents.researcher import format_sources
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Using the analysis and sources provided, write a clear, "
    "well-structured answer for the given audience. Every substantive claim must carry an "
    "inline citation [n] pointing at the numbered source list. End with a 'Sources' section "
    "listing the numbered sources you actually cited. Do not invent sources; if evidence is "
    "weak, say so."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        llm = self._llm or LLMClient()
        with trace_span("writer", {"query": state.request.query}) as span:
            response = llm.complete(
                _SYSTEM_PROMPT,
                (
                    f"Audience: {state.request.audience}\n"
                    f"Query: {state.request.query}\n\n"
                    f"Sources:\n{format_sources(state)}\n\n"
                    f"Analysis:\n{state.analysis_notes or '(none)'}\n\n"
                    f"Research notes:\n{state.research_notes or '(none)'}"
                ),
            )
        state.final_answer = response.content
        record_llm_result(state, AgentName.WRITER, response, span["duration_seconds"])
        state.add_trace_event("writer.done", {"duration_seconds": span["duration_seconds"]})
        return state
