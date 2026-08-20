"""Single-agent baseline: one LLM call does research, analysis, and writing."""

from multi_agent_research_lab.agents.base import BaseAgent, record_llm_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's research question end-to-end: "
    "recall what you know, weigh the evidence, and write a clear, structured answer "
    "for the given audience. Be explicit about uncertainty and avoid making up sources."
)


class SingleAgent(BaseAgent):
    """Baseline agent that answers the query in a single completion (no tools)."""

    name = "single"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        llm = self._llm or LLMClient()
        with trace_span("single_agent", {"query": state.request.query}) as span:
            response = llm.complete(
                _SYSTEM_PROMPT,
                f"Audience: {state.request.audience}\n\nQuestion: {state.request.query}",
            )
        state.final_answer = response.content
        state.record_route(self.name)
        record_llm_result(state, AgentName.SINGLE, response, span["duration_seconds"])
        state.add_trace_event("single_agent.done", {"duration_seconds": span["duration_seconds"]})
        return state
