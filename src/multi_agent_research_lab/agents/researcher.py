"""Researcher agent: collects sources and creates concise research notes."""

from multi_agent_research_lab.agents.base import BaseAgent, record_llm_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a research assistant specialised in gathering evidence. You are given search "
    "results for a query. Write concise research notes: summarise what each source "
    "contributes, keep the [n] source index next to every fact, and note gaps where the "
    "sources do not answer the question. Do NOT analyse or conclude — only collect."
)


def format_sources(state: ResearchState) -> str:
    """Render `state.sources` as a numbered list usable inside prompts and citations."""

    lines = []
    for index, source in enumerate(state.sources, start=1):
        url = f" ({source.url})" if source.url else ""
        lines.append(f"[{index}] {source.title}{url}\n    {source.snippet}")
    return "\n".join(lines)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self._llm = llm
        self._search = search

    def run(self, state: ResearchState) -> ResearchState:
        llm = self._llm or LLMClient()
        search = self._search or SearchClient()
        with trace_span("researcher", {"query": state.request.query}) as span:
            state.sources = search.search(
                state.request.query, max_results=state.request.max_sources
            )
            response = llm.complete(
                _SYSTEM_PROMPT,
                f"Query: {state.request.query}\n\nSearch results:\n{format_sources(state)}",
            )
        state.research_notes = response.content
        record_llm_result(state, AgentName.RESEARCHER, response, span["duration_seconds"])
        state.add_trace_event(
            "researcher.done",
            {"sources": len(state.sources), "duration_seconds": span["duration_seconds"]},
        )
        return state
