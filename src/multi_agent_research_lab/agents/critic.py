"""Critic agent (bonus): validates the final answer before the workflow stops."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import citation_coverage
from multi_agent_research_lab.observability.tracing import trace_span


class CriticAgent(BaseAgent):
    """Checks citation coverage of the final answer and records findings."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic", {"query": state.request.query}) as span:
            coverage = citation_coverage(state)
            uncited = [
                f"[{index}] {source.title}"
                for index, source in enumerate(state.sources, start=1)
                if state.final_answer and f"[{index}]" not in state.final_answer
            ]
            findings = [
                f"citation_coverage={coverage:.0%}" if coverage is not None else "no sources"
            ]
            if uncited:
                findings.append("uncited sources: " + "; ".join(uncited))
            if not state.final_answer:
                findings.append("final answer missing")
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content="\n".join(findings),
                metadata={
                    "citation_coverage": coverage,
                    "duration_seconds": span["duration_seconds"],
                },
            )
        )
        state.add_trace_event("critic.done", {"citation_coverage": coverage})
        return state
