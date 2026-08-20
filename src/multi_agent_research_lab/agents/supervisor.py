"""Supervisor / router: rule-based routing over the shared state.

Rule-based (instead of LLM-based) routing keeps the control loop deterministic, cheap,
and unit-testable — the state itself tells us which stage is missing.
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

# After this many recorded failures, an agent is skipped instead of retried (fallback).
_MAX_AGENT_FAILURES = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        self._max_iterations = (
            max_iterations if max_iterations is not None else get_settings().max_iterations
        )

    def run(self, state: ResearchState) -> ResearchState:
        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event("supervisor.route", {"route": route, "iteration": state.iteration})
        return state

    def decide(self, state: ResearchState) -> str:
        """Pick the next route by inspecting which state fields are still missing."""

        # Guardrail: never loop past the iteration budget.
        if state.iteration >= self._max_iterations:
            return ROUTE_DONE

        if state.final_answer:
            # Bonus quality gate: run the critic exactly once, then stop.
            if ROUTE_CRITIC not in state.route_history and not self._failed_out(state, "critic"):
                return ROUTE_CRITIC
            return ROUTE_DONE

        if not state.sources and not self._failed_out(state, ROUTE_RESEARCHER):
            return ROUTE_RESEARCHER
        if not state.analysis_notes and not self._failed_out(state, ROUTE_ANALYST):
            return ROUTE_ANALYST
        if self._failed_out(state, ROUTE_WRITER):
            # All fallbacks exhausted: stop instead of burning iterations.
            return ROUTE_DONE
        return ROUTE_WRITER

    @staticmethod
    def _failed_out(state: ResearchState, agent_name: str) -> bool:
        """True when an agent already failed enough times to be skipped (fallback)."""

        failures = sum(1 for error in state.errors if error.startswith(f"{agent_name}:"))
        return failures >= _MAX_AGENT_FAILURES
