"""LangGraph workflow: supervisor loop with conditional routing to worker agents.

Keep orchestration here; keep agent internals in `agents/`. The graph shape is:

    supervisor --(route)--> researcher | analyst | writer | critic --> supervisor
    supervisor --(done)---> END
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import traced

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._workers: dict[str, BaseAgent] = {
            ROUTE_RESEARCHER: researcher or ResearcherAgent(),
            ROUTE_ANALYST: analyst or AnalystAgent(),
            ROUTE_WRITER: writer or WriterAgent(),
            ROUTE_CRITIC: critic or CriticAgent(),
        }

    def build(self) -> "StateGraph[ResearchState, None, ResearchState, ResearchState]":
        """Create the LangGraph graph with conditional routing and a stop condition."""

        graph: StateGraph[ResearchState, None, ResearchState, ResearchState] = StateGraph(
            ResearchState
        )
        graph.add_node("supervisor", traced("supervisor")(self._supervisor.run))
        for route, agent in self._workers.items():
            # type ignore: langgraph's add_node overloads reject plain callables under strict mypy
            graph.add_node(route, self._safe_node(route, agent))  # type: ignore[arg-type]
            graph.add_edge(route, "supervisor")
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _last_route,
            {
                ROUTE_RESEARCHER: ROUTE_RESEARCHER,
                ROUTE_ANALYST: ROUTE_ANALYST,
                ROUTE_WRITER: ROUTE_WRITER,
                ROUTE_CRITIC: ROUTE_CRITIC,
                ROUTE_DONE: END,
            },
        )
        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and convert the result back to ResearchState."""

        settings = get_settings()
        compiled = self.build().compile()
        # Each loop is supervisor + worker (2 steps); leave headroom for entry/exit.
        recursion_limit = settings.max_iterations * 2 + 4
        result = compiled.invoke(state, config={"recursion_limit": recursion_limit})
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(dict(result))

    @staticmethod
    def _safe_node(route: str, agent: BaseAgent) -> Callable[[ResearchState], ResearchState]:
        """Wrap a worker so failures land in `state.errors` instead of killing the run.

        The supervisor reads those errors to decide between retry and fallback.
        """

        def node(state: ResearchState) -> ResearchState:
            try:
                return agent.run(state)
            except Exception as exc:  # noqa: BLE001 - route failures back to the supervisor
                logger.exception("agent %s failed", route)
                state.errors.append(f"{route}: {exc}")
                state.add_trace_event(f"{route}.error", {"error": str(exc)})
                return state

        return traced(route)(node)


def _last_route(state: ResearchState | Mapping[str, Any]) -> str:
    """Conditional-edge selector: follow the route the supervisor just recorded."""

    history = (
        state.route_history if isinstance(state, ResearchState) else state.get("route_history", [])
    )
    return history[-1] if history else ROUTE_DONE
