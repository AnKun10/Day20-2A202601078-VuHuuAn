"""Base agent contract and shared helpers."""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""


def record_llm_result(
    state: ResearchState,
    agent: AgentName,
    response: LLMResponse,
    duration_seconds: float | None = None,
) -> None:
    """Append an AgentResult carrying token/cost metadata so benchmarks can aggregate it."""

    state.agent_results.append(
        AgentResult(
            agent=agent,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "duration_seconds": duration_seconds,
            },
        )
    )
