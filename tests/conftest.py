"""Shared offline fakes so tests never hit a network."""

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class FakeLLMClient(LLMClient):
    """Returns canned content; never touches the OpenAI SDK."""

    def __init__(self, content: str = "fake response") -> None:
        self.model = "fake-model"
        self.content = content

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            content=self.content, input_tokens=100, output_tokens=50, cost_usd=0.0001
        )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def mock_search() -> SearchClient:
    return SearchClient(api_key=None, use_mock=True)


@pytest.fixture
def state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
