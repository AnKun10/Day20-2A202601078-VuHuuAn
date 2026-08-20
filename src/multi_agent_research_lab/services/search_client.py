"""Search client abstraction for ResearcherAgent.

Uses Tavily when TAVILY_API_KEY is configured, otherwise falls back to a local mock so the
workflow stays runnable (and testable) offline.
"""

import logging

import httpx

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

_MOCK_SOURCES = [
    SourceDocument(
        title="Mock source: survey article",
        url="https://example.com/survey",
        snippet="A mock survey snippet covering the queried topic at a high level.",
        metadata={"provider": "mock"},
    ),
    SourceDocument(
        title="Mock source: benchmark blog post",
        url="https://example.com/benchmark",
        snippet="A mock blog post comparing approaches with rough benchmark numbers.",
        metadata={"provider": "mock"},
    ),
    SourceDocument(
        title="Mock source: official documentation",
        url="https://example.com/docs",
        snippet="Mock official documentation describing the core concepts and API.",
        metadata={"provider": "mock"},
    ),
]


class SearchClient:
    """Provider-agnostic search client (Tavily or offline mock)."""

    def __init__(self, api_key: str | None = None, use_mock: bool | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.tavily_api_key
        self._timeout = settings.timeout_seconds
        self._use_mock = use_mock if use_mock is not None else not self._api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if self._use_mock:
            logger.info("search.mock query=%r", query)
            return _MOCK_SOURCES[:max_results]

        response = httpx.post(
            _TAVILY_ENDPOINT,
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        documents = [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=item.get("content") or "",
                metadata={"provider": "tavily", "score": item.get("score")},
            )
            for item in payload.get("results", [])
        ]
        logger.info("search.tavily query=%r results=%d", query, len(documents))
        return documents[:max_results]
