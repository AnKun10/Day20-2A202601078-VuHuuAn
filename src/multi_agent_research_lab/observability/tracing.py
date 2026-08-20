"""Tracing hooks.

LangSmith is the primary provider: `init_tracing()` exports the settings the LangSmith SDK
reads from the environment, `traced()` wraps callables as LangSmith runs, and `LLMClient`
wraps the OpenAI SDK so completions appear as nested LLM runs. `trace_span` stays as a
lightweight local fallback that also feeds `ResearchState.trace`.
"""

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Literal, TypeVar, cast

from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def init_tracing() -> None:
    """Enable LangSmith tracing when an API key is configured.

    pydantic-settings reads `.env` for our own config, but the LangSmith SDK reads
    `os.environ` directly — so we export the relevant values here once at startup.
    """

    load_dotenv()
    settings = get_settings()
    if not settings.langsmith_api_key:
        logger.info("tracing disabled: no LANGSMITH_API_KEY configured")
        return
    # Assign explicitly (not setdefault): a stale or empty value loaded from `.env`
    # would otherwise win and the SDK would drop traces with a silent 401.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info("tracing enabled: langsmith project=%s", settings.langsmith_project)


RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def traced(name: str, run_type: RunType = "chain") -> Callable[[F], F]:
    """Decorate a callable as a LangSmith run; no-op when langsmith is unavailable."""

    def decorator(fn: F) -> F:
        try:
            from langsmith import traceable
        except ImportError:  # pragma: no cover - langsmith is an optional extra
            return fn
        return cast(F, traceable(run_type, name=name)(fn))

    return decorator


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal local span used to time steps and enrich `ResearchState.trace`."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("span %s took %.3fs", name, span["duration_seconds"])
