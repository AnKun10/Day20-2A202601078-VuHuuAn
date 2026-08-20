"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
Retry, timeout, token accounting, and cost estimation all live here so agents stay small.
"""

import logging
from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# USD per 1M tokens: (input, output). Used for rough cost estimation in benchmarks.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}

_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def estimate_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimate USD cost from token counts, or None when the model is unknown."""

    pricing = MODEL_PRICING_PER_1M.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI chat completions."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.timeout_seconds)
        if settings.langsmith_api_key:
            # Wrap the SDK client so every completion shows up as an LLM run in LangSmith.
            try:
                from langsmith.wrappers import wrap_openai

                client = wrap_openai(client)
            except ImportError:  # pragma: no cover - langsmith is an optional extra
                logger.warning("langsmith not installed; LLM calls will not be traced")
        self._client = client

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage and estimated cost."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = estimate_cost_usd(self.model, input_tokens, output_tokens)
        logger.info(
            "llm.complete model=%s input_tokens=%s output_tokens=%s cost_usd=%s",
            self.model,
            input_tokens,
            output_tokens,
            cost,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
