"""Allowlisted request controls for logs; never serialize messages, headers or raw bodies."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_loop")
_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
_THINKING_TYPES = {"adaptive", "enabled", "disabled", "between_tools"}


def request_controls(kwargs: dict) -> tuple[int | None, str | None, str | None]:
    """Controls actually sent: SDK extra_body overrides top-level JSON fields."""
    extra = kwargs.get("extra_body")
    body = {**kwargs, **extra} if isinstance(extra, dict) else kwargs
    cap = next((body[k] for k in ("max_completion_tokens", "max_output_tokens", "max_tokens")
                if isinstance(body.get(k), int) and not isinstance(body[k], bool) and body[k] > 0), None)
    thinking = body.get("thinking")
    thinking_type = thinking.get("type") if isinstance(thinking, dict) else None
    if not isinstance(thinking_type, str) or thinking_type not in _THINKING_TYPES:
        thinking_type = None
    output = body.get("output_config")
    reasoning = body.get("reasoning")
    effort = (output.get("effort") if isinstance(output, dict) else None) or (
        reasoning.get("effort") if isinstance(reasoning, dict) else None) or body.get("reasoning_effort")
    if not isinstance(effort, str) or effort not in _EFFORTS:
        effort = None
    return cap, thinking_type, effort


def log_request_controls(agent: Any, kwargs: dict) -> None:
    """One lightweight INFO summary per physical request, after middleware resolution."""
    cap, thinking, effort = request_controls(kwargs)
    agent._last_requested_output_cap = cap
    agent._last_requested_reasoning_effort = effort
    logger.info(
        "Model request controls: api_mode=%r model=%r max_tokens=%s thinking=%s effort=%s",
        agent.api_mode, kwargs.get("model", agent.model), cap if cap is not None else "provider-default",
        thinking or "unspecified", effort or "unspecified",
    )
