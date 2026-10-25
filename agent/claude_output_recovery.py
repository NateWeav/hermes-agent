"""One changed-effort retry for mandatory-thinking Claude, shared by empty and length replies."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

NO_VISIBLE_ANSWER = (
    "⚠️ **No visible answer was produced.** Mandatory-thinking Claude returned no visible "
    "answer or tool call. Recovery stopped instead of repeating the same output budget. "
    "Thinking cannot be disabled on this model. Try a different model, shorten the request, "
    "or review its output-token limit and reasoning effort before retrying."
)


def mandatory_claude(agent: Any) -> bool:
    """Use the adapter's capability policy on native and OpenAI-compatible Claude routes."""
    from agent.anthropic_adapter import (
        _MANDATORY_THINKING_CLAUDE_SUBSTRINGS, _model_matches, _supports_adaptive_thinking,
    )

    model = getattr(agent, "model", "") or ""
    return getattr(agent, "api_mode", None) in {"chat_completions", "anthropic_messages"} and (
        _supports_adaptive_thinking(model) and _model_matches(model, _MANDATORY_THINKING_CLAUDE_SUBSTRINGS)
    )


def recovery_route(agent: Any) -> tuple:
    return (agent.api_mode, getattr(agent, "provider", None), getattr(agent, "base_url", None), agent.model)


def recover_no_visible_claude(agent: Any, response: Any, finish_reason: str, requested_cap=None) -> bool | None:
    """None: unrelated route. True: one low-effort retry armed. False: stop honestly.

    The budget spans both recovery ladders and is per route per turn, not per response.
    An already-low or effort-rejecting route has no supported change to try.
    """
    if not mandatory_claude(agent):
        return None
    route = recovery_route(agent)
    requested_cap = getattr(agent, "_last_requested_output_cap", None) or requested_cap
    attempts = getattr(agent, "_claude_output_recovery_attempts", {})
    sent = getattr(agent, "_wire_reasoning_config", None) or agent.reasoning_config or {}
    effort = str(getattr(agent, "_last_requested_reasoning_effort", None) or sent.get("effort", "default")).lower()
    retry = not (attempts.get(route) or effort in {"low", "minimal"} or
                 getattr(agent, "_reasoning_effort_rejected", False))
    agent._ephemeral_reasoning_off = False
    if retry:
        attempts[route] = 1
        agent._claude_output_recovery_attempts = attempts
        # Keep the failed cap, never multiply a mandatory-thinking stall's token spend.
        agent._claude_output_recovery_pending = (route, requested_cap or agent.max_tokens)
    usage = getattr(response, "usage", None)
    from agent.usage_pricing import normalize_usage

    output = normalize_usage(usage, provider=getattr(agent, "provider", None), api_mode=agent.api_mode).output_tokens if usage else 0
    duration = getattr(agent, "_last_model_request_duration", None)
    elapsed = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "unknown"
    action = "recovery 1/1: effort=low" if retry else "stopping: no supported recovery left"
    cap = requested_cap or agent.max_tokens or "provider default"
    notice = (f"↻ {agent.model}: no visible answer ({finish_reason}); {action}; "
              f"max_tokens={cap}, output_tokens={output or 'unknown'}, request={elapsed}")
    logger.warning("%s", notice)
    agent._emit_diagnostic_wait(notice)
    if not getattr(agent, "thinking_callback", None):
        # Gateway-backed ACP observes status.update, not the optional spinner callback.
        agent._emit_diagnostic_status(notice)
    return retry


def consume_claude_output_recovery(agent: Any) -> dict | None:
    """Consume request-local overrides only on the route that armed them."""
    pending = getattr(agent, "_claude_output_recovery_pending", None)
    agent._claude_output_recovery_pending = None
    if pending is None or not mandatory_claude(agent) or pending[0] != recovery_route(agent):
        return None
    if pending[1] is not None:
        agent._ephemeral_max_output_tokens = pending[1]
    return {"enabled": True, "effort": "low"}
