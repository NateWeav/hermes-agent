"""Safe effective-wire summaries and gateway ACP recovery visibility."""
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_request_summary_is_allowlisted_and_uses_extra_body_precedence(caplog):
    from agent.request_diagnostics import log_request_controls

    agent = SimpleNamespace(api_mode="anthropic_messages", model="claude-opus-5-5")
    caplog.set_level("INFO", logger="agent.conversation_loop")
    log_request_controls(agent, {
        "model": agent.model, "max_tokens": 128000,
        "thinking": {"type": "adaptive", "secret": "thinking-secret"},
        "output_config": {"effort": "high"},
        "extra_body": {"max_tokens": 16384, "output_config": {"effort": "low"}, "api_key": "body-secret"},
        "extra_headers": {"authorization": "header-secret"},
        "messages": [{"role": "user", "content": "prompt-secret"}],
    })
    assert agent._last_requested_output_cap == 16384
    assert "max_tokens=16384 thinking=adaptive effort=low" in caplog.text
    assert "secret" not in caplog.text
    log_request_controls(agent, {"thinking": {"type": "untrusted-secret"}, "reasoning_effort": {"bad": "secret"}})
    assert "secret" not in caplog.text


def test_recovery_reaches_gateway_status_when_no_spinner_callback_is_attached():
    from agent.claude_output_recovery import recover_no_visible_claude
    from agent.status_output import StatusOutputMixin

    class Agent(StatusOutputMixin):
        api_mode = "anthropic_messages"
        model = "claude-opus-5-5"
        max_tokens = 16384
        reasoning_config = {"enabled": True, "effort": "high"}
        thinking_callback = None
        status_callback = MagicMock()
        suppress_status_output = True

        def _touch_activity(self, text):
            pass

    agent = Agent()
    assert recover_no_visible_claude(agent, None, "length") is True
    agent.status_callback.assert_called_once()
    kind, message = agent.status_callback.call_args.args
    assert kind == "lifecycle"
    assert "recovery 1/1" in message and "max_tokens=16384" in message
