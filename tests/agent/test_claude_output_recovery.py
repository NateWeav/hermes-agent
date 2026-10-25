"""Mandatory-thinking recovery must change effort once, not buy repeated full budgets."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent import test_length_continuation_thinking_exhaustion as helpers

loop_agent = helpers.loop_agent
_run = helpers._run


def _response(mode, finish="length", text=""):
    if mode == "anthropic_messages":
        blocks = [SimpleNamespace(type="thinking", thinking="private", signature="test")]
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
        return SimpleNamespace(
            id="msg_test", content=blocks, stop_reason="max_tokens" if finish == "length" else "end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=32000),
        )
    return SimpleNamespace(
        id="chat_test", choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=text, tool_calls=None, reasoning_content="private"))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=32000),
    )


def _configure(agent, mode, effort="high"):
    agent.model = "claude-opus-5-5"
    agent.api_mode = mode
    agent.max_tokens = 32000
    agent.reasoning_config = {"enabled": True, "effort": effort}
    agent._supports_reasoning_extra_body = lambda: True
    agent._is_anthropic_oauth = False
    agent.thinking_callback = MagicMock()
    return agent


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
@pytest.mark.parametrize("finishes", [("length", "length"), ("stop", "stop"), ("length", "stop"), ("stop", "length")])
def test_mandatory_claude_stall_has_one_shared_recovery_budget(loop_agent, mode, finishes, caplog):
    caplog.set_level("INFO", logger="agent.conversation_loop")
    agent = _configure(loop_agent, mode)
    agent.status_callback = MagicMock()
    requests = []

    def call(kwargs):
        requests.append(kwargs)
        return _response(mode, finishes[min(len(requests) - 1, 1)])

    with patch.object(agent, "_interruptible_api_call", side_effect=call):
        result = _run(agent, "answer please")
    assert len(requests) == 2
    assert not result["completed"]
    assert "No visible answer" in result["final_response"]
    assert "none" not in result["final_response"]
    assert "private" not in result["final_response"]
    assert requests[1]["max_tokens"] == 32000
    if mode == "anthropic_messages":
        assert requests[1]["thinking"]["type"] == "adaptive"
        assert requests[1]["output_config"]["effort"] == "low"
        assert requests[0]["system"] == requests[1]["system"]
    else:
        assert requests[1]["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}
        assert requests[0]["messages"][0] == requests[1]["messages"][0]
    notices = " ".join(str(c.args[0]) for c in agent.thinking_callback.call_args_list)
    assert "recovery 1/1" in notices
    assert "effort=low" in notices
    assert "32000" in notices
    assert "request=" in notices
    summaries = [r.getMessage() for r in caplog.records if "Model request controls:" in r.getMessage()]
    assert len(summaries) == 2
    assert "effort=low" in summaries[1] and "max_tokens=32000" in summaries[1]
    assert f"api_mode='{mode}'" in summaries[1]
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
def test_already_low_does_not_repeat_and_new_turn_restores_user_effort(loop_agent, mode):
    agent = _configure(loop_agent, mode, "low")
    with patch.object(agent, "_interruptible_api_call", return_value=_response(mode)) as call:
        result = _run(agent, "answer please")
    assert call.call_count == 1
    assert not result["completed"]
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    with patch.object(agent, "_interruptible_api_call", side_effect=[
        _response(mode), _response(mode, "stop", "Answer.")
    ]) as call:
        result = _run(agent, "new turn")
    assert result["completed"]
    assert call.call_count == 2
    first = call.call_args_list[0].args[0]
    assert (first["output_config"]["effort"] if mode == "anthropic_messages" else
            first["extra_body"]["reasoning"]["effort"]) == "high"


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
def test_known_claude_output_ceiling_applies_to_retry_budget(mode):
    from agent.turn_truncation import boosted_output_cap
    agent = SimpleNamespace(model="claude-sonnet-4-6", api_mode=mode, max_tokens=32000)
    assert boosted_output_cap(agent, 64000, 3) == 64000


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
def test_incomplete_claude_tool_call_at_model_ceiling_is_not_rebilled(loop_agent, mode):
    agent = _configure(loop_agent, mode)
    agent.max_tokens = 128000
    response = _response(mode)
    if mode == "anthropic_messages":
        response.content = [SimpleNamespace(type="tool_use", id="t1", name="terminal", input={"command": "partial"})]
    else:
        response.choices[0].message.tool_calls = [SimpleNamespace(
            id="t1", type="function", function=SimpleNamespace(name="terminal", arguments='{"command":'))]
    with patch.object(agent, "_interruptible_api_call", return_value=response) as call:
        result = _run(agent, "run a command")
    assert call.call_count == 1
    assert not result["completed"]
    assert "not executed" in result["final_response"]


def test_pending_recovery_cannot_leak_across_model_switch():
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    from agent.claude_output_recovery import recover_no_visible_claude
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="custom:claude", model="claude-opus-5-5", base_url="https://example.test",
        max_tokens=16384, reasoning_config={"enabled": True, "effort": "high"},
        _emit_diagnostic_wait=MagicMock(), _emit_diagnostic_status=MagicMock(),
    )
    assert recover_no_visible_claude(agent, None, "length") is True
    agent.model = "claude-sonnet-4-6"
    assert _reasoning_config_for_wire(agent) == agent.reasoning_config
    assert getattr(agent, "_ephemeral_max_output_tokens", None) is None
