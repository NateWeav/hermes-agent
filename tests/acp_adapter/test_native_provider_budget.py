"""Real config -> ACP construction -> SDK wire contracts for named custom providers."""

import copy
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from acp_adapter.session import SessionManager
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = {
        "model": {"provider": "custom:chat-proxy", "default": "gpt-fixture"},
        "agent": {"max_turns": 1000, "disabled_toolsets": ["all"]},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "providers": {
            "chat-proxy": {
                "api": "http://proxy.invalid/v1", "api_key": "fixture-key",
                "transport": "chat_completions", "discover_models": False,
                "models": {"gpt-fixture": {"context_length": 200000}},
            },
            "messages-proxy": {
                "api": "http://proxy.invalid", "api_key": "fixture-key",
                "transport": "anthropic_messages", "discover_models": False,
                "models": {"claude-sonnet-4-5": {"context_length": 200000}},
                "extra_body": {"max_tokens": 16384, "metadata": {"user_id": "fixture-user"}},
            },
        },
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kw: None)

    # Keep the real constructor and resolution; omit unrelated prompt/memory I/O.
    class BudgetAgent(AIAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, skip_context_files=True, skip_memory=True)

    monkeypatch.setattr("run_agent.AIAgent", BudgetAgent)
    db = SessionDB(db_path=tmp_path / "state.db")
    yield config, path, db
    db.close()


def _native(manager, **kwargs):
    return manager._make_agent(
        session_id="native-budget", cwd=".", model="claude-sonnet-4-5",
        requested_provider="custom:messages-proxy", enabled_toolsets=[], **kwargs,
    )


def _wire(kwargs):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "fixture-message", "type": "message", "role": "assistant",
            "model": kwargs["model"], "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    with anthropic.Anthropic(
        api_key="fixture-key", base_url="http://proxy.invalid", max_retries=0, timeout=30,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        client.messages.create(**kwargs)
    return captured[0]


def test_native_budget_and_extra_body_reach_sdk_without_mutating_overrides(runtime):
    _, _, db = runtime
    agent = _native(SessionManager(db=db))
    original = copy.deepcopy(agent.request_overrides)
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[])
    body = _wire(kwargs)
    assert body["max_tokens"] == 16384
    assert body["metadata"] == {"user_id": "fixture-user"}
    assert "extra_body" not in body  # SDK merges it into the JSON object.
    assert agent.request_overrides == original
    assert agent.max_tokens is None
    assert agent.max_iterations == 1000


@pytest.mark.parametrize("recovery_budget", [4096, 32768])
def test_recovery_budget_wins_once_then_provider_budget_returns(runtime, recovery_budget):
    _, _, db = runtime
    agent = _native(SessionManager(db=db))
    agent._ephemeral_max_output_tokens = recovery_budget
    messages = [{"role": "user", "content": "hello"}]
    assert _wire(agent._build_api_kwargs(messages, tools_for_api=[]))["max_tokens"] == recovery_budget
    assert agent._ephemeral_max_output_tokens is None
    assert _wire(agent._build_api_kwargs(messages, tools_for_api=[]))["max_tokens"] == 16384
    assert agent.request_overrides["extra_body"]["max_tokens"] == 16384


@pytest.mark.parametrize("raw, expected", [(1000, 1000), ("17", 17), (None, sys.maxsize),
                                          ("unlimited", sys.maxsize), (True, sys.maxsize),
                                          ("invalid", sys.maxsize)])
def test_acp_uses_shared_turn_limit_validation(runtime, raw, expected):
    config, path, db = runtime
    config["agent"]["max_turns"] = raw
    path.write_text(json.dumps(config), encoding="utf-8")
    agent = _native(SessionManager(db=db))
    assert agent.max_iterations == expected


def test_chat_route_has_no_new_output_cap(runtime, monkeypatch):
    config, path, db = runtime
    config["model"]["max_tokens"] = 7  # Removed global controls must stay inert.
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_MAX_TOKENS", "13")
    agent = SessionManager(db=db)._make_agent(session_id="chat-budget", cwd=".", enabled_toolsets=[])
    assert agent.api_mode == "chat_completions"
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[])
    for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        assert field not in kwargs
        assert field not in kwargs.get("extra_body", {})
    assert agent.max_tokens is None


def test_native_budget_respects_context_window_and_summary(runtime, monkeypatch):
    from agent import chat_completion_helpers as helpers

    _, _, db = runtime
    agent = _native(SessionManager(db=db))
    agent.context_compressor.context_length = 8000
    body = _wire(agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[]))
    assert body["max_tokens"] == 7999
    captured = []

    def summary_call(agent, request_id, kwargs, call, **options):
        captured.append(_wire(kwargs))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="summary")], stop_reason="end_turn")

    monkeypatch.setattr(helpers, "_managed_summary_call", summary_call)
    result = helpers._anthropic_summary_attempt(agent, [{"role": "user", "content": "summarize"}], "summary")(0)
    assert result == "summary"
    assert captured[0]["max_tokens"] == 7999
    assert captured[0]["metadata"] == {"user_id": "fixture-user"}


def test_worker_chat_route_has_no_new_cap_and_endpoint_override_keeps_body_scoped(runtime):
    from gateway.session_worker_construct import routing_agent_kwargs

    config, _, _ = runtime
    plain = routing_agent_kwargs(config, "gpt-fixture", "custom:chat-proxy", "http://proxy.invalid/v1")
    assert plain["api_mode"] == "chat_completions"
    assert not plain["request_overrides"]
    elsewhere = routing_agent_kwargs(config, "claude-sonnet-4-5", "custom:messages-proxy", "http://other.invalid")
    assert not elsewhere["request_overrides"]
    assert elsewhere["api_mode"] is None


def test_named_provider_cold_restore_uses_current_transport_and_identity(runtime):
    _, _, db = runtime
    db.create_session(session_id="restored-budget", source="acp", model="claude-sonnet-4-5", model_config={
        "cwd": ".", "provider": "custom:messages-proxy", "base_url": "http://proxy.invalid",
        "api_mode": "chat_completions",  # Transport persisted before provider config changed.
    })
    manager = SessionManager(db=db)
    state = manager.get_session("restored-budget")
    assert state is not None
    assert state.agent.api_mode == "anthropic_messages"
    assert state.agent.requested_provider == "custom:messages-proxy"
    assert _wire(state.agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[]))["max_tokens"] == 16384
    state.history = [{"role": "user", "content": "hello"}]
    manager.save_session(state.session_id)
    meta = json.loads(db.get_session(state.session_id)["model_config"])
    assert meta["provider"] == "custom:messages-proxy"
    assert "api_key" not in meta


@pytest.mark.asyncio
async def test_gateway_explicit_pick_refreshes_native_provider_for_managed_worker(runtime, monkeypatch):
    """The normal ACP gateway path freezes the picked entry, not its creation-time copy."""
    from agent.managed_worker import _construct_agent
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy, policy_scope
    from gateway.session_worker_construct import routing_agent_kwargs, route_runtime
    from hermes_cli.model_switch import ModelSwitchResult
    from hermes_cli.runtime_provider import frozen_runtime_config

    config, path, db = runtime
    frozen = copy.deepcopy(config)
    frozen["providers"].pop("messages-proxy")
    policy = build_policy({"source": "acp", "cwd": str(path.parent), "model": "gpt-fixture", "toolsets": []},
                          frozen, private_secrets={})
    authority = SimpleNamespace(instance_id="fixture", epoch=1, profile_id="fixture", db=db)
    authority.runner = SimpleNamespace(_resolve_session_agent_runtime=lambda **kw: ("gpt-fixture", {}))
    # Provider/model validation is already covered separately; avoid its live catalog probes.
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, target_provider="custom:messages-proxy", new_model="claude-sonnet-4-5",
        base_url="http://proxy.invalid", api_mode="anthropic_messages", provider_changed=True))
    monkeypatch.setattr("gateway.session_mutation_model._selection_guard_refusal", _no_selection_warning)
    prepared = {"snapshot": {"receipt": {"session_id": "gateway-budget", "policy": asdict(policy)}}}
    selected = restore_policy((await prepare_model(authority, SimpleNamespace(source=None, route="r"),
        {"model": "claude-sonnet-4-5", "provider": "custom:messages-proxy"}, prepared))["policy"])
    selected_config = selected.config(authority)
    assert selected_config["providers"]["messages-proxy"]["extra_body"]["max_tokens"] == 16384
    assert selected.max_turns == 1000
    # A later profile edit cannot silently rewrite the explicit selection in this turn.
    path.write_text(json.dumps({"model": {"default": "other"}}), encoding="utf-8")
    resolved, _ = route_runtime(selected, selected_config, selected.model, "fixture-key", fallback=False)
    assert resolved["api_mode"] == "anthropic_messages"
    with frozen_runtime_config(selected_config), policy_scope(selected, authority=authority):
        agent = _construct_agent({}, AIAgent, selected, model=selected.model,
            provider=selected.provider, base_url=selected.base_url, api_key="fixture-key",
            enabled_toolsets=[], max_iterations=selected.max_turns, skip_context_files=True,
            skip_memory=True, quiet_mode=True, session_id="gateway-budget",
            **routing_agent_kwargs(selected_config, selected.model, selected.provider, selected.base_url))
    assert agent.api_mode == "anthropic_messages"
    assert _wire(agent._build_api_kwargs([{"role": "user", "content": "hello"}], tools_for_api=[]))["max_tokens"] == 16384


async def _no_selection_warning(*args):
    return None
