"""Idle resume adopts current context/output limits without changing the model or prompt."""
import json

from gateway.session_policy import LocalSessionPolicy
from gateway.session_policy_refresh import refreshed_policy


def test_resume_refreshes_context_and_tool_limits_without_changing_prompt_or_route():
    old = {
        'model': {'default': 'claude-opus-5-5', 'provider': 'custom:claude'},
        'agent': {'max_turns': 1000, 'system_prompt': 'stable cached instructions'},
        'compression': {'threshold_tokens': 500000, 'threshold': 0.5},
        'tool_output': {'max_bytes': 50000},
        'file_read_max_chars': 100000,
    }
    policy = LocalSessionPolicy(source='acp', platform='acp', cwd='.', model='claude-opus-5-5',
        toolsets=('terminal',), config_json=json.dumps(old), request_json='{"max_turns":12}',
        terminal_json='{}', editor_mcp_json='{}')
    live = {
        'model': {'default': 'gpt-other'},
        'agent': {'max_turns': 80, 'system_prompt': 'new instructions'},
        'compression': {'threshold_tokens': 120000, 'threshold': 0.5},
        'tool_output': {'max_bytes': 20000},
        'file_read_max_chars': 30000,
    }
    updated, _, _ = refreshed_policy(policy, live, {})
    config = updated.config()
    assert config['compression']['threshold_tokens'] == 120000
    assert config['tool_output']['max_bytes'] == 20000
    assert config['file_read_max_chars'] == 30000
    assert config['model'] == old['model']
    assert config['agent']['system_prompt'] == 'stable cached instructions'
    assert config['agent']['max_turns'] == 1000  # Explicit launch choice is preserved.
    assert json.loads(policy.config_json) == old  # No mutation of the admitted generation.


def test_resume_removed_output_limit_returns_to_default():
    policy = LocalSessionPolicy(source='acp', platform='acp', cwd='.', model='claude-opus-5-5',
        toolsets=('terminal',), config_json='{"tool_output":{"max_bytes":20000},"file_read_max_chars":30000}',
        request_json='{}', terminal_json='{}', editor_mcp_json='{}')
    updated, _, _ = refreshed_policy(policy, {}, {})
    assert 'tool_output' not in updated.config()
    assert 'file_read_max_chars' not in updated.config()
