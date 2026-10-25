"""Gateway-backed ACP must show recovery status without filing it as model reasoning."""
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
async def test_gateway_acp_status_is_visible_without_contaminating_reasoning_or_final():
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    await agent._project({"session_id": "s", "admission_id": "a", "type": "status.update", "payload": {
        "kind": "lifecycle", "text": "Claude recovery 1/1: effort=low; max_tokens=16384; request=100s",
    }})
    agent._conn.session_update.assert_awaited_once()
    update = agent._conn.session_update.call_args.kwargs["update"]
    assert update.session_update == "agent_message_chunk"
    assert "recovery 1/1" in update.content.text
    assert not agent._segments and not agent._streamed
    assert agent._final_remainder("a", {"text": "Answer"}) == "Answer"
