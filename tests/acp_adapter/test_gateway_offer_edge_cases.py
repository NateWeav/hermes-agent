"""Receipt retry provenance and queued ACP heartbeat supersession."""
import asyncio

import acp
import pytest

from tests.acp_adapter.test_retained_identity_retirement import owner
from tests.acp_adapter.test_gateway_completion_ownership import editor, notice, offer
from hermes_cli.gateway_client import GatewayClientError
from hermes_state_runtime import list_session_admissions
from gateway.session_automation import admit_process_notice, prepare_heartbeat
from gateway.run_notifications import GatewayNotificationsMixin
from acp_adapter.gateway_notifications import drain


@pytest.mark.asyncio
async def test_lost_receipt_submit_retry_keeps_automation_without_editor_meta(owner):
    a = await editor(owner)
    await offer(owner)
    await drain(a, owner.sid)
    result = a._conn.ext_notification.await_args.args[1]
    a._event_task = asyncio.create_task(a._events())
    try:
        owner.wire.lose = ('prompt.submit', lambda params: True, 'after')
        with pytest.raises(GatewayClientError):
            await a.prompt([acp.text_block(result['text'])], owner.sid,
                field_meta={'hermes.notificationIds': result['notificationIds']})
        await asyncio.wait_for(a.prompt([acp.text_block(result['text'])], owner.sid), 5)
        rows = list_session_admissions(owner.authority.db, session_id=owner.sid, pending_only=False)
        assert len(rows) == 1
        assert rows[0]['payload']['local_automation_v1']
        assert owner.executed == [result['text']]
    finally:
        a._event_task.cancel()
        await asyncio.gather(a._event_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_receipts_are_ordinary_user_input(owner):
    a = await editor(owner)
    a._event_task = asyncio.create_task(a._events())
    try:
        await asyncio.wait_for(a.prompt([acp.text_block('human')], owner.sid,
            field_meta={'hermes.notificationIds': []}), 5)
        assert owner.executed == ['human']
        row = list_session_admissions(owner.authority.db, session_id=owner.sid, pending_only=False)[0]
        assert 'local_automation_v1' not in row['payload']
    finally:
        a._event_task.cancel()
        await asyncio.gather(a._event_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unclaimed_heartbeat_offers_fold_output_into_latest_tick(owner):
    a = await editor(owner)
    event, raw = notice(owner)
    runner = owner.authority.runner
    runner._completion_delivery_identity = GatewayNotificationsMixin._completion_delivery_identity
    adapter = runner._adapter_for_source(event.source)
    for number, output in [('1', 'first\n'), ('2', 'second\n')]:
        tick = await prepare_heartbeat(runner, {**raw, 'type': 'heartbeat',
            'event_id': number, 'output': output, 'omitted': 0})
        event.text = tick['output']
        await admit_process_notice(runner, owner.authority, adapter, event, tick)
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 1
    result = a._conn.ext_notification.await_args.args[1]
    assert 'first' in result['text'] and 'second' in result['text']
    a._event_task = asyncio.create_task(a._events())
    try:
        await asyncio.wait_for(a.prompt([acp.text_block(result['text'])], owner.sid,
            field_meta={'hermes.notificationIds': result['notificationIds']}), 5)
        assert owner.executed == [result['text']]
    finally:
        a._event_task.cancel()
        await asyncio.gather(a._event_task, return_exceptions=True)
