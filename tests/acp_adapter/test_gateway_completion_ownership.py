"""Real authority/ACP contract: completion offers are not hidden model admissions."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import acp
import pytest

from tests.acp_adapter.test_retained_identity_retirement import owner, Wire
from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_notifications import drain
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session_automation import admit_process_notice, producer_identity, _process_completed
from gateway.session_automation_fence import fence_automation
from gateway.session_contract import SessionRef
from gateway.platforms.event import MessageEvent
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


async def editor(o, wire=None):
    wire = wire or o.wire
    a = GatewayACPAgent()
    a._gateway = wire
    a._conn = SimpleNamespace(ext_notification=AsyncMock(), session_update=AsyncMock())
    a._snapshots[o.sid] = await wire.rpc('session.resume', session_id=o.sid)
    return a


def notice(o, name='job', generation=1):
    live = o.authority.sessions[o.sid]
    raw = {'type': 'completion', 'session_id': name, 'started_at': 1,
           'session_key': live.route, 'producer_turn': {'owner': o.sid, 'generation': generation}}
    event = MessageEvent(text='Job finished: ' + name, source=live.source, internal=True,
        metadata={'gateway_session_key': live.route, 'gateway_session_id': o.sid,
                  'producer_started_at': 1, 'producer_turn': raw['producer_turn']})
    return event, raw


async def offer(o, name='job', generation=1):
    event, raw = notice(o, name, generation)
    o.authority.runner._completion_delivery_identity = GatewayNotificationsMixin._completion_delivery_identity
    adapter = o.authority.runner._adapter_for_source(event.source)
    await admit_process_notice(o.authority.runner, o.authority, adapter, event, raw)
    return event, raw


@pytest.mark.asyncio
async def test_one_client_wake_no_hidden_turn_and_client_prompt_executes_once(owner):
    a = await editor(owner)
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(owner.authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    b = await editor(owner, wire)
    try:
        event, raw = await offer(owner)
        await offer(owner)  # retried producer ACK
        await asyncio.gather(drain(a, owner.sid), drain(b, owner.sid))
        calls = a._conn.ext_notification.await_args_list + b._conn.ext_notification.await_args_list
        assert len(calls) == 1
        assert calls[0].args == ('hermes/notification', {
            'sessionId': owner.sid, 'kind': 'completion', 'title': event.text, 'text': event.text,
            'notificationIds': [producer_identity(owner.authority.runner, raw)]})
        assert list_session_admissions(owner.authority.db, session_id=owner.sid, pending_only=False) == []
        assert owner.executed == []
        # T3 requests the continuation; only this prompt is allowed to execute.
        a._event_task = asyncio.create_task(a._events())
        notice = calls[0].args[1]
        await asyncio.wait_for(a.prompt([acp.text_block(event.text)], owner.sid,
            field_meta={'hermes.notificationIds': notice['notificationIds']}), 5)
        assert owner.executed == [event.text]
        await drain(b, owner.sid)
        assert len(a._conn.ext_notification.await_args_list + b._conn.ext_notification.await_args_list) == 1
    finally:
        if a._event_task:
            a._event_task.cancel()
            await asyncio.gather(a._event_task, return_exceptions=True)
        await wire.viewer.close()


@pytest.mark.asyncio
async def test_stop_fences_before_offer_and_before_delivery(owner):
    a = await editor(owner)
    await offer(owner, generation=0)
    ref = SessionRef('p', owner.sid)
    await fence_automation(owner.authority, ref, 0)
    # Generation 0 and earlier producers cannot wake, even if their offer was persisted first.
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 0
    with pytest.raises(RuntimeStoreError) as refused:
        await offer(owner, 'stopped', 0)
    assert refused.value.reason == 'cancelled_automation'
    assert owner.executed == []
    await offer(owner, 'new-work', 1)
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 1


@pytest.mark.asyncio
async def test_disconnected_claim_reclaimed_and_already_admitted_not_offered(owner):
    a = await editor(owner)
    await offer(owner)
    held = await owner.wire.rpc('session.acp_notification', session_id=owner.sid)
    assert held['notification']
    await owner.wire.viewer.close()
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(owner.authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    b = await editor(owner, wire)
    try:
        await drain(b, owner.sid)
        assert b._conn.ext_notification.await_count == 1
        await drain(b, owner.sid)
        assert b._conn.ext_notification.await_count == 1
        event, raw = notice(owner, 'admitted')
        identity = producer_identity(owner.authority.runner, raw)
        # An autonomous admission from an older runtime is not also a client offer.
        owner.authority._schedule_admitted = lambda *a: None
        await owner.authority.admit_automation(owner.authority.runner._adapter_for_source(event.source), event, identity)
        await offer(owner, 'admitted')
        rows = list_session_admissions(owner.authority.db, session_id=owner.sid)
        assert len(rows) == 1
        owner.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE admission_id=?", (rows[0]['admission_id'],)))
        await drain(b, owner.sid)
        assert b._conn.ext_notification.await_count == 1
    finally:
        await wire.viewer.close()


@pytest.mark.asyncio
async def test_busy_offer_waits_and_failed_delivery_is_retriable(owner):
    a = await editor(owner)
    owner.authority._schedule = lambda *args: None
    await owner.wire.rpc('prompt.submit', session_id=owner.sid, input_id='busy', text='human')
    await offer(owner)
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 0
    owner.authority.db._execute_write(lambda conn: conn.execute(
        "UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE target_session_id=?",
        (owner.sid,)))
    a._conn.ext_notification.side_effect = OSError('detached')
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 1
    a._conn.ext_notification.side_effect = None
    await asyncio.gather(drain(a, owner.sid), drain(a, owner.sid))
    assert a._conn.ext_notification.await_count == 2
    assert owner.executed == []


@pytest.mark.asyncio
async def test_non_acp_keeps_autonomous_admission(owner, tmp_path):
    from gateway.session_local import create_local_session
    owner.sid = create_local_session(owner.authority, owner.wire.viewer.actor, {
        'request_id': 'cli', 'source': 'cli', 'cwd': str(tmp_path),
        'model': 'frozen', 'toolsets': []}).session_id
    owner.authority._schedule_admitted = lambda *args: None
    await offer(owner)
    rows = list_session_admissions(owner.authority.db, session_id=owner.sid)
    assert len(rows) == 1
    assert rows[0]['payload']['local_automation_v1']['identity']


@pytest.mark.asyncio
async def test_completion_offer_suppresses_heartbeat_and_withdraws_waiting_offer(owner):
    a = await editor(owner)
    event, raw = notice(owner)
    tick = {**raw, 'type': 'heartbeat', 'event_id': 'tick-1'}
    adapter = owner.authority.runner._adapter_for_source(event.source)
    owner.authority.runner._completion_delivery_identity = GatewayNotificationsMixin._completion_delivery_identity
    await admit_process_notice(owner.authority.runner, owner.authority, adapter, event, tick)
    await offer(owner)
    assert _process_completed(owner.authority.runner, owner.authority, owner.sid, tick)
    await drain(a, owner.sid)
    assert a._conn.ext_notification.await_count == 1
    assert a._conn.ext_notification.await_args.args[1]['kind'] == 'completion'


@pytest.mark.asyncio
async def test_client_receipt_consumption_is_atomic_and_stop_checked_at_submit(owner):
    a = await editor(owner)
    await offer(owner)
    await drain(a, owner.sid)
    notice = a._conn.ext_notification.await_args.args[1]
    ids = notice['notificationIds']
    a._event_task = asyncio.create_task(a._events())
    try:
        await asyncio.wait_for(a.prompt([acp.text_block(notice['text'])], owner.sid,
            field_meta={'hermes.notificationIds': ids}), 5)
        # Retrying even after the first prompt settled must replay the SAME admission.
        await asyncio.wait_for(a.prompt([acp.text_block(notice['text'])], owner.sid,
            field_meta={'hermes.notificationIds': ids}), 5)
        assert owner.executed == [notice['text']]
        await offer(owner, 'stopped', 0)
        await drain(a, owner.sid)
        stale = a._conn.ext_notification.await_args.args[1]
        generation = owner.authority.db.get_session(owner.sid)['runtime_generation']
        await fence_automation(owner.authority, SessionRef('p', owner.sid), generation)
        with pytest.raises(Exception):
            await asyncio.wait_for(a.prompt([acp.text_block(stale['text'])], owner.sid,
                field_meta={'hermes.notificationIds': stale['notificationIds']}), 5)
        assert owner.executed == [notice['text']]
    finally:
        a._event_task.cancel()
        await asyncio.gather(a._event_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_restart_before_client_receipt_redelivers_then_consumes_once(owner):
    from gateway.session_authority import initialize_session_authority
    a = await editor(owner)
    await offer(owner)
    await drain(a, owner.sid)  # socket send succeeded; T3 has not prompted yet
    first = a._conn.ext_notification.await_args.args[1]
    await owner.wire.viewer.close()
    previous = owner.authority
    # A new process does not reuse the old authority's in-memory LOCAL adapter.
    runner = SimpleNamespace(**vars(previous.runner))
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    owner.authority = await initialize_session_authority(runner,
        profile_id='p', instance_id='replacement')
    assert owner.authority.epoch > previous.epoch
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(owner.authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    b = await editor(owner, wire)
    try:
        await drain(b, owner.sid)
        replacement = b._conn.ext_notification.await_args.args[1]
        assert replacement == first
        b._event_task = asyncio.create_task(b._events())
        await asyncio.wait_for(b.prompt([acp.text_block(first['text'])], owner.sid,
            field_meta={'hermes.notificationIds': first['notificationIds']}), 5)
        await asyncio.wait_for(b.prompt([acp.text_block(first['text'])], owner.sid,
            field_meta={'hermes.notificationIds': first['notificationIds']}), 5)
        assert owner.executed == [first['text']]
        await drain(b, owner.sid)
        assert b._conn.ext_notification.await_count == 1
    finally:
        if b._event_task:
            b._event_task.cancel()
            await asyncio.gather(b._event_task, return_exceptions=True)
        await wire.viewer.close()
