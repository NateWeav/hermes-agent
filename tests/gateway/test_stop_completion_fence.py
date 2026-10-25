"""Stop invalidates old automatic wakes, not the next explicit user request."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_state_runtime import admit_session_input, claim_session_input, list_session_admissions, settle_session_input
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit


def _notice(identity, started=10.0):
    if not identity.startswith('['):
        identity = json.dumps(['completion', identity, started])
    return {'text': 'background finished', 'local_automation_v1': {
        'identity': identity, 'owner': 's', 'route': 'route', 'target': 's',
        'producer_started_at': started}}


@pytest.mark.asyncio
@pytest.mark.parametrize('delivery', ['after_stop', 'before_stop', 'writing_during_stop'])
async def test_stop_completion_race_never_starts_a_hidden_turn(tmp_path, monkeypatch, delivery):
    from gateway import session_automation
    from gateway.session_automation_fence import automation_guard

    db, authority = _authority(tmp_path, monkeypatch)
    authority.runner._cached_agent_for = lambda route: None
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    identity = json.dumps(['completion', 'proc_a02039308079', 10.0])
    with db:
        human = await _submit(authority, 'human')
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        queued = None
        if delivery == 'before_stop':
            queued = admit_session_input(db, epoch=authority.epoch, principal_id='automation:route',
                session_id='s', request_id=identity, payload=_notice(identity))
        # Hold an actual producer admission immediately before its guarded SQLite write.
        original = session_automation.snapshot_automation
        monkeypatch.setattr(session_automation, 'snapshot_automation',
            lambda *args: (_notice(identity), SimpleNamespace(session_key='route')))
        write = db._execute_write
        if delivery == 'writing_during_stop':
            import threading
            in_write, allow_write = threading.Event(), threading.Event()
            def delayed(operation):
                if threading.current_thread().name.startswith('asyncio') and not in_write.is_set():
                    in_write.set()
                    assert allow_write.wait(5)
                return write(operation)
            monkeypatch.setattr(db, '_execute_write', delayed)
            pending = asyncio.create_task(authority.admit_automation(None, SimpleNamespace(source=None), identity))
            assert await asyncio.to_thread(in_write.wait, 5)
        try:
            await authority.interrupt(ACTOR, REF, running['generation'])
        finally:
            if delivery == 'writing_during_stop':
                allow_write.set()
        if delivery == 'writing_during_stop':
            from hermes_state_runtime import RuntimeStoreError
            with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
                await pending
        elif delivery == 'after_stop':
            from hermes_state_runtime import RuntimeStoreError
            with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
                admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                    request_id=identity, payload=_notice(identity),
                    _authorize_write=automation_guard(authority, 's', _notice(identity)))
        if queued:
            row = next(r for r in list_session_admissions(db, session_id='s', pending_only=False)
                       if r['admission_id'] == queued['admission_id'])
            assert (row['status'], row['outcome']) == ('terminal', 'cancelled')
        settle_session_input(db, epoch=authority.epoch, admission_id=human.admission_id,
                             generation=running['generation'], outcome='interrupted')
        assert claim_session_input(db, epoch=authority.epoch, session_id='s') is None
        # No process notification resurrected the stopped generation. A human can still resume.
        await _submit(authority, 'resume')
        fresh = claim_session_input(db, epoch=authority.epoch, session_id='s')
        assert fresh['request_id'] == 'resume' and fresh['generation'] > running['generation']
        monkeypatch.setattr(session_automation, 'snapshot_automation', original)


@pytest.mark.asyncio
async def test_active_and_fresh_work_completes_but_old_work_stays_fenced_after_resume(tmp_path, monkeypatch):
    from gateway.session_automation_fence import automation_guard
    from hermes_state_runtime import RuntimeStoreError
    import time

    db, authority = _authority(tmp_path, monkeypatch)
    authority.runner._cached_agent_for = lambda route: None
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        # No Stop: legitimate active completion is accepted.
        active = _notice('active')
        admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
            request_id='active', payload=active, _authorize_write=automation_guard(authority, 's', active))
        await authority.interrupt(ACTOR, REF, 0)
        await _submit(authority, 'resume')
        fresh = _notice('fresh', time.time() + 1)
        admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
            request_id='fresh', payload=fresh, _authorize_write=automation_guard(authority, 's', fresh))
        # The fence is durable, not a latch cleared by user input or a gateway restart.
        with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
            admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                request_id='old', payload=_notice('old'),
                _authorize_write=automation_guard(authority, 's', _notice('old')))
        assert {r['request_id'] for r in list_session_admissions(db, session_id='s')} == {'resume', 'fresh'}


@pytest.mark.asyncio
async def test_terminal_acp_cancel_fences_producers_without_changing_completed_outcome(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_automation_fence import automation_guard
    from hermes_state_runtime import RuntimeStoreError, get_session_admission

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        human = await _submit(authority, 'finishing')
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        settle_session_input(db, epoch=authority.epoch, admission_id=human.admission_id,
                             generation=running['generation'], outcome='completed')
        await AuthorityConnection.cancel(connection, REF, {'admission_id': human.admission_id})
        assert get_session_admission(db, admission_id=human.admission_id)['outcome'] == 'completed'
        old = _notice('late')
        with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
            admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                request_id='late', payload=old, _authorize_write=automation_guard(authority, 's', old))
        await _submit(authority, 'fresh-user')
        fresh = claim_session_input(db, epoch=authority.epoch, session_id='s')
        # A repeated cancel of the old admission never cancels a new user's generation.
        await AuthorityConnection.cancel(connection, REF, {'admission_id': human.admission_id})
        notice = _notice('new-work')
        notice['local_automation_v1']['producer_turn'] = {'owner': 's', 'generation': fresh['generation']}
        admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
            request_id='new-work', payload=notice, _authorize_write=automation_guard(authority, 's', notice))


@pytest.mark.asyncio
async def test_generation_fence_catches_work_started_by_stopped_turn_after_stop(tmp_path, monkeypatch):
    from gateway.session_automation_context import producer_turn_scope
    from gateway.session_automation_fence import automation_guard, producer_metadata
    from tools.process_registry import ProcessRegistry
    from gateway.run_notifications import GatewayNotificationsMixin
    from hermes_state_runtime import RuntimeStoreError
    import time

    db, authority = _authority(tmp_path, monkeypatch)
    authority.runner._cached_agent_for = lambda route: None
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        await _submit(authority, 'human')
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        await authority.interrupt(ACTOR, REF, running['generation'])
        with producer_turn_scope('s', running['generation']):
            # Tool thread unwinds late and spawns after Stop: timestamps alone would admit it.
            process = ProcessRegistry._new_session('true', '', '', 'route', str(tmp_path))
        assert process.started_at > 0 and process.producer_turn == {'owner': 's', 'generation': running['generation']}
        event = GatewayNotificationsMixin._build_process_completion_event({}, process, process.id)
        payload = _notice('late-spawn', time.time() + 1)
        payload['local_automation_v1'].update(producer_metadata(event))
        with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
            admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                request_id='late-spawn', payload=payload, _authorize_write=automation_guard(authority, 's', payload))
        # Reopening the DB demonstrates the fence is durable, not just a runner flag.
        from hermes_state import SessionDB
        with SessionDB(tmp_path / 'state.db') as reopened:
            with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
                reopened._execute_write(automation_guard(authority, 's', payload))


@pytest.mark.asyncio
async def test_real_notification_admission_preserves_active_delivery_but_drops_stopped_work(tmp_path, monkeypatch):
    from tests.gateway.test_automation_notification_category import _native_owner
    from gateway.run_notifications import _watch_notification_metadata
    from gateway.session_contract import SessionRef
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import RuntimeStoreError

    runner, authority, adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    route, sid = metadata['gateway_session_key'], metadata['gateway_session_id']
    ref = SessionRef(authority.profile_id, authority.logical_owner(sid))
    actor = type(ACTOR)(ACTOR.subject, authority.profile_id, ACTOR.capabilities, ACTOR.transport_id)
    try:
        event = {'type': 'completion', 'session_id': 'proc_active', 'session_key': route,
                 'parent_session_id': sid, 'started_at': 10.0, 'platform': source.platform.value}
        fields = runner._COMPLETION_BATCH_KEY_FIELDS
        before_stop = runner._event_route_key(event, fields)
        assert before_stop == runner._event_route_key({**event, 'started_at': 11.0}, fields)
        assert await runner._inject_watch_notification('active completion', event) is True
        await authority.interrupt(actor, ref, 0)
        import time
        after_stop = {**event, 'started_at': time.time() + 1}
        assert before_stop != runner._event_route_key(after_stop, fields)
        assert runner._event_route_key(event, fields) != runner._event_route_key(after_stop, fields)
        assert await runner._inject_watch_notification('late completion', {**event, 'session_id': 'proc_late'}) is None
        assert list_session_admissions(authority.db, session_id=ref.session_id)[0]['principal_id'].startswith('messaging:')
        assert not any(r['principal_id'].startswith('automation:') for r in list_session_admissions(
            authority.db, session_id=ref.session_id))
        for kind in ('completion', 'async_delegation', 'watch_match', 'watch_disabled', 'heartbeat'):
            old = {**event, 'type': kind, 'dispatched_at': 10.0}
            synthetic = MessageEvent(text='old notice', source=source, internal=True,
                                    metadata=_watch_notification_metadata(old, source))
            with pytest.raises(RuntimeStoreError, match='cancelled_automation'):
                await authority.admit_automation(adapter, synthetic, json.dumps([kind, 'old', '1']))
        # Fresh generation permission wins over an old wall clock; no global stopped latch.
        fresh = {**event, 'session_id': 'proc_fresh', 'producer_turn': {'owner': ref.session_id, 'generation': 1}}
        assert await runner._inject_watch_notification('fresh completion', fresh) is True
    finally:
        authority.db.close()
