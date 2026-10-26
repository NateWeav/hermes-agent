"""Durable ACP completion offers. Only the editor may request their model turn."""
import json
from functools import partial

from gateway.session_automation_fence import automation_guard
from gateway.session_runtime_workers import tracked_write
from hermes_state_runtime import RuntimeStoreError

_PREFIX = 'acp_notification:'


def _key(sid, identity):
    return _PREFIX + json.dumps([sid, identity], separators=(',', ':'))


async def offer(authority, adapter, event, identity, evt):
    from gateway.session_automation import snapshot_automation
    from gateway.session_policy import policy_for_source
    from gateway.config import Platform
    if event.source.platform != Platform.LOCAL:
        return False
    policy = policy_for_source(authority.runner, event.source)
    if policy is None or policy.source != 'acp':
        return False
    payload, _entry = snapshot_automation(authority, adapter, event, identity)
    descriptor = payload['local_automation_v1']
    sid = descriptor['owner']

    value = {'payload': payload, 'kind': evt.get('type', 'completion'),
             'title': event.metadata.get('display_text') or event.text[:200],
             'text': event.text, 'delivered': False, 'claim': None, 'consumed': False}

    def write():
        def commit(conn):
            authority._admission_gate(automation_guard(authority, sid, payload))(conn)
            # An old owner may already have admitted this completion. Never offer it again.
            identities = set(descriptor.get('identities', [identity]))
            for row in conn.execute('SELECT payload_json FROM session_admissions WHERE target_session_id=?', (sid,)):
                old = json.loads(row[0])
                old = old.get('local_automation_v1') or old.get('native_text_v1', {}).get('automation', {})
                if identities.intersection(old.get('identities', [old.get('identity')])):
                    return
            replaced = evt.get('_replaces_offer')
            if replaced:
                old = conn.execute('SELECT value FROM state_meta WHERE key=?', (replaced,)).fetchone()
                prior = json.loads(old[0]) if old else None
                if prior is None or prior.get('claim') or prior.get('delivered'):
                    raise RuntimeStoreError('stale_generation')
                prior.update(delivered=True, withdrawn=True)
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(prior), replaced))
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                         (_key(sid, identity), json.dumps(value)))
        return authority.db._execute_write(commit)

    def publish(_):
        authority.sessions[sid].event_stream.publish(sid,
            {'kind': 'acp_notification', 'text': identity}, event_type='status.update', execution={})
    await tracked_write(authority, write, then=publish, ordered=True)
    return True


async def notification(connection, ref, params):
    """Claim one offer until its client-owned prompt consumes the durable receipt.

    ``ack`` only confirms socket delivery, not consumption. Disconnected claims
    can therefore be reclaimed without losing a result before its prompt arrives.
    """
    authority, actor = connection.authority, connection.actor
    authority.authorize(actor, ref, 'session:submit')
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None or policy.source != 'acp':
        raise RuntimeStoreError('permission_denied')
    if ref.session_id not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    action, identity = params.get('action', 'take'), params.get('identity')

    def write():
        def commit(conn):
            if action == 'take' and conn.execute(
                    "SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal' LIMIT 1",
                    (ref.session_id,)).fetchone():
                return {'notification': None}
            rows = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (_PREFIX + '%',)).fetchall()
            for key, raw in rows:
                sid, item_id = json.loads(key[len(_PREFIX):])
                if sid != ref.session_id or (identity is not None and identity != item_id):
                    continue
                value = json.loads(raw)
                if value['delivered']:
                    continue
                claim = value.get('claim')
                if action != 'take':
                    if claim != actor.transport_id:
                        continue
                    # A socket send is not a client receipt. Keep the offer pending
                    # until prompt.submit consumes it; a detached viewer may redeliver.
                    if action == 'release':
                        value['claim'] = None
                else:
                    if claim and claim in authority.events:
                        continue
                    try:
                        automation_guard(authority, sid, value['payload'])(conn)
                    except RuntimeStoreError as exc:
                        if exc.reason != 'cancelled_automation':
                            raise
                        value['delivered'] = True
                        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(value), key))
                        continue
                    value['claim'] = actor.transport_id
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(value), key))
                if action == 'take':
                    return {'notification': {k: value[k] for k in ('kind', 'title', 'text')}, 'identity': item_id}
                return {'notification': None}
            return {'notification': None}
        return authority.db._execute_write(commit)
    return await tracked_write(authority, write, ordered=True)


def _rows(conn, sid):
    for key, raw in conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (_PREFIX + '%',)):
        owner, identity = json.loads(key[len(_PREFIX):])
        if owner == sid:
            yield key, identity, json.loads(raw)


def contains_identity(authority, sid, identity):
    with authority.db._read_ctx() as conn:
        return any(identity in (value['payload']['local_automation_v1'].get('identities') or [item_id])
                   for _, item_id, value in _rows(conn, sid))


def prepare_tick(authority, sid, process, event):
    """Fold only an unclaimed offer: a claimed one may already be on its editor's wire."""
    from gateway.session_automation import producer_identity
    from tools.process_registry_notifications import join_heartbeat_outputs
    identity = producer_identity(authority.runner, event)
    with authority.db._read_ctx() as conn:
        rows = list(_rows(conn, sid))
    if any(identity in (value['payload']['local_automation_v1'].get('identities') or [item_id])
           for _, item_id, value in rows):
        return True, None
    for key, item_id, value in rows:
        parsed = json.loads(item_id)
        descriptor = value['payload']['local_automation_v1']
        if (len(parsed) == 3 and parsed[:2] == ['heartbeat', process]
                and not value.get('delivered') and not value.get('claim')):
            earlier = descriptor.get('tick')
            if isinstance(earlier, dict):
                identities = sorted(set([identity, *(descriptor.get('identities') or [item_id])]))
                return True, {**event, **join_heartbeat_outputs(earlier, event),
                              '_replaces_offer': key, '_automation_identities': identities}
    return False, event


def tick_claimed(authority, key):
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    return saved is None or bool(json.loads(saved[0]).get('claim') or json.loads(saved[0]).get('delivered'))


async def withdraw_ticks(authority, sid, processes):
    def write():
        def commit(conn):
            for key, identity, value in _rows(conn, sid):
                parsed = json.loads(identity)
                if (len(parsed) == 3 and parsed[0] == 'heartbeat' and parsed[1] in processes
                        and not value.get('consumed')):
                    value['delivered'] = True
                    value['withdrawn'] = True
                    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(value), key))
        return authority.db._execute_write(commit)
    await tracked_write(authority, write, ordered=True)


async def submit_receipts(connection, ref, params):
    """Consume an editor's result receipt in the SAME transaction as its one admission.

    Keep the stored automation provenance so Stop also cancels a continuation queued
    behind human input. Never trust the editor's text or a new random prompt identity.
    """
    from hermes_state_runtime import admit_session_input
    ids = params.get('notification_ids')
    if (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str)
            or params.get('attachments')):
        raise RuntimeStoreError('invalid_params')
    authority, sid, identity = connection.authority, ref.session_id, ids[0]
    authority.authorize(connection.actor, ref, 'session:submit')
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[sid].source)
    if policy is None or policy.source != 'acp':
        raise RuntimeStoreError('permission_denied')
    if sid not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (_key(sid, identity),)).fetchone()
    if saved is None:
        raise RuntimeStoreError('not_found')
    value = json.loads(saved[0])
    if value.get('withdrawn'):
        raise RuntimeStoreError('cancelled_automation')
    payload = value['payload']
    route = payload['local_automation_v1']['route']

    def consume(conn):
        automation_guard(authority, sid, payload)(conn)
        current = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                         (_key(sid, identity),)).fetchone()[0])
        if current.get('withdrawn'):
            raise RuntimeStoreError('cancelled_automation')
        current.update(consumed=True, delivered=True, claim=None)
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(current), _key(sid, identity)))

    return await tracked_write(authority, partial(admit_session_input, authority.db,
        epoch=authority.epoch, principal_id='automation:' + route, session_id=sid,
        request_id=identity, payload=payload, _authorize_write=authority._admission_gate(consume)),
        then=partial(authority._admitted, ref, None), ordered=True,
        after=partial(authority._schedule_admitted, ref))


def handlers(connection):
    return {'session.acp_notification': partial(notification, connection)}
