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
             'text': event.text, 'delivered': False, 'claim': None}

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
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                         (_key(sid, identity), json.dumps(value)))
        return authority.db._execute_write(commit)

    def publish(_):
        authority.sessions[sid].event_stream.publish(sid,
            {'kind': 'acp_notification', 'text': identity}, event_type='status.update', execution={})
    await tracked_write(authority, write, then=publish, ordered=True)
    return True


async def notification(connection, ref, params):
    """Claim/ack one offer. A disconnected viewer's unacknowledged claim is reclaimable."""
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
                    value['delivered'] = action == 'ack'
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


def handlers(connection):
    return {'session.acp_notification': partial(notification, connection)}
