"""Durable Stop boundary for process/delegation wakes on a canonical session FIFO.

Session ownership survives Stop; permission to wake it does not. Keep the cutoff on
its logical owner, across compression and gateway restarts. Never clear it on user
resume: old producers must not acquire the new user's permission to run.
"""
import json
import math
import time

from hermes_state_runtime import RuntimeStoreError

_PREFIX = 'automation_stop:'
_PRODUCERS = frozenset({'completion', 'async_delegation', 'watch_match', 'watch_disabled', 'heartbeat'})


def _descriptor(payload):
    return payload.get('local_automation_v1') or payload.get('native_text_v1', {}).get('automation', {})


def _producer(descriptor):
    try:
        return json.loads(descriptor.get('identity', ''))[0]
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _stale(conn, session_id, descriptor):
    if _producer(descriptor) not in _PRODUCERS:
        return False  # bot DMs, cron, and explicit user input are not process wakes
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + session_id,)).fetchone()
    if saved is None:
        return False
    fence = json.loads(saved[0])
    turn = descriptor.get('producer_turn')
    if isinstance(turn, dict) and turn.get('owner') == session_id and type(turn.get('generation')) is int:
        return turn['generation'] <= fence['generation']
    # An unrelated producer generation cannot be interpreted as this owner's.
    # Legacy and recovered producers retain the spawn/dispatch-time fallback.
    started = descriptor.get('producer_started_at')
    # Missing legacy provenance cannot prove that work was started after Stop.
    return (type(started) not in (int, float) or not math.isfinite(started)
            or started <= fence['cutoff'])


def automation_guard(authority, session_id, payload, guard=None):
    """Check at the actual admission commit, not a preflight separated by awaits."""
    def authorize(conn):
        if _stale(conn, session_id, _descriptor(payload)):
            raise RuntimeStoreError('cancelled_automation', detail='producer predates explicit Stop')
        if guard is not None:
            guard(conn)
    return authorize


async def fence_automation(authority, ref, generation):
    """Fence only the generation the caller authorized; settle queued old notices truthfully.

    Serialized by SQLite with admission writes: an in-flight completion either commits before
    Stop and is cancelled here, or commits after it and its write guard refuses.
    No execution-generation bump: the interrupted turn still owns its settlement.
    """
    from gateway.session_runtime_workers import tracked_write
    cutoff = time.time()
    route = authority.sessions[ref.session_id].route

    def write():
        def commit(conn):
            epoch = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
            if epoch is None or epoch[0] != authority.epoch:
                raise RuntimeStoreError('stale_epoch')
            current = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (ref.session_id,)).fetchone()
            if current is None or current[0] != generation:
                raise RuntimeStoreError('stale_generation')
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + ref.session_id,)).fetchone()
            # A retried Stop must not revoke work a later user started.
            if old is None or json.loads(old[0])['generation'] < generation:
                conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                    (_PREFIX + ref.session_id, json.dumps({'generation': generation, 'cutoff': cutoff})))
            cancelled = []
            for row in conn.execute("SELECT admission_id,payload_json FROM session_admissions WHERE target_session_id=? "
                                    "AND principal_id=? AND status='queued'", (ref.session_id, 'automation:' + route)):
                if _stale(conn, ref.session_id, _descriptor(json.loads(row['payload_json']))):
                    cancelled.append(row['admission_id'])
            for admission_id in cancelled:
                conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE admission_id=?", (admission_id,))
            return cancelled
        return authority.db._execute_write(commit)

    def publish(cancelled):
        for admission_id in cancelled:
            authority._settle_cancelled(ref, admission_id)
    await tracked_write(authority, write, then=publish)


def legacy_batch_boundary(runner, event):
    """Legacy producers can still fan in, provided they agree on Stop eligibility."""
    from gateway.session_authorities import active_authority
    from gateway.session_automation import _route_owner
    authority = active_authority(runner)
    if authority is None:
        return ''
    entry = runner.session_store.lookup_by_session_key(str(event.get('session_key') or ''))
    if entry is None:
        return ''
    descriptor = {'identity': json.dumps([event.get('type')]), **producer_metadata(event)}
    owner = _route_owner(authority, entry)
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + owner,)).fetchone()
        if saved is None:
            return ''
        return f"{json.loads(saved[0])['generation']}:{_stale(conn, owner, descriptor)}"


def producer_metadata(event):
    """The producer's spawn/dispatch time, not the completion's delivery time."""
    return {'producer_started_at': event.get('dispatched_at') if event.get('type') == 'async_delegation'
            else event.get('started_at'), 'producer_turn': event.get('producer_turn')}
