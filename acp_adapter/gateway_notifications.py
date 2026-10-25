"""Project a gateway-owned offer through the existing client-driven ACP extension."""
import logging

logger = logging.getLogger(__name__)


async def drain(agent, sid, gateway=None):
    if agent._conn is None:
        return
    gateway = gateway or agent._gateway
    if gateway is None:
        return
    locks = getattr(agent, '_notification_locks', None)
    if locks is None:
        agent._notification_locks = locks = {}
    import asyncio
    async with locks.setdefault(sid, asyncio.Lock()):
        await _drain(agent, sid, gateway)


async def _drain(agent, sid, gateway):
    try:
        while True:
            offer = await gateway.rpc('session.acp_notification', session_id=sid)
            notice = offer.get('notification')
            if not notice:
                return
            identity = offer['identity']
            try:
                # SDK adds the leading underscore: wire method is _hermes/notification.
                await agent._conn.ext_notification('hermes/notification', {
                    'sessionId': agent._editor_id(sid), **notice})
            except BaseException:
                await gateway.rpc('session.acp_notification', session_id=sid, action='release', identity=identity)
                raise
            await gateway.rpc('session.acp_notification', session_id=sid, action='ack', identity=identity)
    except Exception:
        # Old gateways do not serve this extension. Never infer locally as a fallback.
        logger.warning('ACP completion offer not delivered for %s', sid, exc_info=True)
