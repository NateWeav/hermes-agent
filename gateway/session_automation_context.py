"""A producer's owning turn, carried through executor threads, never a global env flag."""
from contextlib import contextmanager
from contextvars import ContextVar

_turn: ContextVar[dict | None] = ContextVar('automation_producer_turn', default=None)


def producer_turn():
    """The logical FIFO and execution generation that started this work, or None for legacy callers."""
    return _turn.get()


@contextmanager
def producer_turn_scope(session_id, generation):
    token = _turn.set({'owner': session_id, 'generation': generation})
    try:
        yield
    finally:
        _turn.reset(token)
