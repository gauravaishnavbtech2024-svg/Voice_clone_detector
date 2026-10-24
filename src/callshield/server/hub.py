"""AlertHub managing per-user WebSocket subscriptions and alert broadcasts."""

import asyncio
import contextlib
from collections import defaultdict
from typing import Any


class AlertHub:
    """Manages real-time alert subscribers by user_id using threadsafe queues."""

    def __init__(self) -> None:
        self._subscribers: dict[
            str, set[tuple[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop]]
        ] = defaultdict(set)

    def subscribe(
        self,
        user_id: str,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Registers an async queue to receive alerts for user_id."""
        active_loop = loop or asyncio.get_running_loop()
        self._subscribers[user_id].add((queue, active_loop))

    def unsubscribe(
        self,
        user_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Unregisters an async queue."""
        if user_id in self._subscribers:
            self._subscribers[user_id] = {
                (q, loop) for q, loop in self._subscribers[user_id] if q is not queue
            }
            if not self._subscribers[user_id]:
                del self._subscribers[user_id]

    async def publish(self, user_id: str, message: dict[str, Any]) -> None:
        """Dispatches an alert payload to all active subscribers for user_id."""
        if user_id not in self._subscribers:
            return

        for queue, loop in list(self._subscribers[user_id]):
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(queue.put_nowait, message)
