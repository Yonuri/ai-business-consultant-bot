"""Простой in-memory rate limiter для клиентских вопросов."""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(self, cooldown_seconds: float, per_hour: int) -> None:
        self._cooldown = max(0.0, cooldown_seconds)
        self._per_hour = max(1, per_hour)
        self._last_call: dict[int, float] = {}
        self._events: dict[int, deque[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.monotonic()

        if now - self._last_call.get(user_id, 0.0) < self._cooldown:
            return False

        events = self._events.setdefault(user_id, deque())
        while events and now - events[0] > 3600:
            events.popleft()

        if len(events) >= self._per_hour:
            return False

        self._last_call[user_id] = now
        events.append(now)
        return True
