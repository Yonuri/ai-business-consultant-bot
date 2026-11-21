"""Модели данных, общие для слоя БД и сервисов."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Source:
    id: int
    kind: str
    title: str
    location: str
    chunks: int
    added_by: int
    created_at: datetime

    @property
    def is_url(self) -> bool:
        return self.kind == "url"

    @property
    def icon(self) -> str:
        return "🌐" if self.is_url else "📄"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Stats:
    users: int
    questions: int
    sources: int
    chunks: int
    manager_requests: int
