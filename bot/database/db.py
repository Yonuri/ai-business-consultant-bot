"""SQLite-хранилище: пользователи, источники, история диалогов, статистика."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from bot.database.models import ChatMessage, Source, Stats

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    created_at  TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    added_by    INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_kind ON sources(kind);
CREATE INDEX IF NOT EXISTS idx_sources_location ON sources(location);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    question    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at);

CREATE TABLE IF NOT EXISTS manager_requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    username    TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(UTC)


class Database:
    """Тонкая асинхронная обёртка над SQLite.

    Одно соединение + блокировка достаточно для одного процесса бота.
    Для нескольких реплик используйте внешнюю БД.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        logger.info("SQLite подключена: %s", self._path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("База данных не инициализирована")
        return self._conn

    # ------------------------------------------------------------------
    # Пользователи
    # ------------------------------------------------------------------

    async def upsert_user(
        self, user_id: int, username: str | None, first_name: str | None
    ) -> None:
        now = _iso_now()
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO users (id, username, first_name, created_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_seen_at = excluded.last_seen_at
                """,
                (user_id, username, first_name, now, now),
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Источники
    # ------------------------------------------------------------------

    async def add_source(
        self,
        *,
        kind: str,
        title: str,
        location: str,
        chunks: int,
        added_by: int,
    ) -> int:
        async with self._lock:
            cursor = await self.conn.execute(
                """
                INSERT INTO sources (kind, title, location, chunks, added_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, title, location, chunks, added_by, _iso_now()),
            )
            await self.conn.commit()
            return int(cursor.lastrowid or 0)

    async def update_source_chunks(self, source_id: int, chunks: int) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE sources SET chunks = ? WHERE id = ?", (chunks, source_id)
            )
            await self.conn.commit()

    async def find_source(self, *, kind: str, location: str) -> Source | None:
        async with self._lock:
            cursor = await self.conn.execute(
                "SELECT * FROM sources WHERE kind = ? AND location = ? ORDER BY id DESC LIMIT 1",
                (kind, location),
            )
            row = await cursor.fetchone()
        return self._row_to_source(row) if row else None

    async def get_source(self, source_id: int) -> Source | None:
        async with self._lock:
            cursor = await self.conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            )
            row = await cursor.fetchone()
        return self._row_to_source(row) if row else None

    async def list_sources(self) -> list[Source]:
        async with self._lock:
            cursor = await self.conn.execute(
                "SELECT * FROM sources ORDER BY created_at DESC, id DESC"
            )
            rows = await cursor.fetchall()
        return [self._row_to_source(row) for row in rows]

    async def count_sources(self) -> int:
        async with self._lock:
            cursor = await self.conn.execute("SELECT COUNT(*) FROM sources")
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def delete_source(self, source_id: int) -> bool:
        async with self._lock:
            cursor = await self.conn.execute(
                "DELETE FROM sources WHERE id = ?", (source_id,)
            )
            await self.conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_source(row: aiosqlite.Row) -> Source:
        return Source(
            id=int(row["id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            location=str(row["location"]),
            chunks=int(row["chunks"]),
            added_by=int(row["added_by"]),
            created_at=_parse_dt(str(row["created_at"])),
        )

    # ------------------------------------------------------------------
    # История диалогов
    # ------------------------------------------------------------------

    async def add_message(self, user_id: int, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Недопустимая роль сообщения: {role}")
        async with self._lock:
            await self.conn.execute(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, role, content, _iso_now()),
            )
            await self.conn.commit()

    async def get_history(self, user_id: int, limit: int = 8) -> list[ChatMessage]:
        if limit <= 0:
            return []
        async with self._lock:
            cursor = await self.conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [ChatMessage(role=str(r["role"]), content=str(r["content"])) for r in rows]

    async def clear_history(self, user_id: int) -> int:
        async with self._lock:
            cursor = await self.conn.execute(
                "DELETE FROM messages WHERE user_id = ?", (user_id,)
            )
            await self.conn.commit()
            return cursor.rowcount

    async def trim_history(self, user_id: int, keep: int = 40) -> None:
        keep = max(2, keep)
        async with self._lock:
            await self.conn.execute(
                """
                DELETE FROM messages
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM messages
                      WHERE user_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (user_id, user_id, keep),
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    async def log_question(self, user_id: int, question: str) -> None:
        async with self._lock:
            await self.conn.execute(
                "INSERT INTO questions (user_id, question, created_at) VALUES (?, ?, ?)",
                (user_id, question[:2000], _iso_now()),
            )
            await self.conn.commit()

    async def add_manager_request(
        self, user_id: int, username: str | None, payload: str
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO manager_requests (user_id, username, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, username, payload[:4000], _iso_now()),
            )
            await self.conn.commit()

    async def stats(self, *, chunks: int) -> Stats:
        async with self._lock:
            cursor = await self.conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM users) AS users,
                    (SELECT COUNT(*) FROM questions) AS questions,
                    (SELECT COUNT(*) FROM sources) AS sources,
                    (SELECT COUNT(*) FROM manager_requests) AS manager_requests
                """
            )
            row = await cursor.fetchone()
        return Stats(
            users=int(row["users"]) if row else 0,
            questions=int(row["questions"]) if row else 0,
            sources=int(row["sources"]) if row else 0,
            chunks=chunks,
            manager_requests=int(row["manager_requests"]) if row else 0,
        )

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Служебный метод для миграций/скриптов."""
        async with self._lock:
            await self.conn.execute(sql, params)
            await self.conn.commit()
