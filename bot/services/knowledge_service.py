"""Сервис базы знаний: добавление и удаление источников."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

from bot.config import Settings
from bot.database.db import Database
from bot.database.models import Source
from bot.services.rag_engine import RagEngine
from bot.utils.text import split_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: Source
    chunks: int
    replaced: bool


class KnowledgeService:
    def __init__(self, *, db: Database, rag: RagEngine, settings: Settings) -> None:
        self._db = db
        self._rag = rag
        self._settings = settings

    async def ingest(
        self,
        *,
        kind: str,
        title: str,
        location: str,
        text: str,
        added_by: int,
    ) -> IngestResult:
        """Индексирует текст и сохраняет источник.

        Порядок операций безопасен для замены: новый источник добавляется первым,
        и только после успешной индексации удаляется старая версия.
        """
        chunks = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("После разбиения не осталось текста для индексации.")

        existing = await self._db.find_source(kind=kind, location=location)

        source_id = await self._db.add_source(
            kind=kind,
            title=title,
            location=location,
            chunks=0,
            added_by=added_by,
        )
        try:
            count = await self._rag.add_chunks(
                source_id=source_id, title=title, chunks=chunks
            )
        except Exception:
            # Не оставляем «висячую» запись в SQLite и частично записанные векторы.
            with contextlib.suppress(Exception):
                await self._rag.delete_source(source_id)
            await self._db.delete_source(source_id)
            raise

        await self._db.update_source_chunks(source_id, count)

        replaced = False
        if existing is not None and existing.id != source_id:
            replaced = True
            try:
                await self._rag.delete_source(existing.id)
            finally:
                await self._db.delete_source(existing.id)

        source = await self._db.get_source(source_id)
        if source is None:  # pragma: no cover - защита от гонки
            raise RuntimeError("Источник не найден после сохранения.")

        logger.info(
            "Источник добавлен: %s (%s фрагментов, заменён=%s)",
            source.title,
            count,
            replaced,
        )
        return IngestResult(source=source, chunks=count, replaced=replaced)

    async def delete(self, source_id: int) -> Source | None:
        source = await self._db.get_source(source_id)
        if source is None:
            return None
        await self._rag.delete_source(source_id)
        await self._db.delete_source(source_id)
        return source

    async def count_chunks(self) -> int:
        return await self._rag.count()

    async def list_sources(self) -> list[Source]:
        return await self._db.list_sources()

    async def get_source(self, source_id: int) -> Source | None:
        return await self._db.get_source(source_id)
