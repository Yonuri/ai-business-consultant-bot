"""Векторный поиск на ChromaDB с эмбеддингами Gemini.

Chroma хранит только фрагменты и векторы; метаданные источников живут в SQLite.
Все синхронные вызовы Chroma выполняются в отдельном потоке, чтобы не блокировать
event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb

from bot.services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source_id: int
    title: str
    chunk_index: int
    text: str
    score: float  # косинусная близость в диапазоне [-1, 1]


class RagEngine:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        gemini: GeminiClient,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._gemini = gemini
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect_sync)
        await self._verify_dimension()
        logger.info("ChromaDB инициализирована: %s", self._persist_dir)

    def _connect_sync(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    async def _verify_dimension(self) -> None:
        """Ловит смену EMBEDDING_DIM: старый индекс с другой размерностью не примет векторы."""
        if await self.count() == 0:
            return
        sample = await asyncio.to_thread(
            self.collection.get, limit=1, include=["embeddings"]
        )
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return
        stored = len(embeddings[0])
        expected = self._gemini.embedding_dim
        if stored != expected:
            raise RuntimeError(
                f"Размерность индекса ({stored}) не совпадает с EMBEDDING_DIM ({expected}). "
                "Верните прежнее значение EMBEDDING_DIM или пересоберите базу знаний "
                "(удалите каталог data/chroma и загрузите документы заново)."
            )

    @property
    def collection(self):
        if self._collection is None:
            raise RuntimeError("Векторная база не инициализирована")
        return self._collection

    async def count(self) -> int:
        return int(await asyncio.to_thread(self.collection.count))

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    async def add_chunks(
        self,
        *,
        source_id: int,
        title: str,
        chunks: Sequence[str],
    ) -> int:
        """Считает эмбеддинги и добавляет фрагменты. Возвращает их количество."""
        if not chunks:
            return 0

        embeddings = await self._gemini.embed_documents(list(chunks))
        if len(embeddings) != len(chunks):
            raise RuntimeError("Количество эмбеддингов не совпало с количеством фрагментов")

        ids = [f"{source_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {"source_id": int(source_id), "title": title, "chunk_index": index}
            for index in range(len(chunks))
        ]

        async with self._write_lock:
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                documents=list(chunks),
                embeddings=embeddings,
                metadatas=metadatas,
            )
        logger.info("Добавлено %s фрагментов (source_id=%s)", len(chunks), source_id)
        return len(chunks)

    async def delete_source(self, source_id: int) -> None:
        async with self._write_lock:
            await asyncio.to_thread(
                self.collection.delete, where={"source_id": int(source_id)}
            )
        logger.info("Удалены фрагменты source_id=%s", source_id)

    async def reset(self) -> None:
        """Полностью пересоздаёт коллекцию (служебная операция)."""
        async with self._write_lock:
            client = self._client
            if client is not None:
                await asyncio.to_thread(client.delete_collection, self._collection_name)
            await asyncio.to_thread(self._connect_sync)
        logger.info("Векторная база пересоздана: %s", self._collection_name)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.2,
    ) -> list[RetrievedChunk]:
        total = await self.count()
        if total == 0:
            return []

        embedding = await self._gemini.embed_query(query)
        n_results = max(1, min(top_k, total))

        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        # strict=False: при рассинхроне списков просто берём общую длину,
        # а не роняем поиск исключением.
        for text, metadata, distance in zip(
            documents, metadatas, distances, strict=False
        ):
            if not text or not metadata:
                continue
            # В пространстве cosine distance = 1 - similarity.
            score = 1.0 - float(distance)
            if score < min_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    source_id=int(metadata.get("source_id", 0)),
                    title=str(metadata.get("title", "Без названия")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=text,
                    score=score,
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
