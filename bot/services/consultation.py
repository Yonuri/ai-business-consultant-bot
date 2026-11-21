"""Сервис консультации: поиск по базе знаний + генерация ответа Gemini."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.genai import types

from bot import texts
from bot.config import Settings
from bot.database.db import Database
from bot.services.gemini_client import GeminiClient, GeminiError
from bot.services.rag_engine import RagEngine, RetrievedChunk
from bot.utils.text import truncate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    chunks: list[RetrievedChunk]
    grounded: bool  # True, если ответ построен по найденному контексту

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.title not in seen:
                seen.append(chunk.title)
        return seen[:3]


class ConsultantService:
    def __init__(
        self,
        *,
        db: Database,
        rag: RagEngine,
        gemini: GeminiClient,
        settings: Settings,
    ) -> None:
        self._db = db
        self._rag = rag
        self._gemini = gemini
        self._settings = settings

    async def answer(self, user_id: int, question: str) -> Answer:
        """Основной сценарий: вопрос клиента → ответ по базе знаний."""
        question = question.strip()
        history = await self._db.get_history(user_id, limit=self._settings.history_limit)

        chunks = await self._rag.search(
            question,
            top_k=self._settings.rag_top_k,
            min_similarity=self._settings.rag_min_similarity,
        )

        if not chunks:
            logger.info("Нет релевантных фрагментов для вопроса (user_id=%s)", user_id)
            return Answer(text=texts.NO_ANSWER, chunks=[], grounded=False)

        prompt = texts.user_prompt(self._format_context(chunks), question)
        contents = self._build_contents(history, prompt)

        try:
            result = await self._gemini.generate(
                system_prompt=texts.SYSTEM_PROMPT, contents=contents
            )
        except GeminiError:
            logger.exception("Ошибка Gemini при ответе клиенту")
            raise

        raw = result.text
        if not raw or texts.NO_ANSWER_TOKEN in raw or len(raw) < 3:
            return Answer(text=texts.NO_ANSWER, chunks=chunks, grounded=False)

        # Футер с источниками добавляет хэндлер: он же отвечает за экранирование
        # и разметку при отправке в Telegram.
        return Answer(
            text=truncate(raw, self._settings.max_answer_chars),
            chunks=chunks,
            grounded=True,
        )

    # ------------------------------------------------------------------
    # Тестовый режим администратора (без истории клиента)
    # ------------------------------------------------------------------

    async def test_answer(self, question: str) -> Answer:
        chunks = await self._rag.search(
            question,
            top_k=self._settings.rag_top_k,
            min_similarity=self._settings.rag_min_similarity,
        )
        if not chunks:
            return Answer(text=texts.NO_ANSWER, chunks=[], grounded=False)

        result = await self._gemini.generate(
            system_prompt=texts.SYSTEM_PROMPT,
            contents=texts.user_prompt(self._format_context(chunks), question),
        )
        if not result.text or texts.NO_ANSWER_TOKEN in result.text:
            return Answer(text=texts.NO_ANSWER, chunks=chunks, grounded=False)
        return Answer(
            text=truncate(result.text, self._settings.max_answer_chars),
            chunks=chunks,
            grounded=True,
        )

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"[Фрагмент {index} | источник: {chunk.title}]\n{chunk.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_contents(history, prompt: str) -> list[types.Content]:
        contents: list[types.Content] = []
        for message in history:
            role = "user" if message.role == "user" else "model"
            content = message.content.strip()
            if not content:
                continue
            contents.append(
                types.Content(role=role, parts=[types.Part(text=content)])
            )
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents
