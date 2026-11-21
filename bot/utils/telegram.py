"""Вспомогательные функции для Telegram API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from aiogram import Bot
from aiogram.enums import ChatAction, ParseMode
from aiogram.types import Message

from bot.utils.text import esc, split_message

logger = logging.getLogger(__name__)

HTML_SOURCE_FOOTER = "<i>📄 Источник: {sources}</i>"


@contextlib.asynccontextmanager
async def typing(
    bot: Bot,
    chat_id: int,
    action: ChatAction = ChatAction.TYPING,
    interval: float = 4.5,
) -> AsyncIterator[None]:
    """Держит статус «печатает…», пока выполняется блок кода.

    Telegram сбрасывает индикатор примерно через 5 секунд, поэтому действие
    периодически повторяется в фоновой задаче.
    """
    stop = asyncio.Event()

    async def _loop() -> None:
        while not stop.is_set():
            try:
                await bot.send_chat_action(chat_id, action)
            except Exception:  # noqa: BLE001 - индикатор не критичен
                logger.debug("Не удалось отправить chat action", exc_info=True)
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def send_long(
    message: Message,
    text: str,
    limit: int = 4000,
    parse_mode: str | None = None,
) -> None:
    """Отправляет текст, разбивая его на несколько сообщений при необходимости."""
    for part in split_message(text, limit=limit):
        await message.answer(
            part, parse_mode=parse_mode, disable_web_page_preview=True
        )


async def send_answer(message: Message, body: str, sources: list[str]) -> None:
    """Отправляет ответ модели как текст с необязательной сноской об источнике.

    Текст модели экранируется и уходит с ParseMode.HTML: так сноска выделяется
    курсивом, а любые «случайные» теги в ответе Gemini не ломают отправку.
    """
    parts = split_message(body, limit=3500) or [body]
    footer = ""
    if sources:
        footer = "\n\n" + HTML_SOURCE_FOOTER.format(
            sources=esc(", ".join(sources))
        )

    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        await message.answer(
            esc(part) + (footer if is_last else ""),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
