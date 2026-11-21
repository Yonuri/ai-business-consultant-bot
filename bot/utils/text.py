"""Работа с текстом: очистка, чанкинг, разбиение длинных сообщений."""

from __future__ import annotations

import html
import re

_WS_RUN = re.compile(r"[ \t]+")
_LINE_EDGES = re.compile(r" *\n *")
_BLANK_LINES = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def clean_text(text: str) -> str:
    """Нормализует пробелы и переводы строк, сохраняя структуру абзацев."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("​", "")
    text = _WS_RUN.sub(" ", text)
    text = _LINE_EDGES.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Режет текст на перекрывающиеся фрагменты по границам абзацев/предложений."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    # Не даём перекрытию «съесть» весь шаг: иначе чанков будет слишком много.
    step_floor = max(1, chunk_size - overlap)

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Ищем «красивую» границу в последних 40% окна.
            window_start = start + int(chunk_size * 0.6)
            for separator in ("\n\n", "\n", ". ", "! ", "? ", "; ", " "):
                cut = text.rfind(separator, window_start, end)
                if cut != -1:
                    end = cut + len(separator)
                    break
        piece = text[start:end].strip()
        if len(piece) >= 20:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + step_floor)
    return chunks


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Делит готовый ответ на части, чтобы уложиться в лимит Telegram."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def esc(value: object) -> str:
    """Экранирует текст для ParseMode.HTML."""
    return html.escape(str(value), quote=False)
