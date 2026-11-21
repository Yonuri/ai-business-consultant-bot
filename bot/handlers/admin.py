"""Админ-панель внутри Telegram."""

from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from bot import texts
from bot.config import Settings
from bot.database.db import Database
from bot.filters.admin import IsAdmin
from bot.keyboards import admin as kb
from bot.services.consultation import ConsultantService
from bot.services.document_parser import (
    EmptyDocumentError,
    ParseError,
    UnsafeUrlError,
    is_supported,
    parse_file,
    parse_url,
)
from bot.services.gemini_client import GeminiError
from bot.services.knowledge_service import KnowledgeService
from bot.states import AdminStates
from bot.utils.telegram import send_long, typing
from bot.utils.text import esc, truncate
from bot.utils.time import format_dt

logger = logging.getLogger(__name__)

router = Router(name="admin")
router.message.filter(IsAdmin())
router.callback_query.filter(IsAdmin())

KB_PAGE_SIZE = 8
HTML = ParseMode.HTML


def _mb(value: int) -> str:
    return f"{value / (1024 * 1024):.0f}"


# ---------------------------------------------------------------------------
# Вход и меню
# ---------------------------------------------------------------------------

@router.message(Command("admin"))
async def cmd_admin(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        texts.ADMIN_MENU,
        reply_markup=kb.admin_menu(),
        parse_mode=HTML,
        disable_web_page_preview=True,
    )


@router.callback_query(F.data == kb.CB_MENU)
async def show_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await _safe_edit(callback, texts.ADMIN_MENU, kb.admin_menu())
    await callback.answer()


@router.callback_query(F.data == kb.CB_CLOSE)
async def close_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await _safe_edit(callback, "Меню закрыто. Откройте снова командой /admin.")
    await callback.answer()


@router.callback_query(F.data == kb.CB_CANCEL)
async def cancel_action(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await _safe_edit(callback, texts.ADMIN_CANCELLED, kb.admin_menu())
    await callback.answer()


@router.callback_query(F.data == "admin:noop")
async def noop(callback: CallbackQuery) -> None:
    await callback.answer()


# ---------------------------------------------------------------------------
# Добавление документа
# ---------------------------------------------------------------------------

@router.callback_query(F.data == kb.CB_ADD_FILE)
async def ask_file(callback: CallbackQuery, state: FSMContext, settings: Settings) -> None:
    await state.set_state(AdminStates.waiting_document)
    await _safe_edit(
        callback,
        texts.ADMIN_ASK_FILE.format(limit_mb=_mb(settings.max_document_bytes)),
        kb.cancel_keyboard(),
    )
    await callback.answer()


@router.message(StateFilter(AdminStates.waiting_document), F.document)
async def receive_document(
    message: Message,
    state: FSMContext,
    bot: Bot,
    settings: Settings,
    knowledge: KnowledgeService,
) -> None:
    document = message.document
    if document is None:
        return

    filename = document.file_name or "document"
    if not is_supported(filename):
        await message.answer(texts.ADMIN_UNSUPPORTED, reply_markup=kb.cancel_keyboard())
        return

    if document.file_size and document.file_size > settings.max_document_bytes:
        await message.answer(
            texts.ADMIN_FILE_TOO_BIG.format(
                size_mb=document.file_size / (1024 * 1024),
                limit_mb=_mb(settings.max_document_bytes),
            ),
            reply_markup=kb.cancel_keyboard(),
        )
        return

    admin_id = message.from_user.id if message.from_user else 0

    try:
        await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_DOCUMENT)
        downloaded = await bot.download(document)
        data = downloaded.read() if downloaded is not None else b""
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось скачать документ из Telegram")
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error="Не удалось скачать файл из Telegram."),
            reply_markup=kb.cancel_keyboard(),
        )
        return

    if len(data) > settings.max_document_bytes:
        await message.answer(
            texts.ADMIN_FILE_TOO_BIG.format(
                size_mb=len(data) / (1024 * 1024),
                limit_mb=_mb(settings.max_document_bytes),
            ),
            reply_markup=kb.cancel_keyboard(),
        )
        return

    try:
        parsed = await parse_file(filename, data)
    except EmptyDocumentError as exc:
        await message.answer(
            f"⚠️ {esc(exc)}", reply_markup=kb.cancel_keyboard(), parse_mode=HTML
        )
        return
    except ParseError as exc:
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error=esc(exc)),
            reply_markup=kb.cancel_keyboard(),
            parse_mode=HTML,
        )
        return

    async with typing(bot, message.chat.id, ChatAction.TYPING):
        try:
            result = await knowledge.ingest(
                kind="file",
                title=parsed.title,
                location=f"file:{parsed.title.lower()}",
                text=parsed.text,
                added_by=admin_id,
            )
        except GeminiError:
            logger.exception("Ошибка эмбеддингов при загрузке файла")
            await message.answer(
                texts.ADMIN_INGEST_ERROR.format(
                    error="Gemini API недоступен. Проверьте GEMINI_API_KEY и квоты."
                ),
                reply_markup=kb.admin_menu(),
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка индексации файла")
            await message.answer(
                texts.ADMIN_INGEST_ERROR.format(error=esc(str(exc))),
                reply_markup=kb.admin_menu(),
                parse_mode=HTML,
            )
            return

    await state.clear()
    text = texts.ADMIN_FILE_ADDED.format(
        chunks=result.chunks, title=esc(result.source.title)
    )
    if result.replaced:
        text += texts.ADMIN_SOURCE_REPLACED
    await message.answer(text, reply_markup=kb.admin_menu(), parse_mode=HTML)


@router.message(StateFilter(AdminStates.waiting_document))
async def wrong_document_input(message: Message) -> None:
    await message.answer(
        "Пожалуйста, отправьте файл как документ (PDF, DOCX, TXT или MD) "
        "или нажмите «Отмена».",
        reply_markup=kb.cancel_keyboard(),
    )


# ---------------------------------------------------------------------------
# Добавление ссылки
# ---------------------------------------------------------------------------

@router.callback_query(F.data == kb.CB_ADD_URL)
async def ask_url(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_url)
    await _safe_edit(callback, texts.ADMIN_ASK_URL, kb.cancel_keyboard())
    await callback.answer()


@router.message(StateFilter(AdminStates.waiting_url), F.text)
async def receive_url(
    message: Message,
    state: FSMContext,
    bot: Bot,
    settings: Settings,
    knowledge: KnowledgeService,
) -> None:
    raw = (message.text or "").strip()
    if raw.startswith("/"):
        await message.answer(
            "Отправьте ссылку на страницу, например: https://example.com/faq"
        )
        return

    admin_id = message.from_user.id if message.from_user else 0

    try:
        async with typing(bot, message.chat.id, ChatAction.TYPING):
            parsed = await parse_url(
                raw,
                max_bytes=settings.max_url_bytes,
                timeout=settings.http_timeout_seconds,
            )
            result = await knowledge.ingest(
                kind="url",
                title=parsed.title,
                location=raw if "://" in raw else f"https://{raw}",
                text=parsed.text,
                added_by=admin_id,
            )
    except UnsafeUrlError as exc:
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error=esc(exc)),
            reply_markup=kb.cancel_keyboard(),
            parse_mode=HTML,
        )
        return
    except EmptyDocumentError as exc:
        await message.answer(
            f"⚠️ {esc(exc)}", reply_markup=kb.cancel_keyboard(), parse_mode=HTML
        )
        return
    except ParseError as exc:
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error=esc(exc)),
            reply_markup=kb.cancel_keyboard(),
            parse_mode=HTML,
        )
        return
    except GeminiError:
        logger.exception("Ошибка эмбеддингов при индексации URL")
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(
                error="Gemini API недоступен. Проверьте GEMINI_API_KEY и квоты."
            ),
            reply_markup=kb.admin_menu(),
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка индексации URL")
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error=esc(str(exc))),
            reply_markup=kb.admin_menu(),
            parse_mode=HTML,
        )
        return

    await state.clear()
    text = texts.ADMIN_URL_ADDED.format(
        chunks=result.chunks,
        title=esc(result.source.title),
        location=esc(truncate(result.source.location, 120)),
    )
    if result.replaced:
        text += texts.ADMIN_SOURCE_REPLACED
    await message.answer(text, reply_markup=kb.admin_menu(), parse_mode=HTML)


# ---------------------------------------------------------------------------
# Управление базой знаний
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("admin:kb:"))
async def show_kb(
    callback: CallbackQuery,
    state: FSMContext,
    knowledge: KnowledgeService,
    settings: Settings,
) -> None:
    await state.clear()
    page = _parse_page(callback.data, 0)
    sources = await knowledge.list_sources()
    chunks = await knowledge.count_chunks()

    if not sources:
        await _safe_edit(callback, texts.ADMIN_KB_EMPTY, kb.back_to_menu())
        await callback.answer()
        return

    markup, page = kb.source_list(sources, page, KB_PAGE_SIZE)
    lines = [texts.ADMIN_KB_TITLE.format(sources=len(sources), chunks=chunks), ""]

    start = page * KB_PAGE_SIZE
    for index, source in enumerate(sources[start : start + KB_PAGE_SIZE], start=start + 1):
        template = texts.ADMIN_KB_LINE_URL if source.is_url else texts.ADMIN_KB_LINE_FILE
        lines.append(
            template.format(
                index=index,
                title=esc(truncate(source.title, 60)),
                chunks=source.chunks,
                date=format_dt(source.created_at, settings.timezone),
            )
        )

    pages = (len(sources) + KB_PAGE_SIZE - 1) // KB_PAGE_SIZE
    if pages > 1:
        lines.append(texts.ADMIN_KB_PAGE.format(page=page + 1, pages=pages))

    lines.append("\nНажмите на источник, чтобы удалить его.")
    await _safe_edit(callback, "\n".join(lines), markup)
    await callback.answer()


@router.callback_query(F.data.startswith("admin:source:del:"))
async def confirm_delete(
    callback: CallbackQuery,
    knowledge: KnowledgeService,
) -> None:
    parts = (callback.data or "").split(":")
    try:
        source_id = int(parts[3])
        page = int(parts[4]) if len(parts) > 4 else 0
    except (IndexError, ValueError):
        await callback.answer(texts.ADMIN_DELETE_NOT_FOUND, show_alert=True)
        return

    source = await knowledge.get_source(source_id)
    if source is None:
        await callback.answer(texts.ADMIN_DELETE_NOT_FOUND, show_alert=True)
        return

    await _safe_edit(
        callback,
        texts.ADMIN_DELETE_CONFIRM.format(
            icon=source.icon,
            title=esc(source.title),
            chunks=source.chunks,
        ),
        kb.delete_confirm(source, page),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("admin:source:delok:"))
async def delete_source(callback: CallbackQuery, knowledge: KnowledgeService) -> None:
    parts = (callback.data or "").split(":")
    try:
        source_id = int(parts[3])
    except (IndexError, ValueError):
        await callback.answer(texts.ADMIN_DELETE_NOT_FOUND, show_alert=True)
        return

    try:
        source = await knowledge.delete(source_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка удаления источника %s", source_id)
        await callback.answer(
            texts.ADMIN_DELETE_FAILED.format(error=str(exc))[:190], show_alert=True
        )
        return

    if source is None:
        await callback.answer(texts.ADMIN_DELETE_NOT_FOUND, show_alert=True)
        return

    await _safe_edit(
        callback,
        texts.ADMIN_DELETE_DONE.format(title=esc(source.title)),
        kb.back_to_menu(),
    )
    await callback.answer("Удалено")


# ---------------------------------------------------------------------------
# Тест режима клиента
# ---------------------------------------------------------------------------

@router.callback_query(F.data == kb.CB_TEST)
async def ask_test_question(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_test_question)
    await _safe_edit(callback, texts.ADMIN_ASK_TEST, kb.cancel_keyboard())
    await callback.answer()


@router.message(StateFilter(AdminStates.waiting_test_question), F.text)
async def run_test(
    message: Message,
    state: FSMContext,
    bot: Bot,
    consultant: ConsultantService,
    settings: Settings,
) -> None:
    question = (message.text or "").strip()
    if not question:
        return

    try:
        async with typing(bot, message.chat.id, ChatAction.TYPING):
            answer = await consultant.test_answer(question)
    except GeminiError:
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(
                error="Gemini API недоступен. Проверьте GEMINI_API_KEY и квоты."
            ),
            reply_markup=kb.admin_menu(),
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка тестового режима")
        await message.answer(
            texts.ADMIN_INGEST_ERROR.format(error=esc(str(exc))),
            reply_markup=kb.admin_menu(),
            parse_mode=HTML,
        )
        return

    await state.clear()

    if answer.chunks:
        chunk_blocks = "".join(
            texts.ADMIN_TEST_CHUNK.format(
                index=index,
                score=chunk.score,
                title=esc(chunk.title),
                preview=esc(truncate(chunk.text.replace("\n", " "), 220)),
            )
            for index, chunk in enumerate(answer.chunks, start=1)
        )
    else:
        chunk_blocks = texts.ADMIN_TEST_NO_CHUNKS

    result = texts.ADMIN_TEST_RESULT.format(
        answer=esc(answer.text),
        count=len(answer.chunks),
        chunks=chunk_blocks,
    )
    await send_long(message, result, limit=3900, parse_mode=HTML)
    await message.answer(
        f"Минимальный порог релевантности: {settings.rag_min_similarity:.2f}. "
        "В меню — /admin.",
        reply_markup=kb.back_to_menu(),
    )


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

@router.callback_query(F.data == kb.CB_STATS)
async def show_stats(
    callback: CallbackQuery,
    db: Database,
    knowledge: KnowledgeService,
) -> None:
    chunks = await knowledge.count_chunks()
    stats = await db.stats(chunks=chunks)
    await _safe_edit(
        callback,
        texts.ADMIN_STATS.format(
            users=stats.users,
            questions=stats.questions,
            sources=stats.sources,
            chunks=stats.chunks,
            manager_requests=stats.manager_requests,
        ),
        kb.back_to_menu(),
    )
    await callback.answer()


# ---------------------------------------------------------------------------
# Служебное
# ---------------------------------------------------------------------------

def _parse_page(data: str | None, default: int = 0) -> int:
    if not data:
        return default
    try:
        return max(0, int(data.rsplit(":", 1)[-1]))
    except ValueError:
        return default


async def _safe_edit(
    callback: CallbackQuery,
    text: str,
    markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Редактирует сообщение, игнорируя «message is not modified» и устаревшие сообщения."""
    message = callback.message
    if message is None:
        await callback.bot.send_message(
            callback.from_user.id,
            text,
            reply_markup=markup,
            parse_mode=HTML,
            disable_web_page_preview=True,
        )
        return

    try:
        await message.edit_text(
            text,
            reply_markup=markup,
            parse_mode=HTML,
            disable_web_page_preview=True,
        )
    except TelegramBadRequest as exc:
        error = str(exc)
        if "message is not modified" in error:
            return
        logger.info("Редактирование недоступно (%s), отправляю новое сообщение", error)
        await callback.bot.send_message(
            callback.from_user.id,
            text,
            reply_markup=markup,
            parse_mode=HTML,
            disable_web_page_preview=True,
        )
