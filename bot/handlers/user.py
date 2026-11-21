"""Клиентские хэндлеры: приветствие, вопросы, очистка истории, заявка менеджеру."""

from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.config import Settings
from bot.database.db import Database
from bot.keyboards.admin import manager_cancel
from bot.keyboards.reply import main_keyboard
from bot.services.consultation import ConsultantService
from bot.services.gemini_client import GeminiError
from bot.states import UserStates
from bot.utils.ratelimit import RateLimiter
from bot.utils.telegram import send_answer, typing
from bot.utils.text import esc

logger = logging.getLogger(__name__)

router = Router(name="user")


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(
    message: Message,
    state: FSMContext,
    db: Database,
) -> None:
    await state.clear()
    user = message.from_user
    if user is not None:
        await db.upsert_user(user.id, user.username, user.first_name)

    await message.answer(
        texts.WELCOME,
        reply_markup=main_keyboard(),
        disable_web_page_preview=True,
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        texts.HELP, reply_markup=main_keyboard(), disable_web_page_preview=True
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    await state.clear()
    text = texts.MANAGER_CANCELLED if current else "Нечего отменять."
    await message.answer(text, reply_markup=main_keyboard())


@router.message(Command("admin"))
async def admin_no_access(message: Message) -> None:
    """Сюда попадают только не-админы: админский роутер подключён первым."""
    await message.answer(texts.NO_ACCESS, reply_markup=main_keyboard())


# ---------------------------------------------------------------------------
# Кнопки постоянной клавиатуры
# ---------------------------------------------------------------------------

@router.message(F.text == texts.BTN_CLEAR_HISTORY)
async def clear_history(message: Message, state: FSMContext, db: Database) -> None:
    await state.clear()
    user = message.from_user
    removed = await db.clear_history(user.id) if user else 0
    logger.info(
        "История очищена (user_id=%s, удалено=%s)", user.id if user else None, removed
    )
    await message.answer(texts.HISTORY_CLEARED, reply_markup=main_keyboard())


@router.message(F.text == texts.BTN_MANAGER)
async def ask_manager(message: Message, state: FSMContext) -> None:
    await state.set_state(UserStates.waiting_manager_request)
    await message.answer(texts.ASK_MANAGER, reply_markup=manager_cancel())


@router.callback_query(F.data == "user:manager:cancel")
async def cancel_manager_cb(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    if callback.message is not None:
        try:
            await callback.message.edit_text(texts.MANAGER_CANCELLED)
        except Exception:  # noqa: BLE001 - сообщение могло устареть
            await callback.message.answer(texts.MANAGER_CANCELLED)
    await callback.answer()


@router.callback_query(F.data == "user:manager:ask")
async def ask_manager_from_cb(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(UserStates.waiting_manager_request)
    if callback.message is not None:
        await callback.message.answer(texts.ASK_MANAGER, reply_markup=manager_cancel())
    else:
        await callback.bot.send_message(
            callback.from_user.id, texts.ASK_MANAGER, reply_markup=manager_cancel()
        )
    await callback.answer()


# ---------------------------------------------------------------------------
# Заявка менеджеру
# ---------------------------------------------------------------------------

@router.message(StateFilter(UserStates.waiting_manager_request))
async def receive_manager_request(
    message: Message,
    state: FSMContext,
    bot: Bot,
    db: Database,
    settings: Settings,
) -> None:
    payload = (message.text or "").strip()
    if not payload:
        await message.answer(texts.MANAGER_NEED_TEXT, reply_markup=manager_cancel())
        return

    user = message.from_user
    if user is None:
        await message.answer(texts.MANAGER_REQUEST_FAILED)
        return

    await state.clear()
    await db.add_manager_request(user.id, user.username, payload)

    # Пользовательский текст экранируется: иначе HTML-разметка в сообщении
    # клиента могла бы сломать отправку админу.
    admin_text = texts.MANAGER_TO_ADMIN.format(
        name=esc(user.full_name),
        username=esc(f" (@{user.username})" if user.username else ""),
        user_id=user.id,
        payload=esc(payload[:3000]),
    )

    delivered = 0
    for admin_id in settings.admins:
        try:
            await bot.send_message(admin_id, admin_text, parse_mode=ParseMode.HTML)
            delivered += 1
        except Exception:  # noqa: BLE001 - админ мог заблокировать бота
            logger.warning(
                "Не удалось доставить заявку админу %s", admin_id, exc_info=True
            )

    if delivered:
        await message.answer(texts.MANAGER_REQUEST_SENT, reply_markup=main_keyboard())
    else:
        logger.error("Заявка не доставлена ни одному админу (user_id=%s)", user.id)
        await message.answer(texts.MANAGER_REQUEST_FAILED, reply_markup=main_keyboard())


# ---------------------------------------------------------------------------
# Вопросы клиента
# ---------------------------------------------------------------------------

@router.message(F.text & ~F.text.startswith("/"))
async def answer_question(
    message: Message,
    state: FSMContext,
    bot: Bot,
    db: Database,
    settings: Settings,
    consultant: ConsultantService,
    rate_limiter: RateLimiter,
) -> None:
    # Клиентский ввод сбрасывает незавершённые состояния (например, заявку).
    await state.clear()

    user = message.from_user
    if user is None:
        return

    question = (message.text or "").strip()
    if not question:
        return

    await db.upsert_user(user.id, user.username, user.first_name)

    if len(question) > settings.max_question_chars:
        await message.answer(
            texts.QUESTION_TOO_LONG.format(limit=settings.max_question_chars)
        )
        return

    if not rate_limiter.allow(user.id):
        await message.answer(texts.RATE_LIMITED)
        return

    await db.log_question(user.id, question)

    try:
        async with typing(bot, message.chat.id, ChatAction.TYPING):
            answer = await consultant.answer(user.id, question)
    except GeminiError:
        logger.exception("Gemini недоступен (user_id=%s)", user.id)
        await message.answer(texts.ERROR_GENERIC, reply_markup=main_keyboard())
        return
    except Exception:  # noqa: BLE001
        logger.exception("Неожиданная ошибка при ответе клиенту (user_id=%s)", user.id)
        await message.answer(texts.ERROR_GENERIC, reply_markup=main_keyboard())
        return

    await db.add_message(user.id, "user", question)
    await db.add_message(user.id, "assistant", answer.text)
    await db.trim_history(user.id, keep=settings.history_store_limit)

    # Ответ модели отправляется как экранированный HTML: Gemini может вернуть
    # символы, которые Telegram воспримет как разметку.
    await send_answer(
        message,
        answer.text,
        answer.sources if answer.grounded else [],
    )

    if not answer.grounded:
        builder = InlineKeyboardBuilder()
        builder.button(text=texts.BTN_MANAGER, callback_data="user:manager:ask")
        await message.answer(
            "Нужна помощь специалиста?",
            reply_markup=builder.as_markup(),
        )


# ---------------------------------------------------------------------------
# Прочие типы сообщений
# ---------------------------------------------------------------------------

@router.message()
async def non_text(message: Message) -> None:
    await message.answer(texts.NON_TEXT_HINT, reply_markup=main_keyboard())
