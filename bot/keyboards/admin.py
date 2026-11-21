"""Inline-клавиатуры админ-панели и клиентских сценариев."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.database.models import Source
from bot.utils.text import truncate

# Callback-данные
CB_MENU = "admin:menu"
CB_ADD_FILE = "admin:add_file"
CB_ADD_URL = "admin:add_url"
CB_KB = "admin:kb:0"
CB_TEST = "admin:test"
CB_STATS = "admin:stats"
CB_CLOSE = "admin:close"
CB_CANCEL = "admin:cancel"


def admin_menu() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text=texts.ADMIN_MENU_BTN_ADD_FILE, callback_data=CB_ADD_FILE)
    builder.button(text=texts.ADMIN_MENU_BTN_ADD_URL, callback_data=CB_ADD_URL)
    builder.button(text=texts.ADMIN_MENU_BTN_KB, callback_data=CB_KB)
    builder.button(text=texts.ADMIN_MENU_BTN_TEST, callback_data=CB_TEST)
    builder.button(text=texts.ADMIN_MENU_BTN_STATS, callback_data=CB_STATS)
    builder.button(text=texts.ADMIN_MENU_BTN_CLOSE, callback_data=CB_CLOSE)
    builder.adjust(1)
    return builder.as_markup()


def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ В меню", callback_data=CB_MENU)]
        ]
    )


def cancel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✖️ Отмена", callback_data=CB_CANCEL)]
        ]
    )


def source_list(
    sources: list[Source],
    page: int,
    page_size: int,
) -> tuple[InlineKeyboardMarkup, int]:
    """Собирает клавиатуру со списком источников и кнопками удаления."""
    pages = max(1, (len(sources) + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    start = page * page_size
    visible = sources[start : start + page_size]

    rows: list[list[InlineKeyboardButton]] = []
    for index, source in enumerate(visible, start=start + 1):
        rows.append(
            [
                InlineKeyboardButton(
                    text=texts.ADMIN_KB_DELETE_BTN.format(
                        index=index, title=truncate(source.title, 34)
                    ),
                    callback_data=f"admin:source:del:{source.id}:{page}",
                )
            ]
        )

    if pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(text="⬅️", callback_data=f"admin:kb:{page - 1}")
            )
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="admin:noop"))
        if page < pages - 1:
            nav.append(
                InlineKeyboardButton(text="➡️", callback_data=f"admin:kb:{page + 1}")
            )
        rows.append(nav)

    rows.append([InlineKeyboardButton(text="⬅️ В меню", callback_data=CB_MENU)])
    return InlineKeyboardMarkup(inline_keyboard=rows), page


def delete_confirm(source: Source, page: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🗑 Да, удалить",
                    callback_data=f"admin:source:delok:{source.id}:{page}",
                ),
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=f"admin:kb:{page}",
                ),
            ]
        ]
    )


def manager_cancel() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✖️ Отмена", callback_data="user:manager:cancel")]
        ]
    )
