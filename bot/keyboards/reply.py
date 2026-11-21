"""Постоянная Reply-клавиатура клиента."""

from __future__ import annotations

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from bot import texts


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=texts.BTN_CLEAR_HISTORY),
                KeyboardButton(text=texts.BTN_MANAGER),
            ]
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder=texts.INPUT_PLACEHOLDER,
    )
