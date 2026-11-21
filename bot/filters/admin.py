"""Фильтр доступа к админ-функциям."""

from __future__ import annotations

from aiogram.filters import BaseFilter
from aiogram.types import CallbackQuery, Message, TelegramObject

from bot.config import Settings


class IsAdmin(BaseFilter):
    """Пропускает события только от пользователей из ADMIN_IDS."""

    async def __call__(self, event: TelegramObject, settings: Settings) -> bool:
        user = getattr(event, "from_user", None)
        if user is None and isinstance(event, (Message, CallbackQuery)):
            user = event.from_user
        return settings.is_admin(user.id if user else None)
