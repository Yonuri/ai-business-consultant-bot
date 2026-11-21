"""Точка входа: инициализация сервисов и запуск long polling."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import suppress

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand
from pydantic import ValidationError

from bot import texts
from bot.config import Settings, get_settings
from bot.database.db import Database
from bot.handlers import admin, user
from bot.services.consultation import ConsultantService
from bot.services.gemini_client import GeminiClient
from bot.services.knowledge_service import KnowledgeService
from bot.services.rag_engine import RagEngine
from bot.utils.logging import setup_logging
from bot.utils.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

BOT_COMMANDS = [
    BotCommand(command="start", description="Начать диалог"),
    BotCommand(command="help", description="Как пользоваться ботом"),
    BotCommand(command="admin", description="Панель администратора"),
]


async def build_dispatcher(settings: Settings) -> tuple[Dispatcher, Bot, Database]:
    db = Database(settings.sqlite_path)
    await db.connect()

    gemini = GeminiClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        embedding_batch_size=settings.embedding_batch_size,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
        thinking_budget=settings.gemini_thinking_budget,
        retry_attempts=settings.gemini_retry_attempts,
        proxy=settings.https_proxy,
    )

    rag = RagEngine(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        gemini=gemini,
    )
    await rag.connect()

    knowledge = KnowledgeService(db=db, rag=rag, settings=settings)
    consultant = ConsultantService(db=db, rag=rag, gemini=gemini, settings=settings)

    bot = Bot(
        token=settings.bot_token.get_secret_value(),
        session=AiohttpSession(timeout=60),
        # parse_mode не задаём глобально: клиентские ответы модели уходят
        # обычным текстом, а админские сообщения включают HTML явно.
        default=DefaultBotProperties(parse_mode=None),
    )

    # Всё, что передано в Dispatcher(...), доступно хэндлерам по имени аргумента.
    dp = Dispatcher(
        storage=MemoryStorage(),
        settings=settings,
        db=db,
        gemini=gemini,
        rag=rag,
        knowledge=knowledge,
        consultant=consultant,
        rate_limiter=RateLimiter(
            settings.question_cooldown_seconds, settings.questions_per_hour
        ),
    )

    # Порядок важен: админский роутер первым, чтобы его состояния и /admin
    # не перехватывались клиентским catch-all.
    dp.include_router(admin.router)
    dp.include_router(user.router)

    return dp, bot, db


async def on_startup(
    bot: Bot,
    settings: Settings,
    db: Database,
    knowledge: KnowledgeService,
    gemini: GeminiClient,
) -> None:
    me = await bot.get_me()
    logger.info("Бот запущен: @%s (id=%s)", me.username, me.id)
    await bot.set_my_commands(BOT_COMMANDS)

    if not settings.admins:
        logger.warning(
            "ADMIN_IDS пуст: админ-панель (/admin) недоступна, "
            "заявки менеджеру не доставляются."
        )

    ok, message = await gemini.healthcheck()
    if ok:
        logger.info("✅ %s", message)
    else:
        # Не останавливаем бота: он должен принимать документы и заявки даже
        # при проблемах с API, но владелец должен сразу увидеть причину.
        logger.error("❌ Gemini недоступен — бот не сможет отвечать на вопросы.")
        logger.error("   %s", message)

    chunks = await knowledge.count_chunks()
    sources = await db.count_sources()
    logger.info("База знаний: источников=%s, фрагментов=%s", sources, chunks)
    if chunks == 0:
        logger.warning(texts.ADMIN_STARTUP_NO_SOURCES)


async def run() -> None:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()

    dp, bot, db = await build_dispatcher(settings)

    try:
        await on_startup(bot, settings, db, dp["knowledge"], dp["gemini"])
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        logger.info("Остановка бота…")
        with suppress(Exception):
            await bot.session.close()
        await db.close()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Завершение по сигналу")
    except ValidationError as exc:
        # Самая частая ошибка — незаполненный .env. Печатаем её без трейсбека.
        print("\n❌ Не удалось запустить бота: настройки заполнены неверно.\n")
        for error in exc.errors():
            message = error.get("msg", "").removeprefix("Value error, ")
            print(f"  • {message}")
        print("\nОткройте файл .env и заполните обязательные переменные.")
        print("Инструкция: раздел «Где взять ключи» в README.md\n")
        sys.exit(1)
    except Exception:
        logger.exception("Критическая ошибка при запуске")
        sys.exit(1)


if __name__ == "__main__":
    main()
