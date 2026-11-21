"""Конфигурация приложения.

Все настройки читаются из переменных окружения (или файла `.env`).
Обязательные: BOT_TOKEN, GEMINI_API_KEY, ADMIN_IDS.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ADMIN_IDS_SPLIT = re.compile(r"[,;\s]+")

# Признаки незаполненных значений из .env.example: повторяющиеся символы-заглушки.
_PLACEHOLDER_MARKERS = ("xxxxx", "XXXXX", "your_", "YOUR_", "change_me", "CHANGE_ME")

# Google AI Studio выдаёт ключи двух форматов: классический AIza... и новый AQ....
_GEMINI_KEY_PREFIXES = ("AIza", "AQ.")


@lru_cache(maxsize=8)
def parse_admin_ids(raw: str) -> frozenset[int]:
    """Разбирает строку вида "123,456 789" в множество Telegram ID."""
    result: set[int] = set()
    for part in _ADMIN_IDS_SPLIT.split(raw.strip()):
        if not part:
            continue
        candidate = part.lstrip("+")
        if not candidate.isdigit():
            raise ValueError(
                f"Некорректный Telegram ID в ADMIN_IDS: {part!r}. "
                "Ожидается список чисел через запятую, например: 123456789,987654321"
            )
        result.add(int(candidate))
    return frozenset(result)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Обязательные ---
    bot_token: SecretStr = Field(..., min_length=10)
    gemini_api_key: SecretStr = Field(..., min_length=10)
    admin_ids: str = ""

    # --- Gemini ---
    gemini_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(default=1536, ge=1, le=3072)
    embedding_batch_size: int = Field(default=64, ge=1, le=250)
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    gemini_max_output_tokens: int = Field(default=1200, ge=128, le=8192)
    gemini_thinking_budget: int = Field(default=0, ge=-1, le=24576)
    gemini_retry_attempts: int = Field(default=3, ge=1, le=6)

    # --- RAG ---
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)
    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    chroma_collection: str = "company_knowledge"

    # --- История и лимиты ---
    history_limit: int = Field(default=8, ge=0, le=40)
    history_store_limit: int = Field(default=40, ge=8, le=500)
    question_cooldown_seconds: float = Field(default=1.5, ge=0.0, le=30.0)
    questions_per_hour: int = Field(default=40, ge=1, le=1000)
    max_question_chars: int = Field(default=1500, ge=50, le=4000)
    max_answer_chars: int = Field(default=3500, ge=500, le=4096)

    # --- Документы ---
    max_document_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_url_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    http_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)

    # --- Инфраструктура ---
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    timezone: str = "Europe/Moscow"
    # Прокси для запросов к Gemini. Нужен, если регион заблокирован Google.
    https_proxy: str | None = None

    @property
    def admins(self) -> frozenset[int]:
        return parse_admin_ids(self.admin_ids)

    @field_validator("https_proxy", mode="before")
    @classmethod
    def _blank_proxy_to_none(cls, value: object) -> object:
        # Пустое значение в .env (HTTPS_PROXY=) должно означать «без прокси».
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _reject_placeholders(self) -> Settings:
        """Не даёт запуститься с заглушками из .env.example.

        Молчаливый старт с фиктивным токеном выглядит как «бот не отвечает»,
        поэтому падаем сразу и с понятным сообщением.
        """
        problems: list[str] = []
        token = self.bot_token.get_secret_value().strip()
        key = self.gemini_api_key.get_secret_value().strip()

        if any(marker in token for marker in _PLACEHOLDER_MARKERS) or token.count("x") > 5:
            problems.append("BOT_TOKEN выглядит как заглушка — вставьте токен от @BotFather")
        # Google AI Studio выдаёт ключи двух форматов: старые AIza... и новые AQ....
        if not key.startswith(_GEMINI_KEY_PREFIXES):
            problems.append(
                "GEMINI_API_KEY должен начинаться с 'AIza' или 'AQ.' — "
                "возьмите ключ в Google AI Studio"
            )
        elif "XXXX" in key or key.count("X") > 5:
            problems.append(
                "GEMINI_API_KEY выглядит как заглушка — вставьте настоящий ключ из Google AI Studio"
            )
        if not self.admin_ids.strip():
            problems.append("ADMIN_IDS пуст — укажите свой Telegram ID (узнать: @userinfobot)")
        else:
            try:
                parse_admin_ids(self.admin_ids)
            except ValueError as exc:
                problems.append(str(exc))

        if problems:
            raise ValueError(
                "Файл .env заполнен не полностью:\n  - " + "\n  - ".join(problems)
            )
        return self

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "bot.sqlite3"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def is_admin(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.admins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
