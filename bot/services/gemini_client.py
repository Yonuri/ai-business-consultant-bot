"""Клиент Gemini через официальный SDK `google-genai`.

Инкапсулирует:
  * генерацию ответа (с retry и отключённым «размышлением» для скорости);
  * получение эмбеддингов батчами с L2-нормализацией (косинус = скалярное произведение).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# httpx2 (бэкенд google-genai по умолчанию) не поддерживает SOCKS-прокси,
# а httpx поддерживает. Поэтому при работе через SOCKS переключаем бэкенд.
_HTTPX_SOCKS_PREFIXES = ("socks5://", "socks5h://", "socks4://", "socks4a://")


def _prepare_http_backend(proxy: str | None) -> None:
    """Выбирает HTTP-бэкенд google-genai до создания клиента."""
    if proxy and proxy.strip().lower().startswith(_HTTPX_SOCKS_PREFIXES):
        try:
            import socksio  # noqa: F401
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для SOCKS-прокси нужен пакет socksio: pip install \"httpx[socks]\""
            ) from exc
        os.environ["GOOGLE_GENAI_HTTP_CLIENT"] = "httpx"


class GeminiError(RuntimeError):
    """Ошибка обращения к Gemini, которую можно показать пользователю."""


def _normalize_proxy(proxy: str) -> str:
    """Приводит адрес прокси к виду, который понимает httpx.

    Пользователь часто пишет «host:port» без схемы, а для SOCKS-прокси
    (например, локальный Hiddify/Clash/v2ray) нужно socks5://.
    """
    value = proxy.strip()
    if "://" in value:
        return value
    host = value.rsplit(":", 1)[0]
    # Локальные прокси-клиенты почти всегда SOCKS5.
    if host in {"127.0.0.1", "localhost", "::1"}:
        return f"socks5://{value}"
    return f"http://{value}"


def _is_proxy_unreachable(exc: Exception) -> bool:
    """Отличает «прокси не запущен» от обычного сетевого сбоя.

    Такой сбой не лечится повтором: если клиент VPN не запущен, все попытки
    упадут одинаково, а пользователь ждёт лишние секунды.
    """
    # aiohttp: ClientProxyConnectionError, httpx: ProxyError/ProxyConnectError.
    if "proxy" in type(exc).__name__.lower():
        return True

    text = str(exc).lower()
    if "proxy" not in text and "10061" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "cannot connect",
            "connection refused",
            "10061",
            "name or service not known",
        )
    )


def explain_api_error(exc: Exception) -> str:
    """Переводит частые ошибки Gemini API в понятное человеку сообщение."""
    text = str(exc)
    lowered = text.lower()

    # Прокси настроен, но до него не достучались: чаще всего клиент не запущен.
    if "proxy" in type(exc).__name__.lower() or (
        "proxy" in lowered
        and (
            "cannot connect" in lowered
            or "connection refused" in lowered
            or "10061" in lowered
        )
    ):
        return (
            "Не удалось подключиться к прокси из HTTPS_PROXY. Проверьте, что "
            "VPN/прокси-клиент запущен и слушает указанный порт "
            "(python check_proxy.py покажет точную причину)."
        )
    if "location is not supported" in lowered:
        return (
            "Gemini API недоступен из вашего региона. Включите VPN или укажите "
            "HTTPS_PROXY в .env — запросы к Google пойдут через прокси."
        )
    if "no longer available" in lowered or "not found for api version" in lowered:
        model = "модель"
        match = re.search(r"models/([\w.\-]+)", text)
        if match:
            model = f"модель «{match.group(1)}»"
        return (
            f"Указанная {model} недоступна для вашего ключа. Обновите GEMINI_MODEL / "
            "EMBEDDING_MODEL в .env на актуальные значения."
        )
    if "api key not valid" in lowered or "api_key_invalid" in lowered:
        return "GEMINI_API_KEY недействителен. Проверьте ключ в Google AI Studio."
    if "quota" in lowered or "resource_exhausted" in lowered:
        return "Исчерпана квота Gemini API. Проверьте лимиты в Google Cloud или подождите."
    if "permission" in lowered or "403" in lowered:
        return "У ключа нет доступа к этой модели. Проверьте права ключа в Google AI Studio."
    return "Не удалось получить ответ от Gemini."


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedding_model: str,
        embedding_dim: int,
        embedding_batch_size: int = 64,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        thinking_budget: int = 0,
        retry_attempts: int = 3,
        proxy: str | None = None,
    ) -> None:
        # Прокси нужен, когда регион заблокирован для Gemini API (ошибка
        # "User location is not supported"). Передаётся только в async-клиент:
        # синхронные вызовы не используются.
        http_options = (
            types.HttpOptions(async_client_args={"proxy": _normalize_proxy(proxy)})
            if proxy
            else None
        )
        _prepare_http_backend(_normalize_proxy(proxy) if proxy else None)
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._batch_size = max(1, embedding_batch_size)
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget
        self._retry_attempts = max(1, retry_attempts)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def healthcheck(self) -> tuple[bool, str]:
        """Проверяет, что ключ и модели действительно работают.

        Возвращает (успех, сообщение). Вызывается при старте, чтобы проблемы
        с ключом/регионом/квотой были видны сразу, а не при первом вопросе клиента.
        """
        try:
            await self._embed_batch(["проверка"], task_type="RETRIEVAL_QUERY")
        except GeminiError as exc:
            return False, f"Эмбеддинги ({self._embedding_model}): {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Эмбеддинги ({self._embedding_model}): {exc}"

        try:
            await self.generate(
                system_prompt="Ответь одним словом.",
                contents="Проверка связи. Ответь: ок",
            )
        except GeminiError as exc:
            return False, f"Генерация ({self._model}): {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Генерация ({self._model}): {exc}"

        return True, "Gemini API доступен"

    # ------------------------------------------------------------------
    # Генерация ответа
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        system_prompt: str,
        contents: Sequence[types.Content] | str,
    ) -> GenerationResult:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=self._thinking_budget),
        )

        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                text = (response.text or "").strip()
                usage = getattr(response, "usage_metadata", None)
                return GenerationResult(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                )
            except genai_errors.APIError as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                logger.warning(
                    "Gemini API error (attempt %s/%s, status=%s): %s",
                    attempt,
                    self._retry_attempts,
                    status,
                    exc,
                )
                # Ошибки авторизации/валидации повторять бессмысленно.
                if status is not None and status not in _RETRYABLE_STATUS:
                    raise GeminiError(explain_api_error(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - сеть, таймауты, TLS
                last_error = exc
                logger.warning(
                    "Gemini request failed (attempt %s/%s): %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                # Мёртвый прокси не оживёт от повторов — сообщаем сразу.
                if _is_proxy_unreachable(exc):
                    raise GeminiError(explain_api_error(exc)) from exc
            if attempt < self._retry_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise GeminiError(explain_api_error(last_error) if last_error else "Ошибка Gemini") from last_error

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    async def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = await self._client.aio.models.embed_content(
                    model=self._embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._embedding_dim,
                    ),
                )
                embeddings = list(response.embeddings or [])
                if len(embeddings) != len(texts):
                    raise GeminiError(
                        f"Gemini вернул {len(embeddings)} эмбеддингов вместо {len(texts)}"
                    )
                return [self._normalize(e.values or []) for e in embeddings]
            except GeminiError:
                raise
            except genai_errors.APIError as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                logger.warning(
                    "Embedding API error (attempt %s/%s, status=%s): %s",
                    attempt,
                    self._retry_attempts,
                    status,
                    exc,
                )
                if status is not None and status not in _RETRYABLE_STATUS:
                    raise GeminiError(explain_api_error(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Embedding failed (attempt %s/%s): %s", attempt, self._retry_attempts, exc
                )
                if _is_proxy_unreachable(exc):
                    raise GeminiError(explain_api_error(exc)) from exc
            if attempt < self._retry_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise GeminiError(
            explain_api_error(last_error) if last_error else "Ошибка эмбеддингов Gemini"
        ) from last_error

    @staticmethod
    def _normalize(values: Sequence[float]) -> list[float]:
        vector = [float(v) for v in values]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]
