"""Проверка прокси и доступа к Gemini.

Запуск:
    python check_proxy.py                  # проверить текущие настройки из .env
    python check_proxy.py host:port        # проверить конкретный прокси
    python check_proxy.py socks5://host:port

Скрипт показывает:
  * какой IP и страна видны наружу (и через прокси, и напрямую);
  * работают ли генерация и эмбеддинги Gemini.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from urllib.parse import urlparse

import httpx

from bot.config import get_settings
from bot.services.gemini_client import GeminiClient, _normalize_proxy

TRACE_URL = "https://www.cloudflare.com/cdn-cgi/trace"


def _parse_trace(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def probe_port(proxy: str) -> tuple[bool, str]:
    """Проверяет, слушает ли кто-то порт прокси. Отвечает без сетевых запросов."""
    parsed = urlparse(proxy)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False, "в адресе прокси нет хоста или порта"

    try:
        with socket.create_connection((host, port), timeout=3):
            return True, f"порт {host}:{port} открыт"
    except ConnectionRefusedError:
        return False, f"порт {host}:{port} закрыт — клиент прокси не запущен"
    except TimeoutError:
        return False, f"{host}:{port} не отвечает (таймаут)"
    except OSError as exc:
        return False, f"{host}:{port} недоступен: {exc.strerror or exc}"


def show_egress(label: str, proxy: str | None) -> str | None:
    """Печатает IP/страну выхода. Возвращает страну или None при ошибке."""
    try:
        kwargs = {"proxy": proxy} if proxy else {}
        response = httpx.get(TRACE_URL, timeout=20, **kwargs)
        data = _parse_trace(response.text)
        country = data.get("loc", "?")
        print(f"{label}: IP {data.get('ip', '?')} | страна {country} | colo {data.get('colo', '?')}")
        return country
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: не удалось подключиться — {type(exc).__name__}: {exc}")
        return None


async def check_gemini(proxy: str | None) -> bool:
    settings = get_settings()
    client = GeminiClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        retry_attempts=1,
        proxy=proxy,
    )
    ok, message = await client.healthcheck()
    print(f"Gemini: {'✅ ' if ok else '❌ '}{message}")
    return ok


def main() -> int:
    settings = get_settings()
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("Проверка доступа к Gemini API")
    print("=" * 70)

    show_egress("Напрямую", None)

    proxy = _normalize_proxy(arg) if arg else settings.https_proxy
    if proxy:
        print(f"\nПрокси из {'аргумента' if arg else '.env'}: {proxy}")
        listening, detail = probe_port(proxy)
        print(f"  порт: {'✅' if listening else '❌'} {detail}")
        if not listening:
            print("\n  Прокси-клиент не запущен. Что сделать:")
            print("    1. Запустите VPN/прокси-клиент (Hiddify, Clash, v2ray, Nekoray)")
            print("       и включите в нём режим SOCKS5 / «разрешить подключения».")
            print("    2. Узнайте его порт (обычно 1080, 7890 или 2080) и укажите в .env:")
            print("         HTTPS_PROXY=socks5://127.0.0.1:7890")
            print("    3. Проверьте снова: python check_proxy.py")
            print("\n  Либо укажите платный HTTP-прокси с выходом не из РФ:")
            print("         HTTPS_PROXY=http://user:pass@host:port")
            return 1

        country = show_egress("Через прокси", proxy)
        if country in {"RU", "BY", "IR", "KP"}:
            print(f"\n⚠️  Выход через страну {country}: Gemini её блокирует.")
            print("   Выберите сервер в другой стране (ЕС, США, Казахстан, Турция и т.п.).")
    else:
        print("\nПрокси не задан (HTTPS_PROXY пуст) — проверяю прямой доступ.")

    print()
    ok = asyncio.run(check_gemini(proxy))

    print("\n" + "=" * 70)
    if ok:
        print("✅ Всё готово — можно запускать: python main.py")
        return 0
    print("❌ Gemini недоступен. Что делать:")
    print("   1. Включите VPN/прокси с выходом НЕ в России и укажите его в .env:")
    print("        HTTPS_PROXY=socks5://127.0.0.1:1080")
    print("        HTTPS_PROXY=http://user:pass@host:port")
    print("   2. Проверьте: python check_proxy.py socks5://127.0.0.1:1080")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
