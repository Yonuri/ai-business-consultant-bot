"""Парсинг документов и веб-страниц в чистый текст."""

from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from bot.utils.text import clean_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

_USER_AGENT = (
    "Mozilla/5.0 (compatible; KnowledgeBot/1.0; +https://example.com/bot) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class ParseError(ValueError):
    """Ошибка извлечения текста; сообщение показывается администратору."""


class EmptyDocumentError(ParseError):
    """Текст не извлечён (например, скан PDF без текстового слоя)."""


class UnsafeUrlError(ParseError):
    """URL не прошёл проверку безопасности (SSRF)."""


@dataclass(slots=True)
class ParsedDocument:
    title: str
    text: str

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 20


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------

def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported(filename: str) -> bool:
    return extension_of(filename) in SUPPORTED_EXTENSIONS


async def parse_file(filename: str, data: bytes) -> ParsedDocument:
    """Извлекает текст из PDF/DOCX/TXT/MD.

    Парсинг CPU-bound, поэтому выполняется в отдельном потоке.
    """
    ext = extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(
            "Неподдерживаемый формат файла. Разрешены: PDF, DOCX, TXT, MD."
        )
    if not data:
        raise EmptyDocumentError("Файл пустой — текст не распознан.")

    return await asyncio.to_thread(_parse_file_sync, filename, data, ext)


def _parse_file_sync(filename: str, data: bytes, ext: str) -> ParsedDocument:
    title = Path(filename).name
    if ext == ".pdf":
        text = _parse_pdf(data)
    elif ext == ".docx":
        text = _parse_docx(data)
    else:
        text = _parse_plaintext(data)

    text = clean_text(text)
    if len(text) < 20:
        raise EmptyDocumentError(
            "Текст в документе не распознан. "
            "Если это скан PDF без текстового слоя, сначала выполните OCR "
            "(например, через Adobe Acrobat или OCRmyPDF) и загрузите файл заново."
        )
    return ParsedDocument(title=title, text=text)


def _parse_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("Библиотека pypdf не установлена.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ParseError("PDF защищён паролем — текст не извлечён.") from exc
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - битая страница не должна ломать весь файл
                logger.debug("Не удалось извлечь текст со страницы PDF", exc_info=True)
        return "\n\n".join(pages)
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Не удалось прочитать PDF: {exc}") from exc


def _parse_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ParseError("Библиотека python-docx не установлена.") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Не удалось прочитать DOCX: {exc}") from exc

    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_plaintext(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Веб-страницы
# ---------------------------------------------------------------------------

_BLOCK_TAGS = (
    "article", "main", "section", "div", "p", "br", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table", "blockquote",
)
_STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "header", "aside")
_BOILERPLATE_HINTS = re.compile(
    r"(cookie|политик|конфиденциальн|подпис|subscribe|newsletter|breadcrumb|"
    r"меню сайта|навигац|корзин|basket|modal|popup|реклам)",
    re.IGNORECASE,
)
_MULTISPACE = re.compile(r"[ \t]{2,}")


def normalize_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        raise ParseError("Пустая ссылка.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("Поддерживаются только ссылки http:// и https://.")
    if not parsed.hostname:
        raise UnsafeUrlError("Не удалось определить домен в ссылке.")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("Ссылки со встроенными логином/паролем запрещены.")
    return url


async def _ensure_public_host(hostname: str) -> None:
    """Защита от SSRF: запрещаем localhost, приватные и служебные адреса."""
    lowered = hostname.lower().strip(".")
    if lowered in {"localhost", "localhost.localdomain", "metadata.google.internal"}:
        raise UnsafeUrlError("Ссылки на локальные адреса запрещены.")

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ParseError("Не удалось разрешить домен ссылки. Проверьте адрес.") from exc

    if not infos:
        raise ParseError("Домен ссылки не найден.")

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeUrlError("Ссылки на внутренние и служебные адреса запрещены.")


async def parse_url(
    raw_url: str,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    timeout: float = 20.0,
) -> ParsedDocument:
    url = normalize_url(raw_url)
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
    }

    response, content = await _fetch_page(url, headers=headers, max_bytes=max_bytes, timeout=timeout)

    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type or "xml" in content_type or not content_type:
        text = _html_to_text(_decode_body(content, response))
    elif "text/plain" in content_type or "markdown" in content_type:
        text = _decode_body(content, response)
    else:
        raise ParseError(
            f"По ссылке не веб-страница (content-type: {content_type or 'неизвестно'}). "
            "Поддерживаются HTML и текстовые страницы."
        )

    text = clean_text(text)
    if len(text) < 50:
        raise EmptyDocumentError(
            "На странице не найден текст. Возможно, контент подгружается скриптами "
            "или страница защищена от парсинга."
        )

    title = _extract_title(_decode_body(content, response)) or urlparse(str(response.url)).hostname or url
    return ParsedDocument(title=title, text=text)


async def _fetch_page(
    url: str,
    *,
    headers: dict[str, str],
    max_bytes: int,
    timeout: float,
    max_redirects: int = 5,
) -> tuple[httpx.Response, bytes]:
    """Загружает страницу, проверяя КАЖДЫЙ адрес цепочки редиректов на SSRF."""
    current = url
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers=headers,
        ) as client:
            for _ in range(max_redirects + 1):
                host = urlparse(current).hostname or ""
                await _ensure_public_host(host)

                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ParseError("Сайт вернул редирект без адреса.")
                        current = str(response.url.join(location))
                        if urlparse(current).scheme not in {"http", "https"}:
                            raise UnsafeUrlError("Редирект ведёт на неподдерживаемый адрес.")
                        continue

                    if response.status_code >= 400:
                        raise ParseError(
                            f"Сайт ответил кодом {response.status_code}. "
                            "Проверьте ссылку или доступность страницы."
                        )

                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise ParseError("Страница слишком большая для обработки.")

                    # Материализуем ответ без тела, чтобы вернуть заголовки и URL.
                    return response, bytes(body)

            raise ParseError("Слишком много перенаправлений (редиректов).")
    except httpx.HTTPError as exc:
        raise ParseError(f"Не удалось загрузить страницу: {exc}") from exc


def _decode_body(content: bytes, response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _html_to_text(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError as exc:  # pragma: no cover
        raise ParseError("Библиотека beautifulsoup4 не установлена.") from exc

    soup = BeautifulSoup(html_text, "html.parser")

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()

    # Убираем типовые блоки «мусора» по class/id. Обходим снимок списка,
    # чтобы не обращаться к уже удалённым узлам.
    for element in list(soup.find_all(True)):
        if element.decomposed:
            continue
        classes = element.get("class") or []
        element_id = element.get("id") or ""
        marker = " ".join([*classes, element_id]).strip()
        if marker and _BOILERPLATE_HINTS.search(marker):
            element.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_after("\n")

    text = root.get_text(separator=" ")
    text = _MULTISPACE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_title(html_text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title[:200] or None
