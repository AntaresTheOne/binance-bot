"""Сбор новостей из RSS-лент."""

import calendar
import hashlib
import html
import logging
import re
from datetime import datetime, timezone

import feedparser
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; news-reader/1.0)"
# Аннотация из ленты ограничена: длинный текст не меняет оценку заголовка, а запросы дорожают.
SUMMARY_LIMIT = 1000

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value or ""))).strip()


def _published(entry, fallback: datetime) -> datetime:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return fallback
    published = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    # Дата из будущего — ошибка ленты; считаем, что новость вышла сейчас.
    return min(published, fallback)


def parse_feed(source: str, content: bytes, now: datetime) -> list[dict]:
    items = []
    for entry in feedparser.parse(content).entries:
        title = clean_text(entry.get("title", ""))
        url = entry.get("link", "")
        if not title:
            continue
        key = url or f"{source}:{title}"
        items.append(
            {
                "id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
                "source": source,
                "title": title,
                "summary": clean_text(entry.get("summary", ""))[:SUMMARY_LIMIT],
                "url": url,
                "published_at": _published(entry, now),
            }
        )
    return items


def fetch_all(feeds: dict[str, str], timeout: float, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    items: list[dict] = []
    for source, url in feeds.items():
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Лента %s недоступна: %s", source, exc)
            continue
        parsed = parse_feed(source, response.content, now)
        logger.info("%s: получено %d записей", source, len(parsed))
        items.extend(parsed)
    return items
