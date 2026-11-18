"""Правила, превращающие оценку новости в запрет на вход, и запись news_state.json.

Правила заданы кодом, а не моделью: модель не может ни продлить запрет дольше
потолка, ни снять чужой запрет, ни разрешить сделку.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from .scorer import Score

DEFAULT_HOURS = {"medium": 6, "high": 24}


def decide(
    score: Score,
    published_at: datetime,
    now: datetime,
    assets: list[str],
    stake: str,
    max_hours: int,
) -> list[tuple[str, datetime]]:
    """Возвращает список (область запрета, до какого времени). Пусто — запрета нет."""
    if not score.relevant or score.direction != "bearish":
        return []
    blocks = (score.severity == "high" and score.confidence in ("medium", "high")) or (
        score.severity == "medium" and score.confidence == "high"
    )
    if not blocks:
        return []

    hours = score.horizon_hours
    if not 1 <= hours <= max_hours:
        hours = DEFAULT_HOURS[score.severity]
    hours = min(hours, max_hours)

    until = min(published_at, now) + timedelta(hours=hours)
    if until <= now:
        return []  # новость старая, срок запрета уже прошёл

    if "MARKET" in score.assets:
        scopes = ["global"]
    else:
        scopes = [f"{asset}/{stake}" for asset in score.assets if asset in assets]
    return [(scope, until) for scope in scopes]


def build_state(active_vetoes: list[dict], now: datetime) -> dict:
    """Собирает содержимое news_state.json из действующих запретов."""
    state = {
        "updated_at": now.isoformat(),
        "global": {"block_entries": False, "until": None, "reason": ""},
        "pairs": {},
    }
    # Для каждой области берём запрет с самым поздним сроком.
    latest: dict[str, dict] = {}
    for veto in active_vetoes:
        current = latest.get(veto["scope"])
        if current is None or veto["until"] > current["until"]:
            latest[veto["scope"]] = veto
    for scope, veto in latest.items():
        entry = {
            "block_entries": True,
            "until": veto["until"],
            "reason": f'{veto["reason"]} ({veto["source"]}: {veto["title"]})',
        }
        if scope == "global":
            state["global"] = entry
        else:
            state["pairs"][scope] = entry
    return state


def write_state(path: Path, state: dict) -> None:
    """Пишет файл атомарно, чтобы стратегия никогда не прочитала его наполовину."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
