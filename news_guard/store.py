"""Хранилище новостей и запретов в SQLite. Журнал оценок пригодится для проверки качества модуля."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    scored_at    TEXT,
    model        TEXT,
    score_json   TEXT
);
CREATE TABLE IF NOT EXISTS vetoes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    TEXT NOT NULL REFERENCES items(id),
    scope      TEXT NOT NULL,
    until      TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def add_items(self, items: list[dict], now: datetime | None = None) -> int:
        """Добавляет новости, пропуская уже известные. Возвращает число новых."""
        now = now or utcnow()
        added = 0
        for item in items:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO items (id, source, title, summary, url, published_at, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item["id"],
                    item["source"],
                    item["title"],
                    item["summary"],
                    item["url"],
                    item["published_at"].isoformat(),
                    now.isoformat(),
                ),
            )
            added += cur.rowcount
        self.db.commit()
        return added

    def unscored(self, max_age_hours: int, now: datetime | None = None) -> list[dict]:
        now = now or utcnow()
        cutoff = (now - timedelta(hours=max_age_hours)).isoformat()
        rows = self.db.execute(
            "SELECT * FROM items WHERE scored_at IS NULL AND published_at >= ? ORDER BY published_at",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_score(self, item_id: str, score: dict, model: str, now: datetime | None = None) -> None:
        now = now or utcnow()
        self.db.execute(
            "UPDATE items SET scored_at = ?, model = ?, score_json = ? WHERE id = ?",
            (now.isoformat(), model, json.dumps(score, ensure_ascii=False), item_id),
        )
        self.db.commit()

    def add_veto(self, item_id: str, scope: str, until: datetime, reason: str, now: datetime | None = None) -> None:
        now = now or utcnow()
        self.db.execute(
            "INSERT INTO vetoes (item_id, scope, until, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            (item_id, scope, until.isoformat(), reason, now.isoformat()),
        )
        self.db.commit()

    def active_vetoes(self, now: datetime | None = None) -> list[dict]:
        now = now or utcnow()
        rows = self.db.execute(
            "SELECT v.scope, v.until, v.reason, i.title, i.source, i.url "
            "FROM vetoes v JOIN items i ON i.id = v.item_id WHERE v.until > ? ORDER BY v.until DESC",
            (now.isoformat(),),
        ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict:
        total, scored = self.db.execute(
            "SELECT COUNT(*), COUNT(scored_at) FROM items"
        ).fetchone()
        return {"items": total, "scored": scored}
