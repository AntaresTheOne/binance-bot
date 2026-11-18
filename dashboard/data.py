"""Сбор состояния робота для панели: рынок, сигналы стратегии, сделки, новости.

Все источники открываются только на чтение. Если какой-то из них недоступен,
соответствующий раздел возвращается пустым с пояснением, а панель продолжает работать.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA = PROJECT_ROOT / "user_data"

# Окна скользящих средних стратегии VoteTrend (user_data/strategies/trend_news_guard.py).
WINDOWS = (50, 100, 200)
CHART_DAYS = 365
KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
MARKET_TTL = 300  # секунд между обращениями за ценами
HEARTBEAT_MAX_AGE = 180  # секунд без сигнала в журнале, после которых робот считается остановленным

_HEARTBEAT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*Bot heartbeat.*state='(\w+)'")


@dataclass
class Paths:
    config: Path = USER_DATA / "config.json"
    trades_db: Path = USER_DATA / "tradesv3.dryrun.sqlite"
    bot_log: Path = USER_DATA / "logs" / "dryrun.log"
    news_db: Path = USER_DATA / "news.sqlite"
    news_state: Path = USER_DATA / "news_state.json"
    candles: Path = USER_DATA / "data" / "binance"


@dataclass
class MarketCache:
    fetched_at: float = 0.0
    frames: dict = field(default_factory=dict)
    source: str = ""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# ---------- рынок ----------

def fetch_daily(pair: str, limit: int = 700, timeout: float = 15.0) -> pd.DataFrame:
    """Дневные свечи с официального зеркала публичных данных Binance."""
    response = requests.get(
        KLINES_URL,
        params={"symbol": pair.replace("/", ""), "interval": "1d", "limit": limit},
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json()
    frame = pd.DataFrame(
        {
            "date": [datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc) for r in rows],
            "close": [float(r[4]) for r in rows],
            "close_time": [datetime.fromtimestamp(r[6] / 1000, tz=timezone.utc) for r in rows],
        }
    )
    return frame


def load_local_daily(pair: str, candles_dir: Path) -> pd.DataFrame:
    """Запасной источник: история, скачанная Freqtrade для бэктестов."""
    frame = pd.read_feather(candles_dir / f"{pair.replace('/', '_')}-1d.feather")[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["close_time"] = frame["date"] + timedelta(days=1)
    return frame


def get_market(pairs: list[str], paths: Paths, cache: MarketCache, fetch=fetch_daily) -> tuple[dict, str]:
    if cache.frames and time.monotonic() - cache.fetched_at < MARKET_TTL and set(pairs) <= set(cache.frames):
        return cache.frames, cache.source
    frames, source = {}, "зеркало данных Binance"
    for pair in pairs:
        try:
            frames[pair] = fetch(pair)
        except Exception:  # сеть недоступна — берём скачанную историю
            try:
                frames[pair] = load_local_daily(pair, paths.candles)
                source = "локальная история (цены могут быть устаревшими)"
            except Exception:
                continue
    if frames:
        cache.frames, cache.fetched_at, cache.source = frames, time.monotonic(), source
    return frames, source


def pair_signal(pair: str, frame: pd.DataFrame, now: datetime) -> dict:
    """Текущее положение цены относительно средних — то же правило, что в стратегии."""
    closed = frame[frame["close_time"] <= now].reset_index(drop=True)
    price = float(frame["close"].iloc[-1])  # последняя цена, включая незакрытый день
    close = closed["close"]
    smas = {n: close.rolling(n).mean() for n in WINDOWS}
    last = {n: (float(s.iloc[-1]) if len(s) and pd.notna(s.iloc[-1]) else None) for n, s in smas.items()}
    last_close = float(close.iloc[-1])
    above = {n: (last[n] is not None and last_close > last[n]) for n in WINDOWS}
    votes = sum(above.values())

    known = sorted(v for v in last.values() if v is not None)
    # Позиция открыта, пока цена выше хотя бы двух средних из трёх, поэтому
    # уровень смены сигнала — средняя по величине из трёх скользящих средних.
    flip = known[1] if len(known) == len(WINDOWS) else None

    tail = closed.tail(CHART_DAYS)
    idx = tail.index
    series = {
        "dates": [d.date().isoformat() for d in tail["date"]],
        "close": [round(float(v), 2) for v in tail["close"]],
    }
    for n in WINDOWS:
        series[f"sma{n}"] = [None if pd.isna(v) else round(float(v), 2) for v in smas[n].loc[idx]]

    return {
        "pair": pair,
        "price": price,
        "last_close": last_close,
        "last_close_date": closed["date"].iloc[-1].date().isoformat(),
        "smas": {str(n): last[n] for n in WINDOWS},
        "above": {str(n): above[n] for n in WINDOWS},
        "votes": votes,
        "uptrend": votes * 2 > len(WINDOWS),
        "flip_level": flip,
        "distance_pct": ((flip / last_close - 1) * 100) if flip else None,
        "series": series,
    }


# ---------- робот ----------

def bot_status(log_path: Path, now: datetime) -> dict:
    status = {"running": False, "state": None, "last_heartbeat": None, "age_seconds": None}
    if not log_path.exists():
        return status
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 60_000))
            tail = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return status
    for line in reversed(tail):
        match = _HEARTBEAT_RE.match(line)
        if match:
            # Freqtrade пишет в журнал местное время компьютера.
            beat = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").astimezone()
            age = (now - beat).total_seconds()
            status.update(
                state=match.group(2),
                last_heartbeat=iso(beat.astimezone(timezone.utc)),
                age_seconds=int(age),
                running=age < HEARTBEAT_MAX_AGE and match.group(2) == "RUNNING",
            )
            break
    return status


def _rows(db_path: Path, query: str, params: tuple = ()) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in db.execute(query, params)]
        finally:
            db.close()
    except sqlite3.Error:
        return []


def _as_utc(value: str | None) -> str | None:
    """Freqtrade хранит время в UTC без пометки зоны."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return iso(parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc))


def load_trades(db_path: Path, prices: dict[str, float]) -> dict:
    rows = _rows(db_path, "SELECT * FROM trades ORDER BY id DESC")
    open_trades, closed = [], []
    for row in rows:
        base = {
            "id": row["id"],
            "pair": row["pair"],
            "amount": row["amount"],
            "stake": row["stake_amount"],
            "open_rate": row["open_rate"],
            "open_date": _as_utc(row["open_date"]),
        }
        if row["is_open"]:
            price = prices.get(row["pair"])
            profit = (price - row["open_rate"]) * row["amount"] if price else None
            open_trades.append({
                **base,
                "price": price,
                "profit_abs": profit,
                "profit_pct": ((price / row["open_rate"] - 1) * 100) if price else None,
                "stop_price": row.get("stop_loss"),
            })
        else:
            closed.append({
                **base,
                "close_rate": row["close_rate"],
                "close_date": _as_utc(row["close_date"]),
                "profit_abs": row["close_profit_abs"],
                "profit_pct": (row["close_profit"] or 0) * 100,
                "exit_reason": row["exit_reason"],
            })
    return {"open": open_trades, "closed": closed}


# ---------- новости ----------

def load_news(paths: Paths, now: datetime, limit: int = 40) -> dict:
    news = {"updated_at": None, "age_seconds": None, "vetoes": [], "items": [], "counts": {"items": 0, "scored": 0}}
    if paths.news_state.exists():
        try:
            updated = json.loads(paths.news_state.read_text(encoding="utf-8")).get("updated_at")
            if updated:
                stamp = datetime.fromisoformat(updated)
                news["updated_at"] = iso(stamp)
                news["age_seconds"] = int((now - stamp).total_seconds())
        except (OSError, ValueError):
            pass

    counts = _rows(paths.news_db, "SELECT COUNT(*) AS items, COUNT(scored_at) AS scored FROM items")
    if counts:
        news["counts"] = counts[0]

    active = _rows(
        paths.news_db,
        "SELECT v.scope, v.until, v.reason, i.title, i.source, i.url FROM vetoes v "
        "JOIN items i ON i.id = v.item_id WHERE v.until > ? ORDER BY v.until DESC",
        (now.isoformat(),),
    )
    veto_titles = {v["title"] for v in active}
    # Одно событие часто приходит из нескольких лент. Показываем один запрет на область
    # (с самым поздним сроком) и число остальных новостей-оснований.
    by_scope: dict[str, dict] = {}
    for veto in active:
        if veto["scope"] in by_scope:
            by_scope[veto["scope"]]["more"] += 1
        else:
            by_scope[veto["scope"]] = {**veto, "more": 0}
    news["vetoes"] = list(by_scope.values())

    for row in _rows(
        paths.news_db,
        "SELECT title, source, url, published_at, score_json FROM items "
        "WHERE score_json IS NOT NULL ORDER BY published_at DESC LIMIT ?",
        (limit,),
    ):
        try:
            score = json.loads(row["score_json"])
        except ValueError:
            continue
        news["items"].append({
            "title": row["title"],
            "source": row["source"],
            "url": row["url"] if str(row["url"]).startswith(("http://", "https://")) else "",
            "published_at": row["published_at"],
            "relevant": bool(score.get("relevant")),
            "direction": score.get("direction"),
            "severity": score.get("severity"),
            "confidence": score.get("confidence"),
            "assets": score.get("assets") or [],
            "horizon_hours": score.get("horizon_hours"),
            "reason": score.get("reason", ""),
            "veto": row["title"] in veto_titles,
        })
    return news


# ---------- сводка ----------

def fmt_price(value: float) -> str:
    """Число в русской записи: пробел между тысячами, запятая перед дробной частью."""
    text = f"{value:,.0f}" if value >= 1000 else f"{value:,.2f}"
    return text.replace(",", " ").replace(".", ",")


def fmt_percent(value: float) -> str:
    return f"{value:+.1f}%".replace(".", ",").replace("-", "−")


def headline(pairs: list[dict], trades: dict, news: dict, bot: dict) -> dict:
    """Одна фраза простым языком: что робот делает сейчас и чего ждёт."""
    holding = sorted({t["pair"].split("/")[0] for t in trades["open"]})
    global_veto = any(v["scope"] == "global" for v in news["vetoes"])
    vetoed_pairs = {v["scope"] for v in news["vetoes"]}

    if not bot["running"]:
        title = "Робот остановлен"
    elif holding:
        title = "Робот держит " + " и ".join(holding)
    else:
        title = "Робот в деньгах и ждёт сигнала"

    notes = []
    for pair in pairs:
        coin = pair["pair"].split("/")[0]
        level = pair["flip_level"]
        if level is None:
            continue
        if pair["uptrend"]:
            blocked = global_veto or pair["pair"] in vetoed_pairs
            if coin in holding:
                notes.append(f"{coin}: тренд вверх, позиция открыта; продаст, если день закроется ниже ≈ {fmt_price(level)}.")
            elif blocked:
                notes.append(f"{coin}: тренд вверх, но вход запрещён новостным вето.")
            else:
                notes.append(f"{coin}: тренд вверх, робот купит при ближайшей проверке.")
        else:
            notes.append(
                f"{coin}: тренд вниз; купит, когда день закроется выше ≈ {fmt_price(level)} "
                f"(сейчас до этого {fmt_percent(pair['distance_pct'])})."
            )
    return {"title": title, "notes": notes}


def build_state(paths: Paths | None = None, cache: MarketCache | None = None,
                now: datetime | None = None, fetch=fetch_daily) -> dict:
    paths = paths or Paths()
    cache = cache if cache is not None else MarketCache()
    now = now or utcnow()

    config = json.loads(paths.config.read_text(encoding="utf-8")) if paths.config.exists() else {}
    pair_names = config.get("exchange", {}).get("pair_whitelist", ["BTC/USDT", "ETH/USDT"])
    currency = config.get("stake_currency", "USDT")
    start_balance = float(config.get("dry_run_wallet", 1000))

    frames, source = get_market(pair_names, paths, cache, fetch)
    pairs = [pair_signal(name, frames[name], now) for name in pair_names if name in frames]
    prices = {p["pair"]: p["price"] for p in pairs}

    bot = bot_status(paths.bot_log, now)
    trades = load_trades(paths.trades_db, prices)
    news = load_news(paths, now)

    closed_profit = sum(t["profit_abs"] or 0 for t in trades["closed"])
    open_profit = sum(t["profit_abs"] or 0 for t in trades["open"])
    wins = sum(1 for t in trades["closed"] if (t["profit_abs"] or 0) > 0)

    return {
        "generated_at": iso(now),
        "mode": "dry_run" if config.get("dry_run", True) else "live",
        "strategy": config.get("strategy", ""),
        "currency": currency,
        "market_source": source if pairs else "нет данных о ценах",
        "bot": bot,
        "wallet": {
            "start": start_balance,
            "closed_profit": closed_profit,
            "open_profit": open_profit,
            "balance": start_balance + closed_profit + open_profit,
            "profit_pct": (closed_profit + open_profit) / start_balance * 100 if start_balance else 0,
        },
        "stats": {
            "open": len(trades["open"]),
            "closed": len(trades["closed"]),
            "wins": wins,
            "losses": len(trades["closed"]) - wins,
        },
        "headline": headline(pairs, trades, news, bot),
        "pairs": pairs,
        "trades": trades,
        "news": news,
    }
