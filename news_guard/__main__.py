"""Запуск новостного модуля.

    python -m news_guard once            один цикл: сбор, оценка, запись вето
    python -m news_guard once --no-llm   только сбор новостей, без обращения к Claude
    python -m news_guard run             бесконечный цикл (по умолчанию раз в 5 минут)
    python -m news_guard status          действующие запреты и счётчики
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from .config import Settings, load_settings
from .rules import build_state, decide, write_state
from .scorer import ClaudeCliScorer, ClaudeScorer, ScoringError
from .sources import fetch_all
from .store import Store, utcnow

logger = logging.getLogger("news_guard")


def cycle(settings: Settings, store: Store, scorer=None, fetch=fetch_all, now: datetime | None = None) -> dict:
    """Один проход. scorer=None — только сбор. Возвращает сводку для журнала и тестов."""
    now = now or utcnow()
    added = store.add_items(fetch(settings.feeds, settings.http_timeout, now), now)

    scored = vetoes = 0
    pending = store.unscored(settings.max_age_hours, now)
    if scorer is not None and pending:
        by_id = {item["id"]: item for item in pending}
        for start in range(0, len(pending), settings.batch_size):
            batch = pending[start:start + settings.batch_size]
            try:
                scores = scorer.score(batch)
            except ScoringError as exc:
                logger.error("Оценка не выполнена: %s. Новости будут оценены в следующем цикле.", exc)
                break
            for score in scores:
                item = by_id[score.id]
                store.save_score(score.id, score.model_dump(), getattr(scorer, "model", ""), now)
                scored += 1
                published = datetime.fromisoformat(item["published_at"])
                for scope, until in decide(
                    score, published, now, settings.assets, settings.stake_currency, settings.max_veto_hours
                ):
                    store.add_veto(score.id, scope, until, score.reason, now)
                    vetoes += 1
                    logger.warning("ВЕТО %s до %s: %s — %s", scope, until.isoformat(), item["title"], score.reason)

    active = store.active_vetoes(now)
    write_state(settings.state_path, build_state(active, now))
    summary = {
        "new_items": added,
        "pending": len(pending) - scored,
        "scored": scored,
        "new_vetoes": vetoes,
        "active_vetoes": len(active),
    }
    logger.info(
        "Цикл: новых %(new_items)d, оценено %(scored)d, ждут оценки %(pending)d, "
        "новых вето %(new_vetoes)d, действует вето %(active_vetoes)d", summary,
    )
    return summary


def make_scorer(settings: Settings, no_llm: bool):
    if no_llm:
        return None
    backend = settings.backend
    if backend == "auto":
        has_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        backend = "api" if has_key else "cli"
    try:
        if backend == "cli":
            scorer = ClaudeCliScorer(settings)
            logger.info("Оценка через Claude Code по подписке: %s", scorer.executable)
            return scorer
        logger.info("Оценка через Claude API по ключу")
        return ClaudeScorer(settings)
    except ScoringError as exc:
        logger.error("Claude недоступен (%s). Работаю в режиме сбора без оценки.", exc)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news_guard", description="Новостной модуль торгового робота")
    parser.add_argument("command", choices=["once", "run", "status"])
    parser.add_argument("--no-llm", action="store_true", help="только собирать новости, не обращаться к Claude")
    parser.add_argument("--interval", type=int, default=300, help="пауза между циклами в секундах")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings()
    store = Store(settings.db_path)
    try:
        if args.command == "status":
            print("Счётчики:", store.counts())
            active = store.active_vetoes()
            print(f"Действующих запретов: {len(active)}")
            for veto in active:
                print(f'  {veto["scope"]} до {veto["until"]}: {veto["reason"]} [{veto["source"]}: {veto["title"]}]')
            return 0

        scorer = make_scorer(settings, args.no_llm)
        if args.command == "once":
            cycle(settings, store, scorer)
            return 0

        logger.info("Запущен цикл раз в %d с, модель %s", args.interval, getattr(scorer, "model", "нет"))
        while True:
            try:
                cycle(settings, store, scorer)
            except Exception:
                logger.exception("Цикл завершился ошибкой, повтор после паузы")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Остановлено")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
