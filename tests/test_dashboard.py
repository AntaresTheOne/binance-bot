"""Тесты панели. Сеть не используется: цены подменены, базы создаются во временной папке."""

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data import MarketCache, Paths, bot_status, build_state, fmt_percent, fmt_price, pair_signal  # noqa: E402
from news_guard.store import Store  # noqa: E402

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)


def candles(closes: list[float], end: datetime = NOW) -> pd.DataFrame:
    """Дневные свечи, последняя из которых ещё не закрыта."""
    start = end.replace(hour=0, minute=0) - timedelta(days=len(closes) - 1)
    dates = [start + timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes, "close_time": [d + timedelta(days=1) for d in dates]})


class SignalTest(unittest.TestCase):
    def test_rising_market_is_uptrend(self):
        signal = pair_signal("BTC/USDT", candles([100 + i for i in range(260)]), NOW)
        self.assertTrue(signal["uptrend"])
        self.assertEqual(signal["votes"], 3)
        self.assertLess(signal["distance_pct"], 0)  # уровень смены сигнала ниже цены

    def test_falling_market_is_downtrend(self):
        signal = pair_signal("BTC/USDT", candles([400 - i for i in range(260)]), NOW)
        self.assertFalse(signal["uptrend"])
        self.assertEqual(signal["votes"], 0)
        self.assertGreater(signal["distance_pct"], 0)

    def test_flip_level_is_the_middle_average(self):
        signal = pair_signal("BTC/USDT", candles([400 - i for i in range(260)]), NOW)
        self.assertEqual(signal["flip_level"], sorted(signal["smas"].values())[1])

    def test_unfinished_day_is_not_used_for_the_signal(self):
        closes = [100.0] * 259 + [10_000.0]  # всплеск в ещё не закрытом дне
        signal = pair_signal("BTC/USDT", candles(closes), NOW)
        self.assertEqual(signal["price"], 10_000.0)
        self.assertEqual(signal["last_close"], 100.0)
        self.assertFalse(signal["uptrend"])
        self.assertEqual(len(signal["series"]["dates"]), len(signal["series"]["sma200"]))

    def test_number_formatting_is_russian(self):
        self.assertEqual(fmt_price(77791.09), "77 791")
        self.assertEqual(fmt_price(0.5234), "0,52")
        self.assertEqual(fmt_percent(19.84), "+19,8%")
        self.assertEqual(fmt_percent(-3.21), "−3,2%")


class StateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.paths = Paths(
            config=root / "config.json", trades_db=root / "trades.sqlite", bot_log=root / "bot.log",
            news_db=root / "news.sqlite", news_state=root / "news_state.json", candles=root,
        )
        self.paths.config.write_text(json.dumps({
            "strategy": "VoteTrend", "dry_run": True, "dry_run_wallet": 1000, "stake_currency": "USDT",
            "exchange": {"pair_whitelist": ["BTC/USDT"]},
        }), encoding="utf-8")
        self.fetch = lambda pair: candles([100 + i for i in range(260)])

    def state(self):
        return build_state(self.paths, MarketCache(), NOW, self.fetch)

    def test_works_with_nothing_but_config(self):
        state = self.state()
        self.assertEqual(state["wallet"]["balance"], 1000)
        self.assertFalse(state["bot"]["running"])
        self.assertEqual(state["headline"]["title"], "Робот остановлен")
        self.assertEqual(state["trades"], {"open": [], "closed": []})
        json.dumps(state)  # всё состояние должно сериализоваться

    def test_survives_market_outage(self):
        def broken(pair):
            raise OSError("нет сети")
        state = build_state(self.paths, MarketCache(), NOW, broken)
        self.assertEqual(state["pairs"], [])
        self.assertEqual(state["market_source"], "нет данных о ценах")

    def test_trades_and_balance(self):
        db = sqlite3.connect(self.paths.trades_db)
        db.execute("CREATE TABLE trades (id INTEGER, pair TEXT, is_open INTEGER, amount REAL, stake_amount REAL, "
                   "open_rate REAL, close_rate REAL, open_date TEXT, close_date TEXT, close_profit REAL, "
                   "close_profit_abs REAL, exit_reason TEXT, stop_loss REAL)")
        db.execute("INSERT INTO trades VALUES (1,'BTC/USDT',0,1,300,300,330,'2026-10-01 00:00:00','2026-10-20 00:00:00',0.1,30,'trend_lost',225)")
        db.execute("INSERT INTO trades VALUES (2,'BTC/USDT',1,2,700,350,NULL,'2026-11-10 00:00:00',NULL,NULL,NULL,NULL,262)")
        db.commit()
        db.close()
        state = self.state()
        price = state["pairs"][0]["price"]
        self.assertEqual(state["stats"], {"open": 1, "closed": 1, "wins": 1, "losses": 0})
        self.assertAlmostEqual(state["wallet"]["balance"], 1000 + 30 + (price - 350) * 2)
        self.assertEqual(state["trades"]["closed"][0]["open_date"], "2026-10-01T00:00:00+00:00")

    def test_bot_heartbeat_is_read_from_the_log(self):
        beat = (NOW - timedelta(seconds=40)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.paths.bot_log.write_text(
            f"{beat},123 - freqtrade.worker - INFO - Bot heartbeat. PID=1, version='2026.10', state='RUNNING'\n",
            encoding="utf-8")
        status = bot_status(self.paths.bot_log, NOW)
        self.assertTrue(status["running"])
        self.assertEqual(status["age_seconds"], 40)
        self.assertFalse(bot_status(self.paths.bot_log, NOW + timedelta(minutes=10))["running"])

    def test_vetoes_are_merged_per_scope_and_shape_the_headline(self):
        store = Store(self.paths.news_db)
        self.addCleanup(store.close)
        for key in ("a", "b"):
            store.add_items([{"id": key, "source": "S", "title": f"Fed shock {key}", "summary": "",
                              "url": "javascript:alert(1)", "published_at": NOW}], NOW)
            store.save_score(key, {"relevant": True, "direction": "bearish", "severity": "high", "confidence": "high",
                                   "assets": ["MARKET"], "horizon_hours": 24, "reason": "ставка"}, "opus", NOW)
            store.add_veto(key, "global", NOW + timedelta(hours=24 if key == "a" else 12), "ставка", NOW)
        beat = NOW.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.paths.bot_log.write_text(f"{beat},1 - x - INFO - Bot heartbeat. PID=1, state='RUNNING'\n", encoding="utf-8")

        state = self.state()
        self.assertEqual(len(state["news"]["vetoes"]), 1)
        self.assertEqual(state["news"]["vetoes"][0]["more"], 1)
        self.assertIn("запрещён новостным вето", state["headline"]["notes"][0])
        # Ссылки с посторонними схемами в панель не попадают.
        self.assertTrue(all(item["url"] == "" for item in state["news"]["items"]))
        self.assertTrue(all(item["veto"] for item in state["news"]["items"]))


if __name__ == "__main__":
    unittest.main()
