"""
Стратегии для спота Binance (только покупки) с новостным вето.

- VoteTrend      — основная. Дневной тренд: позиция открыта, пока цена выше
                   большинства скользящих средних за 50, 100 и 200 дней.
                   Выбрана по данным 2018–2024 и проверена на 2025–2026 (research/scan.py).
- TrendNewsGuard — первая базовая стратегия на часовых свечах; оставлена для сравнения,
                   преимущества на истории не показала.

Решение о сделке принимают правила. Новостной модуль не торгует сам: он лишь
пишет файл user_data/news_state.json, и стратегия отменяет вход, если там
стоит запрет (вето). В бэктесте вето не применяется — исторических новостей нет.

Формат news_state.json:
{
  "updated_at": "2026-11-19T10:00:00+00:00",
  "global": {"block_entries": false, "until": null, "reason": ""},
  "pairs": {
    "ETH/USDT": {"block_entries": true,
                 "until": "2026-11-19T18:00:00+00:00",
                 "reason": "взлом крупного протокола"}
  }
}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy, informative

logger = logging.getLogger(__name__)


class NewsVetoMixin:
    """Отмена входа по запрету новостного модуля. Общая часть всех стратегий."""

    def _news_block_reason(self, pair: str, now: datetime) -> str | None:
        path = Path(self.config["user_data_dir"]) / "news_state.json"
        if not path.exists():
            return None  # новостной модуль ещё не запущен
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Файл есть, но не читается: безопаснее не входить.
            return f"news_state.json не читается: {exc}"

        for scope in (state.get("global") or {}, (state.get("pairs") or {}).get(pair) or {}):
            if not scope.get("block_entries"):
                continue
            until = scope.get("until")
            if until:
                try:
                    until_dt = datetime.fromisoformat(until)
                    if until_dt.tzinfo is None:
                        until_dt = until_dt.replace(tzinfo=timezone.utc)
                    if now >= until_dt:
                        continue  # запрет истёк
                except ValueError:
                    pass  # непонятная дата — считаем запрет действующим
            return scope.get("reason") or "запрет новостного модуля"
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if self.dp.runmode not in (RunMode.LIVE, RunMode.DRY_RUN):
            return True
        reason = self._news_block_reason(pair, current_time)
        if reason:
            logger.info("Вход в %s отменён новостным вето: %s", pair, reason)
            return False
        return True


class VoteTrend(NewsVetoMixin, IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200

    # Окна скользящих средних в днях. Позиция — когда цена выше большинства из них.
    windows = (50, 100, 200)

    # Доля от «своей половины» депозита, которую пара получает в сделке.
    # 1.0 — как в исследовании; 0.5 примерно вдвое уменьшает и доходность, и просадку.
    allocation = 1.0

    # Выход по сигналу. Фиксированный тейк-профит отключён.
    minimal_roi = {"0": 100}
    use_exit_signal = True
    exit_profit_only = False
    trailing_stop = False

    # Аварийный стоп на бирже на случай обвала между дневными сигналами.
    # В исследовании стопа не было; на истории он срабатывает редко.
    stoploss = -0.25
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        votes = 0
        for window in self.windows:
            dataframe[f"sma{window}"] = ta.SMA(dataframe, timeperiod=window)
            votes = votes + (dataframe["close"] > dataframe[f"sma{window}"]).astype(int)
        dataframe["votes"] = votes
        dataframe["uptrend"] = dataframe["votes"] * 2 > len(self.windows)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["uptrend"] & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "vote_trend")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ~dataframe["uptrend"] & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_lost")
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = min(proposed_stake * self.allocation, max_stake)
        if min_stake and stake < min_stake:
            return 0  # депозит слишком мал для сделки — пропускаем
        return stake


class TrendNewsGuard(NewsVetoMixin, IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200

    # Выход по сигналу или стопу; фиксированный тейк-профит отключён.
    minimal_roi = {"0": 100}
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    # Стоп-лосс хранится на бирже, а не в памяти робота.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    # Доля депозита, которой рискуем в одной сделке.
    risk_per_trade = 0.01

    @property
    def protections(self):
        return [
            # Пауза по паре после любой сделки.
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            # 3 стопа за 3 суток по всем парам -> сутки без новых входов.
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 3,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            # Просадка 10% за 2 недели -> трое суток без новых входов.
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 336,
                "trade_limit": 4,
                "stop_duration_candles": 72,
                "max_allowed_drawdown": 0.10,
            },
        ]

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(24).mean()

        # Режим рынка: восходящий тренд на 4 часах.
        dataframe["uptrend"] = (
            (dataframe["close_4h"] > dataframe["ema200_4h"])
            & (dataframe["ema50_4h"] > dataframe["ema200_4h"])
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["uptrend"]
                & (dataframe["ema20"] > dataframe["ema50"])
                & (dataframe["close"] > dataframe["ema20"])
                & (dataframe["adx"] > 25)
                & (dataframe["rsi"] < 70)
                & (dataframe["volume"] > dataframe["volume_mean"])
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_1h_4h")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema50"])
                | (~dataframe["uptrend"])
            )
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_lost")
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Размер позиции такой, чтобы срабатывание стопа стоило risk_per_trade депозита."""
        total = self.wallets.get_total_stake_amount()
        stake = total * self.risk_per_trade / abs(self.stoploss)
        stake = min(stake, max_stake)
        if min_stake and stake < min_stake:
            return 0  # депозит слишком мал для сделки с таким риском — пропускаем
        return stake
