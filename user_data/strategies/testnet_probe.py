"""
TestnetProbe — пробная стратегия для проверки связи с биржей. Это НЕ торговая стратегия.

Она сразу покупает BTC на небольшую сумму рыночным ордером, ставит стоп на бирже,
через несколько минут продаёт и больше ничего не делает. Так проверяется вся цепочка:
ключи, выставление и исполнение ордеров, стоп на бирже, учёт баланса.

Запускается только на тестовой бирже Binance (testnet) или в тестовом режиме без денег.
На реальном счёте стратегия откажется стартовать.
"""

import logging
from datetime import datetime, timedelta

from pandas import DataFrame

from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class TestnetProbe(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    process_only_new_candles = False
    startup_candle_count = 0

    # Сколько держать пробную позицию и сколько полных кругов «купил — продал» сделать.
    hold_minutes = 3
    max_round_trips = 1

    minimal_roi = {"0": 100}
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True  # нужно, чтобы Freqtrade вызывал custom_exit

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    def bot_start(self, **kwargs) -> None:
        sandbox = self.config.get("exchange", {}).get("ccxt_config", {}).get("sandbox") is True
        if not sandbox and not self.config.get("dry_run"):
            raise OperationalException(
                "TestnetProbe запускается только на тестовой бирже (ccxt_config.sandbox = true) "
                "или в тестовом режиме. На реальном счёте она отключена."
            )
        logger.info("TestnetProbe: тестовая биржа=%s, тестовый режим=%s", sandbox, self.config.get("dry_run"))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["volume"] >= 0, ["enter_long", "enter_tag"]] = (1, "probe")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        finished = len(Trade.get_trades_proxy(is_open=False))
        if finished >= self.max_round_trips:
            return False  # проба уже выполнена
        logger.info("TestnetProbe: пробная покупка %s, количество %s по цене около %s", pair, amount, rate)
        return True

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        if current_time - trade.open_date_utc >= timedelta(minutes=self.hold_minutes):
            return "probe_done"
        return None
