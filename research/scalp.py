"""Проверка скальпинговых подходов на бессрочных фьючерсах Binance (BTC, ETH; 5 и 15 минут).

    python research/scalp.py dev                  все варианты на периоде разработки
    python research/scalp.py holdout NAME [NAME]  названные варианты на контрольном периоде

Данные готовит research/fetch_futures.py. Правила расчёта:
- сигнал считается по закрытию свечи, вход — по открытию следующей; торговля в обе стороны, без плеча;
- стоп и цель проверяются по максимуму и минимуму свечи; если задеты оба, считается стоп;
- издержки «рыночные»: комиссия 0,05% + проскальзывание 0,01% на каждую сторону (0,12% за сделку);
- издержки «лимитные»: 0,02% на сторону без проскальзывания — заведомо оптимистичный сценарий;
- ставка финансирования начисляется, если позиция пережила момент расчёта;
- контрольный период (2025–2026) считается только для вариантов, названных явно.

Критерий, заданный до запуска: вариант проходит, если на периоде разработки при рыночных
издержках средний результат сделки положителен и коэффициент Шарпа выше 1. Семейство
считается устойчивым, если проходит не меньше половины его вариантов.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import talib

DATA = Path(__file__).resolve().parent.parent / "user_data" / "data" / "futures"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TAKER = 0.0005 + 0.0001   # на сторону
MAKER = 0.0002            # на сторону
DEV = ("2021-01-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")


# ---------- данные ----------

def load(symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_feather(DATA / f"{symbol}-5m.feather").set_index("date")
    if timeframe != "5m":
        df = df.resample(timeframe.replace("m", "min")).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    funding = pd.read_feather(DATA / f"{symbol}-funding.feather")
    # Накопленная ставка финансирования на момент открытия каждой свечи.
    cum = np.concatenate([[0.0], funding["rate"].cumsum().to_numpy()])
    df["cumfund"] = cum[np.searchsorted(funding["date"].to_numpy(), df.index.to_numpy(), side="right")]
    df["atr"] = talib.ATR(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    return df


# ---------- стратегии: вход (+1 длинная, -1 короткая) и сигналы выхода по закрытию свечи ----------

def bb_fade(df, k):
    """Возврат к среднему: вход против движения за полосой Боллинджера, выход на средней."""
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    enter = np.where(df["close"] < mid - k * sd, 1, np.where(df["close"] > mid + k * sd, -1, 0))
    return dict(enter=enter, exit_long=(df["close"] >= mid).to_numpy(), exit_short=(df["close"] <= mid).to_numpy(),
                tp=None, sl=2.0, max_bars=48)


def rsi2(df, th):
    """Откат по тренду: RSI(2) в крайней зоне по направлению EMA200."""
    close = df["close"].to_numpy()
    rsi = talib.RSI(close, 2)
    ema = talib.EMA(close, 200)
    enter = np.where((close > ema) & (rsi < th), 1, np.where((close < ema) & (rsi > 100 - th), -1, 0))
    return dict(enter=enter, exit_long=rsi > 50, exit_short=rsi < 50, tp=None, sl=2.5, max_bars=36)


def breakout(df, n):
    """Пробой канала: вход по направлению пробоя максимума или минимума за n свечей."""
    upper = df["high"].rolling(n).max().shift(1)
    lower = df["low"].rolling(n).min().shift(1)
    enter = np.where(df["close"] > upper, 1, np.where(df["close"] < lower, -1, 0))
    return dict(enter=enter, exit_long=None, exit_short=None, tp=2.0, sl=1.0, max_bars=n)


def ema_cross(df, fast, slow):
    """Пересечение скользящих средних с разворотом позиции."""
    close = df["close"].to_numpy()
    above = talib.EMA(close, fast) > talib.EMA(close, slow)
    prev = np.roll(above, 1)
    prev[0] = above[0]
    enter = np.where(above & ~prev, 1, np.where(~above & prev, -1, 0))
    return dict(enter=enter, exit_long=~above, exit_short=above, tp=None, sl=3.0, max_bars=10**9)


def vwap_fade(df, k, window):
    """Возврат к средневзвешенной цене за сутки при отклонении на k ATR."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()
    z = (df["close"] - vwap) / df["atr"]
    enter = np.where(z < -k, 1, np.where(z > k, -1, 0))
    return dict(enter=enter, exit_long=(df["close"] >= vwap).to_numpy(), exit_short=(df["close"] <= vwap).to_numpy(),
                tp=None, sl=2.0, max_bars=96)


def burst(df, m):
    """Импульс: вход по направлению свечи с телом больше m ATR на повышенном объёме."""
    body = df["close"] - df["open"]
    prev_atr = df["atr"].shift(1)
    loud = df["volume"] > 2 * df["volume"].rolling(48).mean().shift(1)
    enter = np.where((body > m * prev_atr) & loud, 1, np.where((body < -m * prev_atr) & loud, -1, 0))
    return dict(enter=enter, exit_long=None, exit_short=None, tp=1.0, sl=1.0, max_bars=12)


def candidates(timeframe: str) -> dict:
    day = 288 if timeframe == "5m" else 96
    tag = timeframe
    return {
        "bb_fade": {f"bb_{k}_{tag}": (bb_fade, (k,)) for k in (2.0, 2.5, 3.0)},
        "rsi2": {f"rsi2_{t}_{tag}": (rsi2, (t,)) for t in (5, 10, 15)},
        "breakout": {f"brk_{n}_{tag}": (breakout, (n,)) for n in (24, 48, 96)},
        "ema_cross": {f"ema_{a}_{b}_{tag}": (ema_cross, (a, b)) for a, b in ((9, 21), (12, 48), (20, 100))},
        "vwap_fade": {f"vwap_{k}_{tag}": (vwap_fade, (k, day)) for k in (2, 3, 4)},
        "burst": {f"burst_{m}_{tag}": (burst, (m,)) for m in (1.5, 2.0, 3.0)},
    }


# ---------- расчёт сделок ----------

def simulate(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr, cumfund = df["atr"].to_numpy(), df["cumfund"].to_numpy()
    enter = spec["enter"]
    n = len(c)
    off = np.zeros(n, dtype=bool)
    exit_long = spec["exit_long"] if spec["exit_long"] is not None else off
    exit_short = spec["exit_short"] if spec["exit_short"] is not None else off
    tp_k, sl_k, max_bars = spec["tp"], spec["sl"], spec["max_bars"]

    entries, exits, sides, gross, fund = [], [], [], [], []
    pos, entry, start, tp, sl = 0, 0.0, 0, 0.0, 0.0
    for i in range(1, n):
        if pos == 0:
            sig = enter[i - 1]
            a = atr[i - 1]
            if sig == 0 or a != a:
                continue
            pos, entry, start = int(sig), o[i], i
            tp = entry + pos * tp_k * a if tp_k else 0.0
            sl = entry - pos * sl_k * a
        price = 0.0
        if pos == 1:
            if l[i] <= sl:
                price = min(o[i], sl)
            elif tp_k and h[i] >= tp:
                price = tp
            elif exit_long[i] or i - start + 1 >= max_bars:
                price = c[i]
        else:
            if h[i] >= sl:
                price = max(o[i], sl)
            elif tp_k and l[i] <= tp:
                price = tp
            elif exit_short[i] or i - start + 1 >= max_bars:
                price = c[i]
        if price:
            entries.append(start)
            exits.append(i)
            sides.append(pos)
            gross.append(pos * (price / entry - 1))
            fund.append(pos * (cumfund[i] - cumfund[start]))  # длинная позиция платит положительную ставку
            pos = 0

    trades = pd.DataFrame({"side": sides, "gross": gross, "funding": fund,
                           "bars": np.array(exits) - np.array(entries) + 1})
    trades["exit_time"] = df.index[exits] if exits else pd.DatetimeIndex([], tz="UTC")
    trades["entry_time"] = df.index[entries] if entries else pd.DatetimeIndex([], tz="UTC")
    return trades


def summarize(per_symbol: list[pd.DataFrame], period: tuple[str, str], minutes: int) -> dict:
    start, end = pd.Timestamp(period[0], tz="UTC"), pd.Timestamp(period[1], tz="UTC") + pd.Timedelta(days=1)
    days = pd.date_range(start, end, freq="D", inclusive="left")
    daily = pd.Series(0.0, index=days)
    pooled = []
    for trades in per_symbol:
        t = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)].copy()
        t["net"] = t["gross"] - 2 * TAKER - t["funding"]
        t["net_maker"] = t["gross"] - 2 * MAKER - t["funding"]
        pooled.append(t)
        # Капитал делится поровну между монетами; сделка идёт на всю долю монеты.
        by_day = t.groupby(t["exit_time"].dt.floor("D"))["net"].sum() / len(per_symbol)
        daily = daily.add(by_day.reindex(days, fill_value=0.0), fill_value=0.0)
    t = pd.concat(pooled)
    years = len(days) / 365
    equity = daily.cumsum()
    return {
        "сделок в год": len(t) / years / len(per_symbol),
        "удержание, мин": t["bars"].mean() * minutes if len(t) else 0,
        "доля прибыльных %": (t["gross"] > 0).mean() * 100 if len(t) else 0,
        "до издержек, б.п.": t["gross"].mean() * 1e4 if len(t) else 0,
        "рыночные, б.п.": t["net"].mean() * 1e4 if len(t) else 0,
        "лимитные, б.п.": t["net_maker"].mean() * 1e4 if len(t) else 0,
        "в год % (рыночные)": daily.sum() / years * 100,
        "Шарп": daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0,
        "просадка %": (equity - equity.cummax()).min() * 100,
    }


def evaluate(names: list[str] | None, period: tuple[str, str]) -> pd.DataFrame:
    rows = []
    for timeframe in ("5m", "15m"):
        families = candidates(timeframe)
        wanted = {name: (fam, spec) for fam, items in families.items() for name, spec in items.items()
                  if names is None or name in names}
        if not wanted:
            continue
        frames = {symbol: load(symbol, timeframe) for symbol in SYMBOLS}
        for name, (family, (func, params)) in wanted.items():
            per_symbol = [simulate(frames[s], func(frames[s], *params)) for s in SYMBOLS]
            rows.append({"вариант": name, "семейство": family, **summarize(per_symbol, period, int(timeframe[:-1]))})
            print(f"  посчитан {name}", file=sys.stderr)
    return pd.DataFrame(rows).set_index("вариант")


def main() -> int:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        table = evaluate(None, DEV)
        print(f"Период разработки {DEV[0]} — {DEV[1]}. Издержки: рыночные 12 б.п. за сделку, лимитные 4 б.п.\n")
        print(table.to_string())
        table["проходит"] = (table["рыночные, б.п."] > 0) & (table["Шарп"] > 1)
        grouped = table.groupby("семейство")
        summary = grouped[["до издержек, б.п.", "рыночные, б.п.", "лимитные, б.п.", "Шарп"]].median()
        summary["прошло вариантов"] = grouped["проходит"].sum().astype(str) + " из " + grouped["проходит"].count().astype(str)
        pd.set_option("display.float_format", "{:.2f}".format)
        print("\nПо семействам (медиана по 6 вариантам):")
        print(summary.sort_values("Шарп", ascending=False).to_string())
    elif mode == "holdout":
        names = sys.argv[2:]
        if not names:
            print("Назовите варианты, выбранные по периоду разработки.")
            return 2
        print(f"Контрольный период {HOLDOUT[0]} — {HOLDOUT[1]}\n")
        print(evaluate(names, HOLDOUT).to_string())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
