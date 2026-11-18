"""Стенд для сравнения стратегий на BTC и ETH (спот, только покупки).

    python research/scan.py dev                  все кандидаты на периоде разработки
    python research/scan.py holdout NAME [NAME]  названные стратегии на контрольном периоде

Правила честности:
- сигнал считается по закрытию бара, сделка исполняется по открытию следующего;
- комиссия 0,1% и проскальзывание 0,05% на каждую сторону сделки;
- контрольный период (2025–2026) считается только для стратегий, названных явно,
  то есть выбранных заранее по периоду разработки.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import talib

DATA = Path(__file__).resolve().parent.parent / "user_data" / "data" / "binance"
PAIRS = ["BTC_USDT", "ETH_USDT"]
COST = 0.001 + 0.0005  # комиссия + проскальзывание, на одну сторону

DEV = ("2018-06-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")


def load(pair: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_feather(DATA / f"{pair}-{timeframe}.feather").set_index("date").sort_index()
    step = pd.Timedelta(timeframe.upper() if timeframe.endswith("d") else timeframe)
    gaps = int((df.index.to_series().diff().dropna() != step).sum())
    if gaps:
        print(f"  внимание: {pair} {timeframe}: пропусков в данных — {gaps}")
    return df


# ---------- стратегии: возвращают желаемую позицию (True/False) на закрытии каждого бара ----------

def sma_trend(df, n):
    return df["close"] > df["close"].rolling(n).mean()


def momentum(df, lookback):
    return df["close"] > df["close"].shift(lookback)


def ema_cross(df, fast, slow):
    return df["close"].ewm(span=fast, adjust=False).mean() > df["close"].ewm(span=slow, adjust=False).mean()


def vote(df, windows=(50, 100, 200)):
    """Позиция, если цена выше большинства скользящих средних."""
    votes = sum((df["close"] > df["close"].rolling(n).mean()).astype(int) for n in windows)
    return votes * 2 > len(windows)


def donchian(df, entry, exit_):
    upper = df["high"].rolling(entry).max().shift(1)
    lower = df["low"].rolling(exit_).min().shift(1)
    close = df["close"].to_numpy()
    up, lo = upper.to_numpy(), lower.to_numpy()
    want = np.zeros(len(df), dtype=bool)
    held = False
    for i in range(len(df)):
        if not held and close[i] > up[i]:
            held = True
        elif held and close[i] < lo[i]:
            held = False
        want[i] = held
    return pd.Series(want, index=df.index)


def dip_buy(df, threshold, max_hold=10):
    """Покупка просадки в восходящем тренде: RSI(2) ниже порога при цене выше SMA200."""
    close = df["close"].to_numpy()
    sma200 = df["close"].rolling(200).mean().to_numpy()
    sma5 = df["close"].rolling(5).mean().to_numpy()
    rsi2 = talib.RSI(close, timeperiod=2)
    want = np.zeros(len(df), dtype=bool)
    held, days = False, 0
    for i in range(len(df)):
        if held:
            days += 1
            if close[i] > sma5[i] or days >= max_hold or close[i] < sma200[i]:
                held = False
        elif close[i] > sma200[i] and rsi2[i] < threshold:
            held, days = True, 0
        want[i] = held
    return pd.Series(want, index=df.index)


CANDIDATES = {
    # семейство: {название: (таймфрейм, функция, параметры)}
    "sma": {f"sma_{n}": ("1d", sma_trend, (n,)) for n in (50, 100, 150, 200)},
    "momentum": {f"mom_{n}": ("1d", momentum, (n,)) for n in (30, 60, 90, 180)},
    "donchian": {f"don_{a}_{b}": ("1d", donchian, (a, b)) for a, b in ((20, 10), (40, 20), (55, 20), (100, 50))},
    "ema_4h": {f"ema4h_{a}_{b}": ("4h", ema_cross, (a, b)) for a, b in ((50, 200), (100, 400), (150, 600))},
    "vote": {
        "vote_50_100_200": ("1d", vote, ((50, 100, 200),)),
        "vote_30_60_120": ("1d", vote, ((30, 60, 120),)),
        "vote_40_80_160": ("1d", vote, ((40, 80, 160),)),
    },
    "dip_buy": {f"dip_{t}": ("1d", dip_buy, (t,)) for t in (5, 10, 15)},
}


# ---------- расчёт ----------

def strategy_returns(df: pd.DataFrame, want: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Доходность по барам и позиция. Позиция на баре t определяется сигналом бара t-1."""
    pos = want.shift(1, fill_value=False).astype(bool)
    prev = pos.shift(1, fill_value=False)
    close, open_ = df["close"], df["open"]
    close_to_close = close.pct_change().fillna(0.0)
    open_to_close = close / open_ - 1
    gap = (open_ / close.shift(1) - 1).fillna(0.0)

    ret = pd.Series(0.0, index=df.index)
    hold = pos & prev
    enter = pos & ~prev
    leave = ~pos & prev
    ret[hold] = close_to_close[hold]
    ret[enter] = (1 + open_to_close[enter]) * (1 - COST) - 1
    ret[leave] = (1 + gap[leave]) * (1 - COST) - 1
    return ret, pos


def daily(series: pd.Series) -> pd.Series:
    return (1 + series).groupby(series.index.floor("D")).prod() - 1


def metrics(ret: pd.Series) -> dict:
    equity = (1 + ret).cumprod()
    years = len(ret) / 365
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    drawdown = (equity / equity.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0
    return {
        "итог %": total * 100,
        "в год %": cagr * 100,
        "просадка %": drawdown * 100,
        "Шарп": sharpe,
        "Кальмар": cagr / abs(drawdown) if drawdown < 0 else 0.0,
    }


def evaluate(names: list[str], period: tuple[str, str]) -> pd.DataFrame:
    flat = {name: spec for family in CANDIDATES.values() for name, spec in family.items()}
    family_of = {name: fam for fam, items in CANDIDATES.items() for name in items}
    data = {(pair, tf): load(pair, tf) for pair in PAIRS for tf in ("1d", "4h")}

    rows, yearly = [], {}

    hold = pd.concat([daily(data[(p, "1d")]["close"].pct_change().fillna(0.0)) for p in PAIRS], axis=1).mean(axis=1)
    hold = hold.loc[period[0]:period[1]]
    rows.append({"стратегия": "держать 50/50", "семейство": "-", **metrics(hold), "в рынке %": 100.0, "сделок": 0})
    yearly["держать 50/50"] = (1 + hold).groupby(hold.index.year).prod() - 1

    for name in names:
        timeframe, func, params = flat[name]
        per_pair, exposure, trades = [], [], 0
        for pair in PAIRS:
            df = data[(pair, timeframe)]
            ret, pos = strategy_returns(df, func(df, *params))  # индикаторы считаются по всей истории
            ret, pos = ret.loc[period[0]:period[1]], pos.loc[period[0]:period[1]]
            per_pair.append(daily(ret))
            exposure.append(pos.mean())
            trades += int((pos & ~pos.shift(1, fill_value=False)).sum())
        portfolio = pd.concat(per_pair, axis=1).mean(axis=1)
        rows.append({
            "стратегия": name, "семейство": family_of[name], **metrics(portfolio),
            "в рынке %": float(np.mean(exposure)) * 100, "сделок": trades,
        })
        yearly[name] = (1 + portfolio).groupby(portfolio.index.year).prod() - 1

    table = pd.DataFrame(rows).set_index("стратегия")
    years = (pd.DataFrame(yearly).T * 100).round(0).astype(int)
    return table.join(years)


def main() -> int:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        names = [name for family in CANDIDATES.values() for name in family]
        table = evaluate(names, DEV)
        print(f"Период разработки {DEV[0]} — {DEV[1]}\n")
        print(table.to_string())
        grouped = table[table["семейство"] != "-"].groupby("семейство")
        summary = grouped[["в год %", "просадка %", "Шарп"]].median()
        summary["Шарп худший"] = grouped["Шарп"].min()
        summary["просадка худшая %"] = grouped["просадка %"].min()
        pd.set_option("display.float_format", "{:.2f}".format)
        print("\nПо семействам (медиана по вариантам и худший вариант):")
        print(summary.sort_values("Шарп", ascending=False).to_string())
    elif mode == "holdout":
        names = sys.argv[2:]
        if not names:
            print("Назовите стратегии, выбранные по периоду разработки.")
            return 2
        print(f"Контрольный период {HOLDOUT[0]} — {HOLDOUT[1]}\n")
        print(evaluate(names, HOLDOUT).to_string())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
