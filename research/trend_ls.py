"""Трендовая стратегия на фьючерсах: стоит ли добавлять короткие позиции.

    python research/trend_ls.py dev                  все режимы на периоде разработки
    python research/trend_ls.py holdout NAME [NAME]  названные варианты на контрольном периоде

Сигнал тот же, что у VoteTrend: сколько из трёх скользящих средних ниже цены закрытия дня.
Режимы отличаются тем, что делать при тренде вниз:
- long   — только длинные позиции, иначе в деньгах (как сейчас, но с издержками фьючерсов);
- ls     — всегда в рынке: длинная при 2–3 голосах, короткая при 0–1;
- strict — длинная при 2–3 голосах, короткая только при 0, при 1 голосе в деньгах;
- half   — как ls, но короткая позиция вдвое меньше длинной.

Правила расчёта: сигнал по закрытию дня, позиция действует со следующего дня, без плеча;
комиссия 0,05% + проскальзывание 0,01% на каждое изменение позиции; ставка финансирования —
фактическая с октября 2020 года, до этого базовая 0,01% за 8 часов. Цены — спотовые дневные
(на дневном масштабе расхождение с фьючерсом пренебрежимо мало).

Критерий, заданный до запуска: режим с короткими позициями принимается, если на периоде
разработки он лучше режима long по коэффициенту Шарпа и его просадка не глубже более чем на
10 процентных пунктов — при каждом из трёх наборов окон.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPOT = ROOT / "user_data" / "data" / "binance"
FUTURES = ROOT / "user_data" / "data" / "futures"
PAIRS = {"BTC_USDT": "BTCUSDT", "ETH_USDT": "ETHUSDT"}
COST = 0.0005 + 0.0001
BASE_FUNDING_PER_DAY = 0.0003
WINDOW_SETS = {"50_100_200": (50, 100, 200), "40_80_160": (40, 80, 160), "30_60_120": (30, 60, 120)}
MODES = ("long", "ls", "strict", "half")
DEV = ("2018-06-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")


def load(pair: str) -> pd.DataFrame:
    df = pd.read_feather(SPOT / f"{pair}-1d.feather").set_index("date").sort_index()
    funding = pd.read_feather(FUTURES / f"{PAIRS[pair]}-funding.feather").set_index("date")["rate"]
    per_day = funding.groupby(funding.index.floor("D")).sum()
    df["funding"] = per_day.reindex(df.index).fillna(BASE_FUNDING_PER_DAY)
    return df


def target(df: pd.DataFrame, windows: tuple, mode: str) -> pd.Series:
    """Желаемая позиция на закрытии дня: +1 длинная, отрицательная — короткая, 0 — в деньгах."""
    votes = sum((df["close"] > df["close"].rolling(n).mean()).astype(int) for n in windows)
    ready = df["close"].rolling(max(windows)).mean().notna()
    up = votes >= 2
    if mode == "long":
        pos = np.where(up, 1.0, 0.0)
    elif mode == "ls":
        pos = np.where(up, 1.0, -1.0)
    elif mode == "strict":
        pos = np.where(up, 1.0, np.where(votes == 0, -1.0, 0.0))
    elif mode == "half":
        pos = np.where(up, 1.0, -0.5)
    else:
        raise ValueError(mode)
    return pd.Series(pos, index=df.index).where(ready, 0.0)


def daily_returns(df: pd.DataFrame, want: pd.Series) -> tuple[pd.Series, pd.Series]:
    pos = want.shift(1).fillna(0.0)                  # позиция в течение дня
    change = (pos - pos.shift(1).fillna(0.0)).abs()  # разворот из +1 в -1 стоит двух сделок
    price = df["close"].pct_change().fillna(0.0)
    return pos * price - change * COST - pos * df["funding"], pos


def metrics(ret: pd.Series) -> dict:
    equity = (1 + ret).cumprod()
    years = len(ret) / 365
    cagr = equity.iloc[-1] ** (1 / years) - 1
    drawdown = (equity / equity.cummax() - 1).min()
    return {
        "итог %": (equity.iloc[-1] - 1) * 100,
        "в год %": cagr * 100,
        "просадка %": drawdown * 100,
        "Шарп": ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0,
    }


def evaluate(names: list[str], period: tuple[str, str]) -> pd.DataFrame:
    data = {pair: load(pair) for pair in PAIRS}
    rows, yearly = [], {}

    hold = pd.concat([d["close"].pct_change().fillna(0.0) for d in data.values()], axis=1).mean(axis=1).loc[period[0]:period[1]]
    rows.append({"вариант": "держать 50/50 (спот)", "режим": "-", **metrics(hold), "в длинной %": 100.0, "в короткой %": 0.0, "сделок": 0})
    yearly["держать 50/50 (спот)"] = (1 + hold).groupby(hold.index.year).prod() - 1

    for name in names:
        mode, windows_key = name.split(":")
        per_pair, long_share, short_share, trades = [], [], [], 0
        for df in data.values():
            ret, pos = daily_returns(df, target(df, WINDOW_SETS[windows_key], mode))
            ret, pos = ret.loc[period[0]:period[1]], pos.loc[period[0]:period[1]]
            per_pair.append(ret)
            long_share.append((pos > 0).mean())
            short_share.append((pos < 0).mean())
            trades += int((pos != pos.shift(1)).sum()) - 1
        portfolio = pd.concat(per_pair, axis=1).mean(axis=1)
        rows.append({"вариант": name, "режим": mode, **metrics(portfolio),
                     "в длинной %": np.mean(long_share) * 100, "в короткой %": np.mean(short_share) * 100, "сделок": trades})
        yearly[name] = (1 + portfolio).groupby(portfolio.index.year).prod() - 1

    table = pd.DataFrame(rows).set_index("вариант")
    return table.join((pd.DataFrame(yearly).T * 100).round(0).astype(int))


def main() -> int:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        names = [f"{m}:{w}" for w in WINDOW_SETS for m in MODES]
        table = evaluate(names, DEV)
        print(f"Период разработки {DEV[0]} — {DEV[1]}\n")
        print(table.to_string())
        print("\nСравнение с режимом long при тех же окнах (разница в Шарпе и в просадке, п.п.):")
        for windows_key in WINDOW_SETS:
            base = table.loc[f"long:{windows_key}"]
            for m in MODES[1:]:
                row = table.loc[f"{m}:{windows_key}"]
                ok = row["Шарп"] > base["Шарп"] and row["просадка %"] >= base["просадка %"] - 10
                print(f"  {m:6s} {windows_key:11s} Шарп {row['Шарп'] - base['Шарп']:+.2f}, просадка {row['просадка %'] - base['просадка %']:+.1f}"
                      f"  -> {'проходит' if ok else 'не проходит'}")
    elif mode == "holdout":
        names = sys.argv[2:]
        if not names:
            print("Назовите варианты в виде режим:окна, например long:50_100_200.")
            return 2
        print(f"Контрольный период {HOLDOUT[0]} — {HOLDOUT[1]}\n")
        print(evaluate(names, HOLDOUT).to_string())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
