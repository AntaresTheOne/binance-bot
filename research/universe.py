"""Проверка стратегии VoteTrend на большем числе монет — без ошибки выживших.

    python research/universe.py fetch              скачать данные (один раз, несколько минут)
    python research/universe.py dev                сравнить размеры списка на периоде разработки
    python research/universe.py holdout N [N ...]  названные размеры на контрольном периоде

Как составляется список монет:
- берутся все пары к USDT, когда-либо торговавшиеся на споте Binance, включая снятые с торгов;
- исключаются стейблкоины, фиатные валюты, токены с плечом, обёрнутые BTC/ETH и токены на золото;
- в первый день каждого квартала монеты ранжируются по обороту за три предыдущих месяца,
  и в список попадают первые N — то есть только то, что было известно на тот день.

Правила расчёта: сигнал VoteTrend (50/100/200 дней) по закрытию дня, позиция со следующего дня;
капитал делится на N равных долей, монета без сигнала оставляет свою долю в деньгах;
издержки на сторону: 0,15% для BTC и ETH, 0,25% для остальных (комиссия 0,1% плюс проскальзывание).
Ежедневное выравнивание долей не моделируется отдельно — это небольшое упрощение в пользу стратегии.

Критерий, заданный до запуска: расширять список стоит, если на периоде разработки
коэффициент Шарпа выше, чем у двух монет, хотя бы на 0,1, просадка не глубже более чем
на 5 процентных пунктов, и то же верно для соседних размеров списка.
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "user_data" / "data" / "universe"
API = "https://data-api.binance.vision/api/v3"
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
WINDOWS = (50, 100, 200)
SIZES = (2, 5, 10, 15, 20, 30)
MAX_SIZE = max(SIZES)
DEV = ("2018-06-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")

NOT_COINS = {
    # стейблкоины
    "USDC", "BUSD", "TUSD", "PAX", "USDP", "DAI", "FDUSD", "UST", "USTC", "SUSD", "USDS", "USDSB",
    "PYUSD", "USD1", "XUSD", "BFUSD", "USDE", "AEUR", "EURI", "RLUSD", "USDX", "GUSD", "FRAX", "LUSD",
    # фиатные валюты
    "EUR", "GBP", "AUD", "BRL", "TRY", "RUB", "UAH", "NGN", "ZAR", "JPY", "ARS", "MXN", "PLN", "RON", "CZK", "COP", "BIDR", "IDRT", "BKRW", "BVND",
    # обёрнутые и застейканные BTC/ETH, золото
    "WBTC", "WBETH", "BETH", "WETH", "STETH", "PAXG", "XAUT",
}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def get(url: str, **params):
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=40)
            if response.status_code in (418, 429):
                time.sleep(10 * (attempt + 1))
                continue
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    return response


def all_symbols() -> list[str]:
    """Все пары к USDT за всю историю: список архива Binance плюс текущий список биржи."""
    names, marker = set(), ""
    while True:
        text = get(BUCKET, delimiter="/", prefix="data/spot/monthly/klines/", marker=marker).text
        found = re.findall(r"<Prefix>data/spot/monthly/klines/([^/<]+)/</Prefix>", text)
        names.update(found)
        if "<IsTruncated>true</IsTruncated>" not in text or not found:
            break
        marker = f"data/spot/monthly/klines/{found[-1]}/"
    info = get(f"{API}/exchangeInfo").json()
    names.update(s["symbol"] for s in info["symbols"])
    coins = []
    for name in sorted(names):
        if not name.endswith("USDT"):
            continue
        base = name[:-4]
        if base in NOT_COINS or LEVERAGED.search(base):
            continue
        coins.append(name)
    return coins


def monthly_volume(symbol: str) -> pd.DataFrame | None:
    response = get(f"{API}/klines", symbol=symbol, interval="1M", limit=1000)
    if response.status_code != 200:
        return None
    rows = response.json()
    if not rows:
        return None
    return pd.DataFrame({
        "symbol": symbol,
        "month": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
        "quote_volume": [float(r[7]) for r in rows],
    })


def daily(symbol: str) -> pd.DataFrame | None:
    frames, start = [], 0
    while True:
        response = get(f"{API}/klines", symbol=symbol, interval="1d", limit=1000, startTime=start)
        if response.status_code != 200:
            break
        rows = response.json()
        if not rows:
            break
        frames.append(pd.DataFrame({
            "symbol": symbol,
            "date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "close": [float(r[4]) for r in rows],
        }))
        if len(rows) < 1000:
            break
        start = rows[-1][0] + 1
    return pd.concat(frames, ignore_index=True) if frames else None


def membership(volume: pd.DataFrame, size: int) -> dict[pd.Timestamp, list[str]]:
    """Состав списка на первый день каждого квартала: первые `size` монет по обороту за три прошлых месяца."""
    table = volume.pivot(index="month", columns="symbol", values="quote_volume").sort_index()
    trailing = table.rolling(3, min_periods=3).sum().shift(1)  # только завершённые месяцы
    result = {}
    for month in trailing.index:
        if month.month not in (1, 4, 7, 10):
            continue
        row = trailing.loc[month].dropna()
        row = row[row > 0]
        if len(row):
            result[month] = list(row.sort_values(ascending=False).index[:size])
    return result


def fetch() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = all_symbols()
    print(f"Пар к USDT за всю историю (после исключений): {len(symbols)}")
    with ThreadPoolExecutor(6) as pool:
        frames = [f for f in pool.map(monthly_volume, symbols) if f is not None]
    volume = pd.concat(frames, ignore_index=True)
    volume.to_feather(OUT / "monthly_volume.feather")
    print(f"Месячные обороты получены для {volume['symbol'].nunique()} пар")

    members = sorted({s for names in membership(volume, MAX_SIZE).values() for s in names})
    print(f"Монет, когда-либо входивших в первые {MAX_SIZE}: {len(members)}")
    with ThreadPoolExecutor(6) as pool:
        frames = [f for f in pool.map(daily, members) if f is not None]
    prices = pd.concat(frames, ignore_index=True)
    prices.to_feather(OUT / "daily_close.feather")
    print(f"Дневные цены: {prices['symbol'].nunique()} монет, {len(prices):,} строк, "
          f"{prices['date'].min():%Y-%m-%d} — {prices['date'].max():%Y-%m-%d}")
    return 0


# ---------- расчёт ----------

def cost_of(symbol: str) -> float:
    return 0.0015 if symbol in ("BTCUSDT", "ETHUSDT") else 0.0025


def simulate(close: pd.DataFrame, volume: pd.DataFrame, size: int) -> tuple[pd.Series, pd.DataFrame]:
    votes = sum((close > close.rolling(n).mean()).astype(int) for n in WINDOWS)
    ready = close.rolling(max(WINDOWS)).mean().notna()
    want = (votes >= 2) & ready

    member = pd.DataFrame(False, index=close.index, columns=close.columns)
    quarters = membership(volume, size)
    starts = sorted(quarters)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else close.index[-1] + pd.Timedelta(days=1)
        names = [s for s in quarters[start] if s in close.columns]
        member.loc[(close.index >= start) & (close.index < end), names] = True

    alive = close.notna()
    held = (want.shift(1, fill_value=False) & member & alive).astype(float)
    ret = close.pct_change(fill_method=None).fillna(0.0)
    change = (held - held.shift(1).fillna(0.0)).abs()
    costs = change * pd.Series({s: cost_of(s) for s in close.columns})
    portfolio = ((held * ret - costs) / size).sum(axis=1)
    return portfolio, held


def metrics(ret: pd.Series) -> dict:
    equity = (1 + ret).cumprod()
    years = len(ret) / 365
    cagr = equity.iloc[-1] ** (1 / years) - 1
    drawdown = (equity / equity.cummax() - 1).min()
    return {"итог %": (equity.iloc[-1] - 1) * 100, "в год %": cagr * 100, "просадка %": drawdown * 100,
            "Шарп": ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0}


def evaluate(sizes: list[int], period: tuple[str, str]) -> pd.DataFrame:
    volume = pd.read_feather(OUT / "monthly_volume.feather")
    prices = pd.read_feather(OUT / "daily_close.feather")
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.iloc[:-1]  # последний день может быть ещё не закрыт
    rows, yearly = [], {}
    for size in sizes:
        ret, held = simulate(close, volume, size)
        ret, held = ret.loc[period[0]:period[1]], held.loc[period[0]:period[1]]
        entries = int(((held > 0) & (held.shift(1).fillna(0.0) == 0)).sum().sum())
        years = len(ret) / 365
        name = f"первые {size}"
        rows.append({"список": name, **metrics(ret),
                     "позиций в среднем": held.sum(axis=1).mean(),
                     "дней без позиций %": (held.sum(axis=1) == 0).mean() * 100,
                     "сделок в год": entries / years,
                     "монет побывало": int((held.sum() > 0).sum())})
        yearly[name] = (1 + ret).groupby(ret.index.year).prod() - 1
    table = pd.DataFrame(rows).set_index("список")
    return table.join((pd.DataFrame(yearly).T * 100).round(0).astype(int))


def main() -> int:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "fetch":
        return fetch()
    if mode == "dev":
        table = evaluate(list(SIZES), DEV)
        print(f"Период разработки {DEV[0]} — {DEV[1]}\n")
        print(table.to_string())
        base = table.loc["первые 2"]
        print("\nСравнение с двумя монетами (разница в Шарпе и в просадке, п.п.):")
        for size in SIZES[1:]:
            row = table.loc[f"первые {size}"]
            ok = row["Шарп"] >= base["Шарп"] + 0.1 and row["просадка %"] >= base["просадка %"] - 5
            print(f"  первые {size:2d}: Шарп {row['Шарп'] - base['Шарп']:+.2f}, просадка {row['просадка %'] - base['просадка %']:+.1f}"
                  f"  -> {'проходит' if ok else 'не проходит'}")
        return 0
    if mode == "holdout":
        sizes = [int(v) for v in sys.argv[2:]]
        if not sizes:
            print("Назовите размеры списка, выбранные по периоду разработки.")
            return 2
        print(f"Контрольный период {HOLDOUT[0]} — {HOLDOUT[1]}\n")
        print(evaluate(sizes, HOLDOUT).to_string())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
