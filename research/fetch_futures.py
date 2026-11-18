"""Скачивает историю бессрочных фьючерсов Binance (USDT-M) из официального публичного архива.

    python research/fetch_futures.py

Архив data.binance.vision отдаёт помесячные файлы свечей и ставок финансирования.
Ключи не нужны. Результат — по одному файлу на монету в user_data/data/futures/.
"""

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/monthly"
OUT = Path(__file__).resolve().parent.parent / "user_data" / "data" / "futures"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAME = "5m"
START, END = "2020-10", "2026-10"

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def fetch_csv(url: str) -> pd.DataFrame | None:
    """Скачивает zip и читает единственный CSV внутри. Отсутствующий месяц — None."""
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                raw = archive.read(archive.namelist()[0])
            # В старых файлах нет строки заголовков, в новых она есть.
            has_header = not raw[:1].isdigit()
            return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == 3:
                raise
    return None


def load_klines(symbol: str) -> pd.DataFrame:
    urls = [f"{BASE}/klines/{symbol}/{TIMEFRAME}/{symbol}-{TIMEFRAME}-{m}.zip" for m in months(START, END)]
    with ThreadPoolExecutor(8) as pool:
        frames = [f for f in pool.map(fetch_csv, urls) if f is not None]
    for frame in frames:
        frame.columns = KLINE_COLUMNS
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    data = data[["date", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})
    return data.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_funding(symbol: str) -> pd.DataFrame:
    urls = [f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip" for m in months(START, END)]
    with ThreadPoolExecutor(8) as pool:
        frames = [f for f in pool.map(fetch_csv, urls) if f is not None]
    for frame in frames:
        frame.columns = ["calc_time", "interval_hours", "rate"]
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["calc_time"], unit="ms", utc=True)
    return data[["date", "rate"]].astype({"rate": float}).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        klines = load_klines(symbol)
        gaps = int((klines["date"].diff().dropna() != pd.Timedelta(TIMEFRAME.replace("m", "min"))).sum())
        klines.to_feather(OUT / f"{symbol}-{TIMEFRAME}.feather")
        funding = load_funding(symbol)
        funding.to_feather(OUT / f"{symbol}-funding.feather")
        print(f"{symbol}: свечей {len(klines):,} ({klines['date'].iloc[0]:%Y-%m-%d} — {klines['date'].iloc[-1]:%Y-%m-%d}), "
              f"пропусков {gaps}; ставок финансирования {len(funding):,}, "
              f"средняя за 8 ч {funding['rate'].mean() * 100:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
