from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from .config import CACHE_PATH, LOOKBACK, MAX_WORKERS, PERIOD, log


def fetch_sp500_tickers() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    df = pd.read_html(StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    log.info("Fetched %d S&P 500 tickers", len(df))
    return df


def download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    if CACHE_PATH.exists() and _age_days(CACHE_PATH) < 1:
        log.info("Using cached prices at %s", CACHE_PATH)
        return _unstack_long(pd.read_parquet(CACHE_PATH))

    log.info("Downloading %d symbols via yfinance...", len(tickers))
    raw = yf.download(
        tickers,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    prices: dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            df = raw[sym].dropna()
        except Exception:
            continue
        if len(df) < LOOKBACK + 10:
            continue
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        prices[sym] = df[["open", "high", "low", "close", "volume"]]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _stack_long(prices).to_parquet(CACHE_PATH)
    log.info("Loaded %d symbols (cached to %s)", len(prices), CACHE_PATH)
    return prices


def fetch_market_caps(tickers: list[str]) -> dict[str, float | None]:
    def fetch(sym: str):
        try:
            return sym, yf.Ticker(sym).info.get("marketCap")
        except Exception:
            return sym, None

    caps: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed(pool.submit(fetch, s) for s in tickers):
            sym, mc = fut.result()
            caps[sym] = mc
    got = sum(1 for v in caps.values() if v is not None)
    log.info("Market caps fetched: %d/%d", got, len(caps))
    return caps


def _age_days(path: Path) -> float:
    return (datetime.now().timestamp() - path.stat().st_mtime) / 86400


def _stack_long(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sym, df in prices.items():
        d = df.copy()
        d["symbol"] = sym
        frames.append(d.reset_index().rename(columns={"Date": "date", "index": "date"}))
    return pd.concat(frames, ignore_index=True)


def _unstack_long(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for sym, g in long_df.groupby("symbol"):
        key = str(sym)
        df = g.drop(columns="symbol").set_index("date").sort_index()
        df.index = pd.to_datetime(df.index)
        prices[key] = df[["open", "high", "low", "close", "volume"]]
    return prices
