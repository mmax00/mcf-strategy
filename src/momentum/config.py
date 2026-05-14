import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "momentum_prices.parquet"
OUTPUT_HTML = ROOT / "momentum_report.html"

PERIOD = "2y"
LOOKBACK = 252

TOP_N = 10
START_CAPITAL = 27_000
W_MIN, W_MAX = 0.05, 0.15

MAX_WORKERS = 20
MC_SAMPLES = 10_000
SYMBOL_CHART_LOOKBACK = 252

SECTOR_COLORS: dict[str, str] = {
    "Energy": "#fef3c7",
    "Materials": "#fef9c3",
    "Industrials": "#e0e7ff",
    "Consumer Discretionary": "#fce7f3",
    "Consumer Staples": "#dcfce7",
    "Health Care": "#fee2e2",
    "Financials": "#d1fae5",
    "Information Technology": "#dbeafe",
    "Communication Services": "#ede9fe",
    "Utilities": "#cffafe",
    "Real Estate": "#ffedd5",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("momentum")
