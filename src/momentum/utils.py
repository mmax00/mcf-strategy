import numpy as np
import pandas as pd

from momentum.config import SECTOR_COLORS


def sector_color(sector: str | float | None) -> str:
    return SECTOR_COLORS.get(str(sector), "#ffffff")


def yahoo_url(sym: str) -> str:
    return f"https://finance.yahoo.com/quote/{sym}"


def tradingview_url(sym: str) -> str:
    # TradingView uses '.' instead of '-' in multi-class tickers (e.g. BRK.B not BRK-B).
    return f"https://www.tradingview.com/symbols/{sym.replace('-', '.')}/"


def _fmt_pct(x, sign: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.1%}" if sign else f"{x:.1%}"


def _fmt_money(x) -> str:
    return "—" if pd.isna(x) else f"${x:,.2f}"


def _fmt_cap(x) -> str:
    return "—" if pd.isna(x) else f"${x / 1e9:,.1f}B"


def _fmt_num(x, d: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def _fmt_score(x) -> str:
    return "—" if pd.isna(x) else f"{x:.3f}"


def _features_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for sym, r in df.iterrows():
        mean_ret = r["mean_return_1y"]
        sec = r.get("sector") or "—"
        rows.append(
            {
                "symbol": sym,
                "yahoo_url": yahoo_url(str(sym)),
                "tv_url": tradingview_url(str(sym)),
                "name": r.get("name") or "—",
                "sector": sec,
                "sector_color": sector_color(sec),
                "price": _fmt_money(r["last_close"]),
                "vol_1y": _fmt_pct(r["vol_1y"]),
                "mean_return": _fmt_pct(mean_ret, sign=True),
                "mean_return_cls": "positive"
                if pd.notna(mean_ret) and mean_ret > 0
                else "negative",
                "beta": _fmt_num(r["beta_1y"]),
                "market_cap": _fmt_cap(r["market_cap"]),
                "composite_score": _fmt_score(r.get("composite_score", np.nan)),
                "ret_6m_risk_adj": _fmt_num(r["ret_6m_risk_adj"]),
                "composite_momentum": _fmt_pct(r["composite_momentum"]),
                "rel_strength": _fmt_pct(r["rel_strength_3m_spy"], sign=True),
                "dist_52w_high": _fmt_pct(r["dist_from_52w_high"], sign=True),
                "breakout": bool(r["breakout_10d_flag"]),
                "ma_cross": bool(r["ma50_gt_ma200_flag"]),
                "passed": bool(r.get("passed", False)),
            }
        )
    return rows


def _alloc_rows(
    alloc: pd.DataFrame,
    meta_df: pd.DataFrame,
    mu: pd.Series,
    vols: pd.Series,
    betas: pd.Series,
) -> list[dict]:
    rows = []
    for sym, r in alloc.iterrows():
        sec = str(meta_df.loc[sym, "sector"])
        rows.append(
            {
                "symbol": sym,
                "yahoo_url": yahoo_url(str(sym)),
                "tv_url": tradingview_url(str(sym)),
                "name": str(meta_df.loc[sym, "name"]),
                "sector": sec,
                "sector_color": sector_color(sec),
                "price": _fmt_money(r["price"]),
                "mean_return": _fmt_pct(mu.loc[sym], sign=True),
                "vol_1y": _fmt_pct(vols.loc[sym]),
                "beta": _fmt_num(betas.loc[sym]),
                "target_weight": _fmt_pct(r["target_weight"]),
                "target_dollars": _fmt_money(r["target_$"]),
                "shares": int(r["shares"]),
                "invested": _fmt_money(r["invested_$"]),
            }
        )
    return rows


def _symbol_feature_list(
    row: pd.Series, alloc_row: pd.Series, mu_val: float, vol_val: float, score: float
) -> list[tuple[str, str]]:
    return [
        ("Sector", str(row.get("sector") or "—")),
        ("Market Cap", _fmt_cap(row["market_cap"])),
        ("Price", _fmt_money(row["last_close"])),
        ("Mean Return 1y", _fmt_pct(mu_val, sign=True)),
        ("Vol 1y", _fmt_pct(vol_val)),
        ("Beta 1y", _fmt_num(row["beta_1y"])),
        ("Composite Score", _fmt_score(score)),
        ("Risk-Adj 6M", _fmt_num(row["ret_6m_risk_adj"])),
        ("Composite Momentum", _fmt_pct(row["composite_momentum"])),
        ("RS vs SPY (3m)", _fmt_pct(row["rel_strength_3m_spy"], sign=True)),
        ("Dist from 52w High", _fmt_pct(row["dist_from_52w_high"], sign=True)),
        ("Breakout (10d)", "Yes" if row["breakout_10d_flag"] else "No"),
        ("SMA 50 > 200", "Yes" if row["ma50_gt_ma200_flag"] else "No"),
        ("Target Weight (MSR)", _fmt_pct(alloc_row["target_weight"])),
        ("Shares (MSR)", str(int(alloc_row["shares"]))),
        ("Invested (MSR)", _fmt_money(alloc_row["invested_$"])),
    ]
