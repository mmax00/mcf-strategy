import numpy as np
import pandas as pd
import scipy.optimize as sco

from .charts import (
    build_correlation_figure,
    build_frontier_figure,
    build_sector_investment_figure,
    build_symbol_chart,
)
from .config import LOOKBACK, START_CAPITAL, W_MAX, W_MIN
from .utils import _symbol_feature_list, sector_color, tradingview_url, yahoo_url


# ---- Optimisation primitives -------------------------------------------------

def annualised_mean(daily_returns: pd.DataFrame, n: int = 252) -> pd.Series:
    return daily_returns.mean() * n


def annualised_cov(daily_returns: pd.DataFrame, n: int = 252) -> pd.DataFrame:
    return daily_returns.cov() * n


def port_return(w, mu) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def port_vol(w, cov) -> float:
    return float(np.sqrt(np.asarray(w) @ np.asarray(cov) @ np.asarray(w)))


def _sum_to_one():
    return [{"type": "eq", "fun": lambda w: w.sum() - 1}]


def min_variance_portfolio(mu, cov, bounds):
    n = len(mu)
    cov_v = np.asarray(cov)
    res = sco.minimize(lambda w: w @ cov_v @ w, np.full(n, 1 / n),
                       bounds=bounds, constraints=_sum_to_one())
    return {"w": res.x, "er": port_return(res.x, mu), "vol": float(np.sqrt(res.fun))}


def max_sharpe_portfolio(mu, cov, rf=0.0, bounds=None):
    n = len(mu)
    mu_v, cov_v = np.asarray(mu), np.asarray(cov)

    def neg_sharpe(w):
        return -(w @ mu_v - rf) / np.sqrt(w @ cov_v @ w)

    res = sco.minimize(neg_sharpe, np.full(n, 1 / n), bounds=bounds, constraints=_sum_to_one())
    return {"w": res.x, "er": port_return(res.x, mu), "vol": port_vol(res.x, cov), "sharpe": -float(res.fun)}


def target_return_portfolio(mu, cov, target, bounds):
    n = len(mu)
    mu_v, cov_v = np.asarray(mu), np.asarray(cov)
    res = sco.minimize(
        lambda w: w @ cov_v @ w, np.full(n, 1 / n), bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ mu_v - target},
        ],
    )
    if not res.success or abs(port_return(res.x, mu) - target) > 1e-4:
        return None
    return {"w": res.x, "er": target, "vol": float(np.sqrt(res.fun))}


def feasible_return_range(mu, bounds) -> tuple[float, float]:
    n = len(mu)
    mu_v = np.asarray(mu)
    lo = sco.minimize(lambda w: w @ mu_v, np.full(n, 1 / n), bounds=bounds, constraints=_sum_to_one()).fun
    hi = -sco.minimize(lambda w: -w @ mu_v, np.full(n, 1 / n), bounds=bounds, constraints=_sum_to_one()).fun
    return float(lo), float(hi)


def efficient_frontier(mu, cov, bounds, step: float = 0.01):
    lo, hi = feasible_return_range(mu, bounds)
    sigmas, mus = [], []
    for m in np.arange(lo + 1e-3, hi - 1e-3, step):
        p = target_return_portfolio(mu, cov, float(m), bounds)
        if p is not None:
            sigmas.append(p["vol"])
            mus.append(m)
    return np.array(sigmas), np.array(mus)


def sample_bounded_cloud(mu, cov, bounds, n_samples: int, rng=None):
    """Rejection-sample from Dirichlet(α=3) so the cloud reaches the EF edges."""
    rng = rng or np.random.default_rng(42)
    n = len(mu)
    lo, hi = bounds[0][0], bounds[0][1]
    mu_v, cov_v = np.asarray(mu), np.asarray(cov)

    collected: list[np.ndarray] = []
    while sum(len(b) for b in collected) < n_samples:
        draw = rng.dirichlet(np.full(n, 3.0), size=n_samples)
        ok = ((draw >= lo) & (draw <= hi)).all(axis=1)
        collected.append(draw[ok])
    W = np.vstack(collected)[:n_samples]
    ret = W @ mu_v
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov_v, W))
    return vol, ret, ret / vol


# ---- Allocation --------------------------------------------------------------

def integer_share_allocation(weights: pd.Series, last_prices: pd.Series, capital: float) -> pd.DataFrame:
    target_dollars = weights * capital
    shares = np.floor(target_dollars / last_prices).astype(int)
    invested = shares * last_prices
    actual_weight = invested / capital
    return pd.DataFrame({
        "price": last_prices,
        "target_weight": weights,
        "target_$": target_dollars,
        "shares": shares,
        "invested_$": invested,
        "actual_weight": actual_weight,
    })


# ---- Per-portfolio orchestration --------------------------------------------

def build_portfolio_analysis(
    label: str,
    prefix: str,
    symbols: list[str],
    prices: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    sp500_meta: pd.DataFrame,
    scored: pd.Series,
) -> dict:
    """Run returns, optimisation, allocation, charts, and per-symbol breakdown for `symbols`."""
    ret = pd.DataFrame({s: prices[s]["close"].pct_change() for s in symbols}).dropna().iloc[-LOOKBACK:]
    mu = annualised_mean(ret)
    cov = annualised_cov(ret)
    bounds = [(W_MIN, W_MAX)] * len(symbols)

    mvp = min_variance_portfolio(mu, cov, bounds)
    msr = max_sharpe_portfolio(mu, cov, rf=0.0, bounds=bounds)

    last_prices = pd.Series({s: prices[s]["close"].iloc[-1] for s in symbols})
    msr_alloc = integer_share_allocation(pd.Series(msr["w"], index=symbols), last_prices, START_CAPITAL)
    mvp_alloc = integer_share_allocation(pd.Series(mvp["w"], index=symbols), last_prices, START_CAPITAL)

    vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.columns)
    betas = features.loc[symbols, "beta_1y"]

    frontier_fig = build_frontier_figure(mu, cov, bounds, mvp, msr)
    corr_fig = build_correlation_figure(ret)
    msr_sector_usd_fig = build_sector_investment_figure(msr_alloc, sp500_meta)
    mvp_sector_usd_fig = build_sector_investment_figure(mvp_alloc, sp500_meta)

    symbol_sections = []
    for sym in symbols:
        row = features.loc[sym]
        sec = str(sp500_meta.loc[sym, "sector"])
        symbol_sections.append({
            "symbol": sym,
            "yahoo_url": yahoo_url(sym),
            "tv_url": tradingview_url(sym),
            "name": str(sp500_meta.loc[sym, "name"]),
            "sector": sec,
            "sector_color": sector_color(sec),
            "chart_id": f"{prefix}-sym-{sym}",
            "_fig": build_symbol_chart(sym, prices[sym]),
            "features": _symbol_feature_list(
                row, msr_alloc.loc[sym], mu.loc[sym], vols.loc[sym], scored.get(sym, np.nan),
            ),
        })

    msr_invested = float(msr_alloc["invested_$"].sum())
    mvp_invested = float(mvp_alloc["invested_$"].sum())

    return {
        "label": label,
        "prefix": prefix,
        "symbols": symbols,
        "mu": mu, "vols": vols, "betas": betas,
        "mvp": mvp, "msr": msr,
        "msr_alloc": msr_alloc, "mvp_alloc": mvp_alloc,
        "msr_invested": msr_invested, "msr_cash_left": START_CAPITAL - msr_invested,
        "mvp_invested": mvp_invested, "mvp_cash_left": START_CAPITAL - mvp_invested,
        "frontier_fig": frontier_fig,
        "corr_fig": corr_fig,
        "msr_sector_usd_fig": msr_sector_usd_fig,
        "mvp_sector_usd_fig": mvp_sector_usd_fig,
        "symbol_sections": symbol_sections,
    }
