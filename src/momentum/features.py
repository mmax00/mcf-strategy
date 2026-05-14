import numpy as np
import pandas as pd

from .config import log


def feat_ret_6m_risk_adj(close: pd.Series, ret: pd.Series) -> float:
    if len(close) < 126:
        return np.nan
    ret_6m = close.iloc[-1] / close.iloc[-126] - 1
    vol_6m = ret.iloc[-126:].std()
    return ret_6m / vol_6m if vol_6m and vol_6m > 0 else np.nan


def feat_composite_momentum(close: pd.Series) -> float:
    def skip_month_ret(lookback: int, skip: int = 21) -> float:
        if len(close) < lookback + skip + 1:
            return np.nan
        return close.iloc[-1 - skip] / close.iloc[-1 - skip - lookback] - 1

    r3, r6, r12 = skip_month_ret(63), skip_month_ret(126), skip_month_ret(252)
    if all(np.isnan([r3, r6, r12])):
        return np.nan
    return float(np.nanmean([r3, r6, r12]))


def feat_ret_1m(close: pd.Series) -> float:
    return close.iloc[-1] / close.iloc[-22] - 1 if len(close) >= 22 else np.nan


def feat_rel_strength_3m_spy(close: pd.Series, spy_close: pd.Series) -> float:
    if len(close) < 64 or len(spy_close) < 64:
        return np.nan
    return (close.iloc[-1] / close.iloc[-64] - 1) - (
        spy_close.iloc[-1] / spy_close.iloc[-64] - 1
    )


def feat_dist_from_52w_high(close: pd.Series) -> float:
    hi_252 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    return close.iloc[-1] / hi_252 - 1


def feat_breakout_10d_flag(close: pd.Series) -> int:
    if len(close) < 262:
        return 0
    rolling_max = close.rolling(252).max()
    return int(bool((close.iloc[-10:] >= rolling_max.iloc[-10:]).any()))


def feat_sma50(close: pd.Series) -> float:
    return close.rolling(50).mean().iloc[-1]


def feat_sma200(close: pd.Series) -> float:
    return close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan


def feat_ma50_gt_ma200_flag(sma50: float, sma200: float) -> int:
    return int(sma50 > sma200) if not np.isnan(sma200) else 0


def feat_atr_14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    if len(close) < 14:
        return np.nan
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.iloc[-14:].mean()


def feat_beta_1y(ret: pd.Series, spy_close: pd.Series) -> float:
    if len(ret) < 252 or len(spy_close) < 252:
        return np.nan
    r_spy = spy_close.pct_change()
    aligned = pd.concat([ret, r_spy], axis=1).dropna().iloc[-252:]
    aligned.columns = ["s", "m"]
    if aligned["m"].var() <= 0:
        return np.nan
    return aligned["s"].cov(aligned["m"]) / aligned["m"].var()


def feat_max_gap_up_10d(open_: pd.Series, close: pd.Series) -> float:
    if len(close) < 11:
        return np.nan
    return (open_.iloc[-10:] / close.shift(1).iloc[-10:] - 1).max()


def feat_vol_1y(ret: pd.Series) -> float:
    return ret.iloc[-252:].std() * np.sqrt(252) if len(ret) >= 252 else np.nan


def feat_mean_return_1y(ret: pd.Series) -> float:
    return ret.iloc[-252:].mean() * 252 if len(ret) >= 252 else np.nan


def feat_avg_dollar_volume_20d(close: pd.Series, volume: pd.Series) -> float:
    return (close.iloc[-20:] * volume.iloc[-20:]).mean() if len(close) >= 20 else np.nan


def compute_features(
    df: pd.DataFrame, spy_close: pd.Series, market_cap: float | None
) -> dict:
    close, high, low = df["close"], df["high"], df["low"]
    open_, volume = df["open"], df["volume"]
    ret = close.pct_change()

    sma50 = feat_sma50(close)
    sma200 = feat_sma200(close)

    return {
        "ret_6m_risk_adj": feat_ret_6m_risk_adj(close, ret),
        "composite_momentum": feat_composite_momentum(close),
        "ret_1m": feat_ret_1m(close),
        "rel_strength_3m_spy": feat_rel_strength_3m_spy(close, spy_close),
        "dist_from_52w_high": feat_dist_from_52w_high(close),
        "breakout_10d_flag": feat_breakout_10d_flag(close),
        "ma50_gt_ma200_flag": feat_ma50_gt_ma200_flag(sma50, sma200),
        "last_close": close.iloc[-1],
        "sma50": sma50,
        "sma200": sma200,
        "avg_dollar_volume_20d": feat_avg_dollar_volume_20d(close, volume),
        "atr_14": feat_atr_14(high, low, close),
        "beta_1y": feat_beta_1y(ret, spy_close),
        "max_gap_up_10d": feat_max_gap_up_10d(open_, close),
        "market_cap": market_cap,
        "vol_1y": feat_vol_1y(ret),
        "mean_return_1y": feat_mean_return_1y(ret),
    }


def build_features_frame(
    prices: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    market_caps: dict[str, float | None],
) -> pd.DataFrame:
    rows = {
        sym: compute_features(df, spy_close, market_caps.get(sym))
        for sym, df in prices.items()
    }
    features = pd.DataFrame.from_dict(rows, orient="index")
    log.info("Features computed for %d stocks", len(features))
    return features


def apply_hard_filters(features: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (features["market_cap"] >= 2e9)
        & (features["avg_dollar_volume_20d"] >= 20e6)
        & (features["last_close"] >= 10)
        & (features["last_close"] > features["sma50"])
        & (features["last_close"] > features["sma200"])
        & (features["max_gap_up_10d"] < 0.15)
        & (features["beta_1y"] < 2.5)
    )
    eligible = features[mask].copy()
    log.info("Passed hard filters: %d / %d", len(eligible), len(features))
    return eligible


def composite_score(eligible: pd.DataFrame) -> pd.Series:
    dist_score = -(eligible["dist_from_52w_high"] + 0.10).abs()
    ranks = pd.DataFrame(
        {
            "r1": eligible["ret_6m_risk_adj"].rank(pct=True),
            "r2": eligible["composite_momentum"].rank(pct=True),
            "r3": eligible["rel_strength_3m_spy"].rank(pct=True),
            "r4": dist_score.rank(pct=True),
        }
    )
    base = ranks.mean(axis=1)
    reversal_penalty = 0.10 * (
        eligible["ret_1m"] >= eligible["ret_1m"].quantile(0.95)
    ).astype(float)
    return (
        base
        + 0.05 * eligible["breakout_10d_flag"]
        + 0.05 * eligible["ma50_gt_ma200_flag"]
        - reversal_penalty
    )


def pick_top_n(scored: pd.Series, n: int) -> list[str]:
    return scored.dropna().sort_values(ascending=False).head(n).index.tolist()


def pick_sector_leaders(scored: pd.Series, meta: pd.DataFrame, n: int) -> list[str]:
    """Top-scoring stock per GICS sector, then fill to n with next-best overall."""
    ordered = scored.dropna().sort_values(ascending=False)
    picked: list[str] = []
    seen: set[str] = set()
    for sym in ordered.index:
        sec = str(meta.loc[sym, "sector"])
        if sec not in seen:
            picked.append(sym)
            seen.add(sec)
    for sym in ordered.index:
        if len(picked) >= n:
            break
        if sym not in picked:
            picked.append(sym)
    return picked[:n]
