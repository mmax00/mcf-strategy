from datetime import datetime

import pandas as pd

from momentum.charts import build_sector_count_figure
from momentum.config import OUTPUT_HTML, START_CAPITAL, TOP_N, W_MAX, W_MIN, log
from momentum.data import download_prices, fetch_market_caps, fetch_sp500_tickers
from momentum.features import (
    apply_hard_filters,
    build_features_frame,
    composite_score,
    pick_sector_leaders,
    pick_top_n,
)
from momentum.portfolio import build_portfolio_analysis
from momentum.report import render_report


def build_display_frame(
    features: pd.DataFrame, scored: pd.Series, sp500_meta: pd.DataFrame
) -> pd.DataFrame:
    df = features.copy()
    df["name"] = sp500_meta["name"]
    df["sector"] = sp500_meta["sector"]
    df["composite_score"] = scored
    df["passed"] = df.index.isin(scored.dropna().index)
    return df.sort_values("composite_score", ascending=False, na_position="last")


def main() -> None:
    sp500 = fetch_sp500_tickers()
    sp500_meta = sp500.set_index("Symbol")[["Security", "GICS Sector"]].rename(
        columns={"Security": "name", "GICS Sector": "sector"}
    )

    prices = download_prices(sp500["Symbol"].tolist() + ["SPY"])
    spy = prices.pop("SPY", None)
    if spy is None:
        raise RuntimeError("SPY data missing — cannot compute relative strength")

    market_caps = fetch_market_caps(list(prices))

    features = build_features_frame(prices, spy["close"], market_caps)
    eligible = apply_hard_filters(features)
    scored = composite_score(eligible)

    display_df = build_display_frame(features, scored, sp500_meta)
    eligible_sectors = sp500_meta.loc[eligible.index, "sector"].value_counts()
    sector_fig = build_sector_count_figure(eligible_sectors)

    top_n_syms = pick_top_n(scored, TOP_N)
    leader_syms = pick_sector_leaders(scored, sp500_meta, TOP_N)
    log.info("Top %d: %s", TOP_N, top_n_syms)
    log.info("Sector leaders: %s", leader_syms)

    portfolios = [
        build_portfolio_analysis(
            "Top 10 by Composite Score",
            "top10",
            top_n_syms,
            prices,
            features,
            sp500_meta,
            scored,
        ),
        build_portfolio_analysis(
            "Sector Leaders",
            "leaders",
            leader_syms,
            prices,
            features,
            sp500_meta,
            scored,
        ),
    ]

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "capital": START_CAPITAL,
        "w_min": W_MIN,
        "w_max": W_MAX,
        "top_n": TOP_N,
        "n_total": len(features),
        "n_eligible": len(eligible),
    }

    html = render_report(display_df, portfolios, sector_fig, sp500_meta, meta)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    log.info("Wrote %s", OUTPUT_HTML)


if __name__ == "__main__":
    main()
