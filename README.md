# mcf-momentum-strategy

End-to-end S&P 500 momentum strategy. Pulls 2 years of daily prices for every ticker, computes per-stock momentum features, filters and scores them, builds two portfolios (top-10 by score, and one leader per GICS sector), runs mean-variance optimisation, and writes a self-contained HTML dashboard.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a `.venv`, installs all dependencies, and installs the `momentum` package in editable mode.

## Run

```bash
uv run main.py
```

The first run takes ~2-3 minutes (downloading 504 symbols via yfinance + fetching market caps). Subsequent runs within 24 hours reuse the cached parquet at `data/momentum_prices.parquet` and finish in ~30 seconds.

Output: `momentum_report.html` at the project root — open it in any browser. It's a fully self-contained dashboard with sortable tables, interactive Plotly charts, tabs per portfolio, and per-symbol breakdowns.

## Project layout

```
src/momentum/
  config.py          constants (capital, weight bounds, lookback, etc.) + logger
  utils.py           URL helpers, string formatters, table-row builders
  data.py            fetch S&P 500 tickers, prices, market caps; parquet cache
  features.py        per-feature functions + hard filters + composite score + pickers
  charts.py          Plotly figure builders (frontier, correlation, sector bars, symbol charts)
  portfolio.py       MVP / max-Sharpe optimisation + integer-share allocation + per-portfolio pipeline
  report.py          Plotly styling + Jinja rendering
  templates/
    report.html.j2   HTML / CSS / JS dashboard template
main.py              entry point: orchestrates the full pipeline
```

## Pipeline

```
fetch_sp500_tickers          # 503 tickers from Wikipedia
        │
download_prices              # 2y daily OHLCV via yfinance, cached to parquet
        │
fetch_market_caps            # yfinance .info.marketCap, threaded
        │
build_features_frame         # 17 features per stock
        │
apply_hard_filters           # liquidity, trend, beta, gap-risk gates
        │
composite_score              # percentile-rank the survivors
        │
pick_top_n          ──┐
pick_sector_leaders ──┴─► build_portfolio_analysis × 2
                            │
                            ├── mean-variance optimisation (MVP + MSR)
                            ├── integer-share allocation at $27k
                            └── per-symbol charts + feature panels
                                    │
                              render_report → momentum_report.html
```

## Features explained

Each feature has its own `feat_*` function in [src/momentum/features.py](src/momentum/features.py). All features look only at the trailing window of prices; "now" means the last bar in the input.

### Momentum signals (what we want high)

| Feature | What it measures | Why |
|---|---|---|
| `feat_ret_6m_risk_adj` | 6-month return divided by 6-month daily volatility | Risk-adjusted version of plain 6m return — penalises stocks that just got lucky with high volatility |
| `feat_composite_momentum` | Mean of 3m, 6m, and 12m "skip-month" returns (each computed `close[-1-21] / close[-1-21-lookback] - 1`) | The skip-month convention drops the most recent ~21 trading days to avoid the well-known short-term reversal effect — classic academic momentum (Jegadeesh & Titman) |
| `feat_rel_strength_3m_spy` | Stock's 3m return minus SPY's 3m return | Did this stock beat the market over the last quarter? Filters out names that are only "up" because everything is up |
| `feat_dist_from_52w_high` | `last_close / 252-day-high − 1` (always ≤ 0) | Closer to zero = closer to the high. The score function rewards stocks sitting ~10% below their high (not at the high, where reversal risk is highest, and not deeply below it, where the trend is broken) |

### Trend confirmation (boolean flags)

| Feature | Meaning |
|---|---|
| `feat_breakout_10d_flag` | 1 if at any point in the last 10 days the close hit a fresh 252-day high |
| `feat_ma50_gt_ma200_flag` | 1 if the 50-day SMA is above the 200-day SMA (the classic "golden cross" state) |

### Risk / liquidity (used by the hard filter)

| Feature | Meaning | Filter threshold |
|---|---|---|
| `feat_sma50`, `feat_sma200` | Simple moving averages | `last_close` must be above both |
| `feat_avg_dollar_volume_20d` | Mean of `close × volume` over the last 20 days | ≥ $20M (filters illiquid names) |
| `feat_atr_14` | 14-day Average True Range | (computed but unused by the filter — exposed for inspection) |
| `feat_beta_1y` | OLS beta vs SPY over the last 252 days | < 2.5 (filters extreme high-beta names) |
| `feat_max_gap_up_10d` | Largest single-day open-gap-up over the last 10 days | < 15% (avoids stocks that just had a news-driven spike) |
| `feat_vol_1y` | Annualised 1y daily-return volatility | (used for display + per-symbol panels) |
| `feat_mean_return_1y` | Annualised 1y mean daily return | (used as μ in the efficient-frontier display) |
| `market_cap` (passed in, not computed from prices) | yfinance `info.marketCap` | ≥ $2B (filters small-caps) |

### Composite score

After the hard filter, [composite_score](src/momentum/features.py) percentile-ranks the survivors on four signals and averages the ranks:

1. `ret_6m_risk_adj`
2. `composite_momentum`
3. `rel_strength_3m_spy`
4. `−|dist_from_52w_high + 0.10|`  (penalises stocks both *too close* to the high and *too far* below it — the sweet spot is ~10% below)

Then:
- `+0.05` if `breakout_10d_flag` is on
- `+0.05` if `ma50_gt_ma200_flag` is on
- `−0.10` if 1-month return is in the top 5% across the eligible universe (short-term reversal penalty — extends the skip-month idea to the *final* selection step)

## Stock pickers

Two portfolios are built side by side from the same scored universe:

- **Top 10 by Composite Score** (`pick_top_n`): the 10 highest-scoring stocks regardless of sector.
- **Sector Leaders** (`pick_sector_leaders`): the highest-scoring stock from each GICS sector, then filled up to 10 with the next-best overall. Gives a more diversified portfolio.

## Portfolio construction

For each picked basket, [build_portfolio_analysis](src/momentum/portfolio.py):

1. Builds the 252-day return matrix and annualises mean (μ) and covariance (Σ).
2. Solves two bounded mean-variance problems with `scipy.optimize.minimize`:
   - **MVP** (Minimum Variance Portfolio): min `wᵀΣw`
   - **MSR** (Max Sharpe Ratio): min `−(wᵀμ − r_f) / √(wᵀΣw)` with `r_f = 0`
   Both subject to `Σwᵢ = 1` and `0.05 ≤ wᵢ ≤ 0.15` (every name gets at least 5%, no name exceeds 15%).
3. Traces the efficient frontier by solving min-variance-given-target-return across the feasible μ range.
4. Samples 10,000 Dirichlet(α=3) weight vectors (rejection-sampled to fit the bounds) to render the "cloud of possibilities" behind the frontier.
5. Converts each portfolio's weights into integer shares at $27,000 capital — `floor(target_$ / last_price)` per stock — so the dashboard shows an actually-executable allocation, not a continuous one.

## Tuning

All knobs are in [src/momentum/config.py](src/momentum/config.py):

| Constant | Default | What it controls |
|---|---|---|
| `PERIOD` | `"2y"` | yfinance download window |
| `LOOKBACK` | `252` | Trading days used for return / cov / beta calculations |
| `TOP_N` | `10` | Stocks per portfolio |
| `START_CAPITAL` | `27_000` | Dollar budget for integer-share allocation |
| `W_MIN`, `W_MAX` | `0.05`, `0.15` | Per-position weight bounds in the optimiser |
| `MAX_WORKERS` | `20` | Threads for market-cap fetching |
| `MC_SAMPLES` | `10_000` | Monte Carlo cloud size on the frontier chart |
| `SYMBOL_CHART_LOOKBACK` | `252` | Bars shown in each per-symbol price chart |

The hard-filter thresholds are inline in [apply_hard_filters](src/momentum/features.py).
