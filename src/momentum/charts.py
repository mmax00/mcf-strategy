import pandas as pd
import plotly.graph_objects as go

from .config import MC_SAMPLES, SYMBOL_CHART_LOOKBACK


def build_frontier_figure(mu, cov, bounds, mvp, msr) -> go.Figure:
    from .portfolio import efficient_frontier, sample_bounded_cloud

    vol_mc, ret_mc, shp_mc = sample_bounded_cloud(mu, cov, bounds, MC_SAMPLES)
    sigma_ef, mi_ef = efficient_frontier(mu, cov, bounds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol_mc.tolist(),
            y=ret_mc.tolist(),
            mode="markers",
            marker=dict(
                size=3,
                color=shp_mc.tolist(),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe", thickness=10),
            ),
            name="possibilities",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sigma_ef.tolist(),
            y=mi_ef.tolist(),
            line=dict(color="#2563eb", width=2.5),
            name="Efficient Frontier",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[mvp["vol"]],
            y=[mvp["er"]],
            mode="markers",
            marker=dict(color="#dc2626", size=14, line=dict(color="#1e293b", width=1)),
            name=f"MVP (μ={mvp['er']:.1%}, σ={mvp['vol']:.1%})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[msr["vol"]],
            y=[msr["er"]],
            mode="markers",
            marker=dict(
                color="#d97706",
                size=18,
                symbol="star",
                line=dict(color="#1e293b", width=1),
            ),
            name=f"Max Sharpe (μ={msr['er']:.1%}, σ={msr['vol']:.1%})",
        )
    )
    fig.update_layout(
        xaxis=dict(title="σ (ann.)", tickformat=".0%"),
        yaxis=dict(title="μ (ann.)", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, font=dict(size=10)),
    )
    return fig


def build_sector_count_figure(sector_counts: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sector_counts.index.astype(str).tolist(),
            y=sector_counts.values.tolist(),
            marker=dict(color="#2563eb"),
            text=[str(int(v)) for v in sector_counts.values],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis=dict(title="count"))
    return fig


def build_sector_investment_figure(
    alloc: pd.DataFrame, sp500_meta: pd.DataFrame
) -> go.Figure:
    sectors = sp500_meta.loc[alloc.index, "sector"]
    sector_usd = alloc["invested_$"].groupby(sectors).sum().sort_values(ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sector_usd.index.tolist(),
            y=sector_usd.values.tolist(),
            marker=dict(color="#16a34a"),
            text=[f"${v:,.0f}" for v in sector_usd.values],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis=dict(title="$ invested", tickprefix="$"))
    return fig


def build_correlation_figure(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values.tolist(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=corr.round(2).values.tolist(),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="ρ", thickness=10),
        )
    )
    fig.update_layout(
        xaxis=dict(type="category", automargin=True),
        yaxis=dict(type="category", autorange="reversed", automargin=True),
    )
    return fig


def build_symbol_chart(
    sym: str, df: pd.DataFrame, lookback: int = SYMBOL_CHART_LOOKBACK
) -> go.Figure:
    recent = df.iloc[-lookback:]
    y = recent["close"]
    y_min, y_max = float(y.min()), float(y.max())
    pad = max((y_max - y_min) * 0.05, y_max * 0.01)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[d.isoformat() for d in recent.index],
            y=y.tolist(),
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            name=sym,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(title="", tickprefix="$", range=[y_min - pad, y_max + pad])
    )
    return fig
