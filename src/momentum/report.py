from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from .utils import _alloc_rows, _features_rows


_PLOTLY_FONT = dict(family="Inter, sans-serif", color="#475569", size=11)
_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
)


def _style_figure(
    fig: go.Figure, margin: dict | None = None, height: int | None = None
) -> go.Figure:
    layout_updates: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_PLOTLY_FONT,
        margin=margin or dict(t=10, r=8, b=30, l=48),
    )
    if height is not None:
        layout_updates["height"] = height
    fig.update_layout(**layout_updates)
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0")
    return fig


def _fig_to_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        include_plotlyjs=False, full_html=False, div_id=div_id, config=_PLOTLY_CONFIG
    )


def finalize_portfolio(p: dict, sp500_meta: pd.DataFrame) -> dict:
    """Apply plotly styling, serialise figures to HTML, build table rows."""
    _style_figure(p["frontier_fig"], margin=dict(t=10, r=8, b=80, l=50), height=440)
    _style_figure(p["corr_fig"], margin=dict(t=10, r=8, b=80, l=70), height=480)
    _style_figure(
        p["msr_sector_usd_fig"], margin=dict(t=30, r=8, b=90, l=70), height=360
    )
    _style_figure(
        p["mvp_sector_usd_fig"], margin=dict(t=30, r=8, b=90, l=70), height=360
    )
    for sec in p["symbol_sections"]:
        _style_figure(sec["_fig"], margin=dict(t=10, r=8, b=32, l=52), height=520)
        sec["chart_html"] = _fig_to_html(sec["_fig"], sec["chart_id"])

    p["frontier_html"] = _fig_to_html(p["frontier_fig"], f"{p['prefix']}-frontier")
    p["corr_html"] = _fig_to_html(p["corr_fig"], f"{p['prefix']}-corr")
    p["msr_sector_usd_html"] = _fig_to_html(
        p["msr_sector_usd_fig"], f"{p['prefix']}-msr-sector-usd"
    )
    p["mvp_sector_usd_html"] = _fig_to_html(
        p["mvp_sector_usd_fig"], f"{p['prefix']}-mvp-sector-usd"
    )

    p["msr_rows"] = _alloc_rows(
        p["msr_alloc"], sp500_meta, p["mu"], p["vols"], p["betas"]
    )
    p["mvp_rows"] = _alloc_rows(
        p["mvp_alloc"], sp500_meta, p["mu"], p["vols"], p["betas"]
    )
    return p


def render_report(
    display_df: pd.DataFrame,
    portfolios: list[dict],
    sector_fig: go.Figure,
    sp500_meta: pd.DataFrame,
    meta: dict,
) -> str:
    _style_figure(sector_fig, margin=dict(t=10, r=8, b=90, l=40), height=480)
    for p in portfolios:
        finalize_portfolio(p, sp500_meta)

    template = _TEMPLATE_ENV.get_template("report.html.j2")
    return template.render(
        meta=meta,
        features_rows=_features_rows(display_df),
        sector_html=_fig_to_html(sector_fig, "sector-chart"),
        portfolios=portfolios,
    )
