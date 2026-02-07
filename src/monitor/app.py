"""Tradingo Backtest Monitor

A dashboard for analyzing trading backtests. Reads backtest results from
ArcticDB and presents session summaries, historical performance, and
position breakdowns.

Usage:
    tradingo-monitor --arctic-uri lmdb://path/to/arctic
    python -m monitor --arctic-uri lmdb://path/to/arctic
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import arcticdb as adb
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, callback, dash_table, dcc, html

# -- Backtest schema (mirrors tradingo.backtest.BACKTEST_FIELDS) --

BACKTEST_FIELDS = (
    "unrealised_pnl",
    "realised_pnl",
    "total_pnl",
    "net_investment",
    "net_position",
    "net_exposure",
    "avg_open_price",
    "stop_trade",
)

SESSION_START_HOUR = 14  # 2pm UTC
SESSION_END_HOUR = 21  # 9pm UTC

ARCTIC_URI: str = ""

pd.options.plotting.backend = "plotly"
pio.templates.default = "plotly_white"


# ---- Data Access (standalone - uses arcticdb directly) ----


def _lib() -> adb.library.Library:
    return adb.Arctic(ARCTIC_URI).get_library("backtest")


def discover_portfolios() -> list[str]:
    """Find all portfolio names by listing symbols matching *.portfolio."""
    symbols = _lib().list_symbols(regex=r".+\.portfolio$")
    return sorted(s.rsplit(".portfolio", 1)[0] for s in symbols)


def read_portfolio(name: str, **kw: Any) -> pd.DataFrame:
    return _lib().read(f"{name}.portfolio", **kw).data


def read_instrument(name: str, field: str, **kw: Any) -> pd.DataFrame:
    return _lib().read(f"{name}.instrument.{field}", **kw).data


def session_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Current/latest trading session (2pm-9pm UTC on last business day)."""
    now = pd.Timestamp.now("UTC")
    today = now.normalize()
    if today.weekday() >= 5:
        today = (today - pd.offsets.BDay(1)).normalize()
    start = today + pd.Timedelta(hours=SESSION_START_HOUR)
    end = today + pd.Timedelta(hours=SESSION_END_HOUR)
    if now < start:
        prev = (today - pd.offsets.BDay(1)).normalize()
        start = prev + pd.Timedelta(hours=SESSION_START_HOUR)
        end = prev + pd.Timedelta(hours=SESSION_END_HOUR)
    return start, end


def fetch_live_positions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch open positions from IG and aggregate by epic.

    Returns:
        net: DataFrame indexed by epic with columns [net_position, avg_price]
        raw: DataFrame of individual trades with [epic, signed_size, level, created]
    """
    from tradingo.sampling.ig import get_ig_service

    svc = get_ig_service()
    positions = svc.fetch_open_positions()
    svc.session.close()

    empty_net = pd.DataFrame(columns=["net_position", "avg_price"])
    empty_raw = pd.DataFrame(columns=["epic", "signed_size", "level", "created"])

    if positions.empty:
        return empty_net, empty_raw

    sign = positions["direction"].map({"BUY": 1.0, "SELL": -1.0})
    positions["signed_size"] = positions["size"].astype(float) * sign
    positions["level"] = positions["level"].astype(float)

    raw = positions[["epic", "signed_size", "level", "createdDateUTC"]].copy()
    raw = raw.rename(columns={"createdDateUTC": "created"})
    raw["created"] = pd.to_datetime(raw["created"], utc=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        sizes = g["size"].astype(float)
        return pd.Series(
            {
                "net_position": g["signed_size"].sum(),
                "avg_price": (g["level"] * sizes).sum() / sizes.sum(),
            }
        )

    net = positions.groupby("epic").apply(_agg, include_groups=False)
    return net, raw


# ---- UI Helpers ----

GREEN = "rgb(0,200,100)"
RED = "rgb(200,50,50)"
GREEN_FILL = "rgba(0,200,100,0.1)"
RED_FILL = "rgba(200,50,50,0.15)"
BLUE = "rgb(55,126,184)"
ORANGE = "rgb(228,120,42)"
SHOW: dict[str, str] = {}
HIDE: dict[str, str] = {"display": "none"}


def kpi_card(title: str, value: str, color: str = "success") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        title,
                        className="mb-1 text-muted",
                        style={"fontSize": "0.8rem"},
                    ),
                    html.H5(value, className="mb-0 fw-bold"),
                ]
            ),
            color=color,
            inverse=True,
        ),
        md=2,
    )


def fmt(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,.0f}"


def empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font_size=14,
    )
    fig.update_layout(height=300, margin=dict(t=30, b=30, l=30, r=30))
    return fig


def pnl_color(val: float) -> str:
    return "success" if val >= 0 else "danger"


def compute_daily_stats(pf: pd.DataFrame) -> dict[str, str]:
    """Compute performance stats from a portfolio summary DataFrame."""
    daily = pf.resample("B").last().ffill()
    daily_pnl = daily.total_pnl.diff().dropna()
    cum_pnl = daily_pnl.cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak

    n = len(daily_pnl)
    wins = int((daily_pnl > 0).sum())
    losses = int((daily_pnl < 0).sum())
    avg_win = float(daily_pnl[daily_pnl > 0].mean()) if wins else 0.0
    avg_loss = float(daily_pnl[daily_pnl < 0].mean()) if losses else 0.0
    pf_ratio = (
        abs(avg_win * wins / (avg_loss * losses))
        if losses and avg_loss != 0
        else float("inf")
    )
    std = float(daily_pnl.std())
    sharpe = float(daily_pnl.mean()) / std * np.sqrt(252) if std > 0 else 0.0

    return {
        "Total PnL": fmt(cum_pnl.iloc[-1]) if len(cum_pnl) else "N/A",
        "Trading Days": str(n),
        "Win Rate": f"{wins / n * 100:.0f}%" if n else "N/A",
        "Avg Win": fmt(avg_win),
        "Avg Loss": fmt(avg_loss),
        "Profit Factor": f"{pf_ratio:.2f}" if pf_ratio != float("inf") else "-",
        "Max Drawdown": fmt(float(dd.min())),
        "Sharpe (ann.)": f"{sharpe:.2f}",
        "Best Day": fmt(float(daily_pnl.max())) if n else "N/A",
        "Worst Day": fmt(float(daily_pnl.min())) if n else "N/A",
    }


# ---- App Layout ----

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Tradingo Monitor",
)

app.layout = dbc.Container(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand("Tradingo Monitor", className="fs-4 fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="portfolio-select",
                                    placeholder="Portfolio...",
                                    style={"minWidth": "250px"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Switch(
                                    id="live-toggle",
                                    label="Live",
                                    value=False,
                                    className="mt-1",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Refresh",
                                    id="refresh-btn",
                                    color="outline-dark",
                                    size="sm",
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="g-2",
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            # dark=True,
            className="mb-3",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Session", tab_id="session"),
                dbc.Tab(label="History", tab_id="history"),
                dbc.Tab(label="Positions", tab_id="positions"),
                dbc.Tab(label="Compare", tab_id="compare"),
            ],
            id="tabs",
            active_tab="session",
            className="mb-3",
        ),
        # ---- Session tab ----
        html.Div(
            id="session-tab",
            children=[
                dbc.Row(
                    dbc.Col(
                        dcc.DatePickerSingle(
                            id="session-date",
                            date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                            display_format="YYYY-MM-DD",
                            placeholder="Session date",
                        ),
                        width="auto",
                    ),
                    className="mb-3",
                ),
                dbc.Row(id="session-kpis", className="mb-3 g-2"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="session-pnl")), lg=8),
                        dbc.Col(
                            dcc.Loading(dcc.Graph(id="session-instrument-bar")), lg=4
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="session-exposure")), lg=6),
                        dbc.Col(dcc.Loading(dcc.Graph(id="session-positions")), lg=6),
                    ]
                ),
            ],
        ),
        # ---- History tab ----
        html.Div(
            id="history-tab",
            children=[
                dbc.Row(
                    dbc.Col(
                        dcc.DatePickerRange(
                            id="history-dates",
                            start_date=(
                                pd.Timestamp.now() - pd.offsets.BDay(60)
                            ).strftime("%Y-%m-%d"),
                            end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                            display_format="YYYY-MM-DD",
                        ),
                        width="auto",
                    ),
                    className="mb-3",
                ),
                dcc.Loading(dcc.Graph(id="history-cumulative")),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="history-daily")), lg=8),
                        dbc.Col(html.Div(id="history-stats"), lg=4),
                    ]
                ),
            ],
        ),
        # ---- Positions tab ----
        html.Div(
            id="positions-tab",
            children=[
                html.Div(id="positions-table-wrapper", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="positions-net")), lg=6),
                        dbc.Col(dcc.Loading(dcc.Graph(id="positions-exposure")), lg=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(dcc.Graph(id="positions-unrealised-pnl")), lg=6
                        ),
                        dbc.Col(dcc.Loading(dcc.Graph(id="positions-total-pnl")), lg=6),
                    ],
                ),
            ],
        ),
        # ---- Compare tab ----
        html.Div(
            id="compare-tab",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="compare-portfolio-select",
                                placeholder="Compare against...",
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dcc.DatePickerRange(
                                id="compare-dates",
                                start_date=(
                                    pd.Timestamp.now() - pd.offsets.BDay(60)
                                ).strftime("%Y-%m-%d"),
                                end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                                display_format="YYYY-MM-DD",
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                    className="mb-3 g-2",
                ),
                dcc.Loading(dcc.Graph(id="compare-cumulative")),
                dbc.Row(
                    [
                        dbc.Col(dcc.Loading(dcc.Graph(id="compare-daily")), lg=7),
                        dbc.Col(html.Div(id="compare-stats"), lg=5),
                    ]
                ),
            ],
        ),
    ],
    fluid=True,
)

server = app.server


# ---- Tab visibility ----


@callback(
    Output("session-tab", "style"),
    Output("history-tab", "style"),
    Output("positions-tab", "style"),
    Output("compare-tab", "style"),
    Input("tabs", "active_tab"),
)
def toggle_tabs(tab: str) -> tuple[dict, dict, dict, dict]:
    return (
        SHOW if tab == "session" else HIDE,
        SHOW if tab == "history" else HIDE,
        SHOW if tab == "positions" else HIDE,
        SHOW if tab == "compare" else HIDE,
    )


# ---- Portfolio discovery ----


@callback(
    Output("portfolio-select", "options"),
    Input("refresh-btn", "n_clicks"),
)
def load_portfolios(_: Any) -> list[str]:
    return discover_portfolios()


@callback(
    Output("portfolio-select", "value"),
    Input("portfolio-select", "options"),
)
def auto_select(options: list[str]) -> Any:
    return options[0] if options else dash.no_update


# ---- Session Tab ----


@callback(
    Output("session-kpis", "children"),
    Output("session-pnl", "figure"),
    Output("session-instrument-bar", "figure"),
    Output("session-exposure", "figure"),
    Output("session-positions", "figure"),
    Input("portfolio-select", "value"),
    Input("tabs", "active_tab"),
    Input("session-date", "date"),
)
def update_session(portfolio: str | None, tab: str, session_date: str | None) -> tuple:
    if not portfolio or tab != "session":
        raise dash.exceptions.PreventUpdate

    if session_date:
        day = pd.Timestamp(session_date).normalize()
        if day.tz is None:
            day = day.tz_localize("UTC")
        start = day + pd.Timedelta(hours=SESSION_START_HOUR)
        end = day + pd.Timedelta(hours=SESSION_END_HOUR)
    else:
        start, end = session_range()
    no_data = (
        [],
        empty_fig("No session data"),
        empty_fig(),
        empty_fig(),
        empty_fig(),
    )

    try:
        pf = read_portfolio(portfolio, date_range=(start, end))
    except Exception:
        return no_data
    if pf.empty:
        return no_data

    # KPIs
    s_pnl = pf.total_pnl.iloc[-1] - pf.total_pnl.iloc[0]
    s_unrealised = pf.unrealised_pnl.iloc[-1]
    s_realised = pf.realised_pnl.iloc[-1] - pf.realised_pnl.iloc[0]
    net_exp = pf.net_exposure.iloc[-1]
    gross_exp = pf.gross_exposure.iloc[-1]

    kpis = [
        kpi_card("Session PnL", fmt(s_pnl), pnl_color(s_pnl)),
        kpi_card("Unrealised", fmt(s_unrealised), pnl_color(s_unrealised)),
        kpi_card("Realised", fmt(s_realised), pnl_color(s_realised)),
        kpi_card("Net Exposure", fmt(net_exp), "info"),
        kpi_card("Gross Exposure", fmt(gross_exp), "info"),
        kpi_card(
            "Session",
            f"{start.strftime('%d %b %H:%M')}-{end.strftime('%H:%M')}",
            "secondary",
        ),
    ]

    # Portfolio PnL (relative to session start)
    pnl_rel = pf.total_pnl - pf.total_pnl.iloc[0]
    positive = pnl_rel.iloc[-1] >= 0
    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Scatter(
            x=pnl_rel.index,
            y=pnl_rel.values,
            mode="lines",
            name="Session PnL",
            fill="tozeroy",
            fillcolor=GREEN_FILL if positive else RED_FILL,
            line_color=GREEN if positive else RED,
        )
    )
    fig_pnl.update_layout(
        title="Session PnL",
        height=400,
        yaxis_title="PnL",
        margin=dict(t=40, b=40),
    )

    # Instrument PnL bar
    try:
        inst_pnl = read_instrument(portfolio, "total_pnl", date_range=(start, end))
        delta = (inst_pnl.ffill().iloc[-1] - inst_pnl.ffill().iloc[0]).dropna()
        delta = delta.sort_values()
        colors = [RED if v < 0 else GREEN for v in delta.values]
        fig_bar = go.Figure(
            go.Bar(
                x=delta.values,
                y=delta.index,
                orientation="h",
                marker_color=colors,
            )
        )
        fig_bar.update_layout(
            title="Instrument PnL",
            height=400,
            margin=dict(t=40, b=40, l=120),
        )
    except Exception:
        fig_bar = empty_fig("No instrument data")

    # Exposure
    fig_exp = go.Figure()
    fig_exp.add_trace(
        go.Scatter(x=pf.index, y=pf.net_exposure, name="Net", mode="lines")
    )
    fig_exp.add_trace(
        go.Scatter(x=pf.index, y=pf.gross_exposure, name="Gross", mode="lines")
    )
    fig_exp.update_layout(title="Exposure", height=350, margin=dict(t=40, b=40))

    # Positions
    try:
        positions = read_instrument(portfolio, "net_position", date_range=(start, end))
        fig_pos = positions.ffill().plot(title="Positions")
        fig_pos.update_layout(height=350, margin=dict(t=40, b=40))
    except Exception:
        fig_pos = empty_fig("No position data")

    return kpis, fig_pnl, fig_bar, fig_exp, fig_pos


# ---- History Tab ----


@callback(
    Output("history-cumulative", "figure"),
    Output("history-daily", "figure"),
    Output("history-stats", "children"),
    Input("portfolio-select", "value"),
    Input("tabs", "active_tab"),
    Input("history-dates", "start_date"),
    Input("history-dates", "end_date"),
    Input("live-toggle", "value"),
)
def update_history(
    portfolio: str | None,
    tab: str,
    start_date: str | None,
    end_date: str | None,
    live: bool = False,
) -> tuple:
    if not portfolio or tab != "history":
        raise dash.exceptions.PreventUpdate

    start = (
        pd.Timestamp(start_date).tz_localize("UTC")
        if start_date
        else pd.Timestamp.now("UTC") - pd.offsets.BDay(60)
    )
    end = (
        pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
        if end_date
        else pd.Timestamp.now("UTC")
    )
    no_data: tuple = (empty_fig("No data"), empty_fig(), html.P("No data"))

    try:
        pf = read_portfolio(portfolio, date_range=(start, end))
    except Exception:
        return no_data
    if pf.empty:
        return no_data

    daily = pf.resample("B").last().ffill()
    daily_pnl = daily.total_pnl.diff().dropna()
    cum_pnl = daily_pnl.cumsum()

    # Cumulative PnL + drawdown
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak

    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=cum_pnl.index,
            y=cum_pnl.values,
            mode="lines",
            name="Cumulative PnL",
            fill="tozeroy",
            fillcolor=GREEN_FILL,
            line_color=GREEN,
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor=RED_FILL,
            line_color=RED,
        )
    )
    fig_cum.update_layout(
        title="Cumulative PnL & Drawdown",
        height=400,
        margin=dict(t=40, b=40),
    )

    # Daily bars
    bar_colors = [GREEN if v >= 0 else RED for v in daily_pnl.values]
    fig_daily = go.Figure(
        go.Bar(x=daily_pnl.index, y=daily_pnl.values, marker_color=bar_colors)
    )
    fig_daily.update_layout(title="Daily PnL", height=350, margin=dict(t=40, b=40))

    # Live position entries overlay
    if live:
        try:
            _, live_raw = fetch_live_positions()
            if not live_raw.empty:
                # Mark position entry times on the cumulative PnL chart
                within = live_raw[
                    (live_raw.created >= start) & (live_raw.created <= end)
                ]
                if not within.empty:
                    # Interpolate PnL at each entry time for y-position
                    pnl_full = pf.total_pnl.ffill().diff().cumsum().fillna(0)
                    entry_pnl = (
                        pnl_full.reindex(pnl_full.index.union(within.created))
                        .interpolate()
                        .reindex(within.created)
                    )

                    labels = [
                        f"{r.epic}<br>{'+' if r.signed_size > 0 else ''}"
                        f"{r.signed_size:.1f} @ {r.level:.2f}"
                        for _, r in within.iterrows()
                    ]
                    fig_cum.add_trace(
                        go.Scatter(
                            x=within.created,
                            y=entry_pnl.values,
                            mode="markers",
                            name="Live entries",
                            marker=dict(
                                size=10,
                                symbol="diamond",
                                color=ORANGE,
                                line=dict(width=1, color="white"),
                            ),
                            text=labels,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )
        except Exception:
            pass  # live data unavailable

    # Performance stats
    stat = compute_daily_stats(pf)
    stats = dbc.Table(
        [html.Tbody([html.Tr([html.Td(k), html.Td(v)]) for k, v in stat.items()])],
        bordered=True,
        hover=True,
        size="sm",
        className="mt-2",
    )

    return fig_cum, fig_daily, html.Div([html.H5("Performance Summary"), stats])


# ---- Positions Tab ----


@callback(
    Output("positions-table-wrapper", "children"),
    Output("positions-net", "figure"),
    Output("positions-exposure", "figure"),
    Output("positions-unrealised-pnl", "figure"),
    Output("positions-total-pnl", "figure"),
    Input("portfolio-select", "value"),
    Input("tabs", "active_tab"),
    Input("live-toggle", "value"),
)
def update_positions(portfolio: str | None, tab: str, live: bool) -> tuple:
    if not portfolio or tab != "positions":
        raise dash.exceptions.PreventUpdate

    end = pd.Timestamp.now("UTC")
    start = end - pd.offsets.BDay(5)
    no_data: tuple = (
        html.P("No data"),
        empty_fig(),
        empty_fig(),
        empty_fig(),
        empty_fig(),
    )

    try:
        positions = read_instrument(portfolio, "net_position", date_range=(start, end))
        total_pnl = read_instrument(portfolio, "total_pnl", date_range=(start, end))
        unrealised = read_instrument(
            portfolio, "unrealised_pnl", date_range=(start, end)
        )
        exposure = read_instrument(portfolio, "net_exposure", date_range=(start, end))
        avg_price = read_instrument(
            portfolio, "avg_open_price", date_range=(start, end)
        )
    except Exception:
        return no_data

    if positions.empty:
        return no_data

    # Fetch live data if toggled on
    live_net = pd.DataFrame(columns=["net_position", "avg_price"])
    live_raw = pd.DataFrame(columns=["epic", "signed_size", "level", "created"])
    if live:
        try:
            live_net, live_raw = fetch_live_positions()
        except Exception:
            pass  # live data unavailable - proceed without

    # Current state table
    latest_pos = positions.ffill().iloc[-1]
    latest_avg = avg_price.ffill().iloc[-1]

    # Include all instruments with backtest OR live positions
    bt_active = set(latest_pos[latest_pos.abs() > 0.001].index)
    live_active = set(live_net.index) if not live_net.empty else set()
    all_active = sorted(bt_active | live_active)

    if not all_active:
        table_content = html.P("No open positions", className="text-muted")
    else:
        table_data: dict[str, list] = {
            "Instrument": all_active,
            "BT Position": [latest_pos.get(i, 0.0) for i in all_active],
            "BT Avg Price": [latest_avg.get(i, float("nan")) for i in all_active],
        }

        if live and not live_net.empty:
            table_data["Live Position"] = [
                live_net.at[i, "net_position"] if i in live_net.index else 0.0
                for i in all_active
            ]
            table_data["Live Avg Price"] = [
                live_net.at[i, "avg_price"] if i in live_net.index else float("nan")
                for i in all_active
            ]
            table_data["Pos Diff"] = [
                table_data["Live Position"][j] - table_data["BT Position"][j]
                for j in range(len(all_active))
            ]
            table_data["Price Diff"] = [
                table_data["Live Avg Price"][j] - table_data["BT Avg Price"][j]
                for j in range(len(all_active))
            ]
        else:
            table_data["Exposure"] = [
                exposure.ffill().iloc[-1].get(i, 0.0) for i in all_active
            ]
            table_data["Unrealised PnL"] = [
                unrealised.ffill().iloc[-1].get(i, 0.0) for i in all_active
            ]
            table_data["Total PnL"] = [
                total_pnl.ffill().iloc[-1].get(i, 0.0) for i in all_active
            ]

        table_df = pd.DataFrame(table_data).round(2)

        conditional_styles = []
        for col in ["Unrealised PnL", "Total PnL", "Pos Diff", "Price Diff"]:
            if col in table_df.columns:
                conditional_styles.extend(
                    [
                        {
                            "if": {"filter_query": f"{{{col}}} > 0", "column_id": col},
                            "color": GREEN,
                        },
                        {
                            "if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
                            "color": RED,
                        },
                    ]
                )

        table_content = dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=[
                (
                    {
                        "name": c,
                        "id": c,
                        "type": "numeric",
                        "format": {"specifier": ",.2f"},
                    }
                    if c != "Instrument"
                    else {"name": c, "id": c}
                )
                for c in table_df.columns
            ],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "right", "padding": "8px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=conditional_styles,
        )

    # Position chart - overlay live step function
    fig_pos = positions.ffill().plot(title="Net Positions (5d)")
    fig_pos.update_layout(height=350, margin=dict(t=40, b=40))

    if live and not live_raw.empty:
        for epic in live_raw["epic"].unique():
            if epic not in positions.columns:
                continue
            trades = live_raw[live_raw.epic == epic].sort_values("created")
            cum_pos = trades.set_index("created")["signed_size"].cumsum()
            # Extend to now so the step is visible
            cum_pos[pd.Timestamp.now("UTC")] = cum_pos.iloc[-1]
            fig_pos.add_trace(
                go.Scatter(
                    x=cum_pos.index,
                    y=cum_pos.values,
                    mode="lines",
                    name=f"{epic} (live)",
                    line=dict(dash="dot", width=2),
                )
            )

    # Exposure chart
    fig_exp = exposure.ffill().plot(title="Net Exposure (5d)")
    fig_exp.update_layout(height=350, margin=dict(t=40, b=40))

    # Per-instrument unrealised PnL (live PnL)
    active_cols = [c for c in positions.columns if c in bt_active | live_active]
    unrealised_active = unrealised.ffill()[
        [c for c in active_cols if c in unrealised.columns]
    ]
    if not unrealised_active.empty:
        fig_unrealised = go.Figure()
        for col in unrealised_active.columns:
            fig_unrealised.add_trace(
                go.Scatter(
                    x=unrealised_active.index,
                    y=unrealised_active[col],
                    mode="lines",
                    name=col,
                )
            )
        fig_unrealised.update_layout(
            title="Unrealised PnL per Position",
            height=350,
            margin=dict(t=40, b=40),
            yaxis_title="PnL",
        )
    else:
        fig_unrealised = empty_fig("No unrealised PnL data")

    # Per-instrument total PnL (theoretical PnL)
    total_pnl_active = total_pnl.ffill()[
        [c for c in active_cols if c in total_pnl.columns]
    ]
    if not total_pnl_active.empty:
        fig_total = go.Figure()
        for col in total_pnl_active.columns:
            fig_total.add_trace(
                go.Scatter(
                    x=total_pnl_active.index,
                    y=total_pnl_active[col],
                    mode="lines",
                    name=col,
                )
            )
        fig_total.update_layout(
            title="Total PnL per Position",
            height=350,
            margin=dict(t=40, b=40),
            yaxis_title="PnL",
        )
    else:
        fig_total = empty_fig("No total PnL data")

    return (
        html.Div([html.H5("Open Positions"), table_content]),
        fig_pos,
        fig_exp,
        fig_unrealised,
        fig_total,
    )


# ---- Compare Tab ----


@callback(
    Output("compare-portfolio-select", "options"),
    Input("portfolio-select", "options"),
)
def compare_portfolio_options(options: list[str]) -> Any:
    return options if options else dash.no_update


@callback(
    Output("compare-cumulative", "figure"),
    Output("compare-daily", "figure"),
    Output("compare-stats", "children"),
    Input("portfolio-select", "value"),
    Input("compare-portfolio-select", "value"),
    Input("tabs", "active_tab"),
    Input("compare-dates", "start_date"),
    Input("compare-dates", "end_date"),
)
def update_compare(
    portfolio_a: str | None,
    portfolio_b: str | None,
    tab: str,
    start_date: str | None,
    end_date: str | None,
) -> tuple:
    if not portfolio_a or not portfolio_b or tab != "compare":
        raise dash.exceptions.PreventUpdate

    start = (
        pd.Timestamp(start_date).tz_localize("UTC")
        if start_date
        else pd.Timestamp.now("UTC") - pd.offsets.BDay(60)
    )
    end = (
        pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
        if end_date
        else pd.Timestamp.now("UTC")
    )
    no_data: tuple = (
        empty_fig("Select two portfolios"),
        empty_fig(),
        html.P("Select two portfolios to compare."),
    )

    try:
        pf_a = read_portfolio(portfolio_a, date_range=(start, end))
        pf_b = read_portfolio(portfolio_b, date_range=(start, end))
    except Exception:
        return no_data
    if pf_a.empty or pf_b.empty:
        return no_data

    # Daily PnL series for each
    daily_a = pf_a.resample("B").last().ffill().total_pnl.diff().dropna()
    daily_b = pf_b.resample("B").last().ffill().total_pnl.diff().dropna()
    cum_a = daily_a.cumsum()
    cum_b = daily_b.cumsum()

    # Overlaid cumulative PnL
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=cum_a.index,
            y=cum_a.values,
            mode="lines",
            name=portfolio_a,
            line_color=BLUE,
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=cum_b.index,
            y=cum_b.values,
            mode="lines",
            name=portfolio_b,
            line_color=ORANGE,
        )
    )
    fig_cum.update_layout(
        title="Cumulative PnL Comparison",
        height=400,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Grouped daily PnL bars
    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(x=daily_a.index, y=daily_a.values, name=portfolio_a, marker_color=BLUE)
    )
    fig_daily.add_trace(
        go.Bar(x=daily_b.index, y=daily_b.values, name=portfolio_b, marker_color=ORANGE)
    )
    fig_daily.update_layout(
        title="Daily PnL",
        barmode="group",
        height=350,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Side-by-side stats
    stats_a = compute_daily_stats(pf_a)
    stats_b = compute_daily_stats(pf_b)
    header = html.Thead(
        html.Tr([html.Th("Metric"), html.Th(portfolio_a), html.Th(portfolio_b)])
    )
    body = html.Tbody(
        [
            html.Tr([html.Td(k), html.Td(stats_a[k]), html.Td(stats_b[k])])
            for k in stats_a
        ]
    )
    stats_table = dbc.Table(
        [header, body],
        bordered=True,
        hover=True,
        size="sm",
        className="mt-2",
    )

    return fig_cum, fig_daily, html.Div([html.H5("Comparison"), stats_table])


# ---- Entry point ----


def main() -> int:
    global ARCTIC_URI

    parser = argparse.ArgumentParser(description="Tradingo Backtest Monitor")
    parser.add_argument(
        "--arctic-uri",
        default=None,
        help="ArcticDB URI (overrides $TP_ARCTIC_URI)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ARCTIC_URI = args.arctic_uri or os.environ.get("TP_ARCTIC_URI", "")
    if not ARCTIC_URI:
        print("Error: provide --arctic-uri or set TP_ARCTIC_URI")
        return 1

    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
