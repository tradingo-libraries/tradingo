import os
import urllib.parse
from dash import Dash, State, html, dcc, callback, Output, Input
from flask import Flask
from pandas.core.generic import weakref
import plotly.express as px
import pandas as pd
from tradingo.api import Tradingo

ARCTIC_URL = os.environ.get(
    "TRADINGO_ARCTIC_URL",
    "lmdb:///home/rory/dev/tradingo-plat/data/prod/tradingo.db",
)

pd.options.plotting.backend = "plotly"


api = Tradingo(
    uri=ARCTIC_URL,
    provider="ig-trading",
    universe="im-multi-asset",
    name="etft",
)

app = Dash()

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(children="Tradingo Monitor", style={"textAlign": "center"}),
                dcc.DatePickerRange(
                    start_date=None,
                    end_date=None,
                    id="date-selection",
                ),
                dcc.Dropdown(
                    value=None,
                    options=api.instruments["im-multi-asset"]().index.to_list(),
                    id="dropdown-selection",
                    multi=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Graph(id="z_score"),
                dcc.Graph(id="net_position"),
                dcc.Graph(id="unrealised_pnl"),
                dcc.Graph(id="total_pnl"),
                dcc.Graph(id="month-to-date"),
            ]
        ),
    ]
)


@callback(
    (
        Output("z_score", "figure"),
        Output("net_position", "figure"),
        Output("unrealised_pnl", "figure"),
        Output("total_pnl", "figure"),
        Output("month-to-date", "figure"),
    ),
    (
        Input("dropdown-selection", "value"),
        Input("date-selection", "start_date"),
        Input("date-selection", "end_date"),
    ),
)
def update_graph(
    value,
    start_date,
    end_date,
):

    def range_breaks(fig):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=[21.5, 13.5], pattern="hour"),
                dict(bounds=["sat", "mon"]),
            ]
        )
        return fig

    value = value or None
    date = pd.Timestamp(start_date or pd.Timestamp.now()).normalize()
    end = pd.Timestamp(
        end_date or (pd.Timestamp.now().normalize() + pd.Timedelta(hours=24))
    ).normalize()
    z_score = api.signals.intraday_momentum.z_score(
        columns=value,
        date_range=(date, end),
    )

    start = z_score.index[0]
    end = z_score.index[-1] + pd.Timedelta(minutes=15)

    unrealised_pnl = (
        api.backtest.intraday.rounded.position.instrument.unrealised_pnl(
            columns=value,
            date_range=(start, end),
        )
        .ffill()
        .fillna(0.0)
        .diff()
        .cumsum()
        .fillna(0.0)
    )

    net_position = api.backtest.intraday.rounded.position.instrument.net_position(
        columns=value,
        date_range=(start, end),
    )
    total_pnl = (
        api.backtest.intraday.rounded.position.portfolio(
            date_range=(
                start,
                end,
            ),
        )
        .total_pnl.ffill()
        .fillna(0.0)
        .diff()
        .cumsum()
        .fillna(0.0)
    )

    inception = pd.Timestamp("2024-11-07 00:00:00+00:00")

    mtd = (
        api.backtest.intraday.rounded.position.portfolio(
            date_range=(
                inception,
                # end - pd.offsets.BDay(30),
                end,
            ),
        )
        .resample(pd.offsets.BDay(1))
        .last()
        .diff()
        .fillna(0)
        .cumsum()
        .total_pnl
    )

    return (
        range_breaks(z_score.plot(title="z_score")),
        range_breaks(net_position.plot(title="net_position")),
        range_breaks(unrealised_pnl.plot(title="unrealised_pnl")),
        range_breaks(total_pnl.plot(title="total_pnl")),
        mtd.plot(title="Returns since inception"),
    )


server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
    )
