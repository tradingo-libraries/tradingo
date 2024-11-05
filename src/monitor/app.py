import os
from dash import Dash, html, dcc, callback, Output, Input
from flask import Flask
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
        html.H1(children="Tradingo Monitor", style={"textAlign": "center"}),
        dcc.DatePickerSingle(date=pd.Timestamp.now().normalize(), id="date-selection"),
        dcc.Dropdown(
            value=None,
            options=api.instruments["im-multi-asset"]().index.to_list(),
            id="dropdown-selection",
            multi=True,
        ),
        dcc.Graph(id="z_score"),
        dcc.Graph(id="net_position"),
        dcc.Graph(id="unrealised_pnl"),
        dcc.Graph(id="total_pnl"),
        dcc.Graph(id="month-to-date"),
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
        Input("date-selection", "date"),
    ),
)
def update_graph(
    value,
    date,
):
    value = value or None
    date = pd.Timestamp(date).normalize()
    end = date + pd.Timedelta(hours=24)
    z_score = api.signals.intraday_momentum.z_score(
        columns=value,
        date_range=(date, end),
    )

    start = z_score.index[0]
    end = z_score.index[-1] + pd.Timedelta(minutes=15)

    unrealised_pnl = api.backtest.intraday.rounded.position.instrument.unrealised_pnl(
        columns=value,
        date_range=(start, end),
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
        .total_pnl.diff()
        .fillna(0)
        .cumsum()
    )

    mtd = (
        api.backtest.intraday.rounded.position.portfolio(
            date_range=(
                date - pd.offsets.Day(30),
                None,
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
        z_score.plot(title="z_score"),
        unrealised_pnl.plot(title="unrealised_pnl"),
        total_pnl.plot(title="total_pnl"),
        net_position.plot(title="net_position"),
        mtd.plot(title="Month to date returns"),
    )


server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
    )
