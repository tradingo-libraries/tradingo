import os
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from tradingo.api import Tradingo

ARCTIC_URL = os.environ.get(
    "TRADINGO_ARCTIC_URL",
    "lmdb:///home/rory/dev/tradingo-plat/data/prod/tradingo.db",
)

pd.options.plotting.backend = "plotly"


api = Tradingo(
    uri=ARCTIC_URL, provider="ig-trading", universe="im-multi-asset", name="etft"
)

app = Dash()

app.layout = html.Div(
    [
        html.H1(children="Tradingo Monitor", style={"textAlign": "center"}),
        dcc.Dropdown(
            value=None,
            options=api.instruments["im-multi-asset"]().index.to_list(),
            id="dropdown-selection",
        ),
        dcc.Graph(id="z_score"),
        dcc.Graph(id="unrealised_pnl"),
        dcc.Graph(id="month-to-date"),
    ]
)


@callback(
    (
        Output("z_score", "figure"),
        Output("unrealised_pnl", "figure"),
        Output("month-to-date", "figure"),
    ),
    Input("dropdown-selection", "value"),
)
def update_graph(value):
    value = value or None
    z_score = api.signals.intraday_momentum.z_score(
        columns=value,
        date_range=(pd.Timestamp.now("utc").normalize() + pd.Timedelta(hours=9), None),
    )

    unrealised_pnl = api.backtest.intraday.rounded.position.instrument.unrealised_pnl(
        columns=value,
        date_range=(pd.Timestamp.now("utc").normalize() + pd.Timedelta(hours=13), None),
    )

    mtd = (
        api.backtest.intraday.rounded.position.portfolio(
            date_range=(
                pd.Timestamp.now("utc").normalize() - pd.offsets.Day(30),
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
        mtd.plot(title="Month to date returns"),
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
    )
