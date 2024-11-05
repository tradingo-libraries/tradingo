from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from tradingo.api import Tradingo

ARCTIC_URL = (
    "s3://s3.us-east-1.amazonaws.com:tradingo-store?aws_auth=true&path_prefix=prod"
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
        dcc.Graph(id="graph-content"),
    ]
)


@callback(
    (Output("graph-content", "figure"),),
    Input("dropdown-selection", "value"),
)
def update_graph(value):
    value = value or None
    z_score = api.signals.intraday_momentum.z_score(
        columns=value,
        date_range=(pd.Timestamp.now("utc").normalize(), None),
    )

    unrealised_pnl = api.backtest.intraday.rounded.position.instrument.unrealised_pnl(
        columns=value,
        date_range=(pd.Timestamp.now("utc").normalize(), None),
    )
    return (z_score.plot(), unrealised_pnl.plot())


if __name__ == "__main__":
    app.run(
        debug=True,
    )
