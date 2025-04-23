import datetime

import dash
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, callback, dcc, html

from tradingo.api import Tradingo
from tradingo.settings import TradingoConfig

config = TradingoConfig.from_env().to_env()

ARCTIC_URL = config.arctic_uri

pd.options.plotting.backend = "plotly"


app = Dash()

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                html.H1(children="Tradingo Monitor", style={"textAlign": "center"}),
                dcc.DatePickerRange(
                    start_date=None,
                    end_date=None,
                    id="date-selection",
                ),
                dcc.Dropdown(
                    options=[],
                    placeholder="Universe",
                    value="im-multi-asset",
                    id="universe-selection",
                    multi=False,
                ),
                dcc.Dropdown(
                    value=None,
                    options=[],
                    placeholder="Symbols",
                    id="asset-selection",
                    multi=True,
                ),
                dcc.Dropdown(
                    value=None,
                    options=[],
                    placeholder="Portfolio",
                    id="portfolio-selection",
                    multi=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Graph(id="day-to-date"),
                dcc.Graph(id="week-to-date"),
                dcc.Graph(id="month-to-date"),
                dcc.Graph(id="year-to-date"),
                dcc.Graph(id="z_score"),
                dcc.Graph(id="net_position"),
                dcc.Graph(id="raw_position"),
                dcc.Graph(id="unrealised_pnl"),
                dcc.Graph(id="total_pnl"),
                dcc.Graph(id="instrument_total_pnl"),
                dcc.Graph(id="gross_margin"),
                dcc.Graph(id="margin"),
                dcc.Graph(id="short_vol"),
                dcc.Graph(id="long_vol"),
            ]
        ),
    ]
)


@callback(
    Output("universe-selection", "options"),
    Input("url", "pathname"),
)
def set_universe_options(_):
    api = Tradingo(
        uri=ARCTIC_URL,
    )
    return api.instruments._library.list_symbols()


@callback(
    Output("asset-selection", "options"),
    Input("universe-selection", "value"),
)
def set_asset_options(universe):
    if not universe:
        return dash.no_update
    api = Tradingo(
        uri=ARCTIC_URL,
    )
    return api.instruments[universe]().index.to_list()


@callback(
    Output("portfolio-selection", "options"),
    Input("universe-selection", "value"),
)
def set_portfolio_options(universe):
    if not universe:
        return dash.no_update
    api = Tradingo(
        uri=ARCTIC_URL,
    )
    return api.portfolio.list()


@callback(
    Output("portfolio-selection", "value"),
    Input("portfolio-selection", "options"),
)
def set_portfolio_value(options):
    if not options:
        return dash.no_update
    return options[0]


@callback(
    (
        Output("z_score", "figure"),
        Output("net_position", "figure"),
        Output("raw_position", "figure"),
        Output("unrealised_pnl", "figure"),
        Output("total_pnl", "figure"),
        Output("instrument_total_pnl", "figure"),
        Output("margin", "figure"),
        Output("gross_margin", "figure"),
        Output("short_vol", "figure"),
        Output("long_vol", "figure"),
        Output("day-to-date", "figure"),
        Output("week-to-date", "figure"),
        Output("month-to-date", "figure"),
        Output("year-to-date", "figure"),
    ),
    (
        Input("asset-selection", "value"),
        Input("date-selection", "start_date"),
        Input("date-selection", "end_date"),
        Input("universe-selection", "value"),
        Input("portfolio-selection", "value"),
    ),
)
def update_graph(
    assets,
    start_date,
    end_date,
    universe,
    portfolio,
):
    if not universe or not portfolio:
        return dash.no_update

    api = Tradingo(
        uri=ARCTIC_URL,
    )

    def range_breaks(fig):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=[22, 13], pattern="hour"),
                dict(bounds=["sat", "mon"], pattern="day of week"),
            ]
        )
        return fig

    assets = assets or None
    date = pd.Timestamp(
        start_date or (pd.Timestamp.now() - pd.offsets.BDay(0))
    ).normalize()
    end = pd.Timestamp(
        end_date or (pd.Timestamp.now().normalize() + pd.Timedelta(hours=24))
    ).normalize()

    z_score = api.signals.intraday_momentum.z_score[universe](
        columns=assets,
        date_range=(date, end),
    )

    short_vol = api.signals.intraday_momentum.short_vol[universe](
        date_range=((end or pd.Timestamp.now()) - pd.offsets.BDay(30), end),
        columns=assets,
    ).resample("B").last() * np.sqrt(252)
    long_vol = api.signals.intraday_momentum.long_vol[universe](
        date_range=((end or pd.Timestamp.now()) - pd.offsets.BDay(30), end),
        columns=assets,
    ).resample("B").last() * np.sqrt(252)

    start = z_score.index[0]
    end = z_score.index[-1] + pd.Timedelta(minutes=15)

    unrealised_pnl = (
        api.backtest[portfolio]
        .rounded.position.instrument.unrealised_pnl(
            columns=assets,
            date_range=(start, end),
        )
        .ffill()
        .fillna(0.0)
        .diff()
        .cumsum()
        .fillna(0.0)
    )

    net_position = api.backtest[portfolio].rounded.position.instrument.net_position(
        columns=assets,
        date_range=(start, end),
    )
    raw_position = api.portfolio[portfolio].raw.position(
        columns=assets,
        date_range=(start, end),
    )
    margin = (
        api.backtest[portfolio]
        .rounded.position.instrument.net_exposure(
            date_range=(start, end),
            columns=assets,
        )
        .abs()
        * 0.05
    )
    gross_margin = (
        api.backtest[portfolio]
        .rounded.position.portfolio(
            date_range=(start, end),
        )
        .gross_exposure.abs()
        * 0.05
    )
    total_pnl = (
        api.backtest[portfolio]
        .rounded.position.portfolio(
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
    instrument_total_pnl = (
        api.backtest[portfolio]
        .rounded.position.instrument.total_pnl(
            date_range=(
                start - pd.offsets.YearBegin(1),
                end,
            ),
        )
        .resample(pd.offsets.BDay(1))
        .last()
        .ffill()
        .fillna(0.0)
        .diff()
        .cumsum()
        .fillna(0.0)
    )

    returns = api.backtest[portfolio].rounded.position.portfolio(
        date_range=(
            end - pd.offsets.BDay(252),
            end,
        ),
    )
    one_year = (
        returns.resample(pd.offsets.BDay(1)).last().diff().fillna(0).cumsum().total_pnl
    )
    one_month = (
        returns.loc[end - pd.offsets.Day(30) : end]
        .resample(pd.offsets.BDay(1))
        .last()
        .diff()
        .fillna(0)
        .cumsum()
        .total_pnl
    )
    one_week = (
        returns.loc[end - pd.offsets.Week(1) : end]
        .resample(pd.offsets.Hour(4))
        .last()
        .diff()
        .fillna(0)
        .cumsum()
        .total_pnl
    )
    one_day = (
        returns[
            (returns.index.time < datetime.time.fromisoformat("21:30"))
            & (returns.index.time > datetime.time.fromisoformat("09:30"))
        ]
        .loc[end - pd.offsets.BDay(1) : end]
        .diff()
        .fillna(0)
        .cumsum()
        .total_pnl
    )

    return (
        range_breaks(z_score.plot(title="z_score")),
        range_breaks(net_position.plot(title="net_position")),
        range_breaks(raw_position.plot(title="raw_position")),
        range_breaks(unrealised_pnl.plot(title="unrealised_pnl")),
        range_breaks(total_pnl.plot(title="total_pnl")),
        instrument_total_pnl.plot(title="instrument_total_pnl"),
        range_breaks(
            (gross_margin - unrealised_pnl.sum(axis=1)).plot(title="total_margin")
        ),
        range_breaks(margin.plot(title="margin")),
        # range_breaks(short_vol.plot(title="short_vol")),
        short_vol.plot(title="short_vol"),
        long_vol.plot(title="long_vol"),
        one_day.plot(title="DTD"),
        one_week.plot(title="WTD"),
        one_month.plot(title="MTD"),
        one_year.plot(title="YTD"),
    )


server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
    )
