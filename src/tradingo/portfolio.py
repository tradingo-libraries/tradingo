import logging
import re

import pandas as pd
import numpy as np

import arcticdb as adb

from tradingo.symbols import lib_provider, symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


@lib_provider(model_signals="signals")
@symbol_provider(
    close="prices/close",
    ivol="prices/ivol",
    vol="signals/vol_128",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    "portfolio/raw.percent",
    "portfolio/raw.shares",
    "signals/raw.signal",
    symbol_prefix="{provider}.{universe}.{name}.",
)
def portfolio_construction(
    name: str,
    model_signals: adb.library.Library,
    close: pd.DataFrame,
    ivol: pd.DataFrame,
    vol: pd.DataFrame,
    instruments: pd.DataFrame,
    provider: str,
    universe: str,
    signal_weights: dict,
    multiplier: float,
    default_weight: float,
    constraints: dict,
    instrument_weights: dict,
    vol_scale: bool,
    aum: float,
    **kwargs,
):

    ivol = ivol.iloc[-1].sort_values(ascending=False)

    weights = pd.Series(multiplier, index=instruments.index)

    instruments["Symbol"] = instruments.index

    for key, weights_config in instrument_weights.items():

        if key not in instruments.columns:
            continue

        weights = weights * instruments.apply(
            lambda i: weights_config.get(i[key], default_weight),
            axis=1,
        )
        logger.info("Weights: %s", weights)

    if signal_weights:

        signal_value = (
            pd.concat(
                (
                    model_signals.read(
                        f"{provider}.{universe}.{signal}"
                        # , columns=symbols
                    ).data
                    * weights
                    * signal_weight
                    for signal, signal_weight in signal_weights.items()
                ),
                keys=signal_weights,
                axis=1,
            )
            .transpose()
            .groupby(level=[1])
            .sum()
            .transpose()
            .ffill()
            .fillna(0)
        )

    else:
        signal_value = weights * pd.DataFrame(
            1,
            index=close.index,
            columns=close.columns,
        )

    if vol_scale:

        weights = weights.div(10 * vol)

    logger.info("signal_data: %s", signal_value)

    positions = signal_value.reindex_like(close).ffill()

    if constraints["long_only"]:
        positions[(positions < 0.0)] = 0.0

    pct_position = positions.div(positions.transpose().sum(), axis=0).fillna(0.0)
    for col in pct_position.columns:
        pct_position.loc[
            pct_position.index == pct_position.first_valid_index(), col
        ] = 0.0

    share_position = (pct_position * aum) / close

    return (pct_position, share_position, signal_value)


@symbol_provider(close="prices/adj_close", symbol_prefix="{provider}.{universe}.")
@symbol_publisher("prices/ivol", symbol_prefix="{provider}.{universe}.")
def instrument_ivol(close, provider, **kwargs):
    pct_returns = np.log(close / close.shift())

    ivols = []

    for symbol in pct_returns.columns:
        universe = pct_returns.drop(symbol, axis=1)

        def vol(uni):
            return (1 - (1 + uni).prod(axis=1).pow(1 / 100)).ewm(10).std()

        ivol = vol(pd.concat((universe, pct_returns[symbol]), axis=1)) - vol(universe)
        ivols.append(ivol.rename(symbol))

    return (pd.concat(ivols, axis=1).rename_axis("Symbol"),)


@symbol_provider(
    close="prices/close",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    "portfolio/raw.percent",
    "portfolio/raw.shares",
    symbol_prefix="{provider}.{universe}.{name}.",
)
def position_from_trades(
    close: pd.DataFrame,
    instruments: pd.DataFrame,
    aum: float,
    trade_file: str,
    **kwargs,
):

    trades = (
        pd.read_csv(trade_file, parse_dates=["Date"])
        .dropna(axis=0, how="all")
        .sort_values(["Date"])
    )
    trades = trades[
        trades["Order type"].isin(["AtBest", "Quote and Deal"])
        & trades["Order status"].eq("Completed")
    ]
    trades["Ticker"] = (
        trades["Investment"].apply(
            lambda t: re.match(".*\(([A-Z]{4})\)", t).groups()[0]
        )
        + ".L"
    )
    position_shares = (
        trades.set_index(["Date", "Ticker"])
        .groupby(["Date", "Ticker"])
        .sum()
        .unstack()["My units"]
        .fillna(0.0)
        .cumsum()
        .reindex_like(
            close,
        )
        .ffill()
        .fillna(0.0)
    )

    position_pct = (position_shares * close) / aum
    return (
        position_pct,
        position_shares,
    )
