import logging
import re
from typing import cast

import numpy as np
import pandas as pd
from arcticdb import VersionedItem
from arcticdb.arctic import Library

from tradingo import symbols

logger = logging.getLogger(__name__)


@symbols.lib_provider(signals="signals")  # pyright: ignore
def portfolio_construction(
    signals: Library,
    close: pd.DataFrame,
    model_weights: dict[str, float],
    multiplier: float,
    aum: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    instruments: pd.DataFrame | None = None,
    default_instrument_weight: float = 1.0,
    instrument_weights: dict[str, dict[str, float]] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Catch all portfolio construction function for basic
    portfolio construction routines

    :param signals: library of where to find signals
    :param close: dataframe of close prices
    :param model_weights: dictionary of model weights. Keys correspond
        to a symbol in the library
    :param multiplier: some scalar to multiply weights by
    :param aum: notional aum value
    :param start_date: the start date to use
    :param end_date: the end date to use
    :param instruments: optional dataframe of instruments for asset type weights
    :param instrument_weights: loading to apply to instruments
    """
    instrument_weights = instrument_weights or {}

    weights = pd.Series(multiplier, index=close.columns)

    if instruments is not None:

        instruments["Symbol"] = instruments.index

        for key, weights_config in instrument_weights.items():
            if key not in instruments.columns:
                continue

            weights = weights * instruments.apply(
                lambda i: weights_config.get(i[key], default_instrument_weight),
                axis=1,
            )
            logger.info("Weights: %s", weights)

    signal_value = (
        pd.concat(
            (
                weight
                * pd.DataFrame(
                    cast(
                        VersionedItem,
                        signals.read(
                            model_name,
                            date_range=(start_date, end_date),
                        ),
                    ).data
                )
                for model_name, weight in model_weights.items()
            ),
            keys=model_weights,
            axis=1,
        )
        .transpose()
        .groupby(level=[1])
        .sum()
        .transpose()
        .ffill()
        .fillna(0)
    )

    positions = signal_value.reindex_like(close).ffill()

    pct_position = positions.div(positions.transpose().sum(), axis=0).fillna(0.0)
    for col in pct_position.columns:
        pct_position.loc[
            pct_position.index == pct_position.first_valid_index(), col
        ] = 0.0

    share_position = (pct_position * aum) / close

    return (
        pct_position,
        share_position,
        positions,
        pct_position.round(decimals=2),
        share_position.round(),
        (multiplier * positions).round(),
        signal_value,
    )


def instrument_ivol(
    close: pd.DataFrame,
) -> pd.DataFrame:
    pct_returns = cast(pd.DataFrame, np.log(close / close.shift()))

    ivols = []

    for symbol in pct_returns.columns:
        universe = pct_returns.drop(symbol, axis=1)

        def vol(uni: pd.DataFrame) -> pd.Series[float]:
            return (1 - (1 + uni).prod(axis=1).pow(1 / 100)).ewm(10).std()

        ivol = vol(pd.concat((universe, pct_returns[symbol]), axis=1)) - vol(universe)
        ivols.append(ivol.rename(symbol))

    return pd.concat(ivols, axis=1).rename_axis("Symbol")


def _parse_ticker(t: str) -> str:
    match = re.match(r".*\(([A-Z]{4})\)", t)
    if match is not None:
        return match.groups()[0]
    return t


def position_from_trades(
    close: pd.DataFrame,
    aum: float,
    trade_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = (
        pd.read_csv(trade_file, parse_dates=["Date"])
        .dropna(axis=0, how="all")
        .sort_values(["Date"])
    )
    trades = trades[
        trades["Order type"].isin(["AtBest", "Quote and Deal"])
        & trades["Order status"].eq("Completed")
    ]
    trades["Ticker"] = trades["Investment"].apply(_parse_ticker) + ".L"
    position_shares = cast(
        pd.DataFrame,
        trades.set_index(["Date", "Ticker"])
        .groupby(["Date", "Ticker"])
        .sum()
        .unstack(),
    )
    position_shares = (
        position_shares[["My units"]]
        .fillna(0.0)
        .cumsum()
        .reindex_like(close)
        .ffill()
        .fillna(0.0)
    )

    position_pct = (position_shares * close) / aum
    return (
        position_pct,
        position_shares,
    )


def point_in_time_position(positions: pd.DataFrame) -> pd.DataFrame:
    return positions.iloc[-1:,]
