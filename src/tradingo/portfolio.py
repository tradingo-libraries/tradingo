import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from arcticdb.arctic import Library

from tradingo import symbols

logger = logging.getLogger(__name__)


@symbols.lib_provider(signals="signals")
def portfolio_construction(
    signals: Library,
    close: pd.DataFrame,
    instruments: pd.DataFrame,
    model_weights: dict[str, float],
    multiplier: float,
    aum: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    default_instrument_weight: float = 1.0,
    instrument_weights: Optional[dict] = None,
):
    instrument_weights = instrument_weights or {}

    weights = pd.Series(multiplier, index=instruments.index)

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
                * signals.read(
                    model_name,
                    date_range=(start_date, end_date),
                ).data
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
        positions.round(),
        signal_value,
    )


def portfolio_optimization(
    close: pd.DataFrame,
    factor_returns: pd.DataFrame,
    optimizer_config: dict,
    rebalance_rule: str,
    min_periods: int,
    aum: float,
):
    import riskfolio as rf

    def get_weights(
        returns,
        factors,
    ):
        port = rf.Portfolio(returns=returns)

        port.assets_stats(method_mu="hist", method_cov="ledoit")
        port.lowerret = 0.00056488 * 1.5

        port.factors = factors

        port.factors_stats(
            method_mu="hist",
            method_cov="ledoit",
            feature_selection="PCR",
        )

        w = port.optimization(
            model="FM",
            rm="MV",
            obj="Sharpe",
            hist=False,
        )
        return (
            w.squeeze() if w is not None else pd.Series(np.nan, index=returns.columns)
        )

    asset_returns = close.pct_change().dropna()
    factor_returns = close.pct_change().dropna().reindex(asset_returns.index)

    data = []
    for i, _ in enumerate(asset_returns.index):
        if i < min_periods:
            data.append(
                pd.Series(np.nan, index=asset_returns.columns).to_frame().transpose()
            )
            continue

        ret_subset = asset_returns.iloc[:i]
        data.append(
            get_weights(ret_subset, factor_returns.loc[ret_subset.index])
            .to_frame()
            .transpose()
        )

    pct_position = pd.concat(data, keys=asset_returns.index).droplevel(1)
    share_position = (pct_position * aum) / close

    return (pct_position, share_position)


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


def _parse_ticker(t: str):
    match = re.match(r".*\(([A-Z]{4})\)", t)
    if match is not None:
        return match.groups()[0]
    return t


def position_from_trades(
    close: pd.DataFrame,
    aum: float,
    trade_file: str,
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
    trades["Ticker"] = trades["Investment"].apply(_parse_ticker) + ".L"
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


def point_in_time_position(positions: pd.DataFrame):
    return ((positions).iloc[-1:,],)
