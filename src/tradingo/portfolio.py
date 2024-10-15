import logging
import re
from typing import Optional

import pandas as pd
import numpy as np
import riskfolio as rf

import arcticdb as adb

from tradingo.symbols import lib_provider, symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


@lib_provider(model_signals="signals")
@symbol_provider(
    close="prices/close",
    vol="signals/vol_128",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    template="{0}/{1}.{2}",
    symbol_prefix="{provider}.{universe}.{name}.",
)
def portfolio_construction(
    name: str,
    model_signals: adb.library.Library,
    close: pd.DataFrame,
    vol: pd.DataFrame,
    instruments: pd.DataFrame,
    provider: str,
    universe: str,
    signal_weights: dict,
    multiplier: float,
    default_weight: float,
    constraints: dict,
    vol_scale: bool,
    aum: float,
    instrument_weights: Optional[dict] = None,
    **kwargs,
):

    instrument_weights = instrument_weights or {}

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
                        f"{provider}.{universe}.{signal}",
                        date_range=(close.index[0], close.index[-1]),
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

    return (
        (pct_position, ("portfolio", "raw", "percent")),
        (share_position, ("portfolio", "raw", "shares")),
        (positions, ("portfolio", "raw", "position")),
        (pct_position.round(decimals=2), ("portfolio", "rounded", "percent")),
        (share_position.round(), ("portfolio", "rounded", "shares")),
        (positions.round(), ("portfolio", "rounded", "position")),
        (signal_value, ("signals", "raw", "signal")),
    )


@symbol_provider(
    close="prices/{provider}.{universe}.close",
    factor_returns="prices/{factor_provider}.{factor_universe}.close",
)
@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    "portfolio/raw.percent",
    "portfolio/raw.shares",
    symbol_prefix="{provider}.{universe}.{name}.",
)
def portfolio_optimization(
    close: pd.DataFrame,
    factor_returns: pd.DataFrame,
    optimizer_config: dict,
    rebalance_rule: str,
    min_periods: int,
    aum: float,
    **kwargs,
):
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


# FIXME: Not idempotent
@symbol_provider(
    positions="portfolio/{stage}?as_of=-1",
    previous_positions="portfolio/{stage}?as_of=-2",
    symbol_prefix="{provider}.{universe}.{name}.",
)
@symbol_publisher(
    "trades/{stage}",
    symbol_prefix="{provider}.{universe}.{name}.",
)
def calculate_trades(
    name: str,
    stage: str,
    positions: pd.DataFrame,
    previous_positions: pd.DataFrame,
    **kwargs,
):

    logger.info(
        "Calculating %s trades for %s",
        stage,
        name,
    )

    return (
        (
            positions
            - previous_positions.reindex_like(positions, method="ffill").fillna(0.0)
        ).iloc[
            -1:,
        ],
    )
