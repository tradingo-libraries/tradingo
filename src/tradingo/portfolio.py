import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def portfolio_construction(
    signals: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    close: pd.DataFrame,
    aum: float,
    multiplier: float,
    model_weights: dict[str, float],
    instrument_weights: dict[str, dict[str, float]] | None = None,
    default_instrument_weight: float = 1.0,
    instruments: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Construct portfolio from models via allocations, signals and overrides.

    This function works as an allocation engine which combines multiple model signals
    and instrument-specific weights to produce final portfolio positions.
    Univariate parameters:
      * multiplier: constant scaling factor for all weights
      * model_weights: allocation for each model's signals
      * instrument_weights: multipliers for specific instrument attributes
      * signals: ArcticDB library containing signals data
    Cross-sectional parameters:
      * portfolio optimisation function

    The univariate position for instrument j from model i is calculated as:

        position_i = multiplier * model_weight[i] * instrument_weight[i][j] * signal[i][j]

    Overall portfolio positions can be allocated by looking at the whole portfolio

    Parameters
    ----------
    signals : Library
        ArcticDB library containing signals data.
    start_date : pd.Timestamp
        Start date for signals.
    end_date : pd.Timestamp
        End date for signals.
    close : pd.DataFrame
        DataFrame of closing prices for instruments.
    aum : float
        Assets under management in currency units.
    multiplier : float
        Constant portfolio multiplier to scale all weights.
    model_weights : dict[str, float]
        Allocation for each model's signals.
    instrument_weights : dict[str, dict[str, float]], optional
        Multiplier configuration for instruments, by default None.
    default_instrument_weight : float, optional
        Default instrument_weights if not specified, by default 1.0.
    instruments : pd.DataFrame
        DataFrame of instrument attributes.
    """

    if not multiplier > 0.0:
        raise ValueError("multiplier must be positive")

    _ = instruments.index.unique() if instruments is not None else None  # no-op

    # read signals dataframes indexed as (time, (model, symbol))
    signals_df: pd.DataFrame = signals.loc[start_date:end_date].rename_axis(
        ["model", "symbol"], axis=1
    )

    # calculate model-specific instrument weights
    instrument_weights = instrument_weights or {}
    instruments_mult = pd.Series(
        {
            (model_name, instrument): weight
            for model_name in instrument_weights
            for instrument, weight in instrument_weights[model_name].items()
        }
    )
    instruments_mult = (
        pd.Series(index=signals_df.columns)
        if instruments_mult.empty
        else instruments_mult.reindex(index=signals_df.columns)
    ).fillna(default_instrument_weight)

    # calculate model weights
    models_mult = (
        pd.Series(model_weights)
        .reindex(signals_df.columns.get_level_values("model"))
        .set_axis(signals_df.columns)
        .rename_axis(signals_df.columns.names)
    )

    # apply weights to signals
    total_mult = models_mult * instruments_mult * multiplier
    signals_df = signals_df.multiply(total_mult, axis=1)

    # TODO: implement cross-sectional portfolio optimisation here

    # aggregate signals to get final position to trade
    signal_value = (
        signals_df.transpose().groupby(level=[1]).sum().transpose().ffill().fillna(0.0)
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
