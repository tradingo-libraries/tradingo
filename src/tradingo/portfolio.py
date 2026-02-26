import logging
import re
from typing import cast

import numba
import numpy as np
import numpy.typing as npt
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


def aggregate_portfolio(
    prices: pd.DataFrame,
    model_weights: dict[str, float],
    gearing: float,
    **models: pd.DataFrame,
) -> pd.DataFrame:
    if not models:
        return pd.DataFrame(columns=prices.columns)
    weights = pd.Series(model_weights)
    model_positions = pd.concat(models, axis=1)
    return (
        (model_positions.mul(weights, level=0) * gearing)
        .transpose()
        .groupby(level=1)
        .sum()
        .reindex(prices.columns)
        .transpose()
        .rename_axis("DateTime")
    )


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


def apply_dealing_rules(
    positions: pd.DataFrame,
    instruments: pd.DataFrame,
    lot_size_col: str = "instrument.lotSize",
    min_deal_size_col: str = "dealingRules.minDealSize.value",
) -> pd.DataFrame:
    """Apply dealing rules to positions.

    Rounds positions to lot size multiples and filters trades below minimum deal size.
    Trades are computed from current adjusted position to target rounded position.

    :param positions: Wide format DataFrame (index=time, columns=symbols)
    :param instruments: DataFrame with dealing rules (index=Symbol)
    :param lot_size_col: Column for lot size (position must be multiple of this)
    :param min_deal_size_col: Column for minimum trade size
    :return: Adjusted positions
    """
    if not lot_size_col:
        lot_sizes = np.asarray(pd.Series(1.0, index=positions.columns))
    else:
        lot_sizes = np.asarray(
            instruments[lot_size_col].reindex(positions.columns).fillna(1).astype(float)
        )
    if not min_deal_size_col:
        min_deals = np.asarray(pd.Series(1.0, index=positions.columns))
    else:
        min_deals = np.asarray(
            instruments[min_deal_size_col]
            .reindex(positions.columns)
            .fillna(0)
            .astype(float)
        )

    rounded: npt.NDArray[np.float64] = (
        np.round(positions.values / lot_sizes) * lot_sizes
    )

    # Apply min deal filter using numpy (vectorized across columns)
    result = _apply_min_deal_filter(rounded, min_deals)

    return pd.DataFrame(result, index=positions.index, columns=positions.columns)


@numba.jit(nopython=True)  # type: ignore
def _apply_min_deal_filter(
    rounded: npt.NDArray[np.float64],
    min_deals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Filter trades below minimum deal size."""
    n_rows, n_cols = rounded.shape
    result = np.empty((n_rows, n_cols), dtype=rounded.dtype)
    current = np.zeros(n_cols, dtype=rounded.dtype)

    for i in range(n_rows):
        target = rounded[i]
        can_trade = np.abs(target - current) >= min_deals
        current = np.where(can_trade, target, current)
        result[i] = current

    return result


def volatility_target(
    model: pd.DataFrame,
    close: pd.DataFrame,
    target_volatility: float,
    window: int = 20,
    annualization_factor: float = 252.0,
    min_periods: int | None = None,
    vol_floor: float = 1e-6,
    vol_cap: float | None = None,
) -> pd.DataFrame:
    """Scale model weights to achieve a target volatility per instrument.

    Computes the rolling realized volatility of each instrument and scales
    the model weights such that ex-post, each instrument would have achieved
    the target volatility.

    The scaling factor is: target_volatility / realized_volatility

    :param model: DataFrame of model weights/signals (index=time, columns=symbols).
        These will be scaled by the volatility adjustment factor.
    :param close: DataFrame of close prices (index=time, columns=symbols).
        Used to compute realized volatility.
    :param target_volatility: The target annualized volatility for each instrument
        (e.g., 0.10 for 10% annualized volatility).
    :param window: Rolling window size for volatility calculation (default: 20).
    :param annualization_factor: Factor to annualize volatility. Use 252 for daily
        data, 52 for weekly, 12 for monthly (default: 252).
    :param min_periods: Minimum number of observations required for volatility
        calculation. Defaults to window size if not specified.
    :param vol_floor: Minimum volatility value to avoid division by very small
        numbers (default: 1e-6). Realized vol is floored at this value.
    :param vol_cap: Optional maximum scaling factor. If specified, the scaling
        factor is capped at vol_cap to prevent extreme leverage.
    :return: DataFrame of scaled model weights with same shape as model.

    Example:
        If an instrument has 5% realized volatility and target is 10%,
        the model weight for that instrument is multiplied by 2.0.

        If realized volatility is 20% and target is 10%,
        the model weight is multiplied by 0.5.
    """
    if min_periods is None:
        min_periods = window

    # Compute log returns
    log_returns = pd.DataFrame(np.log(close / close.shift()))

    # Compute rolling realized volatility (annualized)
    realized_vol = log_returns.rolling(
        window=window, min_periods=min_periods
    ).std() * np.sqrt(annualization_factor)

    # Floor volatility to avoid division by near-zero values
    realized_vol = realized_vol.clip(lower=vol_floor)

    # Compute scaling factor
    scaling_factor = target_volatility / realized_vol

    # Optionally cap the scaling factor to prevent extreme leverage
    if vol_cap is not None:
        scaling_factor = scaling_factor.clip(upper=vol_cap)

    # Align model to close prices and apply scaling
    aligned_model = model.reindex_like(close).ffill()
    scaled_model = aligned_model * scaling_factor.reindex_like(aligned_model)

    return pd.DataFrame(scaled_model.fillna(0.0))
