import logging
import re
from typing import Literal, cast

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
        .transpose()
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
    lot_sizes = np.asarray(
        instruments[lot_size_col].reindex(positions.columns).fillna(1)
    )
    min_deals = np.asarray(
        instruments[min_deal_size_col].reindex(positions.columns).fillna(0)
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
        current = np.where(can_trade, target, current)  # type: ignore
        result[i] = current

    return result


def stop_loss(
    position: pd.DataFrame,
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    aum: float,
    max_percent: float,
    mode: Literal["max-drawdown", "price"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate stop loss levels and apply them to positions.

    Computes stop loss price levels based on a percentage move from entry price,
    and returns a position time series that obeys the stop loss constraints.
    Max-Drawdown Stop Loss Calculation:

    stop_level = current_mid_price + (threshold - cumulative_pnl) / position_size

    Where:
    - threshold = -max_percent * AUM (maximum allowed loss)
    - cumulative_pnl = sum of (price_change × position) since position opened
    - position_size is signed (positive for long, negative for short)

    How it works:
    ┌───────────┬────────────┬──────────┬─────────────────────────────┐
    │ Position  │ Price Move │   PnL    │     Stop Level Location     │
    ├───────────┼────────────┼──────────┼─────────────────────────────┤
    │ Long (+)  │ Falls      │ Loss (-) │ Below current price         │
    ├───────────┼────────────┼──────────┼─────────────────────────────┤
    │ Long (+)  │ Rises      │ Gain (+) │ Further below current price │
    ├───────────┼────────────┼──────────┼─────────────────────────────┤
    │ Short (-) │ Rises      │ Loss (-) │ Above current price         │
    ├───────────┼────────────┼──────────┼─────────────────────────────┤
    │ Short (-) │ Falls      │ Gain (+) │ Further above current price │
    └───────────┴────────────┴──────────┴─────────────────────────────┘
    Example:
    - AUM = 1000, max_percent = 0.05 → threshold = -50
    - Long 10 units at mid-price 100, PnL = 0
    - Stop level = 100 + (-50 - 0) / 10 = 95
    - If price falls to 95, loss = 10 × (95-100) = -50, hitting threshold

    The position is closed (set to 0) when cumulative PnL breaches the threshold.

    :param position: Wide format DataFrame of positions (index=time, columns=symbols).
        Positive values indicate long positions, negative indicate short.
    :param bid: DataFrame of bid prices, same shape as position.
    :param ask: DataFrame of ask prices, same shape as position.
    :param aum: Assets under management (used for max-drawdown mode).
    :param max_percent: Maximum adverse price move allowed as a decimal (e.g., 0.05 for 5%).
        For long positions, stop triggers when price falls by this percentage.
        For short positions, stop triggers when price rises by this percentage.
    :param mode: Stop loss calculation mode:
        - "price": Stop based on percentage price move from entry.
        - "max-drawdown": Stop based on cumulative PnL loss as percentage of AUM.
    :return: Tuple of (stop_lossed_position, stop_levels):
        - stop_lossed_position: Positions set to 0 where stop loss triggered.
        - stop_levels: The calculated stop loss price levels.

    Example (price mode):
        If a long position is opened at bid price 100 with max_percent=0.05,
        the stop level is 95 (100 * (1 - 0.05)). If bid falls below 95,
        the position is closed (set to 0).

        For a short position opened at ask price 100 with max_percent=0.05,
        the stop level is 105 (100 * (1 + 0.05)). If ask rises above 105,
        the position is closed.

    Example (max-drawdown mode):
        With AUM=1000 and max_percent=0.05, maximum allowed loss is 50.
        For a long position of 10 units at mid-price 100:
        - If price falls to 95, cumulative loss = 10 * (95-100) = -50
        - This hits the threshold, so position is closed.
        - Stop level = current_price + (threshold - pnl) / position
                     = 100 + (-50 - 0) / 10 = 95
    """
    if mode == "max-drawdown":
        # Calculate cumulative PnL: price change * lagged position
        mid_price = (ask + bid) / 2
        pnl = (mid_price.diff() * position.shift()).cumsum()

        # Maximum allowed loss threshold
        threshold = -max_percent * aum

        # Stop level is the price at which PnL would hit the threshold:
        # new_pnl = current_pnl + (new_price - current_price) * position
        # At stop: threshold = current_pnl + (stop_price - current_price) * position
        # Solving: stop_price = current_price + (threshold - current_pnl) / position
        # For long (position > 0): stop is below current price
        # For short (position < 0): stop is above current price
        stop_levels = mid_price + (threshold - pnl) / position.where(
            position != 0, np.nan
        )

        # Keep position while PnL hasn't breached threshold
        stop_lossed_position = position.where(pnl > threshold, 0.0)
    else:
        # Get entry price: bid for long positions, ask for short positions
        # Only capture price when position is first opened (previous position was 0)
        px_level = bid.where(position > 0, ask).where(position.shift() == 0.0, np.nan)

        # Calculate stop levels:
        # Long: stop = entry * (1 - max_percent) - triggers when price falls
        # Short: stop = entry * (1 + max_percent) - triggers when price rises
        is_long = position > 0
        stop_multiplier = (1 - max_percent) * is_long + (1 + max_percent) * ~is_long
        idx = cast(pd.DatetimeIndex, position.index)
        stop_levels = (px_level * stop_multiplier).groupby(idx.date).ffill()

        # Keep position while stop not triggered:
        # Long: keep while bid >= stop_level (price hasn't fallen below stop)
        # Short: keep while ask <= stop_level (price hasn't risen above stop)
        keep_long = is_long & (bid >= stop_levels)
        keep_short = (position < 0) & (ask <= stop_levels)
        keep_flat = position == 0
        stop_lossed_position = position.where(keep_long | keep_short | keep_flat, 0.0)

    return stop_lossed_position, stop_levels
