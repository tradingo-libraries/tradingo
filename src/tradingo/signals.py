import logging
import numba
import numpy as np
import math
import functools

import pandas as pd

import pandas_market_calendars as pmc

from tradingo.symbols import symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


@symbol_provider(
    close="prices/adj_close",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_publisher(
    template="signals/vol_{0}",
    symbol_prefix="{provider}.{universe}.",
)
def vol(
    speeds,
    close: pd.DataFrame,
    **kwargs,
):
    returns = np.log(close / close.shift())
    return tuple(
        (
            returns.ewm(halflife=speed).std(),
            (speed,),
        )
        for speed in speeds
    )


@symbol_provider(close="prices/adj_close", symbol_prefix="{provider}.{universe}.")
@symbol_publisher(
    "signals/{signal_name}",
    symbol_prefix="{provider}.{universe}.",
)
def ewmac_signal(
    close: pd.DataFrame,
    speed1: int,
    speed2: int,
    provider: str,
    config_name: str,
    model_name="ewmac",
    signal_name="ewmac_{speed1}_{speed2}",
    **kwargs,
):

    logger.info(
        "Running %s model=%s signal=%s with %s",
        config_name,
        model_name,
        signal_name,
        provider,
    )

    returns = np.log(close / close.shift())

    return (
        (
            close.ewm(halflife=speed2, min_periods=speed2).mean()
            - close.ewm(halflife=speed1, min_periods=speed1).mean()
        ),
    )


@symbol_publisher(
    "signals/{signal}.scaled", symbol_prefix="{provider}.{universe}.{model_name}."
)
@symbol_provider(
    signal="signals/{signal}", symbol_prefix="{provider}.{universe}.{model_name}."
)
def scaled(signal, scale: float, **kwargs):
    return ((signal / signal.abs().max()) * scale,)


@symbol_publisher(
    "signals/{signal}.capped", symbol_prefix="{provider}.{universe}.{model_name}."
)
@symbol_provider(
    signal="signals/{signal}", symbol_prefix="{provider}.{universe}.{model_name}."
)
def capped(signal: pd.Series, cap: float, **kwargs):
    signal[signal.abs() >= cap] = np.sign(signal) * cap
    return (signal,)


@numba.jit
def _linear_buffer(signal: np.ndarray, thresholds: np.ndarray):

    res: np.ndarray = np.copy(signal)
    lower = signal - thresholds
    upper = signal + thresholds
    res[0, :] = signal[0, :]

    for i in range(1, res.shape[0]):
        for j in range(signal.shape[1]):
            trade_in = (
                math.isnan(res[i - 1, j]) or res[i - 1, j] == 0.0
            ) and not math.isnan(signal[i, j])
            if trade_in:
                res[i, j] = signal[i, j]
            elif res[i - 1, j] < lower[i, j]:
                res[i, j] = lower[i, j]
            elif res[i - 1, j] > upper[i, j]:
                res[i, j] = upper[i, j]
            else:
                res[i, j] = res[i - 1, j]
    if signal.ndim == 1:
        return np.squeeze(res)
    return res


@symbol_publisher(
    "{library}/{signal}.buffered", symbol_prefix="{provider}.{universe}.{model_name}."
)
@symbol_provider(
    signal="{library}/{signal}",
    symbol_prefix="{provider}.{universe}.{model_name}.",
)
def buffered(signal: pd.Series | pd.DataFrame, buffer_width, **kwargs):

    signal = signal.ffill().fillna(0.0)

    thresholds = signal.mul(buffer_width).abs()

    buffered = _linear_buffer(signal.to_numpy(), thresholds.to_numpy())

    return (pd.DataFrame(buffered, index=signal.index, columns=signal.columns),)


@symbol_publisher(
    "signals/intraday_momentum",
    "signals/intraday_momentum.z_score",
    "signals/intraday_momentum.short_vol",
    "signals/intraday_momentum.long_vol",
    "signals/intraday_momentum.previous_close_px",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_provider(
    ask_close="prices/ask.close",
    bid_close="prices/bid.close",
    symbol_prefix="{provider}.{universe}.",
)
def intraday_momentum(
    ask_close,
    bid_close,
    calendar="NYSE",
    frequency="15min",
    long_vol=64,
    short_vol=6,
    cap=2,
    threshold=1,
    close_offset_periods: int = 0,
    monotonic: bool = True,
    trading_stop_periods: int = 0,
    trade_once: bool = True,
    **kwargs,
):

    close = (ask_close + bid_close) / 2

    cal = pmc.get_calendar(calendar)
    schedule = cal.schedule(start_date=close.index[0], end_date=close.index[-1])
    trading_index = pmc.date_range(schedule, frequency=frequency).intersection(
        close.index
    )
    close = close.reindex(trading_index)

    close_px = close.groupby(close.index.date).last()

    close_px.index = pd.to_datetime(close_px.index).tz_localize("utc")

    previous_close_px = close_px.shift()

    long_vol = close_px.pct_change().ewm(long_vol, min_periods=long_vol).std()
    short_vol = close_px.pct_change().ewm(short_vol, min_periods=short_vol).std()

    previous_close_px = previous_close_px.reindex(close.index, method="ffill")
    long_vol = long_vol.reindex(close.index, method="ffill")
    short_vol = short_vol.reindex(close.index, method="ffill")
    close_px = close_px.reindex(close.index, method="ffill")

    z_score = ((close - previous_close_px) / previous_close_px) / long_vol

    signal = z_score.copy()

    signal.loc[signal.squeeze().abs() < threshold] = 0
    signal.loc[signal.squeeze().abs() > cap] = np.sign(signal) * cap
    signal = signal.ffill()
    # direction = z_score.copy()
    # direction[(np.sign(direction.shift()) != np.sign(direction))] = np.nan
    signal = signal.replace(0, np.nan).groupby(signal.index.date).ffill().fillna(0)
    direction = (
        np.sign(signal).replace(0, np.nan).groupby(signal.index.date).ffill().fillna(0)
    )

    direction.loc[
        (np.sign(direction.squeeze()) != np.sign(direction.squeeze().shift()))
        & (np.sign(direction.squeeze()) != 0)
        & (np.sign(direction.shift().squeeze()) != 0)
    ] = np.nan
    direction = direction.groupby(direction.index.date).ffill()

    # direction = direction.ffill()
    if monotonic:

        signal_cumabsmax = signal.abs().groupby(signal.index.date).cummax()
        signal = signal_cumabsmax * np.sign(direction)

    # ensure
    signal[(signal.shift().abs() >= threshold) & (np.sign(signal.shift()))]
    gearing = 1  # 0.05
    signal = (gearing * signal) / (short_vol * np.sqrt(252))
    # signal = signal.abs().groupby(signal.index.date).cummax() * np.sign(z_score)
    # signal closes position at close time
    close_at = functools.reduce(
        pd.DatetimeIndex.union,
        (
            schedule.market_close - (i * pd.tseries.frequencies.to_offset(frequency))
            for i in range(1, close_offset_periods + 1)
        ),
        pd.DatetimeIndex(schedule.market_close),
    )
    signal.loc[signal.index.isin(close_at)] = 0.0
    return (
        signal,
        z_score,
        short_vol,
        long_vol,
        previous_close_px,
    )
