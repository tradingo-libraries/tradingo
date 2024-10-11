import logging
import numba
import numpy as np
import math

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
    frequency="15m",
    long_vol=64,
    short_vol=6,
    cap=2,
    threshold=1,
    **kwargs,
):

    close = (ask_close + bid_close) / 2

    cal = pmc.get_calendar(calendar)
    schedule = cal.schedule(start_date=close.index[0], end_date=close.index[-1])
    trading_index = pmc.date_range(schedule, frequency=frequency)

    close = close.reindex(trading_index)

    open_px = close.groupby(close.index.date).first()
    close_px = close.groupby(close.index.date).last()

    close_px.index = pd.to_datetime(close_px.index).tz_localize("utc")
    open_px.index = pd.to_datetime(open_px.index).tz_localize("utc")

    previous_close_px = close_px.shift()

    long_vol = close_px.pct_change().ewm(long_vol, min_periods=long_vol).std()
    short_vol = close_px.pct_change().ewm(short_vol, min_periods=short_vol).std()

    previous_close_px = previous_close_px.reindex(close.index, method="ffill")
    long_vol = long_vol.reindex(close.index, method="ffill")
    short_vol = short_vol.reindex(close.index, method="ffill")
    close_px = close_px.reindex(close.index, method="ffill")

    z_score = ((close - previous_close_px) / previous_close_px) / long_vol

    signal = z_score.copy()
    signal[signal.abs() < threshold] = 0
    signal[signal.abs() > cap] = cap
    signal = signal / (short_vol * np.sqrt(252))
    signal.loc[(signal.squeeze() != 0) & (signal.squeeze().shift() != 0)] = np.nan
    signal = signal.ffill()
    stop = 0.01
    signal = signal.abs().groupby(signal.index.date).cummax() * np.sign(z_score)
    # signal closes position at close time
    signal[signal.index.isin(schedule.market_close)] = 0.0
    return (signal,)
