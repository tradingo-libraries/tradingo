import functools
import logging
import math
import operator
from typing import Optional

import numba
import numpy as np
import pandas as pd
import pandas_market_calendars as pmc
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def vol(
    speeds,
    close: pd.DataFrame,
):
    returns = np.log(close / close.shift())
    return tuple(
        (
            returns.ewm(halflife=speed).std(),
            (speed,),
        )
        for speed in speeds
    )


def ewmac_signal(
    close: pd.DataFrame,
    speed1: int,
    speed2: int,
):

    return (
        (
            close.ewm(halflife=speed2, min_periods=speed2).mean()
            - close.ewm(halflife=speed1, min_periods=speed1).mean()
        ),
    )


def scaled(signal, scale: float):
    return ((signal / signal.abs().max()) * scale,)


def capped(signal: pd.Series, cap: float):
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


def buffered(signal: pd.Series | pd.DataFrame, buffer_width):
    signal = signal.ffill().fillna(0.0)

    thresholds = signal.mul(buffer_width).abs()

    buffered = _linear_buffer(signal.to_numpy(), thresholds.to_numpy())

    return (pd.DataFrame(buffered, index=signal.index, columns=signal.columns),)


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
    incremental: int = 0,
    only_with_close: bool = True,
    vol_floor_window=120,
    vol_floor_quantile=0.75,
    ffill_limit: int = 1,
    start_after: int = 0,
    close_overrides: Optional[dict[str, dict[str, int]]] = None,
    dynamic_floor: int = 0,
):
    close = (
        ((ask_close + bid_close) / 2).groupby(ask_close.index.date).ffill(ffill_limit)
    )

    cal = pmc.get_calendar(calendar)
    schedule = cal.schedule(start_date=close.index[0], end_date=close.index[-1])
    trading_index = pmc.date_range(schedule, frequency=frequency).intersection(
        close.index
    )
    close = close.reindex(trading_index)

    close_px = close.groupby(close.index.date).last()

    close_px.index = pd.to_datetime(close_px.index).tz_localize("utc")

    previous_close_px = close_px.shift()

    returns = np.log(previous_close_px / previous_close_px.shift())

    long_vol = (
        returns.ewm(long_vol, min_periods=long_vol)
        .std()
        .reindex(close.index, method="ffill")
    )
    short_vol = (
        returns.ewm(short_vol, min_periods=short_vol)
        .std()
        .rolling(vol_floor_window, min_periods=1)
        .quantile(q=vol_floor_quantile)
        .reindex(close.index, method="ffill")
    )

    previous_close_px = (
        previous_close_px.reindex(
            close.index,
            method="ffill",
            limit=1,
        )
        .groupby(close.index.date)
        .ffill()
    )

    z_score = ((close - previous_close_px) / previous_close_px) / long_vol

    # apply caps and threshold
    #
    signal = (
        z_score.where(z_score.abs() > threshold, 0)
        .where(z_score.abs() < cap, np.sign(z_score) * cap)
        .groupby(z_score.index.date)
        .ffill()
        .fillna(0)
    )

    signal = signal.where(
        signal.index.to_series().dt.date.eq(signal.index.to_series().dt.date.shift()),
        0,  # make first reading 0.0 position
    ).where(
        np.sign(signal.shift()).eq(np.sign(signal)),
        0,  # make changes in sign 0
    )

    def get_pre_trade_index(periods):
        idx = pd.DatetimeIndex([])

        for i in range(0, periods):
            idx = idx.union(
                schedule.market_open + (pd.tseries.frequencies.to_offset(frequency) * i)
            )

        return pd.to_datetime(idx)

    if start_after:
        idx = get_pre_trade_index(start_after)
        signal.loc[signal.index.isin(idx)] = 0.0

    def _dynamic_floor(series, shift=0):
        return series.groupby(series.index.date).transform(
            lambda i: i.where(
                ~functools.reduce(
                    operator.and_,
                    (
                        (np.sign(i.shift(s)).ne(0) & np.sign(i.shift(s)).ne(np.sign(i)))
                        for s in range(1, shift)
                    ),
                ),
                0,
            )
        )

    if dynamic_floor:
        signal = _dynamic_floor(signal, dynamic_floor)

    # allocate 10_000 notional in unit vol space
    gearing = 10_000 / previous_close_px
    signal = (gearing * signal) / (short_vol * np.sqrt(252))
    # signal = signal.abs().groupby(signal.index.date).cummax() * np.sign(z_score)
    # signal closes position at close time

    def periods_before_close(n):
        return functools.reduce(
            pd.DatetimeIndex.union,
            (
                schedule.market_close
                - (i * pd.tseries.frequencies.to_offset(frequency))
                for i in range(1, n + 1)
            ),
            pd.DatetimeIndex(schedule.market_close),
        )

    close_at = periods_before_close(n=close_offset_periods)

    if monotonic:
        signal_cumabsmax = signal.abs().groupby(signal.index.date).cummax()
        is_long = np.sign(signal).groupby(signal.index.date).cummax()
        is_short = np.sign(signal).groupby(signal.index.date).cummin()
        direction = is_long.where(is_long > 0, is_short)
        signal = signal_cumabsmax * np.sign(direction)

    if incremental:
        no_trade_at = periods_before_close(n=incremental)
        signal.loc[signal.index.isin(no_trade_at)] = 0.0
        signal = signal.groupby(signal.index.date).cumsum() * 1

    if only_with_close:
        has_close = (
            close.groupby(close.index.date)
            .apply(
                lambda i: i[
                    (i.index.tz_convert(cal.close_time.tzinfo).time == cal.close_time)
                ]
                .notna()
                .reindex(i.index, method="bfill")
            )
            .droplevel(0)
        )
        has_close[signal.index.date == close.index.date[-1]] = True
        signal = signal.where(has_close, 0)

    if close_overrides:
        for symbol, time in close_overrides.items():
            time = (
                signal.index.tz_convert(time["timezone"]).normalize()
                + pd.Timedelta(hours=time["hours"], minutes=time["minutes"])
            ).tz_convert("utc")
            signal.loc[(signal.index >= time), symbol] = 0.0

    signal.loc[signal.index.isin(close_at)] = 0.0

    # signal = signal.where((close > previous_close_px) & np.sign(signal) > 0, 0).where(
    #     (close < previous_close_px) & np.sign(signal) < 0, 0
    # )

    return (
        signal,
        z_score,
        short_vol,
        long_vol,
        previous_close_px,
    )


def dynamic_mean_reversion(
    z_score: pd.DataFrame,
    n_lags: int = 30,
) -> tuple[pd.DataFrame]:
    y = (
        (
            z_score.resample(pd.offsets.BDay(1)).last().abs()
            > z_score.resample(pd.offsets.BDay(1)).first().abs()
        )
        .reindex(z_score.index, method="ffill")
        .astype(int)
    )

    X = pd.concat(
        (z_score.shift(i).squeeze() for i in range(1, n_lags)),
        axis=1,
    ).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.5,
        shuffle=False,
    )

    model = MultiOutputClassifier(MLPClassifier())

    model.fit(X_train, y_train)

    return pd.DataFrame(
        model.predict(X_test),
        index=z_score.index,
        columns=z_score.columns,
    )
