import logging
import numba
import numpy as np
import math

import pandas as pd

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
            trade_in = math.isnan(res[i - 1, j]) and not math.isnan(signal[i, j])
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

    thresholds = signal.mul(buffer_width)

    buffered = _linear_buffer(signal.to_numpy(), thresholds.to_numpy())

    return (pd.DataFrame(buffered, index=signal.index, columns=signal.columns),)
