"""Expectation calculation module."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from pandas.core.window.rolling import BaseWindow


def safe_dataframe_window(dataframe, how, kwargs) -> BaseWindow:
    """get a dataframe aggergation window safely"""
    try:
        aggregator: Callable = getattr(dataframe, how)
        agg_buffer: BaseWindow = aggregator(**kwargs)
        assert isinstance(agg_buffer, BaseWindow)
    except AttributeError as ex:
        raise NotImplementedError(f"how={how}") from ex
    except AssertionError as ex:
        raise ValueError(f"how={how} is not a dataframe BaseWindow aggregator") from ex
    return agg_buffer


def expectation(
    dataframe: pd.DataFrame,
    annualisation: int = 260,
    how: str = "exponential",
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate the expectation of a given DataFrame.

    :param dataframe: DataFrame containing the data.
    :param annualisation: The annualisation factor, by default 260.
    :param how: The method of calculating expectation, either "exponential" or "rolling",
        by default "exponential".
    :param **kwargs: keyword arguments
        Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling expectation, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling expectation.
        For exponential expectation, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: The expectation of the data.
    """

    agg_buffer = safe_dataframe_window(dataframe, how, kwargs)
    return annualisation * agg_buffer.mean()
