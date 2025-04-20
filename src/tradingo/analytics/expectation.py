"""Expectation calculation module."""

from __future__ import annotations

import pandas as pd


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

    if how not in {"exponential", "rolling"}:
        raise ValueError(f"'how' must be 'exponential' or 'simple', got {how}")

    if how == "exponential":
        mean = annualisation * dataframe.ewm(**kwargs).mean()

    elif how == "rolling":
        window = kwargs.pop("window", None)
        assert window and window > 0, "Window size must be greater than 0"
        mean = annualisation * dataframe.rolling(window=window, **kwargs).mean()

    return mean
