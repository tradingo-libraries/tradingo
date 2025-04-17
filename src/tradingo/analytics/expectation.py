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

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the data.
    annualisation : int
        The annualisation factor, by default 260.
    how : str
        The method of calculating expectation, either "exponential" or "rolling",
        by default "exponential".
    warmup_threshold : Optional[float]
        The threshold for the warmup window, by default None.
    warmup_since : Optional[str]
        The starting point for the warmup window, by default "any".
    downsample_window : Optional[int]
        The window size for downsampling, by default None.
    downsample_fillna : Optional[float]
        The value to fill NaN values after downsampling, by default None.
    **kwargs : keyword arguments
        Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling expectation, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling expectation.
        For exponential expectation, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    Returns
    -------
    pd.DataFrame
        The expectation of the data.
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
