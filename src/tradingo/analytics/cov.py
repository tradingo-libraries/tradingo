"""Covariance calculation module."""

from typing import Callable

import numpy as np
import pandas as pd
from pandas.core.window.rolling import BaseWindow


def _kronecker_product(dataframe: pd.DataFrame) -> pd.DataFrame:
    """row-by-row outer product"""
    return pd.DataFrame(
        dataframe.apply(lambda x: np.kron(x, x), axis=1, result_type="expand"),
        index=dataframe.index,
        columns=pd.MultiIndex.from_product([dataframe.columns, dataframe.columns]),
    )


def cov(
    dataframe: pd.DataFrame,
    annualisation: int = 260,
    how: str = "ewm",
    demean: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate the covariance of a given DataFrame.

    :param dataframe: DataFrame containing the data.
    :param annualisation: The annualisation factor, by default 260.
    :param how: A DataFrame method which returns a BaseWindow [pandas.core.window.rolling]
        for calculating covariance: { ewm | rolling | expanding }
    :param demean: Whether to demean the data, by default True.
            - True -> E[(X - E[X]) * (Y - E[Y])]
            - False -> E[X * Y]
        When True, it is the covariance.
        When False, we use Kronecker's product (row-wise pairs), from which we calc the mean.
        The result is the second moments, useful for raw signal magnitude, co-movement
        including mean shifts, or feeding into models where you handle the mean explicitly.
        NOTE: because of returns being "somewhere around zero", this '(co)variance'
        accounts for the implicit bias within a timeseries (e.g. a typical long position)

    :param **kwargs: Additional arguments for rolling/ewm/... pandas methods.
        For rolling covariance, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling covariance.
        For exponential covariance, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: The covariance of the data.
    """

    dataframe = dataframe if demean else _kronecker_product(dataframe)

    try:
        agg_buffer: Callable = getattr(dataframe, how)
        agg_buffer: BaseWindow = agg_buffer(**kwargs)
        isinstance(agg_buffer, BaseWindow)
    except AttributeError as ex:
        raise NotImplementedError(f"how={how}") from ex
    except AssertionError as ex:
        raise ValueError(f"how={how} is not a dataframe BaseWindow aggregator") from ex

    agg_method = "cov" if demean else "mean"
    cov_: Callable = getattr(agg_buffer, agg_method)

    return annualisation * cov_()
