"""Covariance calculation module."""

import numpy as np
import pandas as pd


def cov(
    dataframe: pd.DataFrame,
    annualisation: int = 260,
    how: str = "exponential",
    demean: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate the covariance of a given DataFrame.

    Parameters
    ----------
    :param dataframe: DataFrame containing the data.
    :param annualisation: The annualisation factor, by default 260.
    :param how: The method of calculating covariance, either "exponential" or "rolling",
        by default "exponential".
    :param demean: Whether to demean the data, by default True.
        When False, it is the Kronecker's product (product of all the pairs),
        from which we calc the mean from a ewm. In a non-ewm format is E[X * Y].
        The result is the second moments, useful for raw signal magnitude,
        co-movement including mean shifts, or feeding into models where you handle
        the mean explicitly.
        NOTE: because of returns being "somewhere around zero", this '(co)variance'
        accounts for the implicit bias within a timeseries (e.g. a typical long position)

    :param **kwargs: Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling covariance, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling covariance.
        For exponential covariance, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: The covariance of the data.
    """

    if how not in {"exponential", "rolling"}:
        raise ValueError(f"'how' must be 'exponential' or 'simple', got {how}")

    if how == "exponential":
        if demean:
            covar = annualisation * dataframe.ewm(**kwargs).cov()
        else:
            square_returns = pd.DataFrame(
                dataframe.apply(lambda x: np.kron(x, x), axis=1, result_type="expand"),
                index=dataframe.index,
                columns=pd.MultiIndex.from_product(
                    [dataframe.columns, dataframe.columns]
                ),
            )
            covar = annualisation * square_returns.ewm(**kwargs).mean()

    elif how == "rolling":
        window = kwargs.pop("window", None)
        assert window and window > 0, "Window size must be greater than 0"
        covar = annualisation * dataframe.rolling(window=window, **kwargs).cov()

    return covar
