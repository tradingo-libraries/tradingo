import numpy as np
import pandas as pd
from pandas.core.window.rolling import window_aggregations


def omega_ratio(returns: pd.Series, required_return=0.0):
    """
    Calculate the Omega ratio of a strategy.

    :param returns: pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.

    :param required_return: float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.

    :returns omega_ratio : float

    """
    return_threshold = (1 + required_return) ** (1 / 252) - 1
    returns_less_thresh = returns - return_threshold
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns: pd.Series, required_return=0.0):
    """
    Calculate the Sharpe ratio of a strategy.
    """
    return_threshold = (1 + required_return) ** (1 / 252) - 1
    return (returns.mean() * 252 - return_threshold) / (returns.std() * np.sqrt(252))
