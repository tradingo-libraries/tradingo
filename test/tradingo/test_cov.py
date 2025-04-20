import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tradingo.analytics import cov

STDS = np.array([0.1, 0.2, 0.15])
CORRELATIONS = np.array(
    [
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ]
)


def excel_columns(size):
    """generate excel-like columns

    :param size: number of columns

    :returns: an array of column names
    """
    result = []
    for i in range(size):
        col = ""
        x = i
        while True:
            x, r = divmod(x, 26)
            col = chr(65 + r) + col
            if x == 0:
                break
            x -= 1
        result.append(col)
    return result


def cov_from_corr(
    correlation_mtx: np.ndarray, std_array: np.ndarray, deannualisation: int = 260
) -> np.ndarray:
    """
    covariance matrix from a std vector and corr matrix

    :param correlation_mtx: correlation matrix
    :param std_array: standard deviation array
    :param deannualisation: de-annualise covariance (e.g. from annual to daily)
    """
    assert len(correlation_mtx.shape) == 2
    assert all(cs == std_array.size for cs in correlation_mtx.shape)
    assert (correlation_mtx == correlation_mtx.T).all()
    cov_matrix = np.outer(std_array, std_array) * correlation_mtx
    return cov_matrix / deannualisation


def correlated_returns(
    size: int,
    cov_mtx: np.ndarray,
    target_vol: float = 0.1,
    mean: float = 0,
    seed: int = 1234,
) -> pd.DataFrame:
    """
    generate plausible market returns

    :param size: sample size
    :param cov_mtx: covariance matrix, with appropriate annualization
    :param target_vol: target vol, with appropriate annualization
    :param mean: mean of the sample
    :param seed: random seed

    :returns: the correlated returns dataframe
    """

    vol_scale = (target_vol**2) / np.sqrt(cov_mtx.sum().sum())
    cov_mtx = vol_scale * cov_mtx
    mean = mean or [0 for _ in range(len(cov_mtx))]

    gen = np.random.RandomState(seed)
    smp = gen.multivariate_normal(mean, cov_mtx, size)

    index = pd.DatetimeIndex(pd.bdate_range(pd.Timestamp(2020, 1, 2), periods=size))
    columns = excel_columns(len(cov_mtx))

    return pd.DataFrame(smp, index=index, columns=columns)


def test_cov():
    columns = ["A", "B", "C"]
    cov_mtx = cov_from_corr(CORRELATIONS, STDS, deannualisation=260)
    df = correlated_returns(1000, cov_mtx, target_vol=0.1)

    actual_ewm = cov(df, how="ewm", halflife=100)
    expected_ewm = pd.DataFrame(
        [
            [0.00439475, 0.00281318, 0.00079727],
            [0.00281318, 0.01817437, 0.00548258],
            [0.00079727, 0.00548258, 0.0115796],
        ],
        index=pd.MultiIndex.from_product([[pd.Timestamp(2023, 11, 1)], columns]),
        columns=columns,
    )
    assert_frame_equal(actual_ewm.tail(3), expected_ewm)

    actual_rolling = cov(df, how="rolling", window=100)
    expected_rolling = pd.DataFrame(
        [
            [0.00421997, 0.00285151, 0.00144046],
            [0.00285151, 0.01600372, 0.00633751],
            [0.00144046, 0.00633751, 0.01300069],
        ],
        index=pd.MultiIndex.from_product([[pd.Timestamp(2023, 11, 1)], columns]),
        columns=columns,
    )
    assert_frame_equal(actual_rolling.tail(3), expected_rolling)

    actual_expanding = cov(df, how="expanding")
    expected_expanding = pd.DataFrame(
        [
            [0.00444099, 0.00259191, 0.00116196],
            [0.00259191, 0.01852227, 0.00611158],
            [0.00116196, 0.00611158, 0.01096141],
        ],
        index=pd.MultiIndex.from_product([[pd.Timestamp(2023, 11, 1)], columns]),
        columns=columns,
    )
    assert_frame_equal(actual_expanding.tail(3), expected_expanding)
