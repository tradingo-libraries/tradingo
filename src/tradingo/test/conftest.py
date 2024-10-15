import string
import numpy as np
import pandas as pd
import pytest

from tradingo.api import Tradingo
from tradingo.utils import null_instruments


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:

    annual_std = 0.12

    returns = pd.DataFrame(
        np.random.normal(0, annual_std / np.sqrt(260), (100, 60)),
        index=pd.bdate_range(start="2024-01-03 00:00:00+00:00", periods=100),
        columns=[
            "".join(np.random.choice(list(string.ascii_uppercase), 4))
            for _ in range(60)
        ],
    )
    return (1 + returns).cumprod()


@pytest.fixture(scope="session")
def position(prices: pd.DataFrame):
    return pd.DataFrame(
        np.random.choice((1, 0, -1), prices.shape),
        index=prices.index,
        columns=prices.columns,
    ).cumsum()


@pytest.fixture(scope="session")
def tradingo(prices: pd.DataFrame, position: pd.DataFrame) -> Tradingo:
    t = Tradingo(
        name="test", provider="yfinance", universe="etfs", uri="mem://tradingo"
    )
    libraries = ["prices", "signals", "backtest", "portfolio", "instruments"]
    for library in libraries:
        t.create_library(library)
    t.instruments.etfs.update(null_instruments(prices.columns), upsert=True)
    t.prices.close.update(prices, upsert=True)
    t.prices.dividend.update(
        pd.DataFrame(0, index=prices.index, columns=prices.columns),
        upsert=True,
    )
    t.portfolio.model.raw.shares.update(position, upsert=True)
    return t
