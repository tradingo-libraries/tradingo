from test.tradingo.utils import close_position

import numpy as np
import pandas as pd
import pytest

from tradingo import backtest
from tradingo.api import Tradingo


def test_backtest_integration(tradingo: Tradingo) -> None:
    pd.concat(
        backtest.backtest(
            portfolio=tradingo.portfolio.model.raw.shares(),
            bid_close=tradingo.prices.close(),
            ask_close=tradingo.prices.close(),
            dividends=None,
        ),
        keys=("portfolio", *(f"instrument.{i}" for i in backtest.BACKTEST_FIELDS)),
        axis=1,
    )


@pytest.mark.parametrize(
    "prices_,portfolio_,unrealised_pnl,realised_pnl",
    [
        (
            pd.DataFrame(
                1 + 0.01,
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ).cumprod(),
            pd.DataFrame(
                1,
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ).cumprod(),
            "(prices_*portfolio_).squeeze().diff().to_frame('ABCD')",
            "pd.Series(0.0, index=prices_.index).to_frame('ABCD')",
        ),
        (
            pd.DataFrame(
                1 + 0.01,
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ).cumprod(),
            pd.DataFrame(
                range(0, 260),
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ),
            "pd.Series([np.nan, np.nan, *(prices_.diff()*portfolio_.shift()).squeeze().iloc[2:]], index=prices_.index).to_frame('ABCD')",
            "pd.Series(0.0, index=prices_.index).to_frame('ABCD')",
        ),
        (
            pd.DataFrame(
                1 + 0.01,
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ).cumprod(),
            close_position(
                pd.Series(
                    1,
                    index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                    name="ABCD",
                ).cumprod(),
            ).to_frame("ABCD"),
            "pd.Series([*(prices_.diff()*portfolio_.shift()).squeeze().iloc[0:-1], np.nan], index=prices_.index).to_frame('ABCD')",
            "pd.Series([*(0 for _ in range(0, 259)), 12.280985478055433], index=prices_.index).to_frame('ABCD')",
        ),
    ],
)
def test_backtest_smoke(
    tradingo: Tradingo,
    prices_: pd.DataFrame,
    portfolio_: pd.DataFrame,
    unrealised_pnl: pd.DataFrame,
    realised_pnl: pd.Series,
) -> None:
    dividends = pd.DataFrame(0, columns=prices_.columns, index=prices_.index)
    bt = pd.concat(
        backtest.backtest(
            portfolio=portfolio_,
            bid_close=prices_,
            ask_close=prices_,
            dividends=dividends,
        ),
        keys=("portfolio", *(f"instrument.{i}" for i in backtest.BACKTEST_FIELDS)),
        axis=1,
    )

    actual_unrealised = pd.DataFrame(bt["instrument.unrealised_pnl"]).diff()
    expected_unrealised = (
        eval(unrealised_pnl) if isinstance(unrealised_pnl, str) else unrealised_pnl
    )

    pd.testing.assert_series_equal(
        pd.Series(actual_unrealised.squeeze()),
        pd.Series(expected_unrealised.astype("float32").squeeze()),
        check_names=False,
        check_freq=False,
        rtol=1e-4,
    )
    actual_realised = bt["instrument.realised_pnl"].diff().fillna(0.0)
    expected_realised = (
        eval(realised_pnl) if isinstance(realised_pnl, str) else realised_pnl
    )
    pd.testing.assert_series_equal(
        pd.Series(actual_realised.squeeze()),
        pd.Series(expected_realised.astype("float32").squeeze()),
        check_names=False,
        check_freq=False,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "position, prices, unrealised_pnl, realised_pnl, total_pnl",
    (
        (
            pd.DataFrame(
                [0, 1, 1, 1, 0, 0],
                columns=["ABC"],
                index=pd.bdate_range("2024-11-24", periods=6),
            ),
            pd.DataFrame(
                [10, 10, 11, 12, 13, 14],
                columns=["ABC"],
                index=pd.bdate_range("2024-11-24", periods=6),
            ),
            pd.DataFrame(
                [np.nan, 0, 1, 2, np.nan, np.nan],
                columns=["ABC"],
                index=pd.bdate_range("2024-11-24", periods=6),
            ),
            pd.DataFrame(
                [0, 0, 0, 0, 3, 3],
                columns=["ABC"],
                index=pd.bdate_range("2024-11-24", periods=6),
            ),
            pd.DataFrame(
                [0, 0, 1, 2, 3, 3],
                columns=["ABC"],
                index=pd.bdate_range("2024-11-24", periods=6),
            ),
        ),
    ),
)
def test_backtest_all_closed(
    position: pd.DataFrame,
    prices: pd.DataFrame,
    unrealised_pnl: pd.DataFrame,
    realised_pnl: pd.DataFrame,
    total_pnl: pd.DataFrame,
) -> None:
    bt = pd.concat(
        backtest.backtest(
            portfolio=position,
            bid_close=prices,
            ask_close=prices,
            dividends=None,
        ),
        keys=("portfolio", *(f"instrument.{i}" for i in backtest.BACKTEST_FIELDS)),
        axis=1,
    )

    pd.testing.assert_frame_equal(
        bt[["instrument.unrealised_pnl"]].droplevel(0, axis=1),
        unrealised_pnl,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bt[["instrument.realised_pnl"]].droplevel(0, axis=1),
        realised_pnl,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bt[["instrument.total_pnl"]].droplevel(0, axis=1),
        total_pnl,
        check_dtype=False,
    )
