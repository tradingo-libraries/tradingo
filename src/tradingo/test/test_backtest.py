from numpy import divide
import pytest
import pandas as pd

from tradingo import backtest
from tradingo.backtest import PnlSnapshot

from tradingo.test.utils import close_position


def test_pnl_snapshot():
    pnl = PnlSnapshot(pd.Timestamp("2024-04-26 00:00:00+00:00"))

    pnl_1 = pnl.on_trade(100, 1, pd.Timestamp("2024-04-29 00:00:00+00:00")).on_trade(
        100, 1, pd.Timestamp("2024-04-29 00:00:00+00:00")
    )

    assert pnl_1.net_investment == 100
    assert pnl_1.net_position == 2
    assert pnl_1.realised_pnl == 0
    assert pnl_1.unrealised_pnl == 0
    assert pnl_1.avg_open_price == 100

    pnl_2 = pnl_1.on_trade(101, -1, pd.Timestamp("2024-04-30 00:00:00+00:00"))
    assert pnl_2.net_investment == 200
    assert pnl_2.net_position == 1
    assert pnl_2.realised_pnl == 1
    assert pnl_2.unrealised_pnl == 0
    assert pnl_1.avg_open_price == 100
    assert pnl_1.avg_open_price == 100

    pnl_2 = pnl_2.on_trade(101, -1, pd.Timestamp("2024-05-01 00:00:00+00:00"))
    assert pnl_2.net_investment == 200
    assert pnl_2.net_position == 0
    assert pnl_2.realised_pnl == 2
    assert pnl_2.unrealised_pnl == 0
    pnl_2 = pnl_2.on_trade(101, 2, pd.Timestamp("2024-05-01 00:00:00+00:00"))
    assert pnl_2.net_investment == 200


def test_backtest_integration(benchmark, tradingo):

    bt = benchmark(
        backtest.backtest,
        prices="close",
        portfolio="model",
        dividends="dividend",
        start_date=pd.Timestamp("2018-01-01 00:00:00+00:00"),
        end_date=pd.Timestamp.now("utc"),
        name="model",
        stage="raw.shares",
        config_name="test",
        provider="yfinance",
        universe="etfs",
        dry_run=True,
        arctic=tradingo,
    )


def test_backtest_integration_old(benchmark, tradingo):

    bt = benchmark(
        backtest.backtest_old,
        prices="close",
        portfolio="model",
        start_date=pd.Timestamp("2018-01-01 00:00:00+00:00"),
        end_date=pd.Timestamp.now("utc"),
        name="model",
        stage="raw.shares",
        config_name="test",
        provider="yfinance",
        universe="etfs",
        dry_run=True,
        arctic=tradingo,
    )

    assert isinstance(bt, pd.DataFrame)
    assert bt.columns.to_list()


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
            "(prices_*portfolio_).squeeze().diff()",
            "pd.Series(0.0, index=prices_.index)",
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
            "(prices_.diff()*portfolio_.shift()).squeeze()",
            "pd.Series(0.0, index=prices_.index)",
        ),
        (
            pd.DataFrame(
                1 + 0.01,
                index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                columns=["ABCD"],
            ).cumprod(),
            close_position(
                pd.DataFrame(
                    1,
                    index=pd.bdate_range("2020-01-01", periods=260, freq="B"),
                    columns=["ABCD"],
                ).cumprod(),
            ),
            "(prices_.diff()*portfolio_.shift()).squeeze()",
            "pd.Series([*(0 for _ in range(0, 259)), 12.280985478055433], index=prices_.index)",
        ),
    ],
)
def test_backtest_smoke(tradingo, prices_, portfolio_, unrealised_pnl, realised_pnl):

    dividends = pd.DataFrame(0, columns=prices_.columns, index=prices_.index)

    bt = backtest.backtest(
        portfolio=portfolio_,
        prices=prices_,
        dividends=dividends,
        name="test",
        config_name="test",
        dry_run=True,
        start_date=portfolio_.index[0],
        end_date=portfolio_.index[-1],
        stage="raw",
        provider="yfinance",
        universe="etfs",
        arctic=tradingo,
    )

    actual_unrealised = bt["backtest/instrument.unrealised_pnl"].squeeze().diff()
    expected_unrealised = (
        eval(unrealised_pnl) if isinstance(unrealised_pnl, str) else unrealised_pnl
    )

    pd.testing.assert_series_equal(
        actual_unrealised,
        expected_unrealised.astype("float32"),
        check_names=False,
        check_freq=False,
        rtol=1e-4,
    )
    actual_realised = (
        bt["backtest/instrument.realised_pnl"].squeeze().diff().fillna(0.0)
    )
    expected_realised = (
        eval(realised_pnl) if isinstance(realised_pnl, str) else realised_pnl
    )
    pd.testing.assert_series_equal(
        actual_realised,
        expected_realised.astype("float32"),
        check_names=False,
        check_freq=False,
        rtol=1e-4,
    )


if __name__ == "__main__":
    pytest.main()
