import pytest
import pandas as pd
import numpy as np
import pandas_market_calendars as pmc
from tradingo.api import Tradingo

from tradingo import signals


UNIVERSE = "test-universe"
PROVIDER = "test-provider"


def test_buffer_signal(tradingo: Tradingo):

    signals.buffered(
        signal=tradingo.portfolio.model.raw.shares(),
        thresholds=tradingo.portfolio.model.raw.shares() * 0.1,
        dry_run=True,
        start_date="2023-01-01",
        end_date="2024-05-31",
        config_name="test",
        model_name="model",
        library="portfolio",
        buffer_width=0.5,
        provider="yfinance",
        universe="etfs",
        arctic=tradingo,
    )


@pytest.mark.parametrize(
    "ask_close, bid_close, close_offset_periods",
    [
        (
            pd.DataFrame(
                np.cumprod(1 + np.random.normal(0, 0.009, 167905)),
                index=pd.bdate_range(
                    start="2020-01-01 00:00:00+00:00",
                    end="2024-10-15 18:30:00+00:00",
                    freq=pd.offsets.Minute(15),
                ),
                columns=["ABCD"],
            ),
            pd.DataFrame(
                np.cumprod(1 + np.random.normal(0, 0.009, 167905)),
                index=pd.bdate_range(
                    start="2020-01-01 00:00:00+00:00",
                    end="2024-10-15 18:30:00+00:00",
                    freq=pd.offsets.Minute(15),
                ),
                columns=["ABCD"],
            ),
            0,
        ),
        (
            pd.DataFrame(
                np.cumprod(1 + np.random.normal(0, 0.009, 167905)),
                index=pd.bdate_range(
                    start="2020-01-01 00:00:00+00:00",
                    end="2024-10-15 18:30:00+00:00",
                    freq=pd.offsets.Minute(15),
                ),
                columns=["ABCD"],
            ),
            pd.DataFrame(
                np.cumprod(1 + np.random.normal(0, 0.009, 167905)),
                index=pd.bdate_range(
                    start="2020-01-01 00:00:00+00:00",
                    end="2024-10-15 18:30:00+00:00",
                    freq=pd.offsets.Minute(15),
                ),
                columns=["ABCD"],
            ),
            3,
        ),
    ],
)
def test_intraday_momentum(
    ask_close,
    bid_close,
    close_offset_periods,
    tradingo,
):

    result = signals.intraday_momentum(
        ask_close=ask_close,
        bid_close=bid_close,
        universe=UNIVERSE,
        provider=PROVIDER,
        dry_run=True,
        close_offset_periods=close_offset_periods,
        calendar="NYSE",
        arctic=tradingo,
    )

    cal = pmc.get_calendar("NYSE")
    schedule = cal.schedule(start_date="2024-10-14", end_date="2024-10-14")
    trading_index = pmc.date_range(schedule, frequency="15min")

    subset = result.loc[trading_index]

    assert subset[-(close_offset_periods + 1) :]["signals/intraday_momentum"][
        "ABCD"
    ].to_list() == [0 for _ in range(0, 1 + close_offset_periods)]
