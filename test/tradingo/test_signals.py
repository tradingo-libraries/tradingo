import numpy as np
import pandas as pd
import pandas_market_calendars as pmc
import pytest

from tradingo import signals
from tradingo.api import Tradingo

UNIVERSE = "test-universe"
PROVIDER = "test-provider"


def test_buffer_signal(tradingo: Tradingo):
    signals.buffered(
        signal=tradingo.portfolio.model.raw.shares(),
        buffer_width=0.5,
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
):
    result = pd.concat(
        signals.intraday_momentum(
            ask_close=ask_close,
            bid_close=bid_close,
            close_offset_periods=close_offset_periods,
            calendar="NYSE",
        ),
        axis=1,
        keys=(
            "intraday_momentum",
            "z_score",
            "short_vol",
            "long_vol",
            "previous_close_px",
        ),
    )

    cal = pmc.get_calendar("NYSE")
    schedule = cal.schedule(start_date="2024-10-14", end_date="2024-10-14")
    trading_index = pmc.date_range(schedule, frequency="15min")

    subset = result.loc[trading_index]

    assert subset[-(close_offset_periods + 1) :]["intraday_momentum"][
        "ABCD"
    ].to_list() == [0 for _ in range(0, 1 + close_offset_periods)]


if __name__ == "__main__":
    pytest.main([__file__])
