"""static instruments data accessors."""

import logging
from typing import Literal, Optional

import pandas as pd
from yfinance import Ticker

from tradingo.sampling.ig import get_ig_service
from tradingo.sampling.yf import currency_to_symbol

logger = logging.getLogger(__name__)


Provider = Literal[
    "alpha_vantage",
    "cboe",
    "fmp",
    "intrinio",
    "polygon",
    "tiingo",
    "tmx",
    "tradier",
    "yfinance",
]


def download_instruments(
    *,
    index_col: Optional[str] = None,
    html: Optional[str] = None,
    file: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    epics: Optional[list[str]] = None,
):
    if file:
        return (
            pd.read_csv(
                file,
                index_col=index_col,
            ).rename_axis("Symbol"),
        )

    if html:
        return (pd.read_html(html)[0].set_index(index_col).rename_axis("Symbol"),)

    if tickers:
        return (
            (
                pd.DataFrame(
                    {t: Ticker(currency_to_symbol(t)).get_info() for t in tickers}
                )
                .transpose()
                .rename_axis("Symbol")
            ),
        )

    if epics:
        service = get_ig_service()

        return (
            pd.DataFrame((service.fetch_market_by_epic(e)["instrument"] for e in epics))
            .set_index("epic")
            .rename_axis("Symbol", axis=0),
        )

    raise ValueError(file)
