"""Yahoo Finance data provider."""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def sample_equity(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
):
    return (
        yf.download(
            [ticker],
            start=start_date,
            end=end_date,
            interval=interval,
        ).droplevel(0, axis=1),
    )
