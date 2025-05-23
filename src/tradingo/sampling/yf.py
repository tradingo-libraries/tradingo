"""Yahoo Finance data provider."""

import logging

import pandas as pd
import pycountry
import yfinance as yf
from arcticdb.version_store.library import Library

from tradingo import symbols

logger = logging.getLogger(__name__)


_CCY_CODES = {c.alpha_3 for c in pycountry.currencies}


def currency_to_symbol(maybe_currency: str) -> str:
    """Convert a currency pair to a YF currency symbol, prepending '=X'."""
    if (
        len(maybe_currency) == 6
        and maybe_currency[:3] in _CCY_CODES
        and maybe_currency[3:] in _CCY_CODES
    ):
        return maybe_currency + "=X"
    return maybe_currency


def symbol_to_currency(symbol: str) -> str:
    """Convert a currency pair to a YF currency symbol, prepending '=X'."""
    if (
        symbol.endswith("=X")
        and len(symbol) == 8
        and symbol[:3] in _CCY_CODES
        and symbol[3:6] in _CCY_CODES
    ):
        return symbol[:-2]
    return symbol

def _get_ticker(ticker: str) -> str:
    if (ticker_ := currency_to_symbol(ticker)) != ticker:
        logger.info("converting currency ticker %s to %s", ticker, ticker_)
        return ticker_
    return ticker


def sample_equity(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    actions: bool = False,
    repair: bool = False,
):
    """sample one symbol from yahoo finance"""

    ticker = _get_ticker(ticker)

    logger.info(
        "querying yfinance ticker=%s start=%s end=%s interval=%s",
        ticker,
        start_date,
        end_date,
        interval,
    )

    if not end_date:
        raise ValueError("end_date must be defined")

    return (
        yf.download(
            [ticker],
            start=pd.Timestamp(start_date) if start_date else None,
            end=pd.Timestamp(end_date),
            interval=interval,
            actions=actions,
            repair=repair,
            multi_level_index=False,
            threads=False,
            group_by="ticker",
            auto_adjust=True,
            prepost=True,
            progress=False,
            keepna=True,
        ),
    )


@symbols.lib_provider(pricelib="{raw_price_lib}")
def create_universe(
    pricelib: Library,
    instruments: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
):
    """
    Create one arctic symbol for each OHLCV prices from yahoo finance.
    Each symbol contains all tickers defined for the universe.
    """

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    def get_data(symbol: str):
        symbol = _get_ticker(symbol)
        return pricelib.read(symbol, date_range=(start_date, end_date)).data

    result = pd.concat(
        ((get_data(symbol) for symbol in instruments.index.to_list())),
        axis=1,
        keys=instruments.index.to_list(),
    ).reorder_levels([1, 0], axis=1)
    return (
        result["Open"],
        result["High"],
        result["Low"],
        result["Close"],
        result["Volume"],
    )
