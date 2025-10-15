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


def adjust_fx_series(
    fx_series: pd.DataFrame,
    ref_ccy: str,
    add_self: bool = False,
    add_cent: bool = False,
) -> pd.DataFrame:
    """
    Adjust fx_series columns based on the reference currency.

    :param fx_series: DataFrame with XXXYYY columns where XXX and YYY are legs
    :param ref_ccy: Reference currency
    :return: Adjusted fx_series with renamed columns and inverted rates where necessary
    """
    adjusted_fx = fx_series.copy()
    new_columns = {}

    for col in fx_series.columns:
        base, quote = col[:3], col[3:]
        if quote == ref_ccy:
            new_columns[col] = base
        elif base == ref_ccy:
            new_columns[col] = quote
            adjusted_fx[col] = 1.0 / adjusted_fx[col]
        else:
            raise ValueError(
                f"Column {col} does not match reference currency {ref_ccy}"
            )

    adjusted_fx.rename(columns=new_columns, inplace=True)

    if add_self:
        adjusted_fx[ref_ccy] = 1.0

    if add_cent and "GBP" in adjusted_fx.columns:
        adjusted_fx["GBp"] = adjusted_fx["GBP"] * 0.01
    if add_cent and "EUR" in fx_series.columns:
        adjusted_fx["c"] = adjusted_fx["EUR"] * 0.01

    return adjusted_fx.loc[:, ~adjusted_fx.columns.duplicated()]


def _align_series(
    series: pd.Series,
    other: pd.Series | float | int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    align a series to another ahead of returns calculation

    :param series: reference series
    :param other: other series or value to align
    """

    if other is None:
        other = pd.Series(1.0, index=series.index, name=series.name)
    elif isinstance(other, (float, int)):
        other = pd.Series(other, index=series.index, name=series.name)
    elif isinstance(other, pd.Series):
        common_idx = series.index.intersection(other.index)
        series = series.reindex(common_idx).dropna()
        other = other.reindex(common_idx).dropna()
    else:
        raise TypeError(type(other))

    return series, other


def convert_prices_to_ccy(
    instruments: pd.DataFrame,
    prices: pd.DataFrame,
    fx_series: pd.DataFrame,
    currency: str,
) -> pd.DataFrame:
    """
    Convert prices to a common currency using fx_series.

    :param instruments: DataFrame with instrument symbols and their currencies
    :param prices: DataFrame with prices indexed by instrument symbols
    :param fx_series: DataFrame with FX rates indexed by currency pairs (e.g., 'EURUSD=X')
    :param currency: Target currency to convert prices to
    :return: List of DataFrames with prices converted to the target currency
    """

    symbols_ccys = instruments.currency.to_dict()
    ccy_syms_map: dict[str, list[str]] = {}
    for symbol, ccy in symbols_ccys.items():
        if ccy not in ccy_syms_map:
            ccy_syms_map[ccy] = []
        ccy_syms_map[ccy].append(symbol)

    converted = []
    for name, df in prices.items():
        df_fx = adjust_fx_series(
            fx_series[name], currency, add_self=True, add_cent=True
        )
        if set(symbols_ccys) != set(df.columns):
            raise ValueError(
                f"prices columns {set(df.columns)} do not match currency_map symbols {set(symbols_ccys)}"
            )
        if missing_ccys := set(ccy_syms_map).difference(set(df_fx.columns)):
            raise ValueError(f"fx_series columns miss currencies: {missing_ccys}")

        result = []
        for sym in df.columns:
            df_, fx_ = _align_series(df[sym].ffill(), df_fx[symbols_ccys[sym]])
            result.append(df_.mul(fx_).rename(df_.name))
        converted.append(pd.concat(result, axis=1)[df.columns])

    return converted
