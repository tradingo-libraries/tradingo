import dateutil.tz
import logging
from tradingo.symbols import symbol_provider, symbol_publisher
from typing import Literal, Optional

from trading_ig.rest import IGService, ApiExceededException
from trading_ig.config import config

from tenacity import Retrying, wait_exponential, retry_if_exception_type

from arcticdb import LibraryOptions
from yfinance import Ticker

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


FUTURES_FIELDS = [
    "expiration",
    "price",
]


OPTION_FIELDS = [
    "expiration",
    "option_type",
    "strike",
    "open_interest",
    "volume",
    "theoretical_price",
    "last_trade_price",
    "tick",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "open",
    "high",
    "low",
    "prev_close",
    "change",
    "change_percent",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "last_trade_timestamp",
    "dte",
]


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


def get_ig_service() -> IGService:

    retryer = Retrying(
        wait=wait_exponential(),
        retry=retry_if_exception_type(ApiExceededException),
    )

    service = IGService(
        username=config.username,
        password=config.password,
        api_key=config.api_key,
        acc_type=config.acc_type,
        use_rate_limiter=True,
        retryer=retryer,
    )

    service.create_session()
    return service


@symbol_publisher("instruments/{universe}", write_pickle=True)
def download_instruments(
    index_col: str,
    *,
    html: Optional[str] = None,
    file: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    epics: Optional[list[str]] = None,
    **kwargs,
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
                pd.DataFrame({t: Ticker(t).get_info() for t in tickers})
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


@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    template="prices/{0}.{1}",
    symbol_prefix="{provider}.{universe}.",
    library_options=LibraryOptions(dynamic_schema=True),
)
def sample_ig_instruments(
    instruments: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
    interval: str,
    **kwargs,
):

    service = get_ig_service()

    def get_data(symbol):
        try:
            return service.fetch_historical_prices_by_epic(
                symbol,
                end_date=pd.Timestamp(end_date)
                .tz_convert(dateutil.tz.tzlocal())
                .tz_localize(None)
                .isoformat(),
                start_date=(pd.Timestamp(start_date) + pd.Timedelta(seconds=1))
                .tz_convert(dateutil.tz.tzlocal())
                .tz_localize(None)
                .isoformat(),
                resolution=interval,
                wait=0,
            )["prices"]
        except Exception as ex:
            if ex.args and ex.args[0] == "Historical price data not found":
                logger.warning("Historical price data not found %s", symbol)
                return pd.DataFrame(
                    np.nan,
                    columns=pd.MultiIndex.from_tuples(
                        (
                            ("Open", "bid"),
                            ("Open", "ask"),
                            ("High", "bid"),
                            ("High", "ask"),
                            ("Low", "bid"),
                            ("Low", "ask"),
                        ),
                    ),
                    index=pd.DatetimeIndex([], name="DateTime"),
                )  # TODO:
            raise ex

    result = pd.concat(
        ((get_data(symbol) for symbol in instruments.index.to_list())),
        axis=1,
        keys=instruments.index.to_list(),
    ).reorder_levels([1, 2, 0], axis=1)
    result.index = result.index.tz_localize(dateutil.tz.tzlocal()).tz_convert("utc")
    return (
        (result["bid"]["Open"], ("bid", "open")),
        (result["bid"]["High"], ("bid", "high")),
        (result["bid"]["Low"], ("bid", "low")),
        (result["bid"]["Close"], ("bid", "close")),
        (result["ask"]["Open"], ("ask", "open")),
        (result["ask"]["High"], ("ask", "high")),
        (result["ask"]["Low"], ("ask", "low")),
        (result["ask"]["Close"], ("ask", "close")),
        (((result["ask"]["Open"] + result["bid"]["Open"]) / 2), ("mid", "open")),
        (((result["ask"]["High"] + result["bid"]["High"]) / 2), ("mid", "high")),
        (((result["ask"]["Low"] + result["bid"]["Low"]) / 2), ("mid", "low")),
        (((result["ask"]["Close"] + result["bid"]["Close"]) / 2), ("mid", "close")),
        # (result["last"]["Open"], ("last", "open")),
        # (result["last"]["High"], ("last", "high")),
        # (result["last"]["Low"], ("last", "low")),
        # (result["last"]["Close"], ("last", "close")),
        # (result["last"]["Volume"], ("last", "volume")),
    )


@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    "prices/open",
    "prices/high",
    "prices/low",
    "prices/close",
    "prices/adj_close",
    "prices/volume",
    "prices/dividend",
    "prices/split_ratio",
    astype=float,
    symbol_prefix="{provider}.{universe}.",
)
def sample_equity(
    instruments: pd.DataFrame,
    start_date: str,
    end_date: str,
    provider: Provider,
    interval: str = "1d",
    **kwargs,
):
    from openbb import obb

    data = obb.equity.price.historical(  # type: ignore
        instruments.index.to_list(),
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        interval=interval,
    ).to_dataframe()

    data.index = pd.to_datetime(data.index)

    close = data.pivot(columns=["symbol"], values="close")
    close.index = pd.to_datetime(close.index)

    return (
        data.pivot(columns=["symbol"], values="open"),
        data.pivot(columns=["symbol"], values="high"),
        data.pivot(columns=["symbol"], values="low"),
        close,
        100 * (1 + close.pct_change()).cumprod(),
        data.pivot(columns=["symbol"], values="volume"),
        data.pivot(columns=["symbol"], values="dividend"),
        data.pivot(columns=["symbol"], values="split_ratio"),
    )


@symbol_publisher(
    template="options/{0}.{1}",
    symbol_prefix="{provider}.",
    library_options=LibraryOptions(dynamic_schema=True, dedup=True),
)
def sample_options(
    universe: list[str], start_date: str, end_date: str, provider: Provider, **kwargs
):
    from openbb import obb

    out = []

    data = (
        (
            symbol,
            obb.derivatives.options.chains(symbol)  # type: ignore
            .to_dataframe()
            .replace({"call": 1, "put": -1})
            .astype({"expiration": "datetime64[ns]"})
            .assign(timestamp=end_date)
            # .set_index(["timestamp", "expiration", "strike", "option_type"])
            .set_index(["timestamp", "contract_symbol"]).unstack(level=1),
        )
        for symbol in universe
    )

    for symbol, df in data:
        df.index = pd.to_datetime(df.index)
        out.extend((df[field], (symbol, field)) for field in OPTION_FIELDS)

    return out


@symbol_publisher(
    template="futures/{0}.{1}",
    symbol_prefix="{provider}.",
    library_options=LibraryOptions(dynamic_schema=True, dedup=True),
)
def sample_futures(
    universe: list[str], start_date: str, end_date: str, provider: Provider, **kwargs
):
    from openbb import obb

    out = []

    data = (
        (
            symbol,
            obb.derivatives.futures.curve(symbol)
            .to_dataframe()
            .assign(timestamp=end_date)
            .set_index(["timestamp", "symbol"])
            .unstack("symbol"),
        )
        for symbol in universe
    )

    for symbol, df in data:
        df.index = pd.to_datetime(df.index)
        out.extend((df[field], (symbol, field)) for field in FUTURES_FIELDS)

    return out
