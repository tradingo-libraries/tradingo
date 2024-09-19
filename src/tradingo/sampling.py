from tradingo.symbols import symbol_provider, symbol_publisher
from typing import Literal, Optional
from arcticdb import LibraryOptions
from yfinance import Ticker

import pandas as pd


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


@symbol_publisher("instruments/{universe}", write_pickle=True)
def download_instruments(
    index_col: str,
    *,
    html: Optional[str] = None,
    file: Optional[str] = None,
    tickers: Optional[list[str]] = None,
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
    raise ValueError(file)


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
