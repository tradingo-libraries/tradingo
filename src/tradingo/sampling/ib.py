"""Interactive Brokers data sampling utilities"""

import logging
from collections.abc import Hashable
from typing import cast

import pandas as pd
from arcticdb.version_store.library import Library
from ib_insync import IB, Contract, util

from tradingo import symbols
from tradingo.settings import IBTradingConfig

logger = logging.getLogger(__name__)

# Maps interval names (as used in IG) to IB bar size strings
RESOLUTION_MAP: dict[str, str] = {
    "SECOND": "1 secs",
    "MINUTE": "1 min",
    "HOUR": "1 hour",
    "DAY": "1 day",
    "WEEK": "1 week",
    "MONTH": "1 month",
}


def _duration_str(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """Convert a date range to an IB durationStr (e.g. '30 D', '1 Y')."""
    days = max(1, (end_date - start_date).days)
    if days <= 365:
        return f"{days} D"
    return f"{max(1, round(days / 365))} Y"


def get_ib_service(
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    account: str | None = None,
) -> IB:
    """Create and connect to Interactive Brokers gateway."""
    config = IBTradingConfig.from_env()

    ib = IB()
    ib.connect(
        host=host or config.host,
        port=port or config.port,
        clientId=client_id or config.client_id,
        account=account or config.account,
        timeout=config.timeout,
    )

    return ib


def sample_instrument(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval: str,
    currency: str = "USD",
    exchange: str = "SMART",
    sec_type: str = "STK",
    service: IB | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch historical BID and ASK bars for a contract from the IB gateway.

    Returns a (bid, ask) pair of DataFrames with OHLC columns, matching the
    format produced by sampling.ig.sample_instrument.
    """
    service = service or get_ib_service()

    contract = Contract(
        symbol=symbol, secType=sec_type, exchange=exchange, currency=currency
    )
    service.qualifyContracts(contract)

    end_dt = (end_date.tz_convert(None) if end_date.tz else end_date).strftime(
        "%Y%m%d %H:%M:%S"
    )
    duration = _duration_str(start_date, end_date)
    bar_size = RESOLUTION_MAP.get(interval.upper(), interval)

    def _fetch(what: str) -> pd.DataFrame:
        bars = service.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars).set_index("date")
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )[["Open", "High", "Low", "Close"]]
        return cast(pd.DataFrame, df[df.index >= start_date])

    return _fetch("BID"), _fetch("ASK")


@symbols.lib_provider(pricelib="{raw_price_lib}")
def create_universe(
    pricelib: Library,
    instruments: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create wide-format price tables from stored IB price data.

    Mirrors sampling.ig.create_universe — reads bid/ask bars already stored
    in ArcticDB by sample_instrument and assembles them into the standard
    12-DataFrame layout (bid OHLC, ask OHLC, mid OHLC).
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    def get_data(symbol: str) -> pd.DataFrame:
        return pd.concat(
            (
                pd.DataFrame(
                    pricelib.read(
                        f"{symbol}.bid", date_range=(start_date, end_date)
                    ).data
                ),
                pd.DataFrame(
                    pricelib.read(
                        f"{symbol}.ask", date_range=(start_date, end_date)
                    ).data
                ),
            ),
            axis=1,
            keys=("bid", "ask"),
        )

    result = pd.concat(
        (get_data(symbol) for symbol in instruments.index.to_list()),
        axis=1,
        keys=instruments.index.to_list(),
    ).reorder_levels([1, 2, 0], axis=1)

    return (
        pd.DataFrame(result["bid"]["Open"]),
        pd.DataFrame(result["bid"]["High"]),
        pd.DataFrame(result["bid"]["Low"]),
        pd.DataFrame(result["bid"]["Close"]),
        pd.DataFrame(result["ask"]["Open"]),
        pd.DataFrame(result["ask"]["High"]),
        pd.DataFrame(result["ask"]["Low"]),
        pd.DataFrame(result["ask"]["Close"]),
        pd.DataFrame((result["ask"]["Open"] + result["bid"]["Open"]) / 2),
        pd.DataFrame((result["ask"]["High"] + result["bid"]["High"]) / 2),
        pd.DataFrame((result["ask"]["Low"] + result["bid"]["Low"]) / 2),
        pd.DataFrame((result["ask"]["Close"] + result["bid"]["Close"]) / 2),
    )


def get_fills_history(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    ib: IB | None = None,
) -> tuple[tuple[pd.DataFrame, tuple[Hashable, ...]], ...]:
    """Fetch execution history from IB and return one DataFrame per symbol.

    Equivalent to sampling.ig.get_activity_history.
    """
    ib = ib or get_ib_service()
    fills = ib.fills()
    ib.disconnect()

    if not fills:
        return ()

    rows = []
    for fill in fills:
        exec_ = fill.execution
        dt = pd.Timestamp(exec_.time).tz_convert("UTC")
        if dt < from_date or dt > to_date:
            continue
        rows.append(
            {
                "symbol": fill.contract.symbol,
                "DateTime": dt,
                "side": exec_.side,
                # IB uses "BOT"/"SLD" for side; normalise to signed size
                "size": exec_.shares if exec_.side == "BOT" else -exec_.shares,
                "price": exec_.price,
                "commission": fill.commissionReport.commission,
            }
        )

    if not rows:
        return ()

    df = pd.DataFrame(rows)

    return tuple(
        (
            group.set_index("DateTime")
            .sort_index()
            .drop("symbol", axis=1)
            .astype({"size": float, "price": float, "commission": float}),
            (symbol,),
        )
        for symbol, group in df.groupby("symbol")
    )
