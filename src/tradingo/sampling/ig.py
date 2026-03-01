"""IG data accessors"""

import logging
from typing import Hashable, cast

import dateutil.tz
import numpy as np
import pandas as pd
from arcticdb.exceptions import NoSuchVersionException
from arcticdb.version_store.library import Library
from tenacity import Retrying, retry_if_exception_type, wait_exponential
from trading_ig.rest import ApiExceededException, IGService

from tradingo import symbols
from tradingo.settings import IGTradingConfig

logger = logging.getLogger(__name__)


COLUMNS = pd.MultiIndex.from_tuples(
    (
        ("bid", "Open"),
        ("bid", "High"),
        ("bid", "Low"),
        ("bid", "Close"),
        ("ask", "Open"),
        ("ask", "High"),
        ("ask", "Low"),
        ("ask", "Close"),
    ),
)


def get_ig_service(
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    acc_type: str | None = None,
) -> IGService:
    config = IGTradingConfig.from_env()

    retryer = Retrying(
        wait=wait_exponential(),
        retry=retry_if_exception_type(ApiExceededException),
    )

    service = IGService(
        username=username or config.username,
        password=password or config.password,
        api_key=api_key or config.api_key,
        acc_type=acc_type or config.acc_type,
        use_rate_limiter=True,
        retryer=retryer,
    )

    service.create_session()
    return service


def sample_instrument(
    epic: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval: str,
    wait: int = 0,
    service: IGService | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    service = service or get_ig_service()
    try:
        result = (
            service.fetch_historical_prices_by_epic(
                epic,
                end_date=pd.Timestamp(end_date)
                .tz_convert(dateutil.tz.tzlocal())
                .tz_localize(None)
                .isoformat(),
                start_date=(pd.Timestamp(start_date) + pd.Timedelta(seconds=1))
                .tz_convert(dateutil.tz.tzlocal())
                .tz_localize(None)
                .isoformat(),
                resolution=interval,
                wait=wait,
            )["prices"]
            .tz_localize(dateutil.tz.tzlocal())
            .tz_convert("utc")
        )
    except Exception as ex:
        if ex.args and (
            ex.args[0] == "Historical price data not found"
            or ex.args[0] == "error.price-history.io-error"
        ):
            logger.warning("Historical price data not found %s", epic)
            result = pd.DataFrame(
                np.nan,
                columns=COLUMNS,
                index=pd.DatetimeIndex([], name="DateTime", tz="utc"),
            )
            # raise SkipException after return
        else:
            raise ex
    finally:
        # TODO: need to make this session management RAII
        service.session.close()

    return (
        pd.DataFrame(result["bid"]),
        pd.DataFrame(result["ask"]),
    )


@symbols.lib_provider(pricelib="{raw_price_lib}")  # pyright: ignore
def create_universe(
    pricelib: Library,
    instruments: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
    permit_missing: bool = False,
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
    """Create wide format price data tables for instruments.

    Params:
        pricelib: the library.
        instruments: the dataframe of instruments.
        end_date: the end date.
        start_date: the start date.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # TODO: Enhance this with batched reading for speed
    def get_data(symbol: str) -> pd.DataFrame:
        try:
            return pd.concat(
                (
                    cast(
                        pd.DataFrame,
                        pricelib.read(
                            f"{symbol}.bid", date_range=(start_date, end_date)
                        ).data,
                    ),
                    cast(
                        pd.DataFrame,
                        pricelib.read(
                            f"{symbol}.ask", date_range=(start_date, end_date)
                        ).data,
                    ),
                ),
                axis=1,
                keys=("bid", "ask"),
            )
        except NoSuchVersionException as ex:
            if permit_missing:
                return pd.DataFrame(
                    data=[],
                    index=pd.DatetimeIndex([], name="timestamp"),
                    columns=COLUMNS,
                )
            raise ex

    result = pd.concat(
        ((get_data(symbol) for symbol in instruments.index.to_list())),
        axis=1,
        keys=instruments.index.to_list(),
    ).reorder_levels([1, 2, 0], axis=1)
    if permit_missing:
        result = result.reindex(instruments.index)
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


def get_activity_history(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    svc: IGService | None = None,
) -> tuple[tuple[pd.DataFrame, tuple[Hashable]], ...]:
    """
    get activtiy history and return dataframe per asset
    """

    svc = svc or get_ig_service()

    act = svc.fetch_account_activity_by_date(
        from_date=pd.Timestamp(from_date),
        to_date=pd.Timestamp(to_date),
    )
    svc.session.close()
    act["DateTime"] = (
        pd.to_datetime(act["date"] + " " + act["time"])
        .dt.tz_localize(dateutil.tz.tzlocal())
        .dt.tz_convert("utc")
    )

    return tuple(
        (
            data.set_index("DateTime")
            .sort_index()
            .drop(["date", "time", "period", "epic", "marketName"], axis=1)
            .replace("-", np.nan)
            .astype(
                {
                    "size": float,
                    "stop": float,
                    "limit": float,
                },
                errors="ignore",
            ),
            (epic,),
        )
        for epic, data in iter(act.groupby("epic"))
    )
