"""IG data accessors"""

import logging
from typing import Optional

import dateutil.tz
import numpy as np
import pandas as pd
from arcticdb.version_store.library import Library
from tenacity import Retrying, retry_if_exception_type, wait_exponential
from trading_ig.rest import ApiExceededException, IGService

from tradingo import symbols
from tradingo.settings import IGTradingConfig

logger = logging.getLogger(__name__)


def get_ig_service(
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    acc_type: Optional[str] = None,
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
    service: Optional[IGService] = None,
):
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
                columns=pd.MultiIndex.from_tuples(
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
                ),
                index=pd.DatetimeIndex([], name="DateTime", tz="utc"),
            )
            # raise SkipException after return
        else:
            raise ex
    finally:
        # TODO: need to make this session management RAII
        service.session.close()

    return (
        result["bid"],
        result["ask"],
    )


@symbols.lib_provider(pricelib="{raw_price_lib}")
def create_universe(
    pricelib: Library,
    instruments: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp,
):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    def get_data(symbol: str):
        return pd.concat(
            (
                pricelib.read(f"{symbol}.bid", date_range=(start_date, end_date)).data,
                pricelib.read(f"{symbol}.ask", date_range=(start_date, end_date)).data,
            ),
            axis=1,
            keys=("bid", "ask"),
        )

    result = pd.concat(
        ((get_data(symbol) for symbol in instruments.index.to_list())),
        axis=1,
        keys=instruments.index.to_list(),
    ).reorder_levels([1, 2, 0], axis=1)
    return (
        result["bid"]["Open"],
        result["bid"]["High"],
        result["bid"]["Low"],
        result["bid"]["Close"],
        result["ask"]["Open"],
        result["ask"]["High"],
        result["ask"]["Low"],
        result["ask"]["Close"],
        ((result["ask"]["Open"] + result["bid"]["Open"]) / 2),
        ((result["ask"]["High"] + result["bid"]["High"]) / 2),
        ((result["ask"]["Low"] + result["bid"]["Low"]) / 2),
        ((result["ask"]["Close"] + result["bid"]["Close"]) / 2),
    )


def get_activity_history(
    from_date,
    to_date,
    svc: IGService | None = None,
):
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
