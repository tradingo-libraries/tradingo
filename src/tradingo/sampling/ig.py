"""IG data accessors"""

import logging
from typing import Any, Generator, Hashable, TypeVar, cast

import arcticdb as adb
import dateutil.tz
import numpy as np
import pandas as pd
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
    acc_number: str | None = None,
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
        acc_number=acc_number or config.acc_number,
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


T = TypeVar("T")


def batch(iterable: list[T], n: int = 1) -> Generator[list[T], None, None]:
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


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

    symbols_to_read = [(f"{s}.ask", f"{s}.bid") for s in instruments.index.to_list()]

    data = pricelib.read_batch(
        [
            adb.ReadRequest(s, date_range=(start_date, end_date))
            for li in symbols_to_read
            for s in li
        ]
    )
    assert isinstance(data, list)

    data_ = batch(data, n=2)

    def get_data(
        data_pair: list[adb.VersionedItem | adb.DataError],
    ) -> pd.DataFrame:
        bid, ask = data_pair
        if isinstance(bid, adb.DataError) or isinstance(ask, adb.DataError):
            if not permit_missing:
                raise ValueError(f"Missing data {bid=}, {ask=}")
            return pd.DataFrame(
                data=[],
                index=pd.DatetimeIndex([], name="timestamp", tz="utc"),
                columns=COLUMNS,
            )
        return pd.concat(
            (
                cast(pd.DataFrame, bid.data),
                cast(pd.DataFrame, ask.data),
            ),
            axis=1,
            keys=(
                bid.symbol.rsplit(".", maxsplit=1)[-1],
                ask.symbol.rsplit(".", maxsplit=1)[-1],
            ),
        )

    result = pd.concat(
        ((get_data(symbol) for symbol in data_)),
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


# ---------------------------------------------------------------------------
# Deal sampling -> live positions
# ---------------------------------------------------------------------------

# IG interprets activity query bounds in the account's local time, but returns
# activity ``date`` fields in UTC (verified against Airflow task timings in BST).
IG_QUERY_TIMEZONE = "Europe/London"
ACTIVITY_PAGE_SIZE = 500
POSITION_ACTIONS = frozenset(
    {"POSITION_OPENED", "POSITION_CLOSED", "POSITION_PARTIALLY_CLOSED"}
)

DEAL_STR_COLUMNS = (
    "epic",
    "dealId",
    "dealReference",
    "status",
    "type",
    "channel",
    "description",
    "direction",
    "actions",
    "affected_deals",
    "currency",
    "market_name",
)
DEAL_FLOAT_COLUMNS = (
    "size",
    "level",
    "stop_level",
    "limit_level",
    "position_change",
)


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _ig_query_time(ts: pd.Timestamp) -> pd.Timestamp:
    return _utc(ts).tz_convert(IG_QUERY_TIMEZONE).tz_localize(None)


def _fetch_activities(
    svc: IGService,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_chunk: pd.Timedelta = pd.Timedelta("1min"),
) -> list[dict[str, Any]]:
    """Raw v3 activity records in ``[start, end]``.

    ``IGService.fetch_account_activity`` mis-parses IG's ``next`` page link
    (``+01:00`` decodes to a space -> HTTP 400), so a window that needs more
    than one page is bisected until each half fits in a single page.
    """
    previous = svc.return_dataframe
    svc.return_dataframe = False
    try:
        data = svc.fetch_account_activity(
            from_date=_ig_query_time(start).to_pydatetime(),
            to_date=_ig_query_time(end).to_pydatetime(),
            detailed=True,
            page_size=ACTIVITY_PAGE_SIZE,
        )
        activities = list(data["activities"])
        if len(activities) >= ACTIVITY_PAGE_SIZE:
            raise OverflowError("activity page full")
        return activities
    except Exception:
        if end - start <= min_chunk:
            raise
        mid = start + (end - start) / 2
        logger.info("Bisecting activity window %s -> %s", start, end)
        return _fetch_activities(svc, start, mid, min_chunk) + _fetch_activities(
            svc, mid, end, min_chunk
        )
    finally:
        svc.return_dataframe = previous


def _empty_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{c: pd.Series(dtype=str) for c in DEAL_STR_COLUMNS},
            **{c: pd.Series(dtype=float) for c in DEAL_FLOAT_COLUMNS},
        },
        index=pd.DatetimeIndex([], name="DateTime", tz="UTC"),
    )


def activities_to_deals(activities: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalise IG v3 activity records into a fixed-schema deal log.

    ``position_change`` is the signed size for accepted deals that opened or
    closed position units, and 0 for rejections and stop/limit amendments.
    """
    if not activities:
        return _empty_deals()

    rows = []
    for act in activities:
        details = act.get("details") or {}
        actions = details.get("actions") or []
        action_types = [str(a.get("actionType", "")) for a in actions]
        affected = [str(a.get("affectedDealId", "")) for a in actions]
        size = details.get("size")
        direction = details.get("direction") or ""
        moves = (
            act.get("status") == "ACCEPTED"
            and act.get("type") == "POSITION"
            and bool(POSITION_ACTIONS.intersection(action_types))
        )
        sign = {"BUY": 1.0, "SELL": -1.0}.get(direction, 0.0)
        rows.append(
            {
                "DateTime": act.get("date"),
                "epic": act.get("epic") or "",
                "dealId": act.get("dealId") or "",
                "dealReference": details.get("dealReference") or "",
                "status": act.get("status") or "",
                "type": act.get("type") or "",
                "channel": act.get("channel") or "",
                "description": act.get("description") or "",
                "direction": direction,
                "actions": ",".join(action_types),
                "affected_deals": ",".join(affected),
                "currency": details.get("currency") or "",
                "market_name": details.get("marketName") or "",
                "size": size,
                "level": details.get("level"),
                "stop_level": details.get("stopLevel"),
                "limit_level": details.get("limitLevel"),
                "position_change": (
                    sign * float(size) if moves and size is not None else 0.0
                ),
            }
        )

    deals = pd.DataFrame(rows).drop_duplicates(subset="dealId", keep="first")
    deals.index = pd.DatetimeIndex(
        pd.to_datetime(deals.pop("DateTime"), utc=True), name="DateTime"
    )
    for col in DEAL_STR_COLUMNS:
        deals[col] = deals[col].astype(str)
    for col in DEAL_FLOAT_COLUMNS:
        deals[col] = pd.to_numeric(deals[col], errors="coerce").astype(float)
    columns = list(DEAL_STR_COLUMNS + DEAL_FLOAT_COLUMNS)
    return deals.sort_index(kind="stable")[columns]


def open_positions_snapshot(
    svc: IGService, epics: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Net signed position and size-weighted average open level per epic."""
    raw = svc.fetch_open_positions()
    net = pd.Series(0.0, index=epics, dtype=float)
    avg = pd.Series(np.nan, index=epics, dtype=float)
    if raw is None or len(raw) == 0 or "epic" not in raw.columns:
        return net, avg

    raw = raw[raw["epic"].isin(epics)].copy()
    raw["size"] = raw["size"].astype(float)
    raw["level"] = raw["level"].astype(float)
    raw["signed"] = raw["direction"].map({"BUY": 1.0, "SELL": -1.0}) * raw["size"]
    net = net.add(raw.groupby("epic")["signed"].sum(), fill_value=0.0)

    for epic, group in raw.groupby("epic"):
        side = np.sign(net.get(epic, 0.0))
        same_side = group[np.sign(group["signed"]) == side]
        if side and same_side["size"].sum():
            avg[epic] = float(
                (same_side["level"] * same_side["size"]).sum() / same_side["size"].sum()
            )
    return net.reindex(epics), avg.reindex(epics)


def deals_to_positions(
    deals: pd.DataFrame,
    final_position: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Rebuild a wide position history anchored to a known final position.

    ``position(t) = final_position - sum(position_change of deals after t)``,
    so only deals between ``start_date`` and the moment ``final_position`` was
    observed are needed. Rows are emitted at ``start_date``, at every deal in
    ``[start_date, end_date]`` and at ``end_date``.
    """
    epics = list(final_position.index)
    moves = deals.loc[deals["position_change"] != 0, ["epic", "position_change"]]
    moves = moves[moves["epic"].isin(epics)]
    if moves.empty:
        wide = pd.DataFrame(
            columns=epics, index=pd.DatetimeIndex([], tz="UTC"), dtype=float
        )
    else:
        wide = (
            moves.reset_index()
            .pivot_table(
                index="DateTime",
                columns="epic",
                values="position_change",
                aggfunc="sum",
            )
            .reindex(columns=epics)
            .fillna(0.0)
        )

    def position_at(ts: pd.Timestamp) -> pd.Series:
        return final_position - wide[wide.index > ts].sum()

    start = _utc(start_date)
    end = _utc(end_date)
    # Position just after each deal = final minus everything strictly later.
    after = wide.iloc[::-1].cumsum().iloc[::-1] - wide
    at_deals = (final_position - after)[(wide.index >= start) & (wide.index <= end)]

    out = pd.concat(
        [
            position_at(start).to_frame(start).T,
            at_deals,
            position_at(end).to_frame(end).T,
        ]
    )
    out.index = pd.DatetimeIndex(out.index, name="DateTime")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[epics].astype(float).round(8)


def sample_ig_positions(
    instruments: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    service: IGService | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DAG node: sample IG deals and aggregate them into live position symbols.

    Columns are fixed to ``instruments.index`` so the outputs line up with the
    backtest/portfolio symbols for the same universe and keep a static schema.
    Note the IG account is shared: an epic traded by two strategies shows the
    account-level net position in both universes.

    :returns: ``(deals, position, avg_open_price)``

        * deals: long-format deal log (accepted and rejected) in the window.
        * position: wide signed net position per epic (step function).
        * avg_open_price: one-row wide frame at sample time.
    """
    own_service = service is None
    svc = service or get_ig_service()
    epics = [str(e) for e in instruments.index]
    start = _utc(start_date)
    end = _utc(end_date)

    try:
        now = pd.Timestamp.now("UTC")
        final_position, avg_price = open_positions_snapshot(svc, epics)
        all_deals = activities_to_deals(_fetch_activities(svc, start, max(now, end)))
    finally:
        if own_service:
            svc.session.close()

    deals = all_deals[all_deals["epic"].isin(epics)]
    position = deals_to_positions(deals, final_position, start, end)
    window_deals = deals[(deals.index >= start) & (deals.index <= end)]

    avg_open_price = avg_price.to_frame(now).T.astype(float)
    avg_open_price.index = pd.DatetimeIndex(avg_open_price.index, name="DateTime")

    logger.info(
        "Sampled %s deals (%s position-changing) for %s epics; net=%s",
        len(window_deals),
        int((window_deals["position_change"] != 0).sum()),
        len(epics),
        final_position[final_position != 0].to_dict(),
    )
    return window_deals, position, avg_open_price
