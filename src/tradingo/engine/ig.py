"""IG Markets order execution.

``adjust_position_sizes`` is the DAG entrypoint: it reconciles the latest row of a
target position frame against the open positions on the IG account and sends the
market orders needed to close the gap.

Safety rails (all observed failure modes on the live account):

* NaN targets are skipped rather than treated as "go flat".
* Epics whose market is not ``TRADEABLE`` (e.g. ``EDITS_ONLY`` after the close)
  are skipped instead of sending orders IG will reject with ``Market closed``.
* Share/ETF CFDs (``instrument.type == SHARES``) are dealt in whole units - IG
  rejects fractional sizes on them (``Rejected: 11``) despite advertising a
  0.01 lot size.
* ``max_target_age`` refuses to trade on stale targets, e.g. when Airflow is
  catching up on old intervals.
* The returned trade log has a UTC ``DatetimeIndex`` so the publisher appends to
  the history instead of overwriting it.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from typing import Any, Hashable, cast

import numpy as np
import pandas as pd
from trading_ig.rest import IGService

from tradingo.sampling.ig import get_ig_service

logger = logging.getLogger(__name__)

# IG rejects sizes with more decimal places than it supports
# (validation.number.too-many-decimal-places.request.size), and float
# arithmetic on sizes produces values like 14.120000000000001.
SIZE_DECIMALS = 2

TRADEABLE = "TRADEABLE"

# Instrument types IG will only deal in whole units.
WHOLE_UNIT_INSTRUMENT_TYPES = frozenset({"SHARES"})

# IG accepts up to 50 epics per /markets request.
MARKETS_BATCH_SIZE = 50

STOP_LEVEL_TOLERANCE = 1e-6


def _round_size(size: float, decimals: int = SIZE_DECIMALS) -> float:
    return round(float(size), decimals)


def _get_latest_or_none(value: pd.Series | None) -> float | None:
    if value is None:
        return None
    latest = value.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)


def size_decimals(
    instrument: pd.Series | None,
    instrument_type: str | None = None,
) -> int:
    """Number of decimals IG accepts for an order size on this instrument."""
    kind = instrument_type
    if kind is None and instrument is not None:
        kind = cast(str | None, instrument.get("instrument.type"))
    if kind in WHOLE_UNIT_INSTRUMENT_TYPES:
        return 0
    return SIZE_DECIMALS


def close_position(
    deal_id: str | Hashable,
    position: pd.Series,
    svc: IGService,
    size: int | float | None = None,
) -> dict[str, Any] | None:
    direction = "BUY" if position.direction == "SELL" else "SELL"

    try:
        result = svc.close_open_position(
            deal_id=deal_id,
            direction=direction,
            epic=None,
            expiry=None,
            level=None,
            order_type="MARKET",
            size=_round_size(abs(size or position["size"])),
            quote_id=None,
        )
    except Exception as ex:
        if ex.args and ex.args[0] == "error.confirms.deal-not-found":
            logger.warning(f"Deal {deal_id=} not found to cancel.")
            return None
        raise ex

    logger.info(result)
    return dict(result)


def close_all_open_position(
    positions: pd.DataFrame, svc: IGService
) -> list[dict[str, Any]]:
    actions = []

    for deal_id, position in positions.iterrows():
        result = close_position(deal_id=deal_id, position=position, svc=svc)
        if result is not None:
            actions.append(result)

    return actions


def get_current_positions(
    service: IGService,
) -> pd.DataFrame:
    """Open positions indexed by ``(epic, dealId)`` with a signed ``size``."""
    raw = service.fetch_open_positions()
    if raw is None or len(raw) == 0 or "epic" not in raw.columns:
        return pd.DataFrame(
            {
                "direction": pd.Series(dtype=str),
                "size": pd.Series(dtype=float),
                "stopLevel": pd.Series(dtype=float),
            },
            index=pd.MultiIndex.from_tuples([], names=["epic", "dealId"]),
        )
    all_positions = raw.set_index(["epic", "dealId"])
    all_positions["size"] = all_positions["direction"].map(
        {"BUY": 1, "SELL": -1}
    ) * all_positions["size"].astype(float)
    return cast(pd.DataFrame, all_positions)


def _epic_positions(current_position: pd.DataFrame, epic: str) -> pd.DataFrame:
    if epic not in current_position.index.get_level_values("epic"):
        return current_position.iloc[0:0].droplevel("epic")
    return cast(pd.DataFrame, current_position.xs(epic, level="epic"))


def update_open_positions(
    service: IGService,
    epic: str,
    new_level: float | None,
    current_position: pd.DataFrame,
    skip_deals: Iterable[Hashable] = (),
) -> list[dict[str, Any]]:
    """Move the stop on every open deal for ``epic`` to ``new_level``."""
    actions: list[dict[str, Any]] = []

    if new_level is not None and np.isnan(new_level):
        new_level = None

    skip = set(skip_deals)
    positions = _epic_positions(current_position, epic)

    for deal_id, position in positions.iterrows():
        if deal_id in skip:
            continue
        current_stop = position.get("stopLevel")
        if (
            new_level is not None
            and current_stop is not None
            and not pd.isna(current_stop)
            and abs(float(current_stop) - new_level) < STOP_LEVEL_TOLERANCE
        ):
            continue
        try:
            result = service.update_open_position(
                limit_level=None,
                stop_level=new_level,
                deal_id=deal_id,
            )
        except Exception:
            logger.warning(
                "Failed to update stop on %s for %s", deal_id, epic, exc_info=True
            )
            continue
        actions.append(dict(result))

    return actions


def reduce_open_positions(
    service: IGService,
    epic: str,
    quantity: float,
    current_position: pd.DataFrame,
    closed_deals: set[Hashable] | None = None,
    decimals: int = SIZE_DECIMALS,
) -> list[dict[str, Any]]:
    """Close ``quantity`` units of ``epic`` across its open deals, smallest first.

    ``current_position`` sizes are signed; closes always use the absolute size.
    Fully closed deal ids are added to ``closed_deals`` when given.
    """
    positions = _epic_positions(current_position, epic)
    actions = []

    quantity_cxd = 0.0
    ordered = positions.assign(_abs=positions["size"].abs()).sort_values("_abs")

    for deal_id, position in ordered.iterrows():
        open_size = float(position["_abs"])
        to_cancel = _round_size(min(open_size, quantity - quantity_cxd), decimals)
        if to_cancel <= 0:
            break

        result = close_position(
            deal_id=deal_id,
            position=position,
            svc=service,
            size=to_cancel,
        )
        if result is not None:
            actions.append(result)
            quantity_cxd += to_cancel
            if closed_deals is not None and to_cancel >= open_size:
                closed_deals.add(deal_id)

        if quantity_cxd >= quantity:
            break

    return actions


def get_market_status(service: IGService, epics: Iterable[str]) -> dict[str, str]:
    """Map epic -> IG ``marketStatus`` (``TRADEABLE``, ``EDITS_ONLY``, ``CLOSED``...).

    Returns an empty mapping if the lookup fails; callers then trade as before.
    """
    epics = list(epics)
    status: dict[str, str] = {}
    for i in range(0, len(epics), MARKETS_BATCH_SIZE):
        chunk = epics[i : i + MARKETS_BATCH_SIZE]
        try:
            markets = service.fetch_markets_by_epics(",".join(chunk), detailed=False)
        except Exception:
            logger.warning("Failed to fetch market status for %s", chunk, exc_info=True)
            continue
        if not isinstance(markets, list):
            continue
        for market in markets:
            try:
                epic = market["instrument"]["epic"]
                status[epic] = market["snapshot"]["marketStatus"]
            except (KeyError, TypeError):
                continue
    return status


def cli_app() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser()

    app.add_argument("--arctic-uri", required=True)
    app.add_argument("--name", required=True)
    app.add_argument("--universe", required=True)
    app.add_argument("--provider", required=True)
    app.add_argument("--portfolio-name", required=True)
    app.add_argument("--stage", default="rounded.position")

    return app


def get_currency(instrument: pd.Series) -> str:
    currencies = instrument.get("instrument.currencies")
    if isinstance(currencies, list) and currencies:
        default = next((c for c in currencies if c.get("isDefault")), currencies[0])
        code = default.get("code")
        if code:
            return str(code)

    name = str(instrument.get("instrument.name", ""))
    if "$" in name:
        return "USD"
    elif "£" in name:
        return "GBP"
    elif "€" in name:
        return "EUR"
    return "GBP"


def _is_stale(target_positions: pd.DataFrame, max_target_age: str | None) -> bool:
    if not max_target_age or target_positions.empty:
        return False
    latest = pd.Timestamp(target_positions.index[-1])
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    age = pd.Timestamp.now("UTC") - latest
    if age > pd.Timedelta(max_target_age):
        logger.warning(
            "Target positions are stale (latest=%s, age=%s > %s); not trading.",
            latest,
            age,
            max_target_age,
        )
        return True
    return False


def _skipped(epic: str, reason: str, current: float, target: float) -> dict[str, Any]:
    return {
        "date": pd.Timestamp.now("UTC").isoformat(),
        "epic": epic,
        "dealStatus": "SKIPPED",
        "reason": reason,
        "direction": "BUY" if target > current else "SELL",
        "size": abs(target - current),
    }


def _actions_frame(actions: list[dict[str, Any]]) -> pd.DataFrame:
    if not actions:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="DateTime", tz="UTC"))
    frame = pd.DataFrame(actions)
    # IG confirm dates are UTC; a DatetimeIndex makes the publisher append.
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("date"), utc=True, format="ISO8601"),
        name="DateTime",
    )
    return frame.sort_index()


def adjust_position_sizes(
    instruments: pd.DataFrame,
    target_positions: pd.DataFrame,
    stop_levels: pd.DataFrame | None = None,
    service: IGService | None = None,
    current_positions: pd.DataFrame | None = None,
    check_market_status: bool = True,
    max_target_age: str | None = None,
) -> pd.DataFrame:
    """Trade the IG account towards the latest row of ``target_positions``.

    :param instruments: instrument metadata indexed by epic.
    :param target_positions: wide target positions (index=time, columns=epics).
    :param stop_levels: optional wide stop levels, same shape as targets.
    :param check_market_status: skip epics whose market is not ``TRADEABLE``.
    :param max_target_age: e.g. ``"30min"``; skip trading entirely when the
        latest target row is older than this.
    :return: trade log indexed by UTC ``DateTime``.
    """
    service = service or get_ig_service()

    if _is_stale(target_positions, max_target_age):
        service.session.close()
        return _actions_frame([])

    if current_positions is None:
        current_positions = get_current_positions(service)

    actions: list[dict[str, Any]] = []

    signed = current_positions["size"].astype(float)
    current_sizes: pd.Series[float] = (
        signed.groupby(level="epic").sum().reindex(target_positions.columns).fillna(0.0)
        if len(signed)
        else pd.Series(0.0, index=target_positions.columns)
    )

    epics = [str(e) for e in target_positions.columns]
    market_status = get_market_status(service, epics) if check_market_status else {}

    for epic in epics:
        raw_target = target_positions[epic].iloc[-1]
        if pd.isna(raw_target):
            logger.warning("Target for %s is NaN, leaving position unchanged.", epic)
            continue

        instrument = (
            pd.Series(instruments.loc[epic]) if epic in instruments.index else None
        )
        decimals = size_decimals(instrument)
        latest_target = _round_size(raw_target, decimals)
        latest_stop = (
            _get_latest_or_none(stop_levels[epic])
            if stop_levels is not None and epic in stop_levels.columns
            else None
        )
        current_position = float(current_sizes.loc[epic])

        if abs(current_position - latest_target) < 10**-decimals / 2:
            logger.info(
                "%s matches target, nothing to do for %s.", current_position, epic
            )
            needs_trade = False
        else:
            needs_trade = True

        status = market_status.get(epic)
        if status is not None and status != TRADEABLE:
            if needs_trade:
                logger.info(
                    "Market %s for %s; skipping %s -> %s",
                    status,
                    epic,
                    current_position,
                    latest_target,
                )
                actions.append(
                    _skipped(epic, f"MARKET_{status}", current_position, latest_target)
                )
            continue

        closed_all = False
        closed_deals: set[Hashable] = set()

        # changing sides (or going flat), close existing
        if current_position and np.sign(current_position) != np.sign(latest_target):
            logger.info("Closing open position of %s for %s", current_position, epic)
            actions.extend(
                close_all_open_position(
                    _epic_positions(current_positions, epic), service
                )
            )
            current_position = 0.0
            closed_all = True

        if abs(current_position) < abs(latest_target):
            target = _round_size(abs(latest_target) - abs(current_position), decimals)
            side = "BUY" if latest_target > 0 else "SELL"

            logger.info(
                "Increasing target position from %s to %s - %s %s for %s stop @ %s",
                current_position,
                latest_target,
                side,
                target,
                epic,
                latest_stop,
            )
            if instrument is None:
                raise KeyError(f"No instrument metadata for {epic}")

            result = service.create_open_position(
                direction=side,
                currency_code=get_currency(instrument),
                order_type="MARKET",
                expiry=instrument["instrument.expiry"],
                size=float(target),
                epic=epic,
                force_open=bool(latest_stop),
                guaranteed_stop=False,
                level=None,
                limit_distance=None,
                limit_level=None,
                quote_id=None,
                stop_distance=None,
                trailing_stop=False,
                trailing_stop_increment=None,
                stop_level=round(latest_stop, 2) if latest_stop else None,
            )

            if result["dealStatus"] == "REJECTED":
                logger.warning(result)
            else:
                logger.info(result)
            actions.append(dict(result))

        elif abs(current_position) > abs(latest_target):
            reduce_by = _round_size(abs(current_position - latest_target), decimals)

            logger.info(
                "Reducing open positions by %s from %s to %s for %s",
                reduce_by,
                current_position,
                latest_target,
                epic,
            )
            actions.extend(
                reduce_open_positions(
                    service,
                    epic=epic,
                    quantity=reduce_by,
                    current_position=current_positions,
                    closed_deals=closed_deals,
                    decimals=decimals,
                )
            )

        # Move stops on deals that were already open and are still open.
        if latest_stop is not None and latest_target and not closed_all:
            actions.extend(
                update_open_positions(
                    service,
                    epic,
                    latest_stop,
                    current_positions,
                    skip_deals=closed_deals,
                )
            )

    service.session.close()
    return _actions_frame(actions)
