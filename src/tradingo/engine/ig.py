import argparse
import logging
from typing import Any, Hashable, cast

import numpy as np
import pandas as pd
from trading_ig.rest import IGService

from tradingo.sampling.ig import get_ig_service

logger = logging.getLogger(__name__)


def _get_latest_or_none(value: pd.Series | None) -> float | None:
    if value is None:
        return None
    latest = value.iloc[-1]
    if np.isnan(latest):
        return None
    return float(latest)


def close_position(
    deal_id: str | Hashable,
    position: pd.Series,
    svc: IGService,
    size: int | float | None = None,
) -> dict[str, Any]:
    direction = "BUY" if position.direction == "SELL" else "SELL"

    result = svc.close_open_position(
        deal_id=deal_id,
        direction=direction,
        epic=None,
        expiry=None,
        level=None,
        order_type="MARKET",
        size=size or abs(position["size"]),
        quote_id=None,
    )

    logger.info(result)
    return dict(result)


def close_all_open_position(
    positions: pd.DataFrame, svc: IGService
) -> list[dict[str, Any]]:
    epic_positions = positions
    actions = []

    for deal_id, position in epic_positions.iterrows():
        result = close_position(deal_id=deal_id, position=position, svc=svc)
        actions.append(result)

    return actions


def get_current_positions(
    service: IGService,
) -> pd.DataFrame:
    all_positions = service.fetch_open_positions().set_index(["epic", "dealId"])
    all_positions["size"] = (
        all_positions["direction"].map({"BUY": 1, "SELL": -1}) * all_positions["size"]
    )
    return cast(pd.DataFrame, all_positions)


def update_open_positions(
    service: IGService,
    epic: str,
    new_level: float | None,
    current_position: pd.DataFrame,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    if new_level is not None and np.isnan(new_level):
        new_level = None

    if epic not in current_position.columns:
        return actions

    positions = cast(pd.DataFrame, current_position.loc[epic])

    for deal_id, _ in positions.sort_values("size").iterrows():
        result = service.update_open_position(
            limit_level=None,
            stop_level=new_level,
            deal_id=deal_id,
        )
        actions.append(dict(result))

    return actions


def reduce_open_positions(
    service: IGService,
    epic: str,
    quantity: int,
    current_position: pd.DataFrame,
) -> list[dict[str, Any]]:
    positions = current_position.loc[epic]
    actions = []

    quantity_cxd = 0.0

    for deal_id, position in (
        cast(pd.DataFrame, positions).sort_values("size").iterrows()
    ):
        to_cancel = min(position["size"], quantity - quantity_cxd)

        result = close_position(
            deal_id=deal_id,
            position=position,
            svc=service,
            size=to_cancel,
        )
        actions.append(result)

        quantity_cxd += to_cancel

        if quantity_cxd >= quantity:
            break

    return actions


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
    name = str(instrument.name)
    if "$" in name:
        return "USD"
    elif "£" in name:
        return "GBP"
    elif "€" in name:
        return "EUR"
    return "GBP"


def adjust_position_sizes(
    instruments: pd.DataFrame,
    target_positions: pd.DataFrame,
    stop_levels: pd.DataFrame | None,
    service: IGService | None = None,
    current_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    service = service or get_ig_service()
    if current_positions is None:
        current_positions = get_current_positions(service)

    actions: list[dict[str, Any]] = []

    current_sizes: pd.Series[float] = (
        current_positions.groupby(level="epic")["size"]
        .sum()
        .reindex(target_positions.columns)
        .fillna(0.0)
    )

    current_stop_level: pd.Series[float] = (
        current_positions.groupby(level="epic")["size"]
        .first()
        .reindex(target_positions.columns)
    )

    for epic in target_positions:
        latest_target = target_positions[epic].iloc[-1]
        latest_stop = (
            _get_latest_or_none(stop_levels[epic]) if stop_levels is not None else None
        )

        current_position = current_sizes.loc[str(epic)]

        # changing sides, close existing
        if current_position and np.sign(current_position) != np.sign(latest_target):
            logger.info("Closing open position of %s for %s", current_position, epic)
            close_actions = close_all_open_position(
                pd.DataFrame(current_positions.loc[str(epic)]),
                service,
            )
            actions.extend(close_actions)
            current_position = 0.0

        # increasing position
        if abs(current_position) < abs(latest_target):
            target = abs(latest_target) - abs(current_position)
            side = "BUY" if latest_target > 0 else "SELL"

            logger.info(
                "Increasing target position from %s to %s - %s for %s stop @ %s",
                current_position,
                latest_target,
                side,
                epic,
                latest_stop,
            )

            result = service.create_open_position(
                direction=side,
                currency_code=get_currency(pd.Series(instruments.loc[str(epic)])),
                order_type="MARKET",
                expiry=instruments.loc[str(epic)]["instrument.expiry"],
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
            reduce_by = abs(current_position - latest_target)

            logger.info(
                "Reducing open positions by %s from %s to %s for %s",
                reduce_by,
                current_position,
                latest_target,
                epic,
            )
            reduce_actions = reduce_open_positions(
                service,
                epic=str(epic),
                quantity=reduce_by,
                current_position=current_positions,
            )
            actions.extend(reduce_actions)
        else:
            logger.info(
                "%s matches target, nothing to do for %s.",
                current_position,
                epic,
            )

        # Only update stop levels if there are existing positions
        if (
            (latest_target and not np.isnan(latest_target))
            and pd.isna(current_stop_level.loc[str(epic)])
            and current_stop_level.loc[str(epic)] != latest_stop
        ):
            update_actions = update_open_positions(
                service, str(epic), latest_stop, current_positions
            )
            actions.extend(update_actions)

    service.session.close()
    if actions:
        return pd.DataFrame(actions).set_index("date").rename_axis("DateTime")
    else:
        return pd.DataFrame(index=pd.DatetimeIndex(data=[], name="DateTime"))
