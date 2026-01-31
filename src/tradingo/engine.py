import argparse
import logging
from typing import Hashable, cast

import numpy as np
import pandas as pd
from trading_ig.rest import IGService

from tradingo.sampling.ig import get_ig_service

logger = logging.getLogger(__name__)


def close_position(
    deal_id: str | Hashable,
    position: pd.Series,
    svc: IGService,
    size: int | float | None = None,
) -> None:
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


def close_all_open_position(positions: pd.DataFrame, svc: IGService) -> None:
    epic_positions = positions

    for deal_id, position in epic_positions.iterrows():
        close_position(deal_id=deal_id, position=position, svc=svc)


def get_current_positions(
    service: IGService,
) -> pd.DataFrame:
    all_positions = service.fetch_open_positions().set_index(["epic", "dealId"])
    all_positions["size"] = (
        all_positions["direction"].replace({"BUY": 1, "SELL": -1})
        * all_positions["size"]
    )
    return cast(pd.DataFrame, all_positions)


def update_open_positions(
    service: IGService,
    epic: str,
    new_level: float | None,
    current_position: pd.DataFrame,
) -> None:

    if new_level is not None and np.isnan(new_level):
        new_level = None

    positions = cast(pd.DataFrame, current_position.loc[epic])

    for deal_id, _ in positions.sort_values("size").iterrows():
        service.update_open_position(
            limit_level=None,
            stop_level=new_level,
            deal_id=deal_id,
        )


def reduce_open_positions(
    service: IGService,
    epic: str,
    quantity: int,
    current_position: pd.DataFrame,
) -> None:
    positions = current_position.loc[epic]

    quantity_cxd = 0.0

    for deal_id, position in (
        cast(pd.DataFrame, positions).sort_values("size").iterrows()
    ):
        to_cancel = min(position["size"], quantity - quantity_cxd)

        close_position(
            deal_id=deal_id,
            position=position,
            svc=service,
            size=to_cancel,
        )

        quantity_cxd += to_cancel

        if quantity_cxd >= quantity:
            break


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
) -> None:
    service = service or get_ig_service()
    if current_positions is None:
        current_positions = get_current_positions(service)

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
        if stop_levels is not None:
            latest_stop = float(stop_levels[epic].iloc[-1])
        else:
            latest_stop = float(np.nan)

        current_position = current_sizes.loc[str(epic)]

        # changing sides, close existing
        if current_position and np.sign(current_position) != np.sign(latest_target):
            logger.info("Closing open position of %s for %s", current_position, epic)
            close_all_open_position(
                pd.DataFrame(current_positions.loc[str(epic)]),
                service,
            )
            current_position = 0.0

        # increasing position
        if abs(current_position) < abs(latest_target):
            target = abs(latest_target) - abs(current_position)
            side = "BUY" if latest_target > 0 else "SELL"

            logger.info(
                "Increasing target position from %s to %s - %s for %s",
                current_position,
                latest_target,
                side,
                epic,
            )

            result = service.create_open_position(
                direction=side,
                currency_code=get_currency(pd.Series(instruments.loc[str(epic)])),
                order_type="MARKET",
                expiry=instruments.loc[str(epic)].expiry,
                size=target,
                epic=epic,
                force_open=False,
                guaranteed_stop=False,
                level=None,
                limit_distance=None,
                limit_level=None,
                quote_id=None,
                stop_distance=None,
                trailing_stop=None,
                trailing_stop_increment=None,
                stop_level=latest_stop,
            )

            logger.info(result)

        elif abs(current_position) > abs(latest_target):
            reduce_by = abs(current_position - latest_target)

            logger.info(
                "Reducing open positions by %s from %s to %s for %s",
                reduce_by,
                current_position,
                latest_target,
                epic,
            )
            reduce_open_positions(
                service,
                epic=str(epic),
                quantity=reduce_by,
                current_position=current_positions,
            )
        else:
            logger.info(
                "%s matches target, nothing to do for %s.",
                current_position,
                epic,
            )

        # Only update stop levels if there are existing positions
        if (
            not pd.isna(current_stop_level.loc[str(epic)])
            and current_stop_level.loc[str(epic)] != latest_stop
        ):
            update_open_positions(service, str(epic), latest_stop, current_positions)
    service.session.close()
