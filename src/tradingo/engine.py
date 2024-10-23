import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading_ig.rest import IGService
from tradingo.api import Tradingo
from tradingo.sampling import get_ig_service
from tradingo.symbols import symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


def close_position(deal_id, position, svc, size=None):
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


def close_all_open_position(positions, svc):

    epic_positions = positions

    for deal_id, position in epic_positions.iterrows():

        close_position(deal_id=deal_id, position=position, svc=svc)


def get_current_positions(
    service: IGService,
):

    all_positions = service.fetch_open_positions().set_index(["epic", "dealId"])
    all_positions["size"] = (
        all_positions["direction"].replace({"BUY": 1, "SELL": -1})
        * all_positions["size"]
    )
    return all_positions


def reduce_open_positions(
    service: IGService,
    epic: str,
    quantity: int,
):

    positions = get_current_positions(service).loc[epic]

    quantity_cxd = 0.0

    for deal_id, position in positions.sort_values("size").iterrows():

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


def cli_app():
    app = argparse.ArgumentParser()

    app.add_argument("--arctic-uri", required=True)
    app.add_argument("--name", required=True)
    app.add_argument("--universe", required=True)
    app.add_argument("--provider", required=True)
    app.add_argument("--portfolio-name", required=True)
    app.add_argument("--stage", default="rounded.position")

    return app


def get_currency(instrument: pd.Series):
    if "$" in instrument.name:
        return "USD"
    elif "£" in instrument.name:
        return "GBP"
    elif "€" in instrument.name:
        return "EUR"
    return "GBP"


# TODO: #21 - symbol_publisher
@symbol_provider(
    instruments="instruments/{universe}",
    no_date=True,
)
@symbol_provider(
    target_positions="portfolio/{name}.{stage}",
    symbol_prefix="{provider}.{universe}.",
)
def adjust_position_sizes(
    stage: str,
    instruments: pd.DataFrame,
    target_positions: pd.DataFrame,
    universe: str,
    provider: str,
    name: str,
    service: Optional[IGService] = None,
    **kwargs,
):

    logger.info(
        "Adjusting name=%s universe=%s provider=%s stage=%s",
        name,
        universe,
        provider,
        stage,
    )

    service = service or get_ig_service()
    current_positions = get_current_positions(service)

    current_sizes = (
        current_positions.groupby("epic")["size"]
        .sum()
        .reindex(target_positions.columns)
        .fillna(0.0)
    )

    for epic in target_positions:
        latest_target = target_positions[epic].iloc[-1]
        current_position = current_sizes.loc[epic]

        # changing sides, close existing
        if current_position and np.sign(current_position) != np.sign(latest_target):
            logger.info("Closing open position of %s for %s", current_position, epic)
            close_all_open_position(current_positions.loc[epic], service)
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
                currency_code=get_currency(instruments.loc[epic]),
                order_type="MARKET",
                expiry=instruments.loc[epic].expiry,
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
                stop_level=None,
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
                epic=epic,
                quantity=reduce_by,
            )
        else:
            logger.info(
                "%s matches target, nothing to do for %s.",
                current_position,
                epic,
            )


def main():

    args = cli_app().parse_args()

    service = get_ig_service()
