"""Interactive Brokers trading engine"""

import argparse
import logging
from typing import cast

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Stock

from tradingo.sampling.ib import get_ib_service

logger = logging.getLogger(__name__)


def close_position(
    contract: Stock,
    position: pd.Series,
    ib: IB,
    size: int | float | None = None,
) -> None:
    """Close a single position."""
    direction = "BUY" if position["direction"] == "SELL" else "SELL"

    order = MarketOrder(direction, size or abs(position["size"]))
    trade = ib.placeOrder(contract, order)
    ib.runUntil(trade.isDone, timeout=30)

    logger.info(
        "Closed position: %s %s %s shares - Status: %s",
        contract.symbol,
        direction,
        size or abs(position["size"]),
        trade.orderStatus.status,
    )


def close_all_open_position(positions: pd.DataFrame, ib: IB) -> None:
    """Close all open positions."""
    for idx, position in positions.iterrows():
        symbol, currency = cast(tuple[str, str], idx)
        contract = Stock(symbol, "SMART", currency)
        ib.qualifyContracts(contract)
        close_position(contract=contract, position=position, ib=ib)


def get_current_positions(
    ib: IB,
    account: str | None = None,
) -> pd.DataFrame:
    """Fetch current positions from the IB gateway."""
    positions = ib.positions() if account is None else ib.positions(account)

    if not positions:
        return pd.DataFrame()

    position_data = []
    for pos in positions:
        contract = pos.contract
        position_data.append(
            {
                "symbol": contract.symbol,
                "currency": contract.currency,
                "size": pos.position,  # positive = long, negative = short
                "direction": "BUY" if pos.position > 0 else "SELL",
                "avg_price": pos.avgCost,
                "contract": contract,
            }
        )

    return pd.DataFrame(position_data).set_index(["symbol", "currency"])


def reduce_open_positions(
    ib: IB,
    symbol: str,
    currency: str,
    quantity: int | float,
    account: str | None = None,
) -> None:
    """Reduce an open position by a specific quantity."""
    positions = get_current_positions(ib, account)

    try:
        position = cast(pd.Series, positions.loc[(symbol, currency)])
    except KeyError:
        logger.warning("No position found for %s %s", symbol, currency)
        return

    to_cancel = min(abs(float(position["size"])), quantity)
    direction = "SELL" if float(position["size"]) > 0 else "BUY"

    contract = Stock(symbol, "SMART", currency)
    ib.qualifyContracts(contract)

    order = MarketOrder(direction, to_cancel)
    trade = ib.placeOrder(contract, order)
    ib.runUntil(trade.isDone, timeout=30)

    logger.info(
        "Reduced position: %s %s - %s shares - Status: %s",
        symbol,
        direction,
        to_cancel,
        trade.orderStatus.status,
    )


def adjust_position_sizes(
    instruments: pd.DataFrame,
    target_positions: pd.DataFrame,
    ib: IB | None = None,
    account: str | None = None,
) -> None:
    """Adjust live positions to match target positions."""
    ib = ib or get_ib_service()
    current_positions = get_current_positions(ib, account)

    if current_positions.empty:
        current_sizes: pd.Series = pd.Series(0.0, index=target_positions.columns)
    else:
        current_sizes = (
            current_positions["size"]
            .groupby(level="symbol")
            .sum()
            .reindex(target_positions.columns)
            .fillna(0.0)
        )

    for symbol in target_positions:
        latest_target = target_positions[symbol].iloc[-1]
        current_position = float(current_sizes.loc[str(symbol)])

        instrument = instruments.loc[str(symbol)]
        currency = str(instrument.get("currency", "USD"))
        if "currency" not in instrument.index:
            logger.warning(
                "No currency for %s in instruments, defaulting to USD", symbol
            )

        if current_position != 0 and np.sign(current_position) != np.sign(
            latest_target
        ):
            logger.info("Closing open position of %s for %s", current_position, symbol)
            try:
                pos_data = cast(
                    pd.Series, current_positions.loc[(str(symbol), currency)]
                )
                contract = Stock(str(symbol), "SMART", currency)
                ib.qualifyContracts(contract)
                close_position(contract=contract, position=pos_data, ib=ib)
            except KeyError:
                pass
            current_position = 0.0

        if abs(current_position) < abs(latest_target):
            target = abs(latest_target) - abs(current_position)
            side = "BUY" if latest_target > 0 else "SELL"

            logger.info(
                "Increasing position from %s to %s (%s) for %s",
                current_position,
                latest_target,
                side,
                symbol,
            )

            contract = Stock(str(symbol), "SMART", currency)
            ib.qualifyContracts(contract)

            order = MarketOrder(side, target)
            trade = ib.placeOrder(contract, order)
            ib.runUntil(trade.isDone, timeout=30)

            logger.info(
                "Order placed: %s %s %s - Status: %s",
                symbol,
                side,
                target,
                trade.orderStatus.status,
            )

        elif abs(current_position) > abs(latest_target):
            reduce_by = abs(current_position - latest_target)

            logger.info(
                "Reducing position by %s (from %s to %s) for %s",
                reduce_by,
                current_position,
                latest_target,
                symbol,
            )

            reduce_open_positions(
                ib=ib,
                symbol=str(symbol),
                currency=currency,
                quantity=reduce_by,
                account=account,
            )
        else:
            logger.info(
                "%s matches target, nothing to do for %s.", current_position, symbol
            )

    ib.disconnect()


def cli_app() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser("tradingo-engine-ib")

    app.add_argument("--arctic-uri", required=True)
    app.add_argument("--name", required=True, help="Strategy name")
    app.add_argument("--universe", required=True)
    app.add_argument("--provider", required=True)
    app.add_argument("--portfolio-name", required=True)
    app.add_argument("--stage", default="rounded.position")
    app.add_argument("--account", default=None, help="IB account number")

    return app


def main() -> None:
    """Load target positions from ArcticDB and adjust live IB positions."""
    import logging as _logging

    from arcticdb import Arctic

    _logging.basicConfig(level=_logging.INFO)

    args = cli_app().parse_args()

    arctic = Arctic(args.arctic_uri)

    instruments_lib = arctic.get_library("instruments")
    instruments: pd.DataFrame = instruments_lib.read(args.universe).data

    positions_lib = arctic.get_library(args.name)
    target_positions: pd.DataFrame = positions_lib.read(
        f"{args.portfolio_name}.{args.stage}"
    ).data

    adjust_position_sizes(
        instruments=instruments,
        target_positions=target_positions,
        account=args.account,
    )


if __name__ == "__main__":
    main()
