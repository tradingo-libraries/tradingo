import logging
from typing import Optional

import numpy as np
import pandas as pd

from tradingo.symbols import symbol_provider, symbol_publisher
from tradingo import _backtest


logger = logging.getLogger(__name__)


BACKTEST_FIELDS = (
    "unrealised_pnl",
    "realised_pnl",
    "total_pnl",
    "net_investment",
    "net_position",
    "avg_open_price",
    "stop_trade",
)


@symbol_provider(
    portfolio="portfolio/{name}.{stage}",
    bid_close="prices/close",
    ask_close="prices/close",
    dividends="prices/dividend",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_publisher(
    "backtest/portfolio",
    *(f"backtest/instrument.{f}" for f in BACKTEST_FIELDS if f != "date"),
    symbol_prefix="{provider}.{universe}.{name}.{stage}.",
    astype="float64",
)
def backtest(
    *,
    name: str,
    portfolio: pd.DataFrame,
    bid_close: pd.DataFrame,
    ask_close: pd.DataFrame,
    dividends: pd.DataFrame,
    stop_limit: Optional[pd.DataFrame] = None,
    stop_loss: Optional[pd.DataFrame] = None,
    stage: str = "raw",
    **kwargs,
):
    trades = portfolio.ffill().fillna(0.0).diff()
    trades.iloc[0] = portfolio.iloc[0]
    bid_close, ask_close = bid_close.ffill(), ask_close.ffill()

    if dividends is None:
        dividends = pd.DataFrame(
            0,
            index=bid_close.index,
            columns=bid_close.columns,
        )
    if stop_limit is None:
        stop_limit = pd.DataFrame(
            np.nan,
            index=bid_close.index,
            columns=bid_close.columns,
        )
    if stop_loss is None:
        stop_loss = pd.DataFrame(
            np.nan,
            index=bid_close.index,
            columns=bid_close.columns,
        )

    logger.info("running backtest for %s on stage %s with %s", name, stage, kwargs)

    def compute_backtest(inst_trades: pd.Series):

        logger.info("Computing backtest for ticker=%s", inst_trades.name)
        inst_asks = ask_close[inst_trades.name].ffill()
        inst_bids = bid_close[inst_trades.name].ffill()
        inst_limit = stop_limit[inst_trades.name]
        inst_loss = stop_loss[inst_trades.name]

        inst_dividends = dividends[inst_trades.name].fillna(0.0)

        return pd.DataFrame(
            data=_backtest.compute_backtest(
                inst_trades.fillna(0.0).to_numpy().astype("float32"),
                inst_bids.to_numpy().astype("float32"),
                inst_asks.to_numpy().astype("float32"),
                inst_limit.to_numpy().astype("float32"),
                inst_loss.to_numpy().astype("float32"),
                inst_dividends.to_numpy().astype("float32"),
            ),
            index=inst_trades.index,
            columns=BACKTEST_FIELDS,
        )

    backtest = pd.concat(
        (compute_backtest(data) for _, data in trades.items()),
        keys=trades.columns,
        axis=1,
    ).reorder_levels([1, 0], axis=1)

    backtest_fields = (backtest.loc[:, f] for f in BACKTEST_FIELDS if f != "date")

    prices = (ask_close + bid_close) / 2

    net_exposure = (backtest["net_position"] * prices.ffill()).sum(axis=1)
    gross_exposure = (backtest["net_position"].abs() * prices.ffill()).sum(axis=1)

    summary = (
        backtest[
            [
                "net_investment",
                "unrealised_pnl",
                "realised_pnl",
                "total_pnl",
            ]
        ]
        .transpose()
        .groupby(level=0)
        .sum()
        .transpose()
    )
    summary["net_exposure"] = net_exposure
    summary["gross_exposure"] = gross_exposure

    return (summary, *backtest_fields)
