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
    "net_exposure",
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
    dividends: Optional[pd.DataFrame] = None,
    stop_limit: Optional[pd.DataFrame] = None,
    stop_loss: Optional[pd.DataFrame] = None,
    stage: str = "raw",
    price_ffill_limit: int = 0,
    **kwargs,
):

    bid_close = bid_close.groupby(bid_close.index.date).ffill(limit=price_ffill_limit)
    ask_close = ask_close.groupby(bid_close.index.date).ffill(limit=price_ffill_limit)

    mid_close = (bid_close + ask_close) / 2
    bid_close, ask_close = bid_close.reindex(mid_close.index), ask_close.reindex(
        mid_close.index
    )

    # restrict position changes to where we have a price.
    portfolio = (
        portfolio.ffill()
        .reindex(mid_close.index, method="ffill")
        .where(mid_close.notna(), np.nan)
        .ffill()
        .fillna(0.0)
    )

    # portfolio = portfolio.reindex(mid_close.index, method="ffill").fillna(0.0)
    trades = portfolio.diff()
    trades.iloc[0] = portfolio.iloc[0]

    if dividends is None:
        dividends = pd.DataFrame(
            0,
            index=mid_close.index,
            columns=mid_close.columns,
        )
    if stop_limit is None:
        stop_limit = pd.DataFrame(
            np.nan,
            index=mid_close.index,
            columns=mid_close.columns,
        )
    if stop_loss is None:
        stop_loss = pd.DataFrame(
            np.nan,
            index=mid_close.index,
            columns=mid_close.columns,
        )

    logger.info("running backtest for %s on stage %s with %s", name, stage, kwargs)

    def compute_backtest(inst_trades: pd.Series):

        logger.info("Computing backtest for ticker=%s", inst_trades.name)
        inst_mids = mid_close[inst_trades.name].ffill().dropna()
        inst_asks = ask_close[inst_trades.name].reindex(inst_mids.index, method="ffill")
        inst_bids = bid_close[inst_trades.name].reindex(inst_mids.index, method="ffill")
        inst_limit = stop_limit[inst_trades.name].reindex(inst_mids.index)
        inst_loss = stop_loss[inst_trades.name].reindex(inst_mids.index)

        inst_dividends = (
            dividends[inst_trades.name].reindex(mid_close.index).fillna(0.0)
        )

        return pd.DataFrame(
            data=_backtest.compute_backtest(
                inst_trades.reindex(inst_mids.index)
                .fillna(0.0)
                .to_numpy()
                .astype("float32"),
                inst_bids.to_numpy().astype("float32"),
                inst_asks.to_numpy().astype("float32"),
                inst_limit.to_numpy().astype("float32"),
                inst_loss.to_numpy().astype("float32"),
                inst_dividends.to_numpy().astype("float32"),
            ),
            index=inst_mids.index,
            columns=BACKTEST_FIELDS,
        )

    backtest = pd.concat(
        (compute_backtest(data) for _, data in trades.items()),
        keys=trades.columns,
        axis=1,
    ).reorder_levels([1, 0], axis=1)

    backtest_fields = (backtest.loc[:, f] for f in BACKTEST_FIELDS if f != "date")

    net_exposure = (backtest["net_position"] * mid_close).ffill().sum(axis=1)
    gross_exposure = (backtest["net_position"].abs() * mid_close).ffill().sum(axis=1)

    summary = (
        backtest[
            [
                "net_investment",
                "unrealised_pnl",
                "realised_pnl",
                "total_pnl",
            ]
        ]
        .ffill()
        .transpose()
        .groupby(level=0)
        .sum()
        .transpose()
    )
    summary["net_exposure"] = net_exposure
    summary["gross_exposure"] = gross_exposure

    return (summary, *backtest_fields)
