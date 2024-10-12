import logging
from typing import Optional

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
)


@symbol_provider(
    portfolio="portfolio/{name}.{stage}",
    bid="prices/close",
    ask="prices/close",
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
    portfolio: pd.DataFrame,
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    dividends: Optional[pd.DataFrame],
    name: str,
    stage: str = "raw",
    **kwargs,
):
    trades = portfolio.ffill().fillna(0.0).diff()
    bid, ask = bid.ffill(), ask.ffill()

    if dividends is None:
        dividends = pd.DataFrame(0, index=bid.index, columns=bid.columns)

    logger.info("running backtest for %s on stage %s with %s", name, stage, kwargs)

    def compute_backtest(inst_trades: pd.Series):

        logger.info("Computing backtest for ticker=%s", inst_trades.name)
        inst_asks = ask[inst_trades.name].ffill()
        inst_bids = bid[inst_trades.name].ffill()
        inst_dividends = dividends[inst_trades.name].fillna(0.0)
        opening_position = portfolio.loc[
            portfolio[inst_trades.name].first_valid_index(), inst_trades.name
        ]
        if opening_position != 0:
            opening_avg_price = ask.loc[
                portfolio[inst_trades.name].first_valid_index(), inst_trades.name
            ]
        else:
            opening_avg_price = 0

        return pd.DataFrame(
            data=_backtest.compute_backtest(
                opening_position,
                opening_avg_price,
                inst_trades.fillna(0.0).to_numpy().astype("float32"),
                inst_bids.to_numpy().astype("float32"),
                inst_asks.to_numpy().astype("float32"),
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

    prices = (ask + bid) / 2

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
