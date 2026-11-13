import logging
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _backtest  # type: ignore

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

SUMMARY_PNL_FIELDS = ("net_investment", "realised_pnl", "total_pnl", "unrealised_pnl")
SUMMARY_FIELDS = (*SUMMARY_PNL_FIELDS, "net_exposure", "gross_exposure")


def backtest(
    *,
    portfolio: pd.DataFrame,
    bid_close: pd.DataFrame,
    ask_close: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
    stop_limit: Optional[pd.DataFrame] = None,
    stop_loss: Optional[pd.DataFrame] = None,
    price_ffill_limit: int = 0,
) -> tuple[pd.DataFrame, ...]:
    bid_close = bid_close.groupby(cast(pd.DatetimeIndex, bid_close.index).date).ffill(
        limit=price_ffill_limit
    )
    ask_close = ask_close.groupby(cast(pd.DatetimeIndex, ask_close.index).date).ffill(
        limit=price_ffill_limit
    )

    mid_close = (bid_close + ask_close) / 2
    bid_close, ask_close = (
        bid_close.reindex(mid_close.index),
        ask_close.reindex(mid_close.index),
    )

    # restrict position changes to where we have a price.
    portfolio = (
        portfolio.ffill()
        .reindex(mid_close.index, method="ffill")
        .where(mid_close.notna(), np.nan)
        .ffill()
        .fillna(0.0)
    )

    trades = portfolio.diff()
    trades.iloc[0] = portfolio.iloc[0]
    del portfolio

    index = mid_close.index
    columns = trades.columns
    n_rows, n_cols = len(index), len(columns)

    # each instrument is tracked from its first price onwards
    valid = mid_close.notna().to_numpy()
    starts = np.where(valid.any(axis=0), valid.argmax(axis=0), n_rows)
    del valid
    out_start = int(starts.min()) if n_cols else n_rows
    out_index = index[out_start:]
    n_out = len(out_index)

    # Per-field outputs are filled one instrument at a time so no wide
    # (instrument x field) frame is ever materialised.
    # Fortran order => each instrument's column write is contiguous.
    fields = {
        f: np.full((n_out, n_cols), np.nan, dtype="float32", order="F")
        for f in BACKTEST_FIELDS
    }
    totals = {k: np.zeros(n_out, dtype="float64") for k in SUMMARY_FIELDS}

    def accumulate(
        total: npt.NDArray[np.float64], offset: int, values: npt.NDArray[np.float64]
    ) -> None:
        filled = pd.Series(values).ffill().to_numpy()
        total[offset:] += np.where(np.isnan(filled), 0.0, filled)

    def optional(
        frame: Optional[pd.DataFrame], name: object, idx: pd.Index, fill: float
    ) -> npt.NDArray[np.float32]:
        if frame is None:
            return np.full(len(idx), fill, dtype="float32")
        return frame[name].reindex(idx).fillna(fill).to_numpy(dtype="float32")

    for j, name in enumerate(columns):
        start = int(starts[j])
        if start >= n_rows:
            continue
        logger.info("Computing backtest for ticker=%s", name)
        inst_index = index[start:]
        offset = start - out_start

        result = _backtest.compute_backtest(
            trades.iloc[start:, j].fillna(0.0).to_numpy(dtype="float32"),
            bid_close.iloc[start:, j].to_numpy(dtype="float32"),
            ask_close.iloc[start:, j].to_numpy(dtype="float32"),
            optional(stop_limit, name, inst_index, np.nan),
            optional(stop_loss, name, inst_index, np.nan),
            optional(dividends, name, inst_index, 0.0),
        )

        for k, f in enumerate(BACKTEST_FIELDS):
            fields[f][offset:, j] = result[:, k]

        position = result[:, BACKTEST_FIELDS.index("net_position")]
        mids = mid_close.iloc[start:, j].to_numpy()
        for f in SUMMARY_PNL_FIELDS:
            accumulate(totals[f], offset, result[:, BACKTEST_FIELDS.index(f)])
        accumulate(totals["net_exposure"], offset, position * mids)
        accumulate(totals["gross_exposure"], offset, np.abs(position) * mids)

        # unrealised pnl is only meaningful while a position is open
        fields["unrealised_pnl"][offset:, j] = np.where(
            position == 0.0, np.nan, result[:, BACKTEST_FIELDS.index("unrealised_pnl")]
        )
        del result

    summary = pd.DataFrame(
        {
            k: v.astype("float32") if k in SUMMARY_PNL_FIELDS else v
            for k, v in totals.items()
        },
        index=out_index,
    )

    return (
        summary,
        *(
            pd.DataFrame(fields[f], index=out_index, columns=columns, copy=False)
            for f in BACKTEST_FIELDS
        ),
    )
