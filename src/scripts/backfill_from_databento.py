"""Backfill the IG and Yahoo Finance raw price libraries from a Databento DBN file.

Merges Databento OHLCV bars into the SAME raw price libraries/symbols the
live IG and Yahoo Finance sampling pipelines write to, rather than building a
separate Databento-only universe: existing bars from the live provider are
always kept, Databento only fills gaps, via ``DataFrame.combine_first``.

- Yahoo Finance: written straight into ``{yf_raw_lib}/{ticker}`` (same
  Open/High/Low/Close/Volume shape ``sampling.yf.sample_equity`` produces).
- IG: adapted to the bid/ask shape ``sampling.ig.create_universe`` expects via
  ``igtrading_from_bars`` (bid == ask == trade OHLC, Databento has no quoted
  spread) and written to ``{ig_raw_lib}/{epic}.bid`` / ``{ig_raw_lib}/{epic}.ask``.
  Databento's raw symbols don't match IG epics, so the ticker->epic mapping is
  read positionally off two existing ``instruments`` symbols (see
  --tickers-instruments-symbol / --ig-instruments-symbol) rather than guessed
  — they must have been built from the same ordered ticker/epic list.

Usage:

    uv run --group databento python src/scripts/backfill_from_databento.py \\
        data/databento/XNAS.ITCH.ohlcv-1m.2018-07-01_2026-09-18.dbn.zst \\
        --arctic-uri lmdb:///Users/rmcstay/data/tradingo.db \\
        --tickers-instruments-symbol etf-momentum \\
        --ig-instruments-symbol etf-momentum-ig
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import cast

import pandas as pd
from arcticdb import Arctic
from arcticdb.version_store.library import Library

from tradingo.sampling.databento import igtrading_from_bars, ingest_bars

logger = logging.getLogger(__name__)


def _merge_write(lib: Library, symbol: str, new: pd.DataFrame) -> None:
    """Write *new* into *symbol*, preferring any existing values on overlap."""
    if lib.has_symbol(symbol):
        existing = cast(pd.DataFrame, lib.read(symbol).data)
        index_name = existing.index.name
        merged = existing.combine_first(new).sort_index()
        # combine_first unions the two indexes: if the existing and new frames'
        # index names disagree (e.g. yfinance's "Datetime" vs databento's
        # "ts_event"), pandas drops the name, which breaks ArcticDB's
        # incremental `update()` for any downstream symbol keyed on it.
        merged.index.name = index_name
    else:
        merged = new.sort_index()
    lib.write(symbol, merged)
    logger.info(
        "wrote %s/%s: %d rows (%s to %s)",
        lib.name,
        symbol,
        len(merged),
        merged.index.min(),
        merged.index.max(),
    )


def epic_map(
    arctic: Arctic,
    instruments_lib: str,
    tickers_symbol: str,
    ig_symbol: str,
) -> dict[str, str]:
    """Build a ticker -> IG epic map from two instruments symbols with matching row order."""
    lib = arctic.get_library(instruments_lib)
    tickers = cast(pd.DataFrame, lib.read(tickers_symbol).data).index.to_list()
    epics = cast(pd.DataFrame, lib.read(ig_symbol).data).index.to_list()
    if len(tickers) != len(epics):
        raise ValueError(
            f"{tickers_symbol} has {len(tickers)} rows but {ig_symbol} has "
            f"{len(epics)} - can't zip positionally, pass an explicit mapping instead"
        )
    return dict(zip(tickers, epics))


def backfill(
    dbn_path: Path,
    arctic_uri: str,
    schema: str = "ohlcv-1m",
    yf_raw_lib: str = "prices_yfinance",
    ig_raw_lib: str = "prices_igtrading",
    instruments_lib: str = "instruments",
    tickers_instruments_symbol: str = "etf-momentum",
    ig_instruments_symbol: str = "etf-momentum-ig",
    skip_ig: bool = False,
) -> None:
    arctic = Arctic(arctic_uri)
    bars = {symbol: df for df, (symbol,) in ingest_bars(dbn_path, schema=schema)}

    yf_lib = arctic.get_library(yf_raw_lib, create_if_missing=True)

    for ticker, raw in bars.items():
        _merge_write(yf_lib, ticker, raw)

    if skip_ig:
        return

    mapping = epic_map(
        arctic, instruments_lib, tickers_instruments_symbol, ig_instruments_symbol
    )
    ig_lib = arctic.get_library(ig_raw_lib, create_if_missing=True)

    for ticker, raw in bars.items():
        epic = mapping.get(ticker)
        if epic is None:
            logger.warning("no IG epic mapped for %s, skipping IG backfill", ticker)
            continue

        bid, ask = igtrading_from_bars(raw, raw.index.min(), raw.index.max())
        _merge_write(ig_lib, f"{epic}.bid", bid)
        _merge_write(ig_lib, f"{epic}.ask", ask)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dbn_path", type=Path)
    parser.add_argument("--arctic-uri", default=os.environ.get("TP_ARCTIC_URI"))
    parser.add_argument("--schema", default="ohlcv-1m")
    parser.add_argument("--yf-raw-lib", default="prices_yfinance")
    parser.add_argument("--ig-raw-lib", default="prices_igtrading")
    parser.add_argument("--instruments-lib", default="instruments")
    parser.add_argument("--tickers-instruments-symbol", default="etf-momentum")
    parser.add_argument("--ig-instruments-symbol", default="etf-momentum-ig")
    parser.add_argument(
        "--skip-ig",
        action="store_true",
        help="only backfill the Yahoo Finance raw library, leave IG untouched",
    )
    args = parser.parse_args()

    if not args.arctic_uri:
        parser.error("--arctic-uri or TP_ARCTIC_URI must be set")

    backfill(
        dbn_path=args.dbn_path,
        arctic_uri=args.arctic_uri,
        schema=args.schema,
        yf_raw_lib=args.yf_raw_lib,
        ig_raw_lib=args.ig_raw_lib,
        instruments_lib=args.instruments_lib,
        tickers_instruments_symbol=args.tickers_instruments_symbol,
        ig_instruments_symbol=args.ig_instruments_symbol,
        skip_ig=args.skip_ig,
    )


if __name__ == "__main__":
    main()
