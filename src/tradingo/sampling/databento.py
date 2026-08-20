"""Databento historical/batch download provider.

Discovers ``.dbn``/``.dbn.zst`` files produced by a Databento historical
download (e.g. ``Historical().batch.download()`` or a portal download) and
loads them into pandas DataFrames via the ``databento`` client library.

See: https://github.com/databento/databento-python
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import databento as db
import pandas as pd
from arcticdb.version_store.library import Library

from tradingo import symbols

logger = logging.getLogger(__name__)

_DBN_GLOB_PATTERNS = ("*.dbn", "*.dbn.zst")

_OHLCV_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


class ProviderDataError(Exception):
    """Raised if there is an issue with data availability
    or gathering from provider."""


@dataclass(frozen=True)
class DBNFileInfo:
    """Metadata header describing a single DBN file discovered on disk."""

    path: Path
    dataset: str
    schema: str | None
    stype_in: str | None
    stype_out: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    symbols: tuple[str, ...]
    nbytes: int


def _iter_dbn_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found = {p for pattern in _DBN_GLOB_PATTERNS for p in root.rglob(pattern)}
    return sorted(found)


def discover(path: str | Path) -> list[DBNFileInfo]:
    """Discover DBN files under *path* and read their metadata headers.

    *path* may be a single ``.dbn``/``.dbn.zst`` file, or a directory such as
    one produced by ``Historical().batch.download()`` or a Databento portal
    download — any ``manifest.json``/``metadata.json``/``condition.json``
    siblings in the same directory are ignored, only DBN files are inspected.

    Only the DBN metadata header is read (not the record body), so this is
    cheap to run even over a large download.

    :param path: file or directory containing a Databento download
    :return: one :class:`DBNFileInfo` per DBN file found, unsorted
    """
    root = Path(path)
    if not root.exists():
        raise ProviderDataError(f"path does not exist: {root}")

    dbn_paths = _iter_dbn_paths(root)
    if not dbn_paths:
        raise ProviderDataError(f"no .dbn/.dbn.zst files found under {root}")

    infos: list[DBNFileInfo] = []
    for dbn_path in dbn_paths:
        try:
            store = db.DBNStore.from_file(dbn_path)
        except Exception:
            logger.exception("failed to open dbn file %s, skipping", dbn_path)
            continue

        infos.append(
            DBNFileInfo(
                path=dbn_path,
                dataset=store.dataset,
                schema=store.schema.value if store.schema is not None else None,
                stype_in=store.stype_in.value if store.stype_in is not None else None,
                stype_out=store.stype_out.value,
                start=store.start,
                end=store.end,
                symbols=tuple(store.symbols),
                nbytes=store.nbytes,
            )
        )

    logger.info("discovered %d dbn file(s) under %s", len(infos), root)
    return infos


def load(
    path: str | Path,
    *,
    schema: str | None = None,
    symbols: list[str] | None = None,
    map_symbols: bool = True,
) -> pd.DataFrame:
    """Load a Databento historical download at *path* into one DataFrame.

    Discovers every DBN file under *path* (see :func:`discover`), optionally
    filters to files matching *schema* and/or *symbols*, decodes each
    matching file with ``DBNStore.to_df`` and concatenates the results in
    start-time order.

    :param path: file or directory containing a Databento download
    :param schema: only load files whose schema equals this value (e.g.
        ``"ohlcv-1m"``, ``"trades"``, ``"mbp-1"``)
    :param symbols: only load files whose queried symbols intersect this list
    :param map_symbols: resolve instrument_id back to the raw symbol string,
        passed through to ``DBNStore.to_df``
    """
    infos = discover(path)

    if schema is not None:
        infos = [info for info in infos if info.schema == schema]
    if symbols is not None:
        wanted = set(symbols)
        infos = [info for info in infos if wanted & set(info.symbols)]

    if not infos:
        raise ProviderDataError(
            f"no dbn files under {path} match schema={schema!r} symbols={symbols!r}"
        )

    frames: list[pd.DataFrame] = []
    for info in sorted(infos, key=lambda i: i.start):
        store = db.DBNStore.from_file(info.path)
        df = store.to_df(map_symbols=map_symbols)
        df["dataset"] = info.dataset
        frames.append(df)

    return pd.concat(frames).sort_index()


def download_instruments(path: str | Path, currency: str = "USD") -> pd.DataFrame:
    """Build an instruments table from a Databento historical download.

    Unlike the yfinance/IG/IB providers this reads the DBN file metadata
    headers under *path* (dataset, venue, queried raw symbols) rather than
    calling an API — a Databento historical/batch download is a
    self-describing static file, there is nothing further to fetch.

    :param path: file or directory containing a Databento download
    :param currency: currency to record for every symbol found (Databento
        does not report per-instrument currency at this level)
    :return: DataFrame indexed by symbol with dataset/venue/schema columns
    """
    infos = discover(path)

    rows: dict[str, dict[str, object]] = {}
    for info in infos:
        venue = info.dataset.split(".", 1)[0]
        for symbol in info.symbols:
            rows[symbol] = {
                "dataset": info.dataset,
                "venue": venue,
                "schema": info.schema,
                "currency": currency,
                "start_date": info.start,
                "end_date": info.end,
            }

    if not rows:
        raise ProviderDataError(f"no symbols found in databento download at {path}")

    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("symbol")


def ingest_bars(
    path: str | Path,
    schema: str | None = None,
    symbols: list[str] | None = None,
) -> tuple[tuple[pd.DataFrame, tuple[str]], ...]:
    """Load a Databento historical download and split it into one OHLCV
    DataFrame per symbol.

    The whole download is decoded once and grouped by symbol, so this is
    intended to be wired up as a single bulk ingest task, published with
    :func:`tradingo.symbols.symbol_publisher`'s ``template=`` mechanism
    (mirrors ``sampling.ig.get_activity_history``) rather than one task per
    symbol.

    :param path: file or directory containing a Databento download
    :param schema: only ingest files matching this schema (e.g. ``"ohlcv-1m"``)
    :param symbols: only ingest files whose queried symbols intersect this list
    :return: one ``(ohlcv, (symbol,))`` pair per symbol found
    """
    df = load(path, schema=schema, symbols=symbols)

    return tuple(
        (
            group[list(_OHLCV_COLUMNS)].rename(columns=_OHLCV_COLUMNS).sort_index(),
            (str(symbol),),
        )
        for symbol, group in df.groupby("symbol")
    )


@symbols.lib_provider(pricelib="{raw_price_lib}")  # pyright: ignore
def create_universe(
    pricelib: Library,
    instruments: pd.DataFrame,
    end_date: pd.Timestamp | None,
    start_date: pd.Timestamp | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create one arctic symbol for each OHLCV field from Databento bars.
    Each symbol contains all instruments defined for the universe.

    Mirrors sampling.yf.create_universe — reads bars already stored in
    ArcticDB by ingest_bars and assembles them into the standard
    5-DataFrame layout (Open, High, Low, Close, Volume).
    """

    start_date = pd.Timestamp(start_date) if start_date else None
    end_date = pd.Timestamp(end_date) if end_date else None

    def get_data(symbol: str) -> pd.DataFrame:
        df = pd.DataFrame(pricelib.read(symbol, date_range=(start_date, end_date)).data)
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = (
                df.index.tz_localize("UTC")
                if df.index.tz is None
                else df.index.tz_convert("UTC")
            )
        return df

    available_symbols = pricelib.list_symbols()
    if missing_symbols := set(instruments.index.difference(available_symbols)):
        logger.warning(
            "some symbols are missing from the library: %s",
            missing_symbols,
        )

    universe_symbols = [
        symbol for symbol in instruments.index.to_list() if symbol in available_symbols
    ]

    result = pd.concat(
        (get_data(symbol) for symbol in universe_symbols),
        axis=1,
        keys=universe_symbols,
    ).reorder_levels([1, 0], axis=1)
    return (
        cast(pd.DataFrame, result["Open"]),
        cast(pd.DataFrame, result["High"]),
        cast(pd.DataFrame, result["Low"]),
        cast(pd.DataFrame, result["Close"]),
        cast(pd.DataFrame, result["Volume"]),
    )
