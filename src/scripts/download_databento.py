"""Download historical bars for a list of symbols from Databento.

Writes one ``.dbn.zst`` file to *output-dir*, readable by
``tradingo.sampling.databento.discover``/``load``/``ingest_bars``.

Requires the ``DATABENTO_API_KEY`` environment variable (or ``--api-key``).

Usage:

    uv run --group databento python src/scripts/download_databento.py \\
        --dataset GLBX.MDP3 \\
        --schema ohlcv-1d \\
        --start 2020-01-01 \\
        --end 2024-01-01 \\
        --output-dir ./data/databento \\
        ESH1 NQH1 CLH1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import databento as db

logger = logging.getLogger(__name__)


def download(
    symbols: list[str],
    dataset: str,
    schema: str,
    start: str,
    end: str,
    output_dir: Path,
    stype_in: str = "raw_symbol",
    api_key: str | None = None,
) -> Path:
    """Fetch *symbols* from Databento's timeseries API and save as one DBN file.

    :return: path to the written ``.dbn.zst`` file
    """
    client = db.Historical(key=api_key)

    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}.{schema}.{start}_{end}.dbn.zst"
    store.to_file(out_path)

    logger.info("wrote %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+", help="raw symbols/tickers to download")
    parser.add_argument(
        "--dataset", required=True, help="Databento dataset, e.g. GLBX.MDP3"
    )
    parser.add_argument(
        "--schema",
        default="ohlcv-1d",
        help="DBN schema, e.g. ohlcv-1m, ohlcv-1d, trades",
    )
    parser.add_argument("--start", required=True, help="start date, e.g. 2020-01-01")
    parser.add_argument("--end", required=True, help="end date, e.g. 2024-01-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write the .dbn.zst file to",
    )
    parser.add_argument(
        "--stype-in",
        default="raw_symbol",
        help="input symbology type (default: raw_symbol)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Databento API key (defaults to DATABENTO_API_KEY env var)",
    )
    args = parser.parse_args()

    download(
        symbols=args.symbols,
        dataset=args.dataset,
        schema=args.schema,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        stype_in=args.stype_in,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
