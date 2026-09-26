"""Discover the union of tickers across IG, Databento, and Yahoo Finance.

For each candidate ticker, looks up:

- Yahoo Finance (best-match quote via ``yfinance.Search``) — for live pricing
- IG markets (via ``search_markets``) — for execution
- Databento (via ``symbology.resolve`` against one dataset) — for historical bars

There is no cross-provider identifier that ties these together automatically
(IG epics, Databento raw symbols, and Yahoo tickers all use different symbol
schemes), so this produces a *candidate* table for manual review rather than
a confident mapping — IG in particular often returns several markets per
search term (cash vs future, spread bet vs CFD), which is why every IG match
is written to a second file rather than picking one automatically.

Requires IG credentials (``IG_SERVICE_*`` env vars, see
``tradingo.settings.IGTradingConfig``) and ``DATABENTO_API_KEY``.

Usage:

    uv run --group databento python src/scripts/discover_tickers.py \\
        --databento-dataset XNAS.ITCH \\
        --start 2024-01-01 --end 2024-01-02 \\
        --output tickers.csv --ig-candidates-output ig_candidates.csv \\
        SPY QQQ GLD TLT
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import databento as db
import pandas as pd
import yfinance as yf
from trading_ig.rest import IGService

from tradingo.sampling.ig import get_ig_service

logger = logging.getLogger(__name__)


def yfinance_lookup(ticker: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search Yahoo Finance for *ticker*, returning candidate quotes."""
    try:
        quotes = yf.Search(ticker, max_results=max_results).quotes
    except Exception:
        logger.exception("yfinance search failed for %s", ticker)
        return []
    return [
        {
            "yf_symbol": q.get("symbol"),
            "yf_name": q.get("shortname") or q.get("longname"),
            "yf_exchange": q.get("exchange"),
            "yf_quote_type": q.get("quoteType"),
        }
        for q in quotes
    ]


def ig_lookup(
    ticker: str, service: IGService, max_results: int = 5
) -> list[dict[str, Any]]:
    """Search IG markets for *ticker*, returning candidate epics."""
    try:
        markets = service.search_markets(ticker)
    except Exception:
        logger.exception("IG search failed for %s", ticker)
        return []
    if markets is None or markets.empty:
        return []
    return [
        {
            "ig_epic": row.get("epic"),
            "ig_instrument_name": row.get("instrumentName"),
            "ig_instrument_type": row.get("instrumentType"),
            "ig_expiry": row.get("expiry"),
        }
        for row in markets.to_dict("records")[:max_results]
    ]


def databento_lookup(
    ticker: str,
    client: db.Historical,
    dataset: str,
    start_date: str,
    end_date: str,
    stype_in: str = "raw_symbol",
) -> dict[str, Any]:
    """Resolve *ticker* against a Databento dataset, confirming it traded there."""
    try:
        result = client.symbology.resolve(
            dataset=dataset,
            symbols=[ticker],
            stype_in=stype_in,
            stype_out="instrument_id",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        logger.exception("Databento resolve failed for %s", ticker)
        return {"databento_dataset": dataset, "databento_found": False}

    mappings = result.get("result", {}).get(ticker)
    return {
        "databento_dataset": dataset,
        "databento_found": bool(mappings),
        "databento_instrument_id": mappings[0]["s"] if mappings else None,
    }


def discover(
    tickers: list[str],
    databento_dataset: str,
    start_date: str,
    end_date: str,
    max_ig_results: int = 5,
    max_yf_results: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Look up *tickers* across yfinance, IG, and Databento.

    :return: (summary, ig_candidates) - summary has the top yfinance match and
        the databento resolution plus the first IG candidate per ticker;
        ig_candidates has every IG market match found, for manual
        disambiguation.
    """
    service = get_ig_service()
    db_client = db.Historical()

    rows = []
    ig_rows = []
    try:
        for ticker in tickers:
            yf_matches = yfinance_lookup(ticker, max_results=max_yf_results)
            ig_matches = ig_lookup(ticker, service, max_results=max_ig_results)
            db_match = databento_lookup(
                ticker, db_client, databento_dataset, start_date, end_date
            )

            for match in ig_matches:
                ig_rows.append({"ticker": ticker, **match})

            rows.append(
                {
                    "ticker": ticker,
                    **(yf_matches[0] if yf_matches else {}),
                    **db_match,
                    **(ig_matches[0] if ig_matches else {}),
                    "ig_candidate_count": len(ig_matches),
                }
            )
    finally:
        service.session.close()

    return pd.DataFrame(rows), pd.DataFrame(ig_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="candidate tickers to look up")
    parser.add_argument(
        "--databento-dataset",
        required=True,
        help="Databento dataset to resolve against, e.g. XNAS.ITCH, ARCX.PILLAR, GLBX.MDP3",
    )
    parser.add_argument(
        "--start", required=True, help="resolve range start, e.g. 2024-01-01"
    )
    parser.add_argument(
        "--end", required=True, help="resolve range end, e.g. 2024-01-02"
    )
    parser.add_argument(
        "--output", default="tickers.csv", help="summary CSV output path"
    )
    parser.add_argument(
        "--ig-candidates-output",
        default="ig_candidates.csv",
        help="every IG market match found, for manual disambiguation",
    )
    args = parser.parse_args()

    summary, ig_candidates = discover(
        tickers=args.tickers,
        databento_dataset=args.databento_dataset,
        start_date=args.start,
        end_date=args.end,
    )
    summary.to_csv(args.output, index=False)
    ig_candidates.to_csv(args.ig_candidates_output, index=False)
    logger.info("wrote %s and %s", args.output, args.ig_candidates_output)


if __name__ == "__main__":
    main()
