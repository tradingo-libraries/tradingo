"""static instruments data accessors."""

import logging
from typing import Any

import pandas as pd
from yfinance import Ticker

from tradingo.sampling.ig import get_ig_service
from tradingo.sampling.yf import currency_to_symbol

logger = logging.getLogger(__name__)


def _flatten_dict(
    obj: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    items: list[tuple[str, object]] = []
    for key, value in obj.items():
        new_key = sep.join((parent_key, key)) if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def download_instruments(
    *,
    index_col: str | None = None,
    html: str | None = None,
    file: str | None = None,
    tickers: list[str] | str | None = None,
    epics: list[str] | None = None,
) -> pd.DataFrame:
    if file:
        return pd.read_csv(
            file,
            index_col=index_col,
        ).rename_axis("Symbol")

    if html:
        return pd.read_html(html)[0].set_index(index_col).rename_axis("Symbol")

    if tickers is not None:
        if isinstance(tickers, str) and tickers.startswith("http"):
            tickers = pd.read_html(tickers)[0][index_col].to_list()
        if isinstance(tickers, str) and tickers.startswith("file"):
            tickers = pd.read_csv(tickers.split("://")[-1])[index_col].to_list()
        return (
            pd.DataFrame({t: Ticker(currency_to_symbol(t)).get_info() for t in tickers})
            .transpose()
            .rename_axis("Symbol")
        )

    if epics:
        service = get_ig_service()

        return (
            pd.DataFrame(
                (_flatten_dict(service.fetch_market_by_epic(e)) for e in epics)
            )
            .set_index("instrument.epic")
            .rename_axis("Symbol", axis=0)
        )

    raise ValueError(file)
