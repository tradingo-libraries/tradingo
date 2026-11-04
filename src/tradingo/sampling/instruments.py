"""static instruments data accessors."""

import logging
import math
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


def snap_price_scale(ratio: float, tolerance: float = 0.06) -> float:
    """Round an IG/source price ratio to a clean factor.

    Same-instrument mappings differ only by IG's points convention, so the
    ratio is a power of ten up to basis/timing noise (e.g. 99.97 -> 100).
    Proxies (a UCITS share class standing in for a US ETF) have arbitrary
    ratios and are kept at 4 significant figures.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"Invalid price ratio: {ratio}")
    power = 10.0 ** round(math.log10(ratio))
    if abs(ratio / power - 1) <= tolerance:
        return power
    return float(f"{ratio:.4g}")


def measure_ig_price_scales(
    source_prices: pd.Series,
    epics: dict[str, str],
    account_id: str | None = None,
    service: Any = None,
    tolerance: float = 0.06,
) -> pd.DataFrame:
    """Measure ``price_scale`` (IG price / source price) for each mapping.

    Compares IG's current mid to the latest source price, so run it while
    both markets are open. Prices are only quoted for the account type the
    session is on: pass the spread-bet ``account_id`` when the preferred
    account is CFD.

    Example (in-cluster, e.g. via the MCP ``run_python`` tool)::

        mid = api.prices_5min.dukascopy.mid.close().ffill().iloc[-1]
        measure_ig_price_scales(mid, {"USD/JPY": "CS.D.USDJPY.TODAY.IP"},
                                account_id="<spread-bet account>")

    :param source_prices: latest source price, indexed by source symbol.
    :param epics: source symbol -> IG epic.
    :return: frame indexed by source symbol with epic, IG mid, source price,
        measured ratio, snapped ``price_scale`` and market status.
    """
    service = service or get_ig_service()
    if account_id:
        service.switch_account(account_id, False)

    markets: dict[str, dict[str, Any]] = {}
    unique = list(dict.fromkeys(epics.values()))
    for i in range(0, len(unique), 50):
        for market in service.fetch_markets_by_epics(
            ",".join(unique[i : i + 50]), detailed=True
        ):
            markets[market["instrument"]["epic"]] = market

    rows = []
    for source, epic in epics.items():
        market = markets.get(epic)
        if market is None:
            logger.warning("No IG market for %s (%s)", source, epic)
            continue
        snapshot = market["snapshot"]
        bid, offer = snapshot.get("bid"), snapshot.get("offer")
        ig_mid = (bid + offer) / 2 if bid and offer else math.nan
        src = float(source_prices.get(source, math.nan))
        ratio = ig_mid / src if src else math.nan
        rows.append(
            {
                "source": source,
                "epic": epic,
                "name": market["instrument"].get("name"),
                "status": snapshot.get("marketStatus"),
                "ig_mid": ig_mid,
                "source_price": src,
                "ratio": ratio,
                "price_scale": (
                    snap_price_scale(ratio, tolerance)
                    if math.isfinite(ratio) and ratio > 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("source")
