"""Dukascopy data accessors."""

import logging
from datetime import datetime

import dukascopy_python
import pandas as pd
from dukascopy_python import instruments as dukascopy_instruments

logger = logging.getLogger(__name__)

INTERVAL_MAP: dict[str, str] = {
    "1SEC": dukascopy_python.INTERVAL_SEC_1,
    "10SEC": dukascopy_python.INTERVAL_SEC_10,
    "30SEC": dukascopy_python.INTERVAL_SEC_30,
    "1MIN": dukascopy_python.INTERVAL_MIN_1,
    "5MIN": dukascopy_python.INTERVAL_MIN_5,
    "10MIN": dukascopy_python.INTERVAL_MIN_10,
    "15MIN": dukascopy_python.INTERVAL_MIN_15,
    "30MIN": dukascopy_python.INTERVAL_MIN_30,
    "1H": dukascopy_python.INTERVAL_HOUR_1,
    "4H": dukascopy_python.INTERVAL_HOUR_4,
    "1D": dukascopy_python.INTERVAL_DAY_1,
    "1W": dukascopy_python.INTERVAL_WEEK_1,
    "1M": dukascopy_python.INTERVAL_MONTH_1,
}

# Reverse map: instrument value -> constant name
_INSTRUMENT_REVERSE_MAP: dict[str, str] = {
    getattr(dukascopy_instruments, name): name
    for name in dir(dukascopy_instruments)
    if name.startswith("INSTRUMENT_")
}


def _resolve_instrument(epic: str) -> str:
    """Resolve an epic string to a dukascopy instrument constant value.

    Accepts either a raw instrument string (e.g. 'AUD/CAD') which is
    already the value, or a constant name (e.g. 'INSTRUMENT_FX_MAJORS_AUD_CAD').
    """
    if hasattr(dukascopy_instruments, epic):
        return str(getattr(dukascopy_instruments, epic))
    if epic in _INSTRUMENT_REVERSE_MAP:
        return epic
    raise ValueError(
        f"Unknown dukascopy instrument: {epic!r}. "
        "Must be a valid instrument value or constant name."
    )


def sample_instrument(
    epic: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample bid and ask OHLCV data from Dukascopy.

    Args:
        epic: Dukascopy instrument string (e.g. 'AUD/CAD', 'E_Brent').
        start_date: Start of sampling window.
        end_date: End of sampling window.
        interval: Interval key (e.g. '15MIN', '1H', '1D').

    Returns:
        Tuple of (bid, ask) DataFrames with columns [Open, High, Low, Close].
    """
    instrument = _resolve_instrument(epic)
    dk_interval = INTERVAL_MAP.get(interval)
    if dk_interval is None:
        raise ValueError(
            f"Unknown interval {interval!r}. "
            f"Valid intervals: {sorted(INTERVAL_MAP)}"
        )

    start_dt = pd.Timestamp(start_date).to_pydatetime()
    end_dt = pd.Timestamp(end_date).to_pydatetime()

    # Ensure naive datetimes (dukascopy_python expects naive UTC)
    if start_dt.tzinfo is not None:
        start_dt = start_dt.astimezone(datetime.now().astimezone().tzinfo).replace(
            tzinfo=None
        )
    if end_dt.tzinfo is not None:
        end_dt = end_dt.astimezone(datetime.now().astimezone().tzinfo).replace(
            tzinfo=None
        )

    logger.info("Fetching %s [%s] %s -> %s", epic, interval, start_dt, end_dt)

    bid = dukascopy_python.fetch(
        instrument,
        dk_interval,
        dukascopy_python.OFFER_SIDE_BID,
        start_dt,
        end_dt,
    )

    ask = dukascopy_python.fetch(
        instrument,
        dk_interval,
        dukascopy_python.OFFER_SIDE_ASK,
        start_dt,
        end_dt,
    )

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to title case to match IG convention."""
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )
        # Drop volume column if present — not used in downstream pipeline
        if "volume" in df.columns:
            df = df.drop(columns=["volume"])
        # Ensure UTC timezone on index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("utc")
        else:
            df.index = df.index.tz_convert("utc")
        return df

    return (_normalize(bid), _normalize(ask))
