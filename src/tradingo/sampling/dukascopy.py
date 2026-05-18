"""Dukascopy data accessors."""

import logging

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


_INSTRUMENTS: dict[str, dict[str, str]] = {
    # FX Majors
    "AUD/USD": {
        "description": "Australian Dollar / US Dollar",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "USD",
    },
    "EUR/USD": {
        "description": "Euro / US Dollar",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "USD",
    },
    "GBP/USD": {
        "description": "British Pound / US Dollar",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "USD",
    },
    "NZD/USD": {
        "description": "New Zealand Dollar / US Dollar",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "USD",
    },
    "USD/CAD": {
        "description": "US Dollar / Canadian Dollar",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "CAD",
    },
    "USD/CHF": {
        "description": "US Dollar / Swiss Franc",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "CHF",
    },
    "USD/JPY": {
        "description": "US Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Majors",
        "currency": "JPY",
    },
    # FX Metals
    "XAG/USD": {
        "description": "Silver / US Dollar",
        "category": "FX",
        "subcategory": "Metals",
        "currency": "USD",
    },
    "XAU/USD": {
        "description": "Gold / US Dollar",
        "category": "FX",
        "subcategory": "Metals",
        "currency": "USD",
    },
    # FX Crosses
    "AUD/CAD": {
        "description": "Australian Dollar / Canadian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CAD",
    },
    "AUD/CHF": {
        "description": "Australian Dollar / Swiss Franc",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CHF",
    },
    "AUD/JPY": {
        "description": "Australian Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "AUD/NZD": {
        "description": "Australian Dollar / New Zealand Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NZD",
    },
    "AUD/SGD": {
        "description": "Australian Dollar / Singapore Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SGD",
    },
    "CAD/CHF": {
        "description": "Canadian Dollar / Swiss Franc",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CHF",
    },
    "CAD/HKD": {
        "description": "Canadian Dollar / Hong Kong Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "HKD",
    },
    "CAD/JPY": {
        "description": "Canadian Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "CHF/JPY": {
        "description": "Swiss Franc / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "CHF/PLN": {
        "description": "Swiss Franc / Polish Zloty",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "PLN",
    },
    "CHF/SGD": {
        "description": "Swiss Franc / Singapore Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SGD",
    },
    "EUR/AUD": {
        "description": "Euro / Australian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "AUD",
    },
    "EUR/CAD": {
        "description": "Euro / Canadian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CAD",
    },
    "EUR/CHF": {
        "description": "Euro / Swiss Franc",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CHF",
    },
    "EUR/CZK": {
        "description": "Euro / Czech Koruna",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CZK",
    },
    "EUR/DKK": {
        "description": "Euro / Danish Krone",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "DKK",
    },
    "EUR/GBP": {
        "description": "Euro / British Pound",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "GBP",
    },
    "EUR/HKD": {
        "description": "Euro / Hong Kong Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "HKD",
    },
    "EUR/HUF": {
        "description": "Euro / Hungarian Forint",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "HUF",
    },
    "EUR/JPY": {
        "description": "Euro / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "EUR/MXN": {
        "description": "Euro / Mexican Peso",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "MXN",
    },
    "EUR/NOK": {
        "description": "Euro / Norwegian Krone",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NOK",
    },
    "EUR/NZD": {
        "description": "Euro / New Zealand Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NZD",
    },
    "EUR/PLN": {
        "description": "Euro / Polish Zloty",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "PLN",
    },
    "EUR/RUB": {
        "description": "Euro / Russian Ruble",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "RUB",
    },
    "EUR/SEK": {
        "description": "Euro / Swedish Krona",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SEK",
    },
    "EUR/SGD": {
        "description": "Euro / Singapore Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SGD",
    },
    "EUR/TRY": {
        "description": "Euro / Turkish Lira",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "TRY",
    },
    "EUR/ZAR": {
        "description": "Euro / South African Rand",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "ZAR",
    },
    "GBP/AUD": {
        "description": "British Pound / Australian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "AUD",
    },
    "GBP/CAD": {
        "description": "British Pound / Canadian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CAD",
    },
    "GBP/CHF": {
        "description": "British Pound / Swiss Franc",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CHF",
    },
    "GBP/JPY": {
        "description": "British Pound / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "GBP/NZD": {
        "description": "British Pound / New Zealand Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NZD",
    },
    "HKD/JPY": {
        "description": "Hong Kong Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "MXN/JPY": {
        "description": "Mexican Peso / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "NZD/CAD": {
        "description": "New Zealand Dollar / Canadian Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CAD",
    },
    "NZD/CHF": {
        "description": "New Zealand Dollar / Swiss Franc",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CHF",
    },
    "NZD/JPY": {
        "description": "New Zealand Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "NZD/SGD": {
        "description": "New Zealand Dollar / Singapore Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SGD",
    },
    "SGD/JPY": {
        "description": "Singapore Dollar / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "TRY/JPY": {
        "description": "Turkish Lira / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    "USD/BRL": {
        "description": "US Dollar / Brazilian Real",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "BRL",
    },
    "USD/CNH": {
        "description": "US Dollar / Chinese Yuan (Offshore)",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CNH",
    },
    "USD/CZK": {
        "description": "US Dollar / Czech Koruna",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "CZK",
    },
    "USD/DKK": {
        "description": "US Dollar / Danish Krone",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "DKK",
    },
    "USD/HKD": {
        "description": "US Dollar / Hong Kong Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "HKD",
    },
    "USD/HUF": {
        "description": "US Dollar / Hungarian Forint",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "HUF",
    },
    "USD/ILS": {
        "description": "US Dollar / Israeli Shekel",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "ILS",
    },
    "USD/MXN": {
        "description": "US Dollar / Mexican Peso",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "MXN",
    },
    "USD/NOK": {
        "description": "US Dollar / Norwegian Krone",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NOK",
    },
    "USD/NZD": {
        "description": "US Dollar / New Zealand Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "NZD",
    },
    "USD/PLN": {
        "description": "US Dollar / Polish Zloty",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "PLN",
    },
    "USD/RON": {
        "description": "US Dollar / Romanian Leu",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "RON",
    },
    "USD/RUB": {
        "description": "US Dollar / Russian Ruble",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "RUB",
    },
    "USD/SEK": {
        "description": "US Dollar / Swedish Krona",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SEK",
    },
    "USD/SGD": {
        "description": "US Dollar / Singapore Dollar",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "SGD",
    },
    "USD/THB": {
        "description": "US Dollar / Thai Baht",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "THB",
    },
    "USD/TRY": {
        "description": "US Dollar / Turkish Lira",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "TRY",
    },
    "USD/ZAR": {
        "description": "US Dollar / South African Rand",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "ZAR",
    },
    "ZAR/JPY": {
        "description": "South African Rand / Japanese Yen",
        "category": "FX",
        "subcategory": "Crosses",
        "currency": "JPY",
    },
    # Fixed Income
    "BUND.TR/EUR": {
        "description": "German Bund 10Y Futures",
        "category": "Rates",
        "subcategory": "Government Bonds",
        "currency": "EUR",
    },
    "UKGILT.TR/GBP": {
        "description": "UK Gilt 10Y Futures",
        "category": "Rates",
        "subcategory": "Government Bonds",
        "currency": "GBP",
    },
    "USTBOND.TR/USD": {
        "description": "US Treasury Bond 10Y Futures",
        "category": "Rates",
        "subcategory": "Government Bonds",
        "currency": "USD",
    },
    # Commodities - Agricultural
    "COCOA.CMD/USD": {
        "description": "Cocoa Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USD",
    },
    "COFFEE.CMD/USX": {
        "description": "Coffee Arabica Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USX",
    },
    "COTTON.CMD/USX": {
        "description": "Cotton No.2 Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USX",
    },
    "OJUICE.CMD/USX": {
        "description": "Orange Juice Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USX",
    },
    "SOYBEAN.CMD/USX": {
        "description": "Soybean Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USX",
    },
    "SUGAR.CMD/USD": {
        "description": "Sugar No.11 Futures",
        "category": "Commodities",
        "subcategory": "Agricultural",
        "currency": "USD",
    },
    # Commodities - Energy
    "DIESEL.CMD/USD": {
        "description": "Heating Oil / Diesel Futures",
        "category": "Commodities",
        "subcategory": "Energy",
        "currency": "USD",
    },
    "E_Brent": {
        "description": "Brent Crude Oil Futures",
        "category": "Commodities",
        "subcategory": "Energy",
        "currency": "USD",
    },
    "E_Light": {
        "description": "WTI Light Crude Oil Futures",
        "category": "Commodities",
        "subcategory": "Energy",
        "currency": "USD",
    },
    "GAS.CMD/USD": {
        "description": "Natural Gas Futures",
        "category": "Commodities",
        "subcategory": "Energy",
        "currency": "USD",
    },
    # Commodities - Metals
    "COPPER.CMD/USD": {
        "description": "Copper Futures",
        "category": "Commodities",
        "subcategory": "Metals",
        "currency": "USD",
    },
    "XPD.CMD/USD": {
        "description": "Palladium Spot",
        "category": "Commodities",
        "subcategory": "Metals",
        "currency": "USD",
    },
    "XPT.CMD/USD": {
        "description": "Platinum Spot",
        "category": "Commodities",
        "subcategory": "Metals",
        "currency": "USD",
    },
    # Equity Indices - Americas
    "DOLLAR.IDX/USD": {
        "description": "US Dollar Index",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "E_D&J-Ind": {
        "description": "Dow Jones Industrial Average",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "E_NQ-100": {
        "description": "NASDAQ 100",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "E_SandP-500": {
        "description": "S&P 500",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "RUSSELL.IDX/USD": {
        "description": "Russell 2000",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "USSC2000.IDX/USD": {
        "description": "US Small Cap 2000 Index",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    "VOL.IDX/USD": {
        "description": "CBOE Volatility Index (VIX)",
        "category": "Indices",
        "subcategory": "Americas",
        "currency": "USD",
    },
    # Equity Indices - Asia
    "CHI.IDX/USD": {
        "description": "China A50 Index",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "USD",
    },
    "E_H-Kong": {
        "description": "Hang Seng Index",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "HKD",
    },
    "E_N225Jap": {
        "description": "Nikkei 225",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "JPY",
    },
    "E_XJO-ASX": {
        "description": "ASX 200",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "AUD",
    },
    "IND.IDX/USD": {
        "description": "Nifty 50 Index",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "USD",
    },
    "SGD.IDX/SGD": {
        "description": "Straits Times Index",
        "category": "Indices",
        "subcategory": "Asia",
        "currency": "SGD",
    },
    # Equity Indices - Europe
    "E_CAAC-40": {
        "description": "CAC 40",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "E_DAAX": {
        "description": "DAX 40",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "E_DJE50XX": {
        "description": "Euro Stoxx 50",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "E_Futsee-100": {
        "description": "FTSE 100",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "GBP",
    },
    "E_IBC-MAC": {
        "description": "IBEX 35",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "E_SWMI": {
        "description": "Swiss Market Index (SMI)",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "CHF",
    },
    "ITA.IDX/EUR": {
        "description": "FTSE MIB (Italy)",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "NLD.IDX/EUR": {
        "description": "AEX (Netherlands)",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    "PLN.IDX/PLN": {
        "description": "WIG20 (Poland)",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "PLN",
    },
    "PRT.IDX/EUR": {
        "description": "PSI 20 (Portugal)",
        "category": "Indices",
        "subcategory": "Europe",
        "currency": "EUR",
    },
    # Equity Indices - Africa
    "SOA.IDX/ZAR": {
        "description": "JSE Top 40 (South Africa)",
        "category": "Indices",
        "subcategory": "Africa",
        "currency": "ZAR",
    },
}


def get_instruments(epics: list[str]) -> pd.DataFrame:
    """Return static instrument metadata for the given Dukascopy epics."""
    records = []
    for epic in epics:
        meta = _INSTRUMENTS.get(epic)
        if meta is None:
            logger.warning("No static metadata for dukascopy instrument: %s", epic)
            records.append(
                {
                    "name": epic,
                    "description": epic,
                    "category": "",
                    "subcategory": "",
                    "currency": "",
                }
            )
        else:
            records.append({"name": epic, **meta})
    return pd.DataFrame(records).set_index("name").rename_axis("Symbol")


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

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # Ensure naive datetimes (dukascopy_python expects naive UTC)
    if not start_dt.tzinfo:
        raise ValueError("Timestamp should not be naiive")
    if not end_dt.tzinfo:
        raise ValueError("Timestamp should not be naiive")

    start_dt = start_dt.tz_convert("utc")
    end_dt = end_dt.tz_convert("utc")

    logger.info("Fetching %s [%s] %s -> %s", epic, interval, start_dt, end_dt)

    start_pydatetime = start_dt.to_pydatetime()
    end_pydatetime = end_dt.to_pydatetime()

    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"], dtype=float)
    empty.index = pd.DatetimeIndex([], tz="utc")

    try:
        bid = dukascopy_python.fetch(
            instrument,
            dk_interval,
            dukascopy_python.OFFER_SIDE_BID,
            start_pydatetime,
            end_pydatetime,
        )
        ask = dukascopy_python.fetch(
            instrument,
            dk_interval,
            dukascopy_python.OFFER_SIDE_ASK,
            start_pydatetime,
            end_pydatetime,
        )
    except (TypeError, KeyError) as e:
        # dukascopy_python raises TypeError (row[0] is a string) or KeyError: 0
        # (lastUpdates[0] on an empty-dict API response) when no data is available
        logger.warning(
            "No data for %s [%s] %s -> %s: %s",
            epic,
            interval,
            start_dt,
            end_dt,
            e,
        )
        return empty, empty

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
        # Ensure UTC timezone on index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("utc")
        else:
            df.index = df.index.tz_convert("utc")
        return df

    return (_normalize(bid), _normalize(ask))
