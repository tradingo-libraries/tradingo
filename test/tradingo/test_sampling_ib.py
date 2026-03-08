"""Tests for tradingo.sampling.ib

ib_insync is optional, so it is stubbed in sys.modules before import.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

if "ib_insync" not in sys.modules:
    sys.modules["ib_insync"] = MagicMock()

from tradingo.sampling.ib import (  # noqa: E402
    _duration_str,
    download_instruments,
)

# ---------------------------------------------------------------------------
# _duration_str
# ---------------------------------------------------------------------------


class TestDurationStr:
    def test_single_day(self) -> None:
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        assert _duration_str(start, end) == "1 D"

    def test_thirty_days(self) -> None:
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-31", tz="UTC")
        assert _duration_str(start, end) == "30 D"

    def test_exactly_365_days_uses_days(self) -> None:
        # 2023 is not a leap year: 2023-01-01 → 2024-01-01 = 365 days
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-01", tz="UTC")
        assert _duration_str(start, end) == "365 D"

    def test_over_one_year_uses_years(self) -> None:
        # 2022-01-01 → 2024-01-01 spans 2 years (730/731 days)
        start = pd.Timestamp("2022-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-01", tz="UTC")
        assert _duration_str(start, end) == "2 Y"

    def test_same_date_clamps_to_one_day(self) -> None:
        ts = pd.Timestamp("2024-06-01", tz="UTC")
        assert _duration_str(ts, ts) == "1 D"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contract_details(
    symbol: str,
    currency: str = "USD",
    exchange: str = "NASDAQ",
    sec_type: str = "STK",
    long_name: str = "Test Corp",
    min_tick: float = 0.01,
) -> MagicMock:
    det = MagicMock()
    det.contract.symbol = symbol
    det.contract.currency = currency
    det.contract.primaryExch = exchange
    det.contract.exchange = "SMART"
    det.contract.secType = sec_type
    det.longName = long_name
    det.minTick = min_tick
    return det


# ---------------------------------------------------------------------------
# download_instruments
# ---------------------------------------------------------------------------


class TestDownloadInstruments:
    @pytest.fixture
    def mock_ib(self) -> MagicMock:
        ib = MagicMock()
        return ib

    def test_returns_dataframe_indexed_by_symbol(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetails.return_value = [
            _make_contract_details("AAPL", currency="USD", exchange="NASDAQ")
        ]
        result = download_instruments([{"symbol": "AAPL"}], service=mock_ib)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "symbol"
        assert "AAPL" in result.index

    def test_currency_and_exchange_from_contract_details(
        self, mock_ib: MagicMock
    ) -> None:
        mock_ib.reqContractDetails.return_value = [
            _make_contract_details("VOD", currency="GBP", exchange="LSE")
        ]
        result = download_instruments(
            [{"symbol": "VOD", "currency": "GBP", "exchange": "LSE"}],
            service=mock_ib,
        )
        assert result.loc["VOD", "currency"] == "GBP"
        assert result.loc["VOD", "exchange"] == "LSE"

    def test_long_name_and_min_tick_populated(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetails.return_value = [
            _make_contract_details("MSFT", long_name="Microsoft Corp", min_tick=0.01)
        ]
        result = download_instruments([{"symbol": "MSFT"}], service=mock_ib)
        assert result.loc["MSFT", "long_name"] == "Microsoft Corp"
        assert result.loc["MSFT", "min_tick"] == 0.01

    def test_multiple_symbols(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetails.side_effect = [
            [_make_contract_details("AAPL")],
            [_make_contract_details("MSFT")],
        ]
        result = download_instruments(
            [{"symbol": "AAPL"}, {"symbol": "MSFT"}], service=mock_ib
        )
        assert list(result.index) == ["AAPL", "MSFT"]

    def test_falls_back_to_config_when_no_details(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetails.return_value = []
        result = download_instruments(
            [{"symbol": "UNKNOWN", "currency": "EUR", "exchange": "XETRA"}],
            service=mock_ib,
        )
        assert "UNKNOWN" in result.index
        assert result.loc["UNKNOWN", "currency"] == "EUR"
        assert result.loc["UNKNOWN", "exchange"] == "XETRA"
        assert result.loc["UNKNOWN", "long_name"] is None

    def test_disconnects_after_download(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetails.return_value = [_make_contract_details("AAPL")]
        download_instruments([{"symbol": "AAPL"}], service=mock_ib)
        mock_ib.disconnect.assert_called_once()

    def test_uses_primary_exchange_over_exchange(self, mock_ib: MagicMock) -> None:
        det = _make_contract_details("AAPL", exchange="NASDAQ")
        det.contract.primaryExch = "NASDAQ"
        det.contract.exchange = "SMART"
        mock_ib.reqContractDetails.return_value = [det]
        result = download_instruments([{"symbol": "AAPL"}], service=mock_ib)
        assert result.loc["AAPL", "exchange"] == "NASDAQ"
