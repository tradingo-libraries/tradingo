"""Tests for tradingo.sampling.databento

``databento`` is an optional dependency (``uv sync --group databento``);
these tests decode a real DBN fixture rather than mocking the binary
parser, so the whole module is skipped when the group isn't installed.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("databento")

from tradingo.sampling.databento import (  # noqa: E402
    ProviderDataError,
    create_universe,
    discover,
    download_instruments,
    ingest_bars,
    load,
)

FIXTURE_DIR = Path(__file__).parent / "sampling" / "data"
FIXTURE_FILE = FIXTURE_DIR / "test_data.ohlcv-1d.dbn.zst"


class TestDiscover:
    def test_discovers_single_file(self) -> None:
        infos = discover(FIXTURE_FILE)
        assert len(infos) == 1
        info = infos[0]
        assert info.dataset == "GLBX.MDP3"
        assert info.schema == "ohlcv-1d"
        assert info.symbols == ("ESH1",)

    def test_discovers_directory_ignoring_sibling_files(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "data.ohlcv-1d.dbn.zst").write_bytes(
            FIXTURE_FILE.read_bytes()
        )
        infos = discover(tmp_path)
        assert len(infos) == 1

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderDataError, match="does not exist"):
            discover(tmp_path / "nope")

    def test_no_dbn_files_raises(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")
        with pytest.raises(ProviderDataError, match="no .dbn"):
            discover(tmp_path)


class TestLoad:
    def test_loads_dataframe(self) -> None:
        df = load(FIXTURE_FILE)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "symbol" in df.columns
        assert set(df["symbol"]) == {"ESH1"}

    def test_schema_filter_no_match_raises(self) -> None:
        with pytest.raises(ProviderDataError, match="no dbn files"):
            load(FIXTURE_FILE, schema="trades")

    def test_symbols_filter_no_match_raises(self) -> None:
        with pytest.raises(ProviderDataError, match="no dbn files"):
            load(FIXTURE_FILE, symbols=["NOTASYMBOL"])


class TestDownloadInstruments:
    def test_returns_dataframe_indexed_by_symbol(self) -> None:
        result = download_instruments(FIXTURE_FILE)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "symbol"
        assert "ESH1" in result.index
        assert result.loc["ESH1", "dataset"] == "GLBX.MDP3"
        assert result.loc["ESH1", "venue"] == "GLBX"
        assert result.loc["ESH1", "currency"] == "USD"

    def test_custom_currency(self) -> None:
        result = download_instruments(FIXTURE_FILE, currency="EUR")
        assert result.loc["ESH1", "currency"] == "EUR"


class TestIngestBars:
    def test_returns_one_pair_per_symbol(self) -> None:
        result = ingest_bars(FIXTURE_FILE)
        assert len(result) == 1
        df, (symbol,) = result[0]
        assert symbol == "ESH1"
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert not df.empty

    def test_symbols_filter(self) -> None:
        result = ingest_bars(FIXTURE_FILE, symbols=["ESH1"])
        assert len(result) == 1
        assert result[0][1] == ("ESH1",)


class TestCreateUniverse:
    """Tests for create_universe function."""

    @staticmethod
    def _make_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [1.0] * len(index),
                "High": [1.0] * len(index),
                "Low": [1.0] * len(index),
                "Close": [1.0] * len(index),
                "Volume": [1.0] * len(index),
            },
            index=index,
        )

    def _make_pricelib(self, data_by_symbol: dict[str, pd.DataFrame]) -> MagicMock:
        pricelib = MagicMock()
        pricelib.list_symbols.return_value = list(data_by_symbol.keys())

        def read(symbol: str, date_range: Any = None) -> MagicMock:
            item = MagicMock()
            item.data = data_by_symbol[symbol]
            return item

        pricelib.read.side_effect = read
        return pricelib

    def test_assembles_wide_ohlcv_tables(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        pricelib = self._make_pricelib(
            {"AAA": self._make_ohlcv(idx), "BBB": self._make_ohlcv(idx)}
        )
        instruments = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="symbol"))

        open_, high, low, close, volume = getattr(create_universe, "__wrapped__")(
            pricelib=pricelib,
            instruments=instruments,
            end_date=None,
            start_date=None,
        )

        assert list(open_.columns) == ["AAA", "BBB"]
        assert list(volume.columns) == ["AAA", "BBB"]
        assert len(open_) == 3

    def test_missing_symbol_logged_and_skipped(self, caplog: Any) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        pricelib = self._make_pricelib({"AAA": self._make_ohlcv(idx)})
        instruments = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="symbol"))

        open_, *_ = getattr(create_universe, "__wrapped__")(
            pricelib=pricelib,
            instruments=instruments,
            end_date=None,
            start_date=None,
        )

        assert list(open_.columns) == ["AAA"]
