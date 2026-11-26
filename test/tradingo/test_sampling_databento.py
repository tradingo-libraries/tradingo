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

db = pytest.importorskip("databento")

from tradingo.sampling import databento  # noqa: E402
from tradingo.sampling.databento import (  # noqa: E402
    ProviderDataError,
    create_quote_universe,
    create_universe,
    discover,
    download_bars,
    download_instruments,
    get_cost,
    ingest_bars,
    ingest_quotes,
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
        (batch,) = list(ingest_bars(FIXTURE_FILE))
        assert len(batch) == 1
        df, (symbol,) = batch[0]
        assert symbol == "ESH1"
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert not df.empty

    def test_symbols_filter(self) -> None:
        (batch,) = list(ingest_bars(FIXTURE_FILE, symbols=["ESH1"]))
        assert len(batch) == 1
        assert batch[0][1] == ("ESH1",)

    def test_chunks_cover_whole_download(self) -> None:
        whole = load(FIXTURE_FILE)
        batches = list(ingest_bars(FIXTURE_FILE, chunk_size=2))
        assert len(batches) == -(-len(whole) // 2)
        chunked = pd.concat(df for batch in batches for df, _ in batch)
        expected = whole[["open", "high", "low", "close", "volume"]]
        assert chunked.to_numpy().tolist() == expected.to_numpy().tolist()
        assert chunked.index.equals(expected.index)


class TestIngestQuotes:
    def test_rejects_non_quote_schema(self) -> None:
        with pytest.raises(ProviderDataError, match="not a quote schema"):
            ingest_quotes(FIXTURE_FILE, schema="ohlcv-1m")

    def test_keeps_best_bid_ask_per_symbol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        idx = pd.date_range("2024-01-01 14:30", periods=2, freq="min", tz="UTC")
        decoded = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "price": [10.0, 20.0],
                "bid_px_00": [9.9, 19.9],
                "ask_px_00": [10.1, 20.1],
                "bid_sz_00": [100, 200],
                "ask_sz_00": [110, 210],
            },
            index=idx,
        )
        info = MagicMock(path=Path("x.dbn.zst"), nbytes=1)
        monkeypatch.setattr(databento, "_select", lambda *a, **k: [info])
        store = MagicMock()
        store.to_df.return_value = iter([decoded])
        monkeypatch.setattr(db.DBNStore, "from_file", lambda p: store)

        (batch,) = list(ingest_quotes("x", schema="bbo-1m"))

        assert [symbol for _, symbol in batch] == [("AAA",), ("BBB",)]
        df, _ = batch[0]
        assert list(df.columns) == ["Bid", "Ask", "BidSize", "AskSize"]
        assert df.iloc[0].tolist() == [9.9, 10.1, 100, 110]


class TestCreateQuoteUniverse:
    def test_assembles_wide_quote_tables(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        quotes = pd.DataFrame(
            {"Bid": 1.0, "Ask": 2.0, "BidSize": 3.0, "AskSize": 4.0}, index=idx
        )
        pricelib = MagicMock()
        pricelib.list_symbols.return_value = ["AAA", "BBB"]
        pricelib.read.side_effect = lambda symbol, date_range=None: MagicMock(
            data=quotes
        )
        instruments = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="symbol"))

        bid, ask, bid_size, ask_size = getattr(create_quote_universe, "__wrapped__")(
            pricelib=pricelib,
            instruments=instruments,
            end_date=None,
            start_date=None,
        )

        assert list(bid.columns) == ["AAA", "BBB"]
        assert (bid == 1.0).all().all()
        assert (ask == 2.0).all().all()
        assert (bid_size == 3.0).all().all()
        assert (ask_size == 4.0).all().all()


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

    def test_batch_freq_yields_disjoint_windows(self) -> None:
        idx = pd.date_range("2024-01-01", "2024-12-31", freq="D", tz="UTC")
        data = {"AAA": self._make_ohlcv(idx), "BBB": self._make_ohlcv(idx[:100])}
        pricelib = MagicMock()
        pricelib.list_symbols.return_value = list(data)

        def read(symbol: str, date_range: Any = None) -> MagicMock:
            item = MagicMock()
            lo, hi = date_range
            item.data = data[symbol].loc[lo:hi]
            return item

        pricelib.read.side_effect = read
        instruments = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="symbol"))

        windows = list(
            getattr(create_universe, "__wrapped__")(
                pricelib=pricelib,
                instruments=instruments,
                start_date=pd.Timestamp("2024-01-01", tz="UTC"),
                end_date=pd.Timestamp("2024-12-31", tz="UTC"),
                batch_freq="QS",
            )
        )

        assert len(windows) == 4
        closes = [close for _, _, _, close, _ in windows]
        assert all(list(close.columns) == ["AAA", "BBB"] for close in closes)
        combined = pd.concat(closes)
        assert combined.index.equals(idx)
        assert combined["BBB"].notna().sum() == 100

    def test_batch_freq_needs_dates(self) -> None:
        pricelib = self._make_pricelib({})
        with pytest.raises(ValueError, match="batch_freq"):
            getattr(create_universe, "__wrapped__")(
                pricelib=pricelib,
                instruments=pd.DataFrame(index=pd.Index([], name="symbol")),
                start_date=None,
                end_date=None,
                batch_freq="QS",
            )


REQUEST: dict[str, Any] = {
    "dataset": "XNAS.ITCH",
    "symbols": ["SPY", "QQQ"],
    "schema": "ohlcv-1m",
    "start": "2024-01-01",
    "end": "2024-02-01",
}


class TestGetCost:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        client = MagicMock()
        client.metadata.get_cost.return_value = 12.5
        client.metadata.get_billable_size.return_value = 1_000
        client.metadata.get_record_count.return_value = 40
        monkeypatch.setattr("tradingo.sampling.databento._historical", lambda _: client)
        return client

    def test_returns_quote(self, client: MagicMock) -> None:
        out = get_cost(**REQUEST)
        assert out.iloc[0].to_dict() == {
            "cost_usd": 12.5,
            "billable_bytes": 1_000,
            "record_count": 40,
            "n_symbols": 2,
        }
        assert isinstance(out.index, pd.DatetimeIndex)


class TestDownloadBars:
    def test_downloads_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda path, **_: Path(
            path
        ).write_bytes(FIXTURE_FILE.read_bytes())
        monkeypatch.setattr("tradingo.sampling.databento._historical", lambda _: client)

        request = {k: v for k, v in REQUEST.items()}
        download_bars(**request, output_dir=tmp_path)
        download_bars(**request, output_dir=tmp_path)

        assert client.timeseries.get_range.call_count == 1
        (written,) = tmp_path.iterdir()
        assert written.name == "XNAS.ITCH.ohlcv-1m.2024-01-01_2024-02-01.dbn.zst"
        assert len(discover(tmp_path)) == 1


def _info(
    dataset: str, schema: str, start: str, end: str, syms: tuple[str, ...]
) -> databento.DBNFileInfo:
    return databento.DBNFileInfo(
        path=Path(f"{start}_{end}.dbn.zst"),
        dataset=dataset,
        schema=schema,
        stype_in="raw_symbol",
        stype_out="instrument_id",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        symbols=syms,
        nbytes=0,
    )


class TestChunks:
    def test_splits_at_quarter_starts_keeping_ends(self) -> None:
        assert databento._chunks("2024-01-15", "2024-08-01", "QS") == [
            ("2024-01-15", "2024-04-01"),
            ("2024-04-01", "2024-07-01"),
            ("2024-07-01", "2024-08-01"),
        ]

    def test_aligned_range(self) -> None:
        assert databento._chunks("2024-01-01", "2024-07-01", "QS") == [
            ("2024-01-01", "2024-04-01"),
            ("2024-04-01", "2024-07-01"),
        ]

    def test_short_range_and_none_are_one_chunk(self) -> None:
        assert databento._chunks("2024-01-01", "2024-02-01", "QS") == [
            ("2024-01-01", "2024-02-01")
        ]
        assert databento._chunks("2019-01-01", "2026-11-20", None) == [
            ("2019-01-01", "2026-11-20")
        ]


HALF_YEAR = {**REQUEST, "start": "2024-01-01", "end": "2024-07-01"}


class TestChunkedDownload:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        client = MagicMock()
        client.timeseries.get_range.side_effect = lambda path, **_: Path(
            path
        ).write_bytes(FIXTURE_FILE.read_bytes())
        monkeypatch.setattr("tradingo.sampling.databento._historical", lambda _: client)
        return client

    def test_one_file_per_chunk(self, tmp_path: Path, client: MagicMock) -> None:
        download_bars(**HALF_YEAR, output_dir=tmp_path)

        calls = client.timeseries.get_range.call_args_list
        assert [(c.kwargs["start"], c.kwargs["end"]) for c in calls] == [
            ("2024-01-01", "2024-04-01"),
            ("2024-04-01", "2024-07-01"),
        ]
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "XNAS.ITCH.ohlcv-1m.2024-01-01_2024-04-01.dbn.zst",
            "XNAS.ITCH.ohlcv-1m.2024-04-01_2024-07-01.dbn.zst",
        ]
        assert Path(calls[0].kwargs["path"]).name.endswith(".dbn.zst.part")

    def test_resumes_after_failed_chunk(
        self, tmp_path: Path, client: MagicMock
    ) -> None:
        write = client.timeseries.get_range.side_effect

        def fail_second(path: Path, **kw: Any) -> None:
            if kw["start"] == "2024-04-01":
                raise ConnectionError("stream dropped")
            write(path, **kw)

        client.timeseries.get_range.side_effect = fail_second
        with pytest.raises(ConnectionError):
            download_bars(**HALF_YEAR, output_dir=tmp_path)

        client.timeseries.get_range.reset_mock()
        client.timeseries.get_range.side_effect = write
        download_bars(**HALF_YEAR, output_dir=tmp_path)

        (call,) = client.timeseries.get_range.call_args_list
        assert (call.kwargs["start"], call.kwargs["end"]) == (
            "2024-04-01",
            "2024-07-01",
        )

    def test_skips_chunks_an_existing_file_covers(
        self, tmp_path: Path, client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = [
            # an earlier, differently chunked download: Q1 for a superset of symbols
            _info(
                "XNAS.ITCH",
                "ohlcv-1m",
                "2023-12-01",
                "2024-04-01",
                ("SPY", "QQQ", "IWM"),
            ),
            # Q2 files that don't count: missing QQQ, other schema, other dataset
            _info("XNAS.ITCH", "ohlcv-1m", "2024-04-01", "2024-07-01", ("SPY",)),
            _info("XNAS.ITCH", "bbo-1m", "2024-04-01", "2024-07-01", ("SPY", "QQQ")),
            _info("GLBX.MDP3", "ohlcv-1m", "2024-04-01", "2024-07-01", ("SPY", "QQQ")),
        ]
        monkeypatch.setattr("tradingo.sampling.databento.discover", lambda _: existing)

        download_bars(**HALF_YEAR, output_dir=tmp_path)

        (call,) = client.timeseries.get_range.call_args_list
        assert (call.kwargs["start"], call.kwargs["end"]) == (
            "2024-04-01",
            "2024-07-01",
        )


FULL_YEAR = {**REQUEST, "start": "2024-01-01", "end": "2025-01-01"}


class TestGetCostResume:
    @pytest.fixture
    def meta(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        client = MagicMock()
        client.metadata.get_cost.return_value = 1.5
        client.metadata.get_billable_size.return_value = 100
        client.metadata.get_record_count.return_value = 10
        monkeypatch.setattr("tradingo.sampling.databento._historical", lambda _: client)
        meta: MagicMock = client.metadata
        return meta

    def test_prices_only_missing_spans(
        self, tmp_path: Path, meta: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Q2 already on disk: Q1 and Q3..Q4 remain, priced as two spans
        on_disk = [
            _info("XNAS.ITCH", "ohlcv-1m", "2024-04-01", "2024-07-01", ("SPY", "QQQ"))
        ]
        monkeypatch.setattr("tradingo.sampling.databento.discover", lambda _: on_disk)

        out = get_cost(**FULL_YEAR, output_dir=tmp_path)

        assert [
            (c.kwargs["start"], c.kwargs["end"]) for c in meta.get_cost.call_args_list
        ] == [("2024-01-01", "2024-04-01"), ("2024-07-01", "2025-01-01")]
        assert out.iloc[0]["cost_usd"] == 3.0
        assert out.iloc[0]["record_count"] == 20

    def test_fully_downloaded_costs_nothing(
        self, tmp_path: Path, meta: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_disk = [
            _info("XNAS.ITCH", "ohlcv-1m", "2019-01-01", "2026-01-01", ("SPY", "QQQ"))
        ]
        monkeypatch.setattr("tradingo.sampling.databento.discover", lambda _: on_disk)

        out = get_cost(**FULL_YEAR, output_dir=tmp_path)

        meta.get_cost.assert_not_called()
        assert out.iloc[0]["cost_usd"] == 0.0

    def test_adjacent_files_cover_a_straddling_chunk(
        self, tmp_path: Path, meta: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a recovered partial + its tail meet mid-quarter (2024-05-20)
        on_disk = [
            _info("XNAS.ITCH", "ohlcv-1m", "2024-05-20", "2025-01-01", ("SPY", "QQQ")),
            _info("XNAS.ITCH", "ohlcv-1m", "2019-01-01", "2024-05-20", ("SPY", "QQQ")),
        ]
        monkeypatch.setattr("tradingo.sampling.databento.discover", lambda _: on_disk)

        out = get_cost(**FULL_YEAR, output_dir=tmp_path)

        meta.get_cost.assert_not_called()
        assert out.iloc[0]["cost_usd"] == 0.0

    def test_gap_between_files_is_priced(
        self, tmp_path: Path, meta: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_disk = [
            _info("XNAS.ITCH", "ohlcv-1m", "2024-01-01", "2024-05-01", ("SPY", "QQQ")),
            _info("XNAS.ITCH", "ohlcv-1m", "2024-06-01", "2025-01-01", ("SPY", "QQQ")),
        ]
        monkeypatch.setattr("tradingo.sampling.databento.discover", lambda _: on_disk)

        get_cost(**FULL_YEAR, output_dir=tmp_path)

        (call,) = meta.get_cost.call_args_list
        assert (call.kwargs["start"], call.kwargs["end"]) == (
            "2024-04-01",
            "2024-07-01",
        )

    def test_without_output_dir_prices_whole_range(self, meta: MagicMock) -> None:
        get_cost(**FULL_YEAR)

        (call,) = meta.get_cost.call_args_list
        assert (call.kwargs["start"], call.kwargs["end"]) == (
            "2024-01-01",
            "2025-01-01",
        )
