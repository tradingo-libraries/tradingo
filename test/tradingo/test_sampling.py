from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from arcticdb import ReadRequest

from tradingo.sampling.yf import (
    ProviderDataError,
    _align_series,
    _get_ticker,
    adjust_fx_series,
    convert_prices_to_ccy,
    create_universe,
    currency_to_symbol,
    sample_equity,
    symbol_to_currency,
)

# =============================================================================
# Yahoo Finance Tests (yf.py)
# =============================================================================


class TestCurrencyConversion:
    """Tests for currency_to_symbol and symbol_to_currency functions."""

    def test_handle_currency_ticker_valid_ticker(self) -> None:
        symbols = "USDJPY=X"
        result = symbol_to_currency(symbols)
        assert (
            result == "USDJPY"
        ), "The ticker should be correctly converted to currency pair."

    def test_handle_currency(self) -> None:
        ticker = "GBPUSD"
        result = currency_to_symbol(ticker)
        assert result == "GBPUSD=X", "The ticker is a currency pair, should end by =X."

    def test_handle_not_a_currency(self) -> None:
        ticker = "NOTCCY"
        result = currency_to_symbol(ticker)
        assert result == "NOTCCY", "The ticker is not a currency pair."

    def test_currency_to_symbol_eurusd(self) -> None:
        assert currency_to_symbol("EURUSD") == "EURUSD=X"

    def test_currency_to_symbol_invalid_length(self) -> None:
        assert currency_to_symbol("EURO") == "EURO"  # Too short
        assert currency_to_symbol("GBPUSDEUR") == "GBPUSDEUR"  # Too long

    def test_symbol_to_currency_not_currency(self) -> None:
        assert symbol_to_currency("AAPL") == "AAPL"
        assert symbol_to_currency("MSFT=X") == "MSFT=X"  # Wrong length


class TestGetTicker:
    """Tests for _get_ticker helper."""

    def test_get_ticker_logs(self) -> None:
        t = _get_ticker("USDEUR")
        assert t.endswith("=X")

    def test_get_ticker_non_currency(self) -> None:
        t = _get_ticker("AAPL")
        assert t == "AAPL"


class TestSampleEquity:
    """Tests for sample_equity function."""

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_calls_download(
        self,
        mock_download: MagicMock,
    ) -> None:
        mock_download.return_value = pd.DataFrame({"Close": [1, 2]})
        df = sample_equity("USDEUR", "2020-01-01", "2020-01-02")
        assert isinstance(df, pd.DataFrame)
        mock_download.assert_called_once()

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_with_actions(self, mock_download: MagicMock) -> None:
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0], "Dividends": [0.0, 0.5]},
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], tz="UTC", name="Date"),
        )
        df = sample_equity("AAPL", "2020-01-01", "2020-01-03", actions=True)
        assert isinstance(df, pd.DataFrame)
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["actions"] is True

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_returns_none_raises_error(
        self, mock_download: MagicMock
    ) -> None:
        mock_download.return_value = None
        with pytest.raises(ProviderDataError, match="Yahoo Finance returned no data"):
            sample_equity("INVALID", "2020-01-01", "2020-01-02")

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_localizes_naive_index(
        self, mock_download: MagicMock
    ) -> None:
        # Return DataFrame with naive datetime index
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="Date"),
        )
        df = sample_equity("AAPL", "2020-01-01", "2020-01-03")
        assert cast(pd.DatetimeIndex, df.index).tz is not None
        assert str(cast(pd.DatetimeIndex, df.index).tz) == "UTC"

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_converts_to_utc(self, mock_download: MagicMock) -> None:
        # Return DataFrame with non-UTC timezone
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02"], tz="US/Eastern", name="Date"
            ),
        )
        df = sample_equity("AAPL", "2020-01-01", "2020-01-03")
        assert str(cast(pd.DatetimeIndex, df.index).tz) == "UTC"

    def test_sample_equity_no_end_date_raises(self) -> None:
        with pytest.raises(ValueError, match="end_date must be defined"):
            sample_equity("AAPL", "2020-01-01", "")

    @patch("tradingo.sampling.yf.yf.download")
    def test_sample_equity_interval_parameter(self, mock_download: MagicMock) -> None:
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="Date"),
        )
        sample_equity("AAPL", "2020-01-01", "2020-01-02", interval="1h")
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["interval"] == "1h"


class TestAdjustFxSeries:
    """Tests for adjust_fx_series function."""

    def test_adjust_fx_series_basic(self) -> None:
        df = pd.DataFrame({"EURUSD": [2.0, 4.0], "USDEUR": [0.5, 0.25]})
        out = adjust_fx_series(df, ref_ccy="USD", add_self=True)
        expected = pd.DataFrame({"EUR": [2.0, 4.0], "USD": [1.0, 1.0]})
        pd.testing.assert_frame_equal(out, expected)

    def test_adjust_fx_series_inversion(self) -> None:
        df = pd.DataFrame({"USDEUR": [0.5, 0.25]})
        out = adjust_fx_series(df, ref_ccy="USD")
        expected = pd.DataFrame({"EUR": [2.0, 4.0]})
        pd.testing.assert_frame_equal(out, expected)

    def test_adjust_fx_series_add_cent_gbp(self) -> None:
        df = pd.DataFrame({"GBPUSD": [1.5, 1.6]})
        out = adjust_fx_series(df, ref_ccy="USD", add_cent=True)
        assert "GBp" in out.columns
        pd.testing.assert_series_equal(
            out["GBp"], pd.Series([0.015, 0.016], name="GBp")
        )

    def test_adjust_fx_series_add_cent_eur(self) -> None:
        # Note: The source code has inconsistent handling here -
        # it checks fx_series.columns (original) for "EUR" but should check adjusted_fx.columns
        # For now, we test the current behavior where "EUR" must be in original columns
        # This test verifies the current behavior even if it's potentially buggy
        df = pd.DataFrame({"EURUSD": [1.1, 1.2]})
        out = adjust_fx_series(df, ref_ccy="USD", add_cent=True)
        # EUR is transformed to column "EUR" but the check is on original fx_series
        # so "c" column is NOT added (current behavior)
        assert "c" not in out.columns
        assert "EUR" in out.columns

    def test_adjust_fx_series_invalid_column_raises(self) -> None:
        df = pd.DataFrame({"EURJPY": [130.0, 131.0]})
        with pytest.raises(ValueError, match="does not match reference currency"):
            adjust_fx_series(df, ref_ccy="USD")


class TestAlignSeries:
    """Tests for _align_series function."""

    def test_align_series_basic(self) -> None:
        s = pd.Series([1, 2, 3])
        o = 2
        s1, s2 = _align_series(s, o)
        assert (s2 == 2).all()
        s1, s2 = _align_series(s)
        assert (s2 == 1).all()

    def test_align_series_with_none(self) -> None:
        s = pd.Series([1, 2, 3], name="test")
        s1, s2 = _align_series(s, None)
        assert (s2 == 1.0).all()
        assert s2.name == "test"

    def test_align_series_with_series(self) -> None:
        s1 = pd.Series([1, 2, 3], index=pd.date_range("2026-05-19", periods=3))
        s2 = pd.Series([4, 5, 6], index=pd.date_range("2026-05-20", periods=3))
        result1, result2 = _align_series(s1, s2)
        pd.testing.assert_index_equal(
            result1.index,
            pd.date_range("2026-05-20", periods=3),
            check_names=False,
        )
        pd.testing.assert_index_equal(
            result2.index,
            pd.date_range("2026-05-20", periods=3),
            check_names=False,
        )

    def test_align_series_invalid_type_raises(self) -> None:
        s = pd.Series([1, 2, 3])
        with pytest.raises(TypeError):
            _align_series(s, "invalid")  # type: ignore


class TestConvertPricesToCcy:
    """Tests for convert_prices_to_ccy function.

    Note: This function expects `prices` to be a dict-like object where
    .items() returns (name, DataFrame) pairs, not a single DataFrame.
    The actual interface appears to expect a dict of DataFrames keyed by
    price type (e.g., {"Open": df, "Close": df}).
    """

    def test_convert_prices_to_ccy_with_dict_prices(self) -> None:
        # prices should be a dict-like where each value is a DataFrame
        # with columns matching instrument symbols
        instruments = pd.DataFrame(
            {"currency": ["EUR"]},
            index=["AAPL"],
        )
        # prices is a dict where keys are price types and values are DataFrames
        # with columns matching the instruments index
        prices = {
            "close": pd.DataFrame(
                {"AAPL": [100.0, 101.0]},
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]),
            )
        }
        fx_series = {
            "close": pd.DataFrame(
                {"EURUSD": [1.1, 1.2]},
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]),
            )
        }
        result = convert_prices_to_ccy(instruments, prices, fx_series, "USD")
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)


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

    def test_mixed_tz_naive_and_aware_symbols_concat(self) -> None:
        """A tz-naive symbol must not break concat with tz-aware symbols."""
        naive_idx = pd.date_range("2024-01-01", periods=3, freq="D")
        aware = self._make_ohlcv(naive_idx.tz_localize("UTC"))
        naive = self._make_ohlcv(naive_idx)
        pricelib = self._make_pricelib({"AAA": aware, "BBB": naive})
        instruments = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="ticker"))

        open_, *_ = getattr(create_universe, "__wrapped__")(
            pricelib=pricelib,
            instruments=instruments,
            end_date=None,
            start_date=None,
        )

        assert isinstance(open_.index, pd.DatetimeIndex)
        assert str(open_.index.tz) == "UTC"
        assert list(open_.columns) == ["AAA", "BBB"]

    def test_non_utc_tz_converted_to_utc(self) -> None:
        """A non-UTC tz-aware symbol is converted to UTC."""
        ny_idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern")
        utc_idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        pricelib = self._make_pricelib(
            {"NY": self._make_ohlcv(ny_idx), "UTC": self._make_ohlcv(utc_idx)}
        )
        instruments = pd.DataFrame(index=pd.Index(["NY", "UTC"], name="ticker"))

        open_, *_ = getattr(create_universe, "__wrapped__")(
            pricelib=pricelib,
            instruments=instruments,
            end_date=None,
            start_date=None,
        )

        assert str(open_.index.tz) == "UTC"


# =============================================================================
# IG Trading Tests (ig.py)
# =============================================================================


class TestGetIgService:
    """Tests for get_ig_service function."""

    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_creates_session(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_ig_service

        mock_config.return_value = MagicMock(
            username="user",
            password="pass",
            api_key="key",
            acc_type="DEMO",
        )
        mock_service = MagicMock()
        mock_ig_service_cls.return_value = mock_service

        service = get_ig_service()

        mock_service.create_session.assert_called_once()
        assert service == mock_service
        # The limiter's login sleep costs ~2s per DAG task; off unless asked for.
        assert mock_ig_service_cls.call_args[1]["use_rate_limiter"] is False

    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_rate_limiter_opt_in(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_ig_service

        get_ig_service(use_rate_limiter=True)

        assert mock_ig_service_cls.call_args[1]["use_rate_limiter"] is True

    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_uses_override_credentials(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_ig_service

        mock_config.return_value = MagicMock(
            username="default_user",
            password="default_pass",
            api_key="default_key",
            acc_type="DEMO",
        )

        get_ig_service(
            username="override_user",
            password="override_pass",
            api_key="override_key",
            acc_type="LIVE",
        )

        call_kwargs = mock_ig_service_cls.call_args[1]
        assert call_kwargs["username"] == "override_user"
        assert call_kwargs["password"] == "override_pass"
        assert call_kwargs["api_key"] == "override_key"
        assert call_kwargs["acc_type"] == "LIVE"

    @pytest.mark.parametrize(
        "profile, prefix",
        [("demo", "IG_SERVICE"), ("prod", "IG_PROD"), ("PROD", "IG_PROD")],
    )
    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_profile_selects_credentials(
        self,
        mock_ig_service_cls: MagicMock,
        mock_config: MagicMock,
        profile: str,
        prefix: str,
    ) -> None:
        from tradingo.sampling.ig import get_ig_service

        get_ig_service(profile=profile)

        assert mock_config.call_args[1]["app_prefix"] == prefix

    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_backoff_is_bounded(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock
    ) -> None:
        from tradingo.ig_rest import ApiExceededException
        from tradingo.sampling.ig import get_ig_service

        get_ig_service(max_attempts=3, max_wait=0)
        retryer = mock_ig_service_cls.call_args[1]["retryer"]
        calls = MagicMock(side_effect=ApiExceededException())

        # Gives up after max_attempts and re-raises the IG error itself.
        with pytest.raises(ApiExceededException):
            retryer(calls)
        assert calls.call_count == 3

    @staticmethod
    def _run_on_fake_clock(
        retryer: Any, window: float
    ) -> tuple[list[float], list[float], float | None]:
        """Run ``retryer`` on a fake clock against IG answering
        ``ApiExceededException`` for the first ``window`` seconds.

        Returns (attempt times, sleeps, time of the successful attempt or None).
        """
        from tradingo.ig_rest import ApiExceededException

        clock = [0.0]
        attempts: list[float] = []
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        def request() -> float:
            attempts.append(clock[0])
            if clock[0] < window:
                raise ApiExceededException()
            return clock[0]

        with patch("time.monotonic", lambda: clock[0]):
            try:
                return attempts, sleeps, retryer.copy(sleep=sleep)(request)
            except ApiExceededException:
                return attempts, sleeps, None

    @pytest.mark.parametrize("seed", range(20))
    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_backoff_outlasts_the_allowance_minute(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock, seed: int
    ) -> None:
        # IG's allowances are a rolling minute. On 2026-11-26 the demo key was
        # refused for ~40s after each quarter-hour, and a 6-attempt budget (~13s)
        # gave up inside it: every trade task failed at login.
        import random

        from tradingo.sampling.ig import get_ig_service

        random.seed(seed)
        get_ig_service()
        retryer = mock_ig_service_cls.call_args[1]["retryer"]

        attempts, sleeps, recovered = self._run_on_fake_clock(retryer, window=60.0)
        assert recovered is not None and 60.0 <= recovered <= 90.0
        # Every wait keeps its fixed half: no burst of rejected requests.
        assert all(s >= min(2 ** (n - 1), 10.0) for n, s in enumerate(sleeps, 1))
        assert len(attempts) <= 12

    @pytest.mark.parametrize("seed", range(20))
    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_backoff_stops_by_max_retry_time(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock, seed: int
    ) -> None:
        import random

        from tradingo.sampling.ig import get_ig_service

        random.seed(seed)
        get_ig_service()
        retryer = mock_ig_service_cls.call_args[1]["retryer"]

        attempts, _, recovered = self._run_on_fake_clock(retryer, window=1e9)
        assert recovered is None
        assert attempts[-1] <= 90.0
        assert 7 <= len(attempts) <= 12

    @patch("tradingo.sampling.ig.IGTradingConfig.from_env")
    @patch("tradingo.sampling.ig.IGService")
    def test_get_ig_service_backoff_only_retries_allowance(
        self, mock_ig_service_cls: MagicMock, mock_config: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_ig_service

        get_ig_service(max_wait=0)
        retryer = mock_ig_service_cls.call_args[1]["retryer"]
        calls = MagicMock(side_effect=ValueError("bad credentials"))

        with pytest.raises(ValueError):
            retryer(calls)
        assert calls.call_count == 1


class TestIGProfiles:
    """IGTradingConfig.for_profile reads one credential set per profile."""

    ENV = {
        "IG_SERVICE_USERNAME": "demo-user",
        "IG_SERVICE_PASSWORD": "demo-pass",
        "IG_SERVICE_API_KEY": "demo-key",
        "IG_SERVICE_ACC_TYPE": "DEMO",
        "IG_PROD_USERNAME": "prod-user",
        "IG_PROD_PASSWORD": "prod-pass",
        "IG_PROD_API_KEY": "prod-key",
        "IG_PROD_ACC_TYPE": "LIVE",
        "IG_PROD_ACC_NUMBER": "ABC12",
    }

    def test_demo_is_default_and_reads_ig_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tradingo.settings import IGTradingConfig

        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)

        config = IGTradingConfig.for_profile()
        assert (config.username, config.acc_type) == ("demo-user", "DEMO")
        assert config.acc_number is None

    def test_prod_reads_ig_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tradingo.settings import IGTradingConfig

        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)

        config = IGTradingConfig.for_profile("prod")
        assert (config.username, config.api_key) == ("prod-user", "prod-key")
        assert (config.acc_type, config.acc_number) == ("LIVE", "ABC12")

    def test_missing_profile_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tradingo.settings import EnvProviderError, IGTradingConfig

        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)

        with pytest.raises(EnvProviderError):
            IGTradingConfig.for_profile("staging")

    @pytest.mark.parametrize("profile", ["", "service"])
    def test_invalid_profile(self, profile: str) -> None:
        from tradingo.settings import ig_profile_prefix

        with pytest.raises(ValueError):
            ig_profile_prefix(profile)


class TestSampleInstrument:
    """Tests for sample_instrument function."""

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_sample_instrument_returns_bid_ask(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import sample_instrument

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        # Create mock response with proper structure
        prices_df = pd.DataFrame(
            {
                ("bid", "Open"): [100.0, 101.0],
                ("bid", "High"): [102.0, 103.0],
                ("bid", "Low"): [99.0, 100.0],
                ("bid", "Close"): [101.0, 102.0],
                ("ask", "Open"): [100.5, 101.5],
                ("ask", "High"): [102.5, 103.5],
                ("ask", "Low"): [99.5, 100.5],
                ("ask", "Close"): [101.5, 102.5],
            },
            index=pd.DatetimeIndex(
                ["2020-01-01 10:00", "2020-01-01 11:00"], name="DateTime"
            ),
        )
        prices_df.columns = pd.MultiIndex.from_tuples(prices_df.columns)  # type: ignore
        mock_service.fetch_historical_prices_by_epic.return_value = {
            "prices": prices_df
        }

        bid, ask = sample_instrument(
            "IX.D.FTSE.DAILY.IP",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-01-02", tz="UTC"),
            "HOUR",
        )

        assert isinstance(bid, pd.DataFrame)
        assert isinstance(ask, pd.DataFrame)
        assert list(bid.columns) == ["Open", "High", "Low", "Close"]
        assert list(ask.columns) == ["Open", "High", "Low", "Close"]
        mock_service.session.close.assert_called_once()

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_sample_instrument_no_data_returns_empty(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import sample_instrument

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.fetch_historical_prices_by_epic.side_effect = Exception(
            "Historical price data not found"
        )

        bid, ask = sample_instrument(
            "IX.D.INVALID.EPIC",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-01-02", tz="UTC"),
            "HOUR",
        )

        assert bid.empty
        assert ask.empty
        mock_service.session.close.assert_called_once()

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_sample_instrument_io_error_returns_empty(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import sample_instrument

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.fetch_historical_prices_by_epic.side_effect = Exception(
            "error.price-history.io-error"
        )

        bid, ask = sample_instrument(
            "IX.D.INVALID.EPIC",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-01-02", tz="UTC"),
            "HOUR",
        )

        assert bid.empty
        assert ask.empty

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_sample_instrument_unexpected_error_raises(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import sample_instrument

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.fetch_historical_prices_by_epic.side_effect = ValueError(
            "Unexpected error"
        )

        with pytest.raises(ValueError, match="Unexpected error"):
            sample_instrument(
                "IX.D.FTSE.DAILY.IP",
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-01-02", tz="UTC"),
                "HOUR",
            )

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_sample_instrument_uses_provided_service(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import sample_instrument

        custom_service = MagicMock()
        prices_df = pd.DataFrame(
            {
                ("bid", "Open"): [100.0],
                ("bid", "High"): [102.0],
                ("bid", "Low"): [99.0],
                ("bid", "Close"): [101.0],
                ("ask", "Open"): [100.5],
                ("ask", "High"): [102.5],
                ("ask", "Low"): [99.5],
                ("ask", "Close"): [101.5],
            },
            index=pd.DatetimeIndex(["2020-01-01 10:00"], name="DateTime"),
        )
        prices_df.columns = pd.MultiIndex.from_tuples(prices_df.columns)  # type: ignore
        custom_service.fetch_historical_prices_by_epic.return_value = {
            "prices": prices_df
        }

        sample_instrument(
            "IX.D.FTSE.DAILY.IP",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-01-02", tz="UTC"),
            "HOUR",
            service=custom_service,
        )

        # Should not call get_ig_service when service is provided
        mock_get_service.assert_not_called()
        custom_service.fetch_historical_prices_by_epic.assert_called_once()


class TestGetActivityHistory:
    """Tests for get_activity_history function."""

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_get_activity_history_groups_by_epic(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_activity_history

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        activity_df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
                "time": ["10:00:00", "11:00:00", "09:00:00"],
                "epic": ["EPIC1", "EPIC1", "EPIC2"],
                "marketName": ["Market1", "Market1", "Market2"],
                "period": ["-", "-", "-"],
                "size": ["1.0", "2.0", "3.0"],
                "stop": ["-", "100", "200"],
                "limit": ["50", "-", "150"],
            }
        )
        mock_service.fetch_account_activity_by_date.return_value = activity_df

        result = get_activity_history(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-03"),
        )

        assert len(result) == 2  # Two EPICs
        mock_service.session.close.assert_called_once()

    @patch("tradingo.sampling.ig.get_ig_service")
    def test_get_activity_history_uses_provided_service(
        self, mock_get_service: MagicMock
    ) -> None:
        from tradingo.sampling.ig import get_activity_history

        custom_service = MagicMock()
        activity_df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "time": ["10:00:00"],
                "epic": ["EPIC1"],
                "marketName": ["Market1"],
                "period": ["-"],
                "size": ["1.0"],
                "stop": ["-"],
                "limit": ["50"],
            }
        )
        custom_service.fetch_account_activity_by_date.return_value = activity_df

        get_activity_history(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            svc=custom_service,
        )

        mock_get_service.assert_not_called()
        custom_service.fetch_account_activity_by_date.assert_called_once()


# =============================================================================
# Instruments Tests (instruments.py)
# =============================================================================


class TestDownloadInstruments:
    """Tests for download_instruments function."""

    @patch("tradingo.sampling.instruments.pd.read_csv")
    def test_download_instruments_from_file(self, mock_read_csv: MagicMock) -> None:
        from tradingo.sampling.instruments import download_instruments

        mock_read_csv.return_value = pd.DataFrame(
            {"name": ["Apple", "Microsoft"], "ticker": ["AAPL", "MSFT"]},
        ).set_index("ticker")

        result = download_instruments(file="instruments.csv", index_col="ticker")

        mock_read_csv.assert_called_once_with("instruments.csv", index_col="ticker")
        assert result.index.name == "Symbol"

    @patch("tradingo.sampling.instruments.pd.read_html")
    def test_download_instruments_from_html(self, mock_read_html: MagicMock) -> None:
        from tradingo.sampling.instruments import download_instruments

        mock_read_html.return_value = [
            pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Name": ["Apple", "Microsoft"]})
        ]

        result = download_instruments(
            html="https://example.com/table.html", index_col="Symbol"
        )

        mock_read_html.assert_called_once_with("https://example.com/table.html")
        assert result.index.name == "Symbol"

    @patch("tradingo.sampling.instruments.Ticker")
    def test_download_instruments_from_tickers_list(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        from tradingo.sampling.instruments import download_instruments

        mock_ticker_aapl = MagicMock()
        mock_ticker_aapl.get_info.return_value = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
        }
        mock_ticker_msft = MagicMock()
        mock_ticker_msft.get_info.return_value = {
            "symbol": "MSFT",
            "shortName": "Microsoft Corp",
        }

        def ticker_side_effect(ticker: str) -> str:
            if ticker == "AAPL":
                return mock_ticker_aapl
            return mock_ticker_msft

        mock_ticker_cls.side_effect = ticker_side_effect

        result = download_instruments(tickers=["AAPL", "MSFT"])

        assert len(result) == 2
        assert result.index.name == "Symbol"
        assert mock_ticker_cls.call_count == 2

    @patch("tradingo.sampling.instruments.pd.read_html")
    @patch("tradingo.sampling.instruments.Ticker")
    def test_download_instruments_from_url_tickers(
        self, mock_ticker_cls: MagicMock, mock_read_html: MagicMock
    ) -> None:
        from tradingo.sampling.instruments import download_instruments

        mock_read_html.return_value = [pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})]

        mock_ticker = MagicMock()
        mock_ticker.get_info.return_value = {"symbol": "TEST", "shortName": "Test"}
        mock_ticker_cls.return_value = mock_ticker

        result = download_instruments(
            tickers="https://example.com/tickers.html",
            index_col="Symbol",
        )

        mock_read_html.assert_called_once()
        assert result.index.name == "Symbol"

    @patch("tradingo.sampling.instruments.get_ig_service")
    def test_download_instruments_from_epics(self, mock_get_service: MagicMock) -> None:
        from tradingo.sampling.instruments import download_instruments

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        def fetch_market_side_effect(epic: str) -> dict[str, Any]:
            return {
                "instrument": {
                    "epic": epic,
                    "name": f"Market {epic}",
                    "type": "INDICES",
                }
            }

        mock_service.fetch_market_by_epic.side_effect = fetch_market_side_effect

        result = download_instruments(epics=["EPIC1", "EPIC2"])

        assert len(result) == 2
        assert result.index.name == "Symbol"
        assert mock_service.fetch_market_by_epic.call_count == 2

    def test_download_instruments_no_source_raises(self) -> None:
        from tradingo.sampling.instruments import download_instruments

        with pytest.raises(ValueError):
            download_instruments()


# =============================================================================
# Create Universe Tests (yf.py and ig.py)
# =============================================================================


class TestYfCreateUniverse:
    """Tests for Yahoo Finance create_universe function."""

    def test_create_universe_reads_from_library(self) -> None:
        mock_lib = MagicMock()

        # Create mock price data
        price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], tz="UTC", name="Date"),
        )

        mock_versioned_item = MagicMock()
        mock_versioned_item.data = price_data
        mock_lib.read.return_value = mock_versioned_item
        mock_lib.list_symbols.return_value = ["AAPL"]

        instruments = pd.DataFrame(
            {"name": ["Apple"]},
            index=pd.Index(["AAPL"], name="Symbol"),
        )

        # Call the underlying function directly (bypass decorator)
        result = create_universe.__wrapped__(  # type: ignore
            mock_lib,
            instruments,
            pd.Timestamp("2020-01-03", tz="UTC"),
            pd.Timestamp("2020-01-01", tz="UTC"),
        )

        assert len(result) == 5  # Open, High, Low, Close, Volume
        mock_lib.read.assert_called()

    def test_create_universe_returns_single_index_columns(self) -> None:
        """Ensure create_universe returns DataFrames with single-level column index."""
        mock_lib = MagicMock()

        # Create mock price data for multiple tickers
        price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], tz="UTC", name="Date"),
        )

        mock_versioned_item = MagicMock()
        mock_versioned_item.data = price_data
        mock_lib.read.return_value = mock_versioned_item
        mock_lib.list_symbols.return_value = ["AAPL", "MSFT"]

        instruments = pd.DataFrame(
            {"name": ["Apple", "Microsoft"]},
            index=pd.Index(["AAPL", "MSFT"], name="Symbol"),
        )

        open_df, high_df, low_df, close_df, volume_df = create_universe.__wrapped__(  # type: ignore
            mock_lib,
            instruments,
            pd.Timestamp("2020-01-03", tz="UTC"),
            pd.Timestamp("2020-01-01", tz="UTC"),
        )

        # Verify all returned DataFrames have single-level column index (not MultiIndex)
        for df, name in [
            (open_df, "Open"),
            (high_df, "High"),
            (low_df, "Low"),
            (close_df, "Close"),
            (volume_df, "Volume"),
        ]:
            assert not isinstance(
                df.columns, pd.MultiIndex
            ), f"{name} DataFrame has MultiIndex columns, expected single-level Index"
            assert isinstance(df.columns, pd.Index)
            assert list(df.columns) == ["AAPL", "MSFT"]


class TestIgCreateUniverse:
    """Tests for IG create_universe function."""

    def test_create_universe_reads_bid_ask(self) -> None:
        from tradingo.sampling.ig import create_universe

        mock_lib = MagicMock()

        bid_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
            },
            index=pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02"], tz="UTC", name="DateTime"
            ),
        )
        ask_data = pd.DataFrame(
            {
                "Open": [100.5, 101.5],
                "High": [102.5, 103.5],
                "Low": [99.5, 100.5],
                "Close": [101.5, 102.5],
            },
            index=pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02"], tz="UTC", name="DateTime"
            ),
        )

        def read_side_effect(
            symbol: str | ReadRequest,
            date_range: Any = None,
        ) -> MagicMock:
            mock_item = MagicMock()
            if ".bid" in symbol:
                mock_item.data = bid_data
            else:
                mock_item.data = ask_data
            if isinstance(symbol, ReadRequest):
                symbol = symbol.symbol
            mock_item.symbol = symbol
            return mock_item

        mock_lib.read.side_effect = read_side_effect
        mock_lib.read_batch.side_effect = lambda s: [read_side_effect(i) for i in s]

        instruments = pd.DataFrame(
            {"name": ["FTSE 100"]},
            index=pd.Index(["IX.D.FTSE.DAILY.IP"], name="Symbol"),
        )

        # Call the underlying function directly (bypass decorator)
        result = create_universe.__wrapped__(  # type: ignore
            mock_lib,
            instruments,
            pd.Timestamp("2020-01-03", tz="UTC"),
            pd.Timestamp("2020-01-01", tz="UTC"),
        )

        assert len(result) == 12  # 4 bid + 4 ask + 4 mid
        assert mock_lib.read_batch.call_count == 1  # bid and ask

    @pytest.mark.parametrize("name", ["DateTime", "timestamp"])
    def test_create_universe_keeps_index_name_with_missing_member(
        self, name: str
    ) -> None:
        """A missing member's placeholder must not drop the members' index name."""
        from arcticdb import Arctic

        from tradingo.sampling.ig import create_universe

        lib = Arctic("mem://ig-universe").get_library(
            f"raw-{name}", create_if_missing=True
        )
        index = pd.date_range(
            "2026-11-26", periods=3, freq="15min", tz="UTC", name=name
        )
        bars = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=index
        )
        lib.write("PRESENT.bid", bars)
        lib.write("PRESENT.ask", bars)
        instruments = pd.DataFrame(
            index=pd.Index(["PRESENT", "MISSING"], name="Symbol")
        )

        result = create_universe.__wrapped__(  # type: ignore
            lib, instruments, index[-1], index[0], permit_missing=True
        )

        for frame in result:
            assert frame.index.name == name
            assert list(frame.columns) == ["PRESENT", "MISSING"]


class TestMeasureIgPriceScales:
    """Measure IG price / source price for dukascopy and yfinance mappings."""

    def test_measure_ig_price_scales(self) -> None:
        from tradingo.sampling.instruments import measure_ig_price_scales

        def market(epic: str, bid: float | None, offer: float | None) -> Any:
            return {
                "instrument": {"epic": epic, "name": epic},
                "snapshot": {"bid": bid, "offer": offer, "marketStatus": "TRADEABLE"},
            }

        service = MagicMock()
        service.fetch_markets_by_epics.return_value = [
            market("CS.D.USDJPY.TODAY.IP", 15840.0, 15841.0),
            market("CS.D.USDCAD.TODAY.IP", 13972.0, 13973.0),
            market("CC.D.VIX.USS.IP", None, None),
        ]
        prices = pd.Series({"USD/JPY": 158.42, "USD/CAD": 1.39716, "VOL": 17.6})

        result = measure_ig_price_scales(
            prices,
            {
                "USD/JPY": "CS.D.USDJPY.TODAY.IP",
                "USD/CAD": "CS.D.USDCAD.TODAY.IP",
                "VOL": "CC.D.VIX.USS.IP",
                "XXX": "NOT.A.MARKET",
            },
            account_id="SB1",
            service=service,
        )

        service.switch_account.assert_called_once_with("SB1", False)
        # raw ratio, not rounded
        assert result.loc["USD/JPY", "price_scale"] == pytest.approx(15840.5 / 158.42)
        assert result.loc["USD/CAD", "price_scale"] == pytest.approx(13972.5 / 1.39716)
        assert pd.isna(result.loc["VOL", "price_scale"])  # closed market
        assert "XXX" not in result.index


if __name__ == "__main__":
    pytest.main([__file__])


class TestDukascopySampleInstrument:
    """``tradingo.sampling.dukascopy.sample_instrument`` against a fake API."""

    START = pd.Timestamp("2026-11-23 10:00:00", tz="utc")
    END = pd.Timestamp("2026-11-23 10:01:00", tz="utc")

    @staticmethod
    def _ms(ts: str) -> int:
        return int(pd.Timestamp(ts, tz="utc").timestamp() * 1000)

    def _row(self, ts: str, price: float = 1.1) -> list[float]:
        return [self._ms(ts), price, price, price, price, 10.0]

    def _patch(self, monkeypatch: pytest.MonkeyPatch, pages: list[Any]) -> list[int]:
        import dukascopy_python

        cursors: list[int] = []
        responses = iter(pages)

        def fake_fetch(**kwargs: Any) -> Any:
            cursors.append(kwargs["last_update"])
            return next(responses)

        monkeypatch.setattr(dukascopy_python, "_fetch", fake_fetch)
        monkeypatch.setattr("time.sleep", lambda _: None)
        return cursors

    def _sample(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from tradingo.sampling.dukascopy import sample_instrument

        return sample_instrument("EUR/USD", self.START, self.END, "10SEC")

    def test_rate_limit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tradingo.sampling.dukascopy import DukascopyAPIError

        error = {"error": "Too Many Requests (Rate limited)."}
        cursors = self._patch(monkeypatch, [error])
        with pytest.raises(DukascopyAPIError, match="Rate limited"):
            self._sample()
        assert len(cursors) == 1  # not retried

    def test_window_ending_short_keeps_bars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The provider has bars up to 10:00:20 only; the next page is {}.
        page = [self._row("2026-11-23 10:00:10"), self._row("2026-11-23 10:00:20")]
        self._patch(monkeypatch, [page, {}, page, {}])
        bid, ask = self._sample()
        assert list(bid.index) == [
            pd.Timestamp("2026-11-23 10:00:10", tz="utc"),
            pd.Timestamp("2026-11-23 10:00:20", tz="utc"),
        ]
        assert list(bid.columns[:4]) == ["Open", "High", "Low", "Close"]
        assert ask.equals(bid)

    def test_pages_forward_and_stops_past_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = [self._row("2026-11-23 10:00:00"), self._row("2026-11-23 10:00:30")]
        # The next page repeats the cursor row, then runs past the end.
        second = [self._row("2026-11-23 10:00:30"), self._row("2026-11-23 10:01:00")]
        second.append(self._row("2026-11-23 10:01:10"))
        cursors = self._patch(monkeypatch, [first, second, first, second])
        bid, _ = self._sample()
        assert [t.strftime("%H:%M:%S") for t in bid.index] == [
            "10:00:00",
            "10:00:30",
            "10:01:00",
        ]
        assert cursors[:2] == [
            self._ms("2026-11-23 10:00"),
            self._ms("2026-11-23 10:00:30"),
        ]

    def test_no_data_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, [{}, []])
        bid, ask = self._sample()
        assert bid.empty and ask.empty
        assert isinstance(bid.index, pd.DatetimeIndex) and str(bid.index.tz) == "UTC"
        assert all(pd.api.types.is_float_dtype(t) for t in bid.dtypes)

    def test_network_error_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dukascopy_python

        page = [self._row("2026-11-23 10:00:10")]
        calls = iter([ConnectionError("reset"), page, {}, page, {}])

        def flaky(**kwargs: Any) -> Any:
            out = next(calls)
            if isinstance(out, Exception):
                raise out
            return out

        monkeypatch.setattr(dukascopy_python, "_fetch", flaky)
        monkeypatch.setattr("time.sleep", lambda _: None)
        bid, _ = self._sample()
        assert len(bid) == 1
