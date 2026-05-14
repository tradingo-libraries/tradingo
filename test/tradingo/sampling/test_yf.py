"""Tests for tradingo.sampling.yf (ArcticDB-free functions)."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingo.sampling.yf import (
    ProviderDataError,
    _align_series,
    _get_ticker,
    adjust_fx_series,
    convert_prices_to_ccy,
    currency_to_symbol,
    sample_equity,
    symbol_to_currency,
)

IDX = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")


# ---------------------------------------------------------------------------
# currency_to_symbol
# ---------------------------------------------------------------------------


def test_currency_pair_gets_suffix() -> None:
    assert currency_to_symbol("GBPUSD") == "GBPUSD=X"


def test_non_currency_passthrough() -> None:
    assert currency_to_symbol("AAPL") == "AAPL"
    assert currency_to_symbol("EGLN.L") == "EGLN.L"


def test_invalid_ccy_codes_passthrough() -> None:
    # 6 chars but not valid ISO-4217 codes
    assert currency_to_symbol("ABCDEF") == "ABCDEF"


def test_partial_ccy_passthrough() -> None:
    # only first leg is a valid currency
    assert currency_to_symbol("GBPXYZ") == "GBPXYZ"


# ---------------------------------------------------------------------------
# symbol_to_currency
# ---------------------------------------------------------------------------


def test_yf_ccy_symbol_stripped() -> None:
    assert symbol_to_currency("GBPUSD=X") == "GBPUSD"


def test_non_ccy_symbol_passthrough() -> None:
    assert symbol_to_currency("AAPL") == "AAPL"
    assert symbol_to_currency("EGLN.L") == "EGLN.L"


def test_roundtrip() -> None:
    assert symbol_to_currency(currency_to_symbol("EURUSD")) == "EURUSD"


# ---------------------------------------------------------------------------
# _get_ticker
# ---------------------------------------------------------------------------


def test_get_ticker_converts_currency() -> None:
    assert _get_ticker("GBPUSD") == "GBPUSD=X"


def test_get_ticker_leaves_equity_unchanged() -> None:
    assert _get_ticker("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# sample_equity
# ---------------------------------------------------------------------------


def _make_yf_response(tz: str | None = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B", tz=tz)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 1000},
        index=idx,
    )


def test_sample_equity_returns_utc_dataframe() -> None:
    with patch("tradingo.sampling.yf.yf.download", return_value=_make_yf_response()):
        result = sample_equity("AAPL", "2024-01-01", "2024-01-10")
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"


def test_sample_equity_raises_without_end_date() -> None:
    with pytest.raises(ValueError, match="end_date"):
        sample_equity("AAPL", "2024-01-01", "")


def test_sample_equity_raises_on_none_response() -> None:
    with patch("tradingo.sampling.yf.yf.download", return_value=None):
        with pytest.raises(ProviderDataError):
            sample_equity("AAPL", "2024-01-01", "2024-01-10")


def test_sample_equity_localizes_naive_index() -> None:
    df = _make_yf_response(tz=None)
    assert isinstance(df.index, pd.DatetimeIndex)
    df.index = df.index.tz_localize(None)
    with patch("tradingo.sampling.yf.yf.download", return_value=df):
        result = sample_equity("AAPL", "2024-01-01", "2024-01-10")
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None


def test_sample_equity_converts_currency_ticker() -> None:
    """Currency pair is converted to YF format before download."""
    with patch(
        "tradingo.sampling.yf.yf.download", return_value=_make_yf_response()
    ) as mock_dl:
        sample_equity("GBPUSD", "2024-01-01", "2024-01-10")
    assert mock_dl.call_args[0][0] == ["GBPUSD=X"]


# ---------------------------------------------------------------------------
# adjust_fx_series
# ---------------------------------------------------------------------------


def test_quote_leg_is_ref_ccy_renamed() -> None:
    """USDGBP with ref=GBP → column renamed to USD (no inversion)."""
    fx = pd.DataFrame({"USDGBP": 0.8}, index=IDX)
    result = adjust_fx_series(fx, ref_ccy="GBP")
    assert "USD" in result.columns
    assert result["USD"].iloc[0] == pytest.approx(0.8)


def test_base_leg_is_ref_ccy_inverted() -> None:
    """GBPUSD with ref=GBP → column renamed to USD and rate inverted."""
    fx = pd.DataFrame({"GBPUSD": 1.25}, index=IDX)
    result = adjust_fx_series(fx, ref_ccy="GBP")
    assert "USD" in result.columns
    assert result["USD"].iloc[0] == pytest.approx(1.0 / 1.25)


def test_add_self_adds_ref_ccy_column() -> None:
    fx = pd.DataFrame({"USDGBP": 0.8}, index=IDX)
    result = adjust_fx_series(fx, ref_ccy="GBP", add_self=True)
    assert "GBP" in result.columns
    assert result["GBP"].iloc[0] == pytest.approx(1.0)


def test_add_cent_adds_gbp_pence_column() -> None:
    fx = pd.DataFrame({"USDGBP": 0.8}, index=IDX)
    result = adjust_fx_series(fx, ref_ccy="GBP", add_self=True, add_cent=True)
    assert "GBp" in result.columns
    assert result["GBp"].iloc[0] == pytest.approx(0.01)


def test_unrelated_ccy_raises() -> None:
    fx = pd.DataFrame({"EURUSD": 1.1}, index=IDX)
    with pytest.raises(ValueError, match="reference currency"):
        adjust_fx_series(fx, ref_ccy="GBP")


# ---------------------------------------------------------------------------
# _align_series
# ---------------------------------------------------------------------------


def test_align_with_none_creates_ones() -> None:
    s = pd.Series([1.0, 2.0, 3.0], index=IDX[:3])
    _, other = _align_series(s, None)
    assert (other == 1.0).all()
    assert other.index.equals(s.index)


def test_align_with_scalar() -> None:
    s = pd.Series([1.0, 2.0], index=IDX[:2])
    _, other = _align_series(s, 0.5)
    assert (other == 0.5).all()


def test_align_series_intersects_index() -> None:
    s1 = pd.Series([1.0, 2.0, 3.0], index=IDX[:3])
    s2 = pd.Series([10.0, 20.0], index=IDX[1:3])
    a, b = _align_series(s1, s2)
    assert len(a) == 2
    assert len(b) == 2


def test_align_series_drops_nan() -> None:
    s1 = pd.Series([1.0, np.nan, 3.0], index=IDX[:3])
    s2 = pd.Series([10.0, 20.0, 30.0], index=IDX[:3])
    a, b = _align_series(s1, s2)
    assert len(a) == 2
    assert not a.isna().any()


def test_align_with_invalid_type_raises() -> None:
    s = pd.Series([1.0], index=IDX[:1])
    with pytest.raises(TypeError):
        _align_series(s, "invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# convert_prices_to_ccy
# ---------------------------------------------------------------------------


def make_instruments(*symbols: str, currency: str = "USD") -> pd.DataFrame:
    return pd.DataFrame({"currency": {s: currency for s in symbols}})


def test_basic_usd_to_gbp() -> None:
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"A": [100.0] * 10}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 0.8}, index=IDX)}
    result = convert_prices_to_ccy(instruments, prices, fx, currency="GBP")
    assert result[0]["A"].iloc[0] == pytest.approx(80.0)


def test_gbp_instrument_unchanged() -> None:
    """GBP instrument with add_self=True FX of 1.0 → price unchanged."""
    instruments = make_instruments("A", currency="GBP")
    prices = {"close": pd.DataFrame({"A": [200.0] * 10}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 0.8}, index=IDX)}
    result = convert_prices_to_ccy(instruments, prices, fx, currency="GBP")
    assert result[0]["A"].iloc[0] == pytest.approx(200.0)


def test_multiple_price_bars_returned() -> None:
    instruments = make_instruments("A", currency="USD")
    prices = {
        "open": pd.DataFrame({"A": [99.0] * 10}, index=IDX),
        "close": pd.DataFrame({"A": [100.0] * 10}, index=IDX),
    }
    fx = {
        "open": pd.DataFrame({"USDGBP": 0.8}, index=IDX),
        "close": pd.DataFrame({"USDGBP": 0.8}, index=IDX),
    }
    result = convert_prices_to_ccy(instruments, prices, fx, currency="GBP")
    assert len(result) == 2


def test_mixed_currencies() -> None:
    instruments = pd.DataFrame({"currency": {"A": "USD", "B": "EUR"}})
    prices = {"close": pd.DataFrame({"A": [100.0] * 10, "B": [200.0] * 10}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 0.8, "EURGBP": 0.9}, index=IDX)}
    result = convert_prices_to_ccy(instruments, prices, fx, currency="GBP")
    assert result[0]["A"].iloc[0] == pytest.approx(80.0)
    assert result[0]["B"].iloc[0] == pytest.approx(180.0)


def test_gap_within_ffill_limit_filled() -> None:
    values = [100.0] * 4 + [np.nan] * 3 + [100.0] * 3
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"A": values}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 1.0}, index=IDX)}
    result = convert_prices_to_ccy(
        instruments, prices, fx, currency="GBP", ffill_limit=10
    )
    assert not result[0]["A"].isna().any()


def test_gap_beyond_ffill_limit_stays_nan() -> None:
    values = [100.0] * 2 + [np.nan] * 6 + [100.0] * 2
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"A": values}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 1.0}, index=IDX)}
    result = convert_prices_to_ccy(
        instruments, prices, fx, currency="GBP", ffill_limit=3
    )
    assert result[0]["A"].isna().any()


def test_trailing_staleness_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    values = [100.0] * 7 + [np.nan] * 3
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"A": values}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 1.0}, index=IDX)}
    with caplog.at_level(logging.WARNING):
        convert_prices_to_ccy(instruments, prices, fx, currency="GBP", ffill_limit=10)
    assert "trailing stale" in caplog.text.lower()


def test_historical_gap_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    values = [100.0] * 3 + [np.nan] * 2 + [100.0] * 5
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"A": values}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 1.0}, index=IDX)}
    with caplog.at_level(logging.WARNING):
        convert_prices_to_ccy(instruments, prices, fx, currency="GBP", ffill_limit=10)
    assert "trailing stale" not in caplog.text.lower()


def test_mismatched_symbols_raises() -> None:
    instruments = make_instruments("A", currency="USD")
    prices = {"close": pd.DataFrame({"B": [100.0] * 10}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 0.8}, index=IDX)}
    with pytest.raises(ValueError, match="do not match"):
        convert_prices_to_ccy(instruments, prices, fx, currency="GBP")


def test_missing_fx_ccy_raises() -> None:
    instruments = make_instruments("A", currency="EUR")
    prices = {"close": pd.DataFrame({"A": [100.0] * 10}, index=IDX)}
    fx = {"close": pd.DataFrame({"USDGBP": 0.8}, index=IDX)}
    with pytest.raises(ValueError, match="miss currencies"):
        convert_prices_to_ccy(instruments, prices, fx, currency="GBP")
