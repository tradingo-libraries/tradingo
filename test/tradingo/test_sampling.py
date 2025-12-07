from unittest.mock import patch

import pandas as pd
import pytest

from tradingo.sampling.yf import (
    _align_series,
    _get_ticker,
    adjust_fx_series,
    currency_to_symbol,
    sample_equity,
    symbol_to_currency,
)


def test_handle_currency_ticker_valid_ticker():
    symbols = "USDJPY=X"
    result = symbol_to_currency(symbols)
    assert (
        result == "USDJPY"
    ), "The ticker should be correctly converted to currency pair."


def test_handle_currency():
    ticker = "GBPUSD"
    result = currency_to_symbol(ticker)
    assert result == "GBPUSD=X", "The ticker is a currency pair, whould end by =X."


def test_handle_not_a_currency():
    ticker = "NOTCCY"
    result = currency_to_symbol(ticker)
    assert result == "NOTCCY", "The ticker is not a currency pair."


def test_get_ticker_logs():
    t = _get_ticker("USDEUR")
    assert t.endswith("=X")


@patch("tradingo.sampling.yf.yf.download")
def test_sample_equity_calls_download(mock_download):
    mock_download.return_value = pd.DataFrame({"Close": [1, 2]})
    (df,) = sample_equity("USDEUR", "2020-01-01", "2020-01-02")
    assert isinstance(df, pd.DataFrame)
    mock_download.assert_called_once()


def test_adjust_fx_series_basic():
    df = pd.DataFrame({"EURUSD": [2.0, 4.0], "USDEUR": [0.5, 0.25]})
    out = adjust_fx_series(df, ref_ccy="USD", add_self=True)
    expected = pd.DataFrame({"EUR": [2.0, 4.0], "USD": [1.0, 1.0]})
    pd.testing.assert_frame_equal(out, expected)


def test_align_series_basic():
    s = pd.Series([1, 2, 3])
    o = 2
    s1, s2 = _align_series(s, o)
    assert (s2 == 2).all()
    s1, s2 = _align_series(s)
    assert (s2 == 1).all()


if __name__ == "__main__":
    pytest.main([__file__])
