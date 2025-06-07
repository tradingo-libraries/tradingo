import pytest

from tradingo.sampling.yf import currency_to_symbol, symbol_to_currency


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


if __name__ == "__main__":
    pytest.main([__file__])
