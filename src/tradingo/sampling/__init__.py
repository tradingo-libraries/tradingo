"""sampling API for data providers"""
from typing import Literal


Provider = Literal[
    "alpha_vantage",
    "cboe",
    "fmp",
    "ig",
    "intrinio",
    "polygon",
    "tiingo",
    "tmx",
    "tradier",
    "yfinance",
]
