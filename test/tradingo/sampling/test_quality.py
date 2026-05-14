"""Tests for tradingo.sampling.quality"""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tradingo.sampling.quality import DataQualityError, check_price_staleness


def make_prices(trailing_nans: dict[str, int], length: int = 20) -> pd.DataFrame:
    """Build a price DataFrame where each column has the given number of trailing NaNs."""
    idx = pd.date_range("2024-01-01", periods=length, freq="B")
    data = {}
    for col, n_nans in trailing_nans.items():
        values = np.ones(length)
        if n_nans:
            values[-n_nans:] = np.nan
        data[col] = values
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# No staleness
# ---------------------------------------------------------------------------


def test_no_staleness_returns_none() -> None:
    prices = make_prices({"A": 0, "B": 0})
    check_price_staleness(prices, universe_name="test", ffill_limit=10)


def test_no_staleness_sends_no_email() -> None:
    prices = make_prices({"A": 0})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        check_price_staleness(
            prices,
            universe_name="test",
            ffill_limit=10,
            alert_recipient="a@b.com",
        )
    mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Approaching limit (trailing_nan == ffill_limit - 1)
# ---------------------------------------------------------------------------


def test_approaching_limit_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    prices = make_prices({"EGLN.L": 9})
    with caplog.at_level(logging.WARNING):
        check_price_staleness(prices, universe_name="test", ffill_limit=10)
    assert "approaching" in caplog.text.lower()


def test_approaching_limit_sends_email() -> None:
    prices = make_prices({"EGLN.L": 9})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        check_price_staleness(
            prices,
            universe_name="test",
            ffill_limit=10,
            alert_recipient="a@b.com",
        )
    mock_send.assert_called_once()
    body: str = mock_send.call_args[1]["body"]
    assert "approaching" in body.lower()


def test_approaching_limit_does_not_raise() -> None:
    prices = make_prices({"EGLN.L": 9})
    check_price_staleness(prices, universe_name="test", ffill_limit=10)


def test_approaching_limit_no_email_without_recipient() -> None:
    prices = make_prices({"EGLN.L": 9})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        check_price_staleness(prices, universe_name="test", ffill_limit=10)
    mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Exceeded limit (trailing_nan >= ffill_limit)
# ---------------------------------------------------------------------------


def test_exceeded_limit_raises() -> None:
    prices = make_prices({"EGLN.L": 10})
    with pytest.raises(DataQualityError, match="ffill exhausted"):
        check_price_staleness(prices, universe_name="test", ffill_limit=10)


def test_exceeded_limit_sends_email_before_raising() -> None:
    prices = make_prices({"EGLN.L": 10})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        with pytest.raises(DataQualityError):
            check_price_staleness(
                prices,
                universe_name="test",
                ffill_limit=10,
                alert_recipient="a@b.com",
            )
    mock_send.assert_called_once()


def test_exceeded_limit_email_contains_html_table() -> None:
    prices = make_prices({"EGLN.L": 10})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        with pytest.raises(DataQualityError):
            check_price_staleness(
                prices,
                universe_name="test",
                ffill_limit=10,
                alert_recipient="a@b.com",
            )
    body: str = mock_send.call_args[1]["body"]
    assert "<table" in body
    assert "EGLN.L" in body


def test_exceeded_limit_only_stale_instruments_in_email() -> None:
    """Instruments at zero staleness should not appear in the alert table."""
    prices = make_prices({"EGLN.L": 10, "WCOM.L": 0})
    with patch("tradingo.notifications.email.send_email") as mock_send:
        with pytest.raises(DataQualityError):
            check_price_staleness(
                prices,
                universe_name="test",
                ffill_limit=10,
                alert_recipient="a@b.com",
            )
    body: str = mock_send.call_args[1]["body"]
    assert "EGLN.L" in body
    assert "WCOM.L" not in body


# ---------------------------------------------------------------------------
# Email send failure is swallowed
# ---------------------------------------------------------------------------


def test_email_failure_does_not_suppress_error() -> None:
    prices = make_prices({"EGLN.L": 10})
    with patch(
        "tradingo.notifications.email.send_email", side_effect=ConnectionError("down")
    ):
        with pytest.raises(DataQualityError):
            check_price_staleness(
                prices,
                universe_name="test",
                ffill_limit=10,
                alert_recipient="a@b.com",
            )
