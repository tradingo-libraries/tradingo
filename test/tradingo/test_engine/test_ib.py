"""Tests for tradingo.engine.ib - IB position management.

ib_insync is an optional dependency, so we stub it in sys.modules before
importing the module under test. All IB interaction is mocked.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub ib_insync so the module can be imported without the package installed
# ---------------------------------------------------------------------------
if "ib_insync" not in sys.modules:
    sys.modules["ib_insync"] = MagicMock()

from tradingo.engine.ib import (  # noqa: E402
    adjust_position_sizes,
    close_all_open_position,
    close_position,
    get_current_positions,
    reduce_open_positions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_order_args() -> tuple[str, float]:
    """Return (action, quantity) from the most recent MarketOrder() call."""
    call = sys.modules["ib_insync"].MarketOrder.call_args
    return str(call.args[0]), float(call.args[1])


def _make_position(symbol: str, currency: str, size: float) -> MagicMock:
    """Build a mock ib_insync Position object."""
    pos = MagicMock()
    pos.contract.symbol = symbol
    pos.contract.currency = currency
    pos.position = size
    pos.avgCost = 150.0
    return pos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_ib_insync_mocks() -> None:
    """Reset ib_insync class mocks between tests so call_args is fresh."""
    sys.modules["ib_insync"].MarketOrder.reset_mock()
    sys.modules["ib_insync"].Stock.reset_mock()


@pytest.fixture
def mock_trade() -> MagicMock:
    trade = MagicMock()
    trade.isDone.return_value = True
    trade.orderStatus.status = "Filled"
    return trade


@pytest.fixture
def mock_ib(mock_trade: MagicMock) -> MagicMock:
    ib = MagicMock()
    ib.placeOrder.return_value = mock_trade
    return ib


@pytest.fixture
def instruments() -> pd.DataFrame:
    return pd.DataFrame(
        {"currency": ["USD", "USD"]},
        index=pd.Index(["AAPL", "MSFT"], name="symbol"),
    )


# ---------------------------------------------------------------------------
# TestGetCurrentPositions
# ---------------------------------------------------------------------------


class TestGetCurrentPositions:
    def test_returns_empty_dataframe_when_no_positions(
        self, mock_ib: MagicMock
    ) -> None:
        mock_ib.positions.return_value = []
        assert get_current_positions(mock_ib).empty

    def test_long_position_has_positive_size(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 100.0)]
        result = get_current_positions(mock_ib)
        assert result.loc[("AAPL", "USD"), "size"] == 100.0
        assert result.loc[("AAPL", "USD"), "direction"] == "BUY"

    def test_short_position_has_negative_size(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = [_make_position("MSFT", "USD", -50.0)]
        result = get_current_positions(mock_ib)
        assert result.loc[("MSFT", "USD"), "size"] == -50.0
        assert result.loc[("MSFT", "USD"), "direction"] == "SELL"

    def test_avg_price_is_not_divided_by_position(self, mock_ib: MagicMock) -> None:
        """avgCost from IB is already per-share — must not be divided again."""
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 10.0)]
        result = get_current_positions(mock_ib)
        assert result.loc[("AAPL", "USD"), "avg_price"] == 150.0

    def test_filters_by_account(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = []
        get_current_positions(mock_ib, account="U123456")
        mock_ib.positions.assert_called_once_with("U123456")

    def test_multi_symbol_positions(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = [
            _make_position("AAPL", "USD", 100.0),
            _make_position("MSFT", "USD", -50.0),
        ]
        result = get_current_positions(mock_ib)
        assert len(result) == 2
        assert result.loc[("AAPL", "USD"), "size"] == 100.0
        assert result.loc[("MSFT", "USD"), "size"] == -50.0


# ---------------------------------------------------------------------------
# TestClosePosition
# ---------------------------------------------------------------------------


class TestClosePosition:
    def test_closes_buy_position_with_sell_order(self, mock_ib: MagicMock) -> None:
        position = pd.Series({"direction": "BUY", "size": 100.0})
        close_position(contract=MagicMock(), position=position, ib=mock_ib)
        action, qty = _market_order_args()
        assert action == "SELL"
        assert qty == 100.0

    def test_closes_sell_position_with_buy_order(self, mock_ib: MagicMock) -> None:
        position = pd.Series({"direction": "SELL", "size": -50.0})
        close_position(contract=MagicMock(), position=position, ib=mock_ib)
        action, qty = _market_order_args()
        assert action == "BUY"
        assert qty == 50.0

    def test_partial_close_uses_explicit_size(self, mock_ib: MagicMock) -> None:
        position = pd.Series({"direction": "BUY", "size": 100.0})
        close_position(contract=MagicMock(), position=position, ib=mock_ib, size=30.0)
        _, qty = _market_order_args()
        assert qty == 30.0

    def test_waits_for_fill(self, mock_ib: MagicMock, mock_trade: MagicMock) -> None:
        position = pd.Series({"direction": "BUY", "size": 10.0})
        close_position(contract=MagicMock(), position=position, ib=mock_ib)
        mock_ib.runUntil.assert_called_once_with(mock_trade.isDone, timeout=30)


# ---------------------------------------------------------------------------
# TestCloseAllOpenPosition
# ---------------------------------------------------------------------------


class TestCloseAllOpenPosition:
    def test_iterates_multi_index_correctly(self, mock_ib: MagicMock) -> None:
        """Regression: symbol/currency must be unpacked from the index, not row."""
        positions = pd.DataFrame(
            {"direction": ["BUY", "SELL"], "size": [100.0, -50.0]},
            index=pd.MultiIndex.from_tuples(
                [("AAPL", "USD"), ("MSFT", "USD")],
                names=["symbol", "currency"],
            ),
        )
        close_all_open_position(positions, mock_ib)
        assert mock_ib.placeOrder.call_count == 2

    def test_qualifies_contracts_before_closing(self, mock_ib: MagicMock) -> None:
        positions = pd.DataFrame(
            {"direction": ["BUY"], "size": [10.0]},
            index=pd.MultiIndex.from_tuples(
                [("AAPL", "USD")], names=["symbol", "currency"]
            ),
        )
        close_all_open_position(positions, mock_ib)
        mock_ib.qualifyContracts.assert_called_once()


# ---------------------------------------------------------------------------
# TestReduceOpenPositions
# ---------------------------------------------------------------------------


class TestReduceOpenPositions:
    def test_reduces_long_position_with_sell(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 100.0)]
        reduce_open_positions(mock_ib, symbol="AAPL", currency="USD", quantity=40)
        action, qty = _market_order_args()
        assert action == "SELL"
        assert qty == 40.0

    def test_reduces_short_position_with_buy(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", -80.0)]
        reduce_open_positions(mock_ib, symbol="AAPL", currency="USD", quantity=30)
        action, qty = _market_order_args()
        assert action == "BUY"
        assert qty == 30.0

    def test_caps_reduction_at_position_size(self, mock_ib: MagicMock) -> None:
        """Cannot reduce more than the current position."""
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 20.0)]
        reduce_open_positions(mock_ib, symbol="AAPL", currency="USD", quantity=50)
        _, qty = _market_order_args()
        assert qty == 20.0  # capped at position size

    def test_no_order_when_position_not_found(self, mock_ib: MagicMock) -> None:
        mock_ib.positions.return_value = []
        reduce_open_positions(mock_ib, symbol="AAPL", currency="USD", quantity=10)
        mock_ib.placeOrder.assert_not_called()


# ---------------------------------------------------------------------------
# TestAdjustPositionSizes
# ---------------------------------------------------------------------------


class TestAdjustPositionSizes:
    def test_opens_new_long_position(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        mock_ib.positions.return_value = []
        target = pd.DataFrame(
            {"AAPL": [0.0, 100.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        action, qty = _market_order_args()
        assert action == "BUY"
        assert qty == 100.0

    def test_opens_new_short_position(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        mock_ib.positions.return_value = []
        target = pd.DataFrame(
            {"AAPL": [0.0, -50.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        action, qty = _market_order_args()
        assert action == "SELL"
        assert qty == 50.0

    def test_increases_existing_position(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 30.0)]
        target = pd.DataFrame(
            {"AAPL": [0.0, 50.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        action, qty = _market_order_args()
        assert action == "BUY"
        assert qty == 20.0  # 50 - 30

    def test_closes_side_then_opens_opposite(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        """Flipping from long to short: close existing then open short."""
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 30.0)]
        target = pd.DataFrame(
            {"AAPL": [0.0, -20.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        # placeOrder called twice: close long + open short
        assert mock_ib.placeOrder.call_count >= 2

    def test_no_order_when_position_matches_target(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        mock_ib.positions.return_value = [_make_position("AAPL", "USD", 100.0)]
        target = pd.DataFrame(
            {"AAPL": [0.0, 100.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        mock_ib.placeOrder.assert_not_called()

    def test_disconnects_after_adjustment(
        self, mock_ib: MagicMock, instruments: pd.DataFrame
    ) -> None:
        mock_ib.positions.return_value = []
        target = pd.DataFrame(
            {"AAPL": [0.0, 0.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        adjust_position_sizes(instruments, target, ib=mock_ib)
        mock_ib.disconnect.assert_called_once()
