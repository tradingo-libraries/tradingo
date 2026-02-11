"""Tests for tradingo.engine module - position management."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingo.engine import (
    adjust_position_sizes,
    close_all_open_position,
    close_position,
    get_currency,
    get_current_positions,
    reduce_open_positions,
)


@pytest.fixture
def mock_ig_service() -> MagicMock:
    """Create a mock IGService."""
    service = MagicMock()
    service.session = MagicMock()
    return service


@pytest.fixture
def instruments() -> pd.DataFrame:
    """Sample instruments dataframe."""
    return pd.DataFrame(
        {
            "expiry": ["DFB", "DFB"],
            "name": ["Test £ Asset", "Test $ Asset"],
        },
        index=pd.Index(["IX.D.FTSE.DAILY.IP", "IX.D.DOW.DAILY.IP"], name="epic"),
    )


class TestGetCurrentPositions:
    """Tests for get_current_positions function."""

    def test_returns_positions_with_signed_size(
        self, mock_ig_service: MagicMock
    ) -> None:
        """Test that positions are returned with correctly signed sizes."""
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP", "IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001", "DEAL002"],
                "direction": ["BUY", "SELL"],
                "size": [2.0, 1.0],
            }
        )

        result = get_current_positions(mock_ig_service)

        assert result.loc[("IX.D.FTSE.DAILY.IP", "DEAL001"), "size"] == 2.0
        assert result.loc[("IX.D.FTSE.DAILY.IP", "DEAL002"), "size"] == -1.0


class TestClosePosition:
    """Tests for close_position function."""

    def test_closes_buy_position_with_sell(self, mock_ig_service: MagicMock) -> None:
        """Test that a BUY position is closed with a SELL order."""
        position = pd.Series({"direction": "BUY", "size": 2.0})

        close_position(deal_id="DEAL001", position=position, svc=mock_ig_service)

        mock_ig_service.close_open_position.assert_called_once_with(
            deal_id="DEAL001",
            direction="SELL",
            epic=None,
            expiry=None,
            level=None,
            order_type="MARKET",
            size=2.0,
            quote_id=None,
        )

    def test_closes_sell_position_with_buy(self, mock_ig_service: MagicMock) -> None:
        """Test that a SELL position is closed with a BUY order."""
        position = pd.Series({"direction": "SELL", "size": -3.0})

        close_position(deal_id="DEAL002", position=position, svc=mock_ig_service)

        mock_ig_service.close_open_position.assert_called_once_with(
            deal_id="DEAL002",
            direction="BUY",
            epic=None,
            expiry=None,
            level=None,
            order_type="MARKET",
            size=3.0,
            quote_id=None,
        )

    def test_closes_partial_position(self, mock_ig_service: MagicMock) -> None:
        """Test partial position close with explicit size."""
        position = pd.Series({"direction": "BUY", "size": 5.0})

        close_position(
            deal_id="DEAL001", position=position, svc=mock_ig_service, size=2.0
        )

        mock_ig_service.close_open_position.assert_called_once()
        call_kwargs = mock_ig_service.close_open_position.call_args[1]
        assert call_kwargs["size"] == 2.0


class TestCloseAllOpenPosition:
    """Tests for close_all_open_position function."""

    def test_closes_all_positions_for_epic(self, mock_ig_service: MagicMock) -> None:
        """Test that all positions for an epic are closed."""
        positions = pd.DataFrame(
            {
                "direction": ["BUY", "BUY"],
                "size": [2.0, 3.0],
            },
            index=pd.Index(["DEAL001", "DEAL002"], name="dealId"),
        )

        close_all_open_position(positions, mock_ig_service)

        assert mock_ig_service.close_open_position.call_count == 2


class TestReduceOpenPositions:
    """Tests for reduce_open_positions function."""

    def test_reduces_position_across_multiple_deals(
        self, mock_ig_service: MagicMock
    ) -> None:
        """Test reducing positions spread across multiple deals."""
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP", "IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001", "DEAL002"],
                "direction": ["BUY", "BUY"],
                "size": [2.0, 3.0],
            }
        )

        reduce_open_positions(mock_ig_service, epic="IX.D.FTSE.DAILY.IP", quantity=4)

        # Should close positions starting from smallest
        assert mock_ig_service.close_open_position.call_count >= 1


class TestGetCurrency:
    """Tests for get_currency function."""

    def test_usd_currency(self) -> None:
        """Test USD currency detection."""
        instrument = pd.Series(name="Test $ Asset")
        assert get_currency(instrument) == "USD"

    def test_gbp_currency(self) -> None:
        """Test GBP currency detection."""
        instrument = pd.Series(name="Test £ Asset")
        assert get_currency(instrument) == "GBP"

    def test_eur_currency(self) -> None:
        """Test EUR currency detection."""
        instrument = pd.Series(name="Test € Asset")
        assert get_currency(instrument) == "EUR"

    def test_default_gbp(self) -> None:
        """Test default to GBP when no currency symbol found."""
        instrument = pd.Series(name="Test Asset")
        assert get_currency(instrument) == "GBP"


class TestAdjustPositionSizes:
    """Tests for adjust_position_sizes - the main position management function."""

    @pytest.fixture
    def target_positions(self) -> pd.DataFrame:
        """Sample target positions dataframe."""
        return pd.DataFrame(
            {
                "IX.D.FTSE.DAILY.IP": [0.0, 3.0],
                "IX.D.DOW.DAILY.IP": [0.0, 2.0],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

    def test_open_new_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test opening a new position when no current position exists."""
        # No existing positions
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {"epic": [], "dealId": [], "direction": [], "size": []}
        )

        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        # Should create a new position
        mock_ig_service.create_open_position.assert_called_once()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "BUY"
        assert call_kwargs["size"] == 3.0
        assert call_kwargs["epic"] == "IX.D.FTSE.DAILY.IP"

    def test_open_short_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test opening a new short position."""
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {"epic": [], "dealId": [], "direction": [], "size": []}
        )

        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, -2.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 2.0

    def test_close_existing_position_when_changing_sides(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test that existing position is closed when changing from long to short."""
        # Existing long position
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [2.0],
            }
        )

        # Target is short
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, -3.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        # Should close existing position first
        mock_ig_service.close_open_position.assert_called()
        # Then open new short position
        mock_ig_service.create_open_position.assert_called()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 3.0

    def test_increase_existing_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test increasing an existing position - adds a new position order."""
        # Existing position of 2
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [2.0],
            }
        )

        # Target is 5 (increase by 3)
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 5.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        # Should not close any position
        mock_ig_service.close_open_position.assert_not_called()
        # Should create new position for the difference
        mock_ig_service.create_open_position.assert_called_once()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "BUY"
        assert call_kwargs["size"] == 3.0  # 5 - 2 = 3

    def test_increase_short_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test increasing a short position (more negative)."""
        # Existing short position of -2
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["SELL"],
                "size": [2.0],
            }
        )

        # Target is -5 (increase short by 3)
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, -5.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.create_open_position.assert_called_once()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 3.0

    def test_decrease_existing_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test decreasing an existing position - partially closes."""
        # Existing position of 5
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [5.0],
            }
        )

        # Target is 2 (decrease by 3)
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        # Should not create new position
        mock_ig_service.create_open_position.assert_not_called()
        # Should close part of existing position
        mock_ig_service.close_open_position.assert_called()

    def test_decrease_position_across_multiple_deals(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test decreasing position when split across multiple deals."""
        # Two existing positions totaling 5
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP", "IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001", "DEAL002"],
                "direction": ["BUY", "BUY"],
                "size": [2.0, 3.0],
            }
        )

        # Target is 1 (decrease by 4, closing multiple positions)
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 1.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        # Should close positions to reduce from 5 to 1
        assert mock_ig_service.close_open_position.call_count >= 1

    def test_no_change_when_position_matches_target(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test that no action is taken when position matches target."""
        # Existing position of 3
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [3.0],
            }
        )

        # Target is also 3
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.create_open_position.assert_not_called()
        mock_ig_service.close_open_position.assert_not_called()

    def test_session_closed_after_adjustment(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        """Test that the session is closed after adjustments."""
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {"epic": [], "dealId": [], "direction": [], "size": []}
        )

        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 1.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.session.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
