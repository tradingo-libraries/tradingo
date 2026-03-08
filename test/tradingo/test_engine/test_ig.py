"""Tests for tradingo.engine.ig - IG position management."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingo.engine.ig import (
    adjust_position_sizes,
    close_all_open_position,
    close_position,
    get_currency,
    get_current_positions,
    reduce_open_positions,
)


@pytest.fixture
def mock_ig_service() -> MagicMock:
    service = MagicMock()
    service.session = MagicMock()
    return service


@pytest.fixture
def instruments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiry": ["DFB", "DFB"],
            "name": ["Test £ Asset", "Test $ Asset"],
        },
        index=pd.Index(["IX.D.FTSE.DAILY.IP", "IX.D.DOW.DAILY.IP"], name="epic"),
    )


class TestGetCurrentPositions:

    def test_returns_positions_with_signed_size(
        self, mock_ig_service: MagicMock
    ) -> None:
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

    def test_closes_buy_position_with_sell(self, mock_ig_service: MagicMock) -> None:
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
        position = pd.Series({"direction": "BUY", "size": 5.0})

        close_position(
            deal_id="DEAL001", position=position, svc=mock_ig_service, size=2.0
        )

        call_kwargs = mock_ig_service.close_open_position.call_args[1]
        assert call_kwargs["size"] == 2.0


class TestCloseAllOpenPosition:

    def test_closes_all_positions_for_epic(self, mock_ig_service: MagicMock) -> None:
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

    def test_reduces_position_across_multiple_deals(
        self, mock_ig_service: MagicMock
    ) -> None:
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP", "IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001", "DEAL002"],
                "direction": ["BUY", "BUY"],
                "size": [2.0, 3.0],
            }
        )

        reduce_open_positions(mock_ig_service, epic="IX.D.FTSE.DAILY.IP", quantity=4)

        assert mock_ig_service.close_open_position.call_count >= 1


class TestGetCurrency:

    def test_usd_currency(self) -> None:
        assert get_currency(pd.Series(name="Test $ Asset")) == "USD"

    def test_gbp_currency(self) -> None:
        assert get_currency(pd.Series(name="Test £ Asset")) == "GBP"

    def test_eur_currency(self) -> None:
        assert get_currency(pd.Series(name="Test € Asset")) == "EUR"

    def test_default_gbp(self) -> None:
        assert get_currency(pd.Series(name="Test Asset")) == "GBP"


class TestAdjustPositionSizes:

    @pytest.fixture
    def target_positions(self) -> pd.DataFrame:
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
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {"epic": [], "dealId": [], "direction": [], "size": []}
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

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
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [2.0],
            }
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, -3.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.close_open_position.assert_called()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 3.0

    def test_increase_existing_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [2.0],
            }
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 5.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.close_open_position.assert_not_called()
        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "BUY"
        assert call_kwargs["size"] == 3.0  # 5 - 2 = 3

    def test_increase_short_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["SELL"],
                "size": [2.0],
            }
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, -5.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        call_kwargs = mock_ig_service.create_open_position.call_args[1]
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 3.0

    def test_decrease_existing_position(
        self,
        mock_ig_service: MagicMock,
        instruments: pd.DataFrame,
    ) -> None:
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [5.0],
            }
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.create_open_position.assert_not_called()
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
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {
                "epic": ["IX.D.FTSE.DAILY.IP"],
                "dealId": ["DEAL001"],
                "direction": ["BUY"],
                "size": [3.0],
            }
        )
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
        mock_ig_service.fetch_open_positions.return_value = pd.DataFrame(
            {"epic": [], "dealId": [], "direction": [], "size": []}
        )
        target = pd.DataFrame(
            {"IX.D.FTSE.DAILY.IP": [0.0, 1.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        adjust_position_sizes(instruments, target, mock_ig_service)

        mock_ig_service.session.close.assert_called_once()
