"""Tests for portfolio construction functions.

This module provides comprehensive tests for the portfolio_construction function,
illustrating what each parameter achieves and how they interact.

NOTE: The current implementation computes `weights` from `multiplier` and
`instrument_weights` but does not apply them to the final positions.
These tests document the actual behavior.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from arcticdb import Arctic

from tradingo.portfolio import portfolio_construction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arctic() -> Arctic:
    """Create an in-memory ArcticDB instance."""
    return Arctic("mem://portfolio-test")


@pytest.fixture
def date_range() -> pd.DatetimeIndex:
    """Standard date range for tests."""
    return pd.bdate_range(start="2024-01-01", periods=20, tz="UTC")


@pytest.fixture
def symbols() -> list[str]:
    """Standard symbol list for tests."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN"]


@pytest.fixture
def close_prices(date_range: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Sample close prices for testing.

    Creates a DataFrame with realistic price movements:
    - AAPL: starts at 100, trends up
    - MSFT: starts at 200, relatively flat
    - GOOGL: starts at 150, trends down then up
    - AMZN: starts at 180, volatile
    """
    np.random.seed(42)
    n = len(date_range)

    prices = pd.DataFrame(
        {
            "AAPL": 100 + np.cumsum(np.random.randn(n) * 2 + 0.5),
            "MSFT": 200 + np.cumsum(np.random.randn(n) * 1.5),
            "GOOGL": 150 + np.cumsum(np.random.randn(n) * 2 - 0.2),
            "AMZN": 180 + np.cumsum(np.random.randn(n) * 3),
        },
        index=date_range,
    )
    return prices


@pytest.fixture
def momentum_signal(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """A momentum signal: positive for trending up, negative for trending down.

    Signal values represent conviction:
    - AAPL: strong positive (trending up)
    - MSFT: weak positive (flat but slightly up)
    - GOOGL: negative (trending down)
    - AMZN: mixed (volatile)
    """
    n = len(date_range)
    return pd.DataFrame(
        {
            "AAPL": np.linspace(0.5, 1.0, n),  # Increasing conviction
            "MSFT": np.full(n, 0.3),  # Weak constant signal
            "GOOGL": np.linspace(-0.2, -0.5, n),  # Increasing negative
            "AMZN": np.sin(np.linspace(0, 2 * np.pi, n)) * 0.5,  # Oscillating
        },
        index=date_range,
    )


@pytest.fixture
def value_signal(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """A value signal: identifies undervalued assets.

    Represents a different view than momentum:
    - AAPL: negative (overvalued after run-up)
    - MSFT: positive (fairly valued)
    - GOOGL: strong positive (undervalued)
    - AMZN: weak positive
    """
    n = len(date_range)
    return pd.DataFrame(
        {
            "AAPL": np.full(n, -0.3),  # Overvalued
            "MSFT": np.full(n, 0.4),  # Fair value
            "GOOGL": np.full(n, 0.8),  # Undervalued
            "AMZN": np.full(n, 0.2),  # Slightly undervalued
        },
        index=date_range,
    )


@pytest.fixture
def instruments_df(symbols: list[str]) -> pd.DataFrame:
    """Instrument metadata for testing instrument weights.

    Contains:
    - asset_type: equity or etf
    - sector: tech, consumer, etc.
    - market_cap: large, mid, small
    """
    return pd.DataFrame(
        {
            "asset_type": ["equity", "equity", "equity", "equity"],
            "sector": ["tech", "tech", "tech", "consumer"],
            "market_cap": ["large", "large", "large", "large"],
        },
        index=pd.Index(symbols, name="Symbol"),
    )


@pytest.fixture
def signals_library(
    arctic: Arctic,
    momentum_signal: pd.DataFrame,
    value_signal: pd.DataFrame,
) -> None:
    """Populate the signals library with test data."""
    lib = arctic.get_library("signals", create_if_missing=True)
    lib.write("momentum", momentum_signal)
    lib.write("value", value_signal)


# ---------------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------------


class TestBasicPortfolioConstruction:
    """Tests for basic portfolio construction functionality."""

    def test_single_model_equal_weights(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
        momentum_signal: pd.DataFrame,
    ) -> None:
        """Test portfolio construction with a single signal model.

        When using a single model with weight 1.0, the output positions
        should directly reflect the signal values.
        """
        start_date = date_range[0]
        end_date = date_range[-1]
        aum = 100_000.0

        (
            pct_position,
            share_position,
            positions,
            _pct_rounded,
            _share_rounded,
            _positions_rounded,
            signal_value,
        ) = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=aum,
            start_date=start_date,
            end_date=end_date,
        )

        # All outputs should have the same shape as close prices
        assert pct_position.shape == close_prices.shape
        assert share_position.shape == close_prices.shape
        assert positions.shape == close_prices.shape

        # signal_value should match the momentum signal (with weight 1.0)
        # Column order may differ due to groupby, so sort columns for comparison
        # Note: signal_value index may be subset of close_prices index
        expected_signal = momentum_signal.ffill().fillna(0)
        common_index = signal_value.index.intersection(expected_signal.index)
        pd.testing.assert_frame_equal(
            signal_value.loc[common_index].sort_index(axis=1),
            expected_signal.loc[common_index].sort_index(axis=1),
        )

        # Percentage positions should sum to ~1 (normalized)
        # First row should be zero (set in function)
        assert (pct_position.iloc[0] == 0).all()

        # Share positions = (pct_position * aum) / close
        expected_shares = (pct_position * aum) / close_prices
        pd.testing.assert_frame_equal(share_position, expected_shares)

    def test_multiple_models_weighted(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
        momentum_signal: pd.DataFrame,
        value_signal: pd.DataFrame,
    ) -> None:
        """Test portfolio construction with multiple weighted models.

        model_weights parameter allows combining multiple signals:
        - {"momentum": 0.6, "value": 0.4} means 60% momentum + 40% value

        The resulting signal is the weighted sum of individual signals.
        """
        start_date = date_range[0]
        end_date = date_range[-1]

        (
            _pct_position,
            _,
            _positions,
            _,
            _,
            _,
            signal_value,
        ) = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 0.6, "value": 0.4},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # The combined signal should be: 0.6 * momentum + 0.4 * value
        expected_signal = (0.6 * momentum_signal + 0.4 * value_signal).ffill().fillna(0)
        # Column order may differ, so sort columns
        # Note: signal_value index may be subset of close_prices index
        common_index = signal_value.index.intersection(expected_signal.index)
        pd.testing.assert_frame_equal(
            signal_value.loc[common_index].sort_index(axis=1),
            expected_signal.loc[common_index].sort_index(axis=1),
        )

        # For GOOGL: momentum is negative, value is positive
        # Combined signal for GOOGL should be less extreme than either
        # momentum: -0.2 to -0.5, value: 0.8
        # Combined: 0.6*(-0.35) + 0.4*0.8 = -0.21 + 0.32 = 0.11 (roughly)
        assert signal_value["GOOGL"].mean() > -0.2  # Less negative than pure momentum


class TestModelWeightsParameter:
    """Tests demonstrating the model_weights parameter effect.

    model_weights specifies how different signal models are combined:
    - {"model_a": 1.0}: Use only model_a
    - {"model_a": 0.5, "model_b": 0.5}: Equal blend
    - {"model_a": 0.7, "model_b": 0.3}: 70/30 blend

    The weights are applied directly to each model's signal before summing.
    """

    def test_model_weights_scale_signals(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Model weights directly scale signal contributions."""
        start_date = date_range[0]
        end_date = date_range[-1]

        # Single model with weight 1.0
        result_1 = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # Same model with weight 2.0
        result_2 = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 2.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # Signal values should be 2x
        pd.testing.assert_frame_equal(
            result_2[6].sort_index(axis=1),  # signal_value
            (result_1[6] * 2).sort_index(axis=1),
        )

        # But percentage positions are normalized, so they should be the same
        pd.testing.assert_frame_equal(
            result_2[0].sort_index(axis=1),  # pct_position
            result_1[0].sort_index(axis=1),
        )


class TestAUMParameter:
    """Tests demonstrating the AUM (Assets Under Management) parameter.

    AUM represents the total capital to allocate. It affects:
    - share_position = (pct_position * aum) / close

    Higher AUM means more shares for the same percentage allocation.
    """

    def test_aum_scales_share_positions(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """AUM directly scales the number of shares."""
        start_date = date_range[0]
        end_date = date_range[-1]

        small_aum = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=50_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        large_aum = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # Share positions should be 2x for double AUM
        pd.testing.assert_frame_equal(
            large_aum[1].sort_index(axis=1),  # share_position
            (small_aum[1] * 2).sort_index(axis=1),
        )

        # Percentage positions should be the same regardless of AUM
        pd.testing.assert_frame_equal(
            large_aum[0].sort_index(axis=1),  # pct_position
            small_aum[0].sort_index(axis=1),
        )


class TestDateRangeFiltering:
    """Tests for start_date and end_date parameters.

    These parameters control which portion of the signal data is used:
    - Signals are read from the library within [start_date, end_date]
    - Useful for backtesting specific periods
    """

    def test_date_range_filters_signal_data(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Date range parameters filter the signal data used.

        When a narrower date range is specified, signals are only read
        within that range. Positions are then forward-filled for dates
        beyond the end_date.
        """
        # Use only first half of the date range
        mid_point = len(date_range) // 2
        start_date = date_range[0]
        end_date = date_range[mid_point]

        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # Positions are reindexed to close_prices and forward-filled
        positions = result[2]

        # After end_date, positions should be constant (forward-filled from last signal)
        last_position_at_end = positions.loc[end_date]
        for date in date_range[mid_point + 1 :]:
            pd.testing.assert_series_equal(
                cast(pd.Series, positions.loc[date]),
                cast(pd.Series, last_position_at_end),
                check_names=False,
            )


class TestOutputStructure:
    """Tests verifying the structure and meaning of each output.

    portfolio_construction returns 7 DataFrames:
    1. pct_position: Percentage allocation per instrument (normalized)
    2. share_position: Number of shares = (pct * aum) / price
    3. positions: Raw position values from combined signals
    4. pct_rounded: pct_position rounded to 2 decimal places
    5. share_rounded: share_position rounded to whole numbers
    6. positions_rounded: positions rounded to whole numbers
    7. signal_value: The raw combined signal before normalization
    """

    def test_output_tuple_length(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Function returns exactly 7 DataFrames."""
        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        assert len(result) == 7
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_pct_position_normalization(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """pct_position is normalized by dividing by sum of positions.

        The normalization uses: positions.div(positions.transpose().sum())
        This means each row's positions are divided by their sum.
        """
        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        pct_position = result[0]

        # First row is explicitly set to 0
        assert (pct_position.iloc[0] == 0).all()

    def test_share_position_calculation(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """share_position = (pct_position * aum) / close."""
        aum = 100_000.0
        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=aum,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        pct_position = result[0]
        share_position = result[1]

        expected_shares = (pct_position * aum) / close_prices
        pd.testing.assert_frame_equal(
            share_position.sort_index(axis=1), expected_shares.sort_index(axis=1)
        )

    def test_rounded_outputs(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Verify rounding is applied correctly to outputs."""
        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        pct_position = result[0]
        share_position = result[1]
        positions = result[2]
        pct_rounded = result[3]
        share_rounded = result[4]
        positions_rounded = result[5]

        # pct_rounded should be pct_position rounded to 2 decimals
        pd.testing.assert_frame_equal(
            pct_rounded.sort_index(axis=1),
            pct_position.round(decimals=2).sort_index(axis=1),
        )

        # share_rounded should be share_position rounded to integers
        pd.testing.assert_frame_equal(
            share_rounded.sort_index(axis=1), share_position.round().sort_index(axis=1)
        )

        # positions_rounded should be positions rounded to integers
        pd.testing.assert_frame_equal(
            positions_rounded.sort_index(axis=1), positions.round().sort_index(axis=1)
        )


class TestPositionsFromSignals:
    """Tests verifying how positions are derived from signals."""

    def test_positions_forward_filled(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Positions are forward-filled from signal values."""
        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        positions = result[2]
        signal_value = result[6]

        # Positions should be signal_value reindexed to close and forward-filled
        expected = signal_value.reindex_like(close_prices).ffill()
        pd.testing.assert_frame_equal(
            positions.sort_index(axis=1), expected.sort_index(axis=1)
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_instrument(
        self,
        arctic: Arctic,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Portfolio with a single instrument."""
        close = pd.DataFrame(
            {"ONLY": range(100, 120)},
            index=date_range,
        )

        # Write single-column signal
        lib = arctic.get_library("signals", create_if_missing=True)
        signal = pd.DataFrame({"ONLY": [1.0] * len(date_range)}, index=date_range)
        lib.write("single", signal)

        result = portfolio_construction(
            arctic=arctic,
            close=close,
            model_weights={"single": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        # Single instrument should get 100% allocation (after first row)
        pct_position = result[0]
        assert (pct_position.iloc[1:]["ONLY"] == 1.0).all()

    def test_zero_signal_values(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Handle signals with zero values (no position)."""
        lib = arctic.get_library("signals", create_if_missing=True)
        zero_signal = pd.DataFrame(0.0, index=date_range, columns=close_prices.columns)
        lib.write("zero", zero_signal)

        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"zero": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        # All positions should be zero
        assert (result[2] == 0).all().all()

    def test_negative_signals_short_positions(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Negative signal values create short positions."""
        lib = arctic.get_library("signals", create_if_missing=True)
        short_signal = pd.DataFrame(
            {
                "AAPL": [-1.0] * len(date_range),  # Short
                "MSFT": [1.0] * len(date_range),  # Long
                "GOOGL": [-0.5] * len(date_range),  # Half short
                "AMZN": [0.5] * len(date_range),  # Half long
            },
            index=date_range,
        )
        lib.write("short", short_signal)

        result = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"short": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        positions = result[2]

        # AAPL should be negative (short)
        assert (positions["AAPL"].iloc[1:] < 0).all()

        # MSFT should be positive (long)
        assert (positions["MSFT"].iloc[1:] > 0).all()

    def test_unbalanced_model_weights(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Model weights don't need to sum to 1."""
        # Weights summing to more than 1 (leverage)
        result_high = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 0.8, "value": 0.8},  # Sum = 1.6
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        # Weights summing to less than 1 (cash position implied)
        result_low = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 0.3, "value": 0.2},  # Sum = 0.5
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        # Higher weights should result in larger signal values
        assert result_high[6].abs().sum().sum() > result_low[6].abs().sum().sum()


class TestIntegrationWithLibProvider:
    """Tests verifying integration with the @lib_provider decorator."""

    def test_signals_library_created_if_missing(
        self,
        close_prices: pd.DataFrame,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """The signals library is created if it doesn't exist."""
        # Fresh Arctic instance without any libraries
        fresh_arctic = Arctic("mem://fresh-test")

        # Create signals library and write data
        lib = fresh_arctic.get_library("signals", create_if_missing=True)
        signal = pd.DataFrame(1.0, index=date_range, columns=close_prices.columns)
        lib.write("test", signal)

        # Should work without errors
        result = portfolio_construction(
            arctic=fresh_arctic,
            close=close_prices,
            model_weights={"test": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
        )

        assert result[0].shape == close_prices.shape

    def test_custom_signals_library_name(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Can override the signals library name via kwargs."""
        # Create a custom library
        custom_lib = arctic.get_library("custom-signals", create_if_missing=True)
        signal = pd.DataFrame(2.0, index=date_range, columns=close_prices.columns)
        custom_lib.write("custom", signal)

        # Use custom library name (the lib_provider decorator allows this)
        result = portfolio_construction(  # type: ignore
            arctic=arctic,
            close=close_prices,
            model_weights={"custom": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=date_range[0],
            end_date=date_range[-1],
            signals="custom-signals",  # Override default library  # type: ignore
        )

        # Should have used the custom signal (value 2.0)
        signal_value = result[6]
        assert (signal_value == 2.0).all().all()


class TestUnusedParametersDocumentation:
    """Tests documenting parameters that are computed but not applied.

    NOTE: The current implementation computes `weights` from `multiplier`
    and `instrument_weights` but does not apply them to the final output.
    These tests document this behavior.
    """

    def test_multiplier_is_computed_but_not_applied(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Document that multiplier does not affect output.

        The multiplier parameter creates a weights Series, but this
        weights Series is never applied to positions or signals.
        """
        start_date = date_range[0]
        end_date = date_range[-1]

        result_1x = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        result_2x = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=2.0,  # Different multiplier
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        # Positions are the same regardless of multiplier
        # (this documents current behavior, not necessarily intended behavior)
        pd.testing.assert_frame_equal(
            result_1x[2].sort_index(axis=1),  # positions
            result_2x[2].sort_index(axis=1),
        )

    def test_instrument_weights_computed_but_not_applied(
        self,
        arctic: Arctic,
        close_prices: pd.DataFrame,
        signals_library: None,
        instruments_df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Document that instrument_weights do not affect output.

        The instrument_weights parameter is used to compute a weights
        Series, but this is never applied to the final positions.
        """
        start_date = date_range[0]
        end_date = date_range[-1]

        result_no_weights = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        result_with_weights = portfolio_construction(
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
            instruments=instruments_df.copy(),
            instrument_weights={"sector": {"tech": 0.5, "consumer": 2.0}},
        )

        # Positions are the same with or without instrument_weights
        # (this documents current behavior, not necessarily intended behavior)
        pd.testing.assert_frame_equal(
            result_no_weights[2].sort_index(axis=1),  # positions
            result_with_weights[2].sort_index(axis=1),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
