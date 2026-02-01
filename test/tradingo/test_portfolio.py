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

from tradingo.portfolio import (
    aggregate_portfolio,
    apply_dealing_rules,
    portfolio_construction,
)

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
            "instrument.lotSize": [1, 1, 1, 1],
            "dealingRules.minDealSize.value": [0, 0, 0, 0],
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Model weights directly scale signal contributions."""
        start_date = date_range[0]
        end_date = date_range[-1]

        # Single model with weight 1.0
        result_1 = portfolio_construction(
            instruments=instruments_df,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """AUM directly scales the number of shares."""
        start_date = date_range[0]
        end_date = date_range[-1]

        small_aum = portfolio_construction(
            instruments=instruments_df,
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=50_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        large_aum = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Function returns exactly 7 DataFrames."""
        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """pct_position is normalized by dividing by sum of positions.

        The normalization uses: positions.div(positions.transpose().sum())
        This means each row's positions are divided by their sum.
        """
        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """share_position = (pct_position * aum) / close."""
        aum = 100_000.0
        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Verify rounding is applied correctly to outputs."""
        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Positions are forward-filled from signal values."""
        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Handle signals with zero values (no position)."""
        lib = arctic.get_library("signals", create_if_missing=True)
        zero_signal = pd.DataFrame(0.0, index=date_range, columns=close_prices.columns)
        lib.write("zero", zero_signal)

        result = portfolio_construction(
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Model weights don't need to sum to 1."""
        # Weights summing to more than 1 (leverage)
        result_high = portfolio_construction(
            instruments=instruments_df,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
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
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Can override the signals library name via kwargs."""
        # Create a custom library
        custom_lib = arctic.get_library("custom-signals", create_if_missing=True)
        signal = pd.DataFrame(2.0, index=date_range, columns=close_prices.columns)
        custom_lib.write("custom", signal)

        # Use custom library name (the lib_provider decorator allows this)
        result = portfolio_construction(  # type: ignore
            instruments=instruments_df,
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
        instruments_df: pd.DataFrame,
    ) -> None:
        """Document that multiplier does not affect output.

        The multiplier parameter creates a weights Series, but this
        weights Series is never applied to positions or signals.
        """
        start_date = date_range[0]
        end_date = date_range[-1]

        result_1x = portfolio_construction(
            instruments=instruments_df,
            arctic=arctic,
            close=close_prices,
            model_weights={"momentum": 1.0},
            multiplier=1.0,
            aum=100_000.0,
            start_date=start_date,
            end_date=end_date,
        )

        result_2x = portfolio_construction(
            instruments=instruments_df,
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
            instruments=instruments_df,
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


class TestApplyDealingRules:
    """Tests for apply_dealing_rules function.

    This function applies dealing rules to positions:
    - Rounds positions to lot size multiples
    - Filters out trades smaller than minimum deal size
    """

    @pytest.mark.parametrize(
        "theoretical_position,expected_position,lot_size,min_deal_size",
        [
            # Basic lot size rounding (lot_size=2, min_deal=0)
            pytest.param(
                [0.0, 2.0, 4.0, 6.0],
                [0.0, 2.0, 4.0, 6.0],
                2.0,
                0.0,
                id="already_aligned_to_lot_size",
            ),
            pytest.param(
                [0.0, 2.5, 4.7, 6.3],
                [0.0, 2.0, 4.0, 6.0],
                2.0,
                0.0,
                id="round_down_to_lot_size",
            ),
            pytest.param(
                [0.0, 2.6, 4.8, 6.6],
                [0.0, 2.0, 4.0, 6.0],
                2.0,
                0.0,
                id="round_nearest_to_lot_size",
            ),
            pytest.param(
                [0.0, 3.0, 5.0, 7.0],
                [0.0, 4.0, 4.0, 8.0],
                2.0,
                0.0,
                id="round_half_to_even",
            ),
            # Lot size = 1 (no rounding effect)
            pytest.param(
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                1.0,
                0.0,
                id="lot_size_1_no_rounding",
            ),
            # Minimum deal size filtering (lot_size=1, min_deal=2)
            # Trades < min_deal are filtered, position stays the same
            pytest.param(
                [0.0, 3.0, 4.0, 7.0],
                [0.0, 3.0, 3.0, 7.0],
                1.0,
                2.0,
                id="filter_small_trade_in_middle",
            ),
            pytest.param(
                [0.0, 5.0, 6.0, 7.0],
                [0.0, 5.0, 5.0, 7.0],
                1.0,
                2.0,
                id="small_trade_filtered_but_accumulated_trade_executes",
            ),
            pytest.param(
                [0.0, 1.0, 2.0, 5.0],
                [0.0, 0.0, 2.0, 5.0],
                1.0,
                2.0,
                id="accumulated_trade_executes_when_threshold_met",
            ),
            # Combined lot size and min deal size
            pytest.param(
                [0.0, 2.5, 4.5, 8.5],
                [0.0, 2.0, 4.0, 8.0],
                2.0,
                2.0,
                id="lot_rounding_and_min_deal",
            ),
            pytest.param(
                [0.0, 2.5, 4.5, 6.5],
                [0.0, 2.0, 4.0, 6.0],
                2.0,
                2.0,
                id="each_trade_meets_threshold",
            ),
            # First row filtering (banker's rounding: 0.5->0, 1.5->2, 2.5->2, 3.5->4)
            pytest.param(
                [1.0, 3.0, 5.0, 7.0],
                [0.0, 4.0, 4.0, 8.0],
                2.0,
                3.0,
                id="first_row_below_threshold_stays_zero",
            ),
            pytest.param(
                [4.0, 6.0, 8.0, 10.0],
                [4.0, 4.0, 8.0, 8.0],
                2.0,
                3.0,
                id="small_trades_filtered_large_trades_execute",
            ),
            # Negative positions (short selling)
            pytest.param(
                [0.0, -2.0, -4.0, -6.0],
                [0.0, -2.0, -4.0, -6.0],
                2.0,
                0.0,
                id="negative_positions_lot_aligned",
            ),
            pytest.param(
                [0.0, -5.0, -6.0, -10.0],
                [0.0, -4.0, -6.0, -10.0],
                2.0,
                2.0,
                id="negative_positions_lot_rounded_and_traded",
            ),
        ],
    )
    def test_apply_dealing_rules_single_instrument(
        self,
        theoretical_position: list[float],
        expected_position: list[float],
        lot_size: float,
        min_deal_size: float,
    ) -> None:
        """Test dealing rules on a single instrument."""

        dates = pd.date_range("2024-01-01", periods=len(theoretical_position))
        positions = pd.DataFrame({"SYM": theoretical_position}, index=dates)
        instruments = pd.DataFrame(
            {
                "instrument.lotSize": [lot_size],
                "dealingRules.minDealSize.value": [min_deal_size],
            },
            index=pd.Index(["SYM"], name="Symbol"),
        )

        result = apply_dealing_rules(positions, instruments)

        expected = pd.DataFrame({"SYM": expected_position}, index=dates)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_dealing_rules_multiple_instruments(self) -> None:
        """Test dealing rules applied independently per instrument."""

        dates = pd.date_range("2024-01-01", periods=4)
        positions = pd.DataFrame(
            {
                "A": [0.0, 2.5, 4.5, 6.5],  # lot_size=2, min_deal=0
                "B": [0.0, 1.0, 3.0, 5.0],  # lot_size=1, min_deal=2
            },
            index=dates,
        )
        instruments = pd.DataFrame(
            {
                "instrument.lotSize": [2.0, 1.0],
                "dealingRules.minDealSize.value": [0.0, 2.0],
            },
            index=pd.Index(["A", "B"], name="Symbol"),
        )

        result = apply_dealing_rules(positions, instruments)

        expected = pd.DataFrame(
            {
                "A": [0.0, 2.0, 4.0, 6.0],  # rounded to lot_size=2
                "B": [0.0, 0.0, 3.0, 5.0],  # first trade filtered (1 < min_deal=2)
            },
            index=dates,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_dealing_rules_missing_instruments(self) -> None:
        """Missing instruments default to lot_size=1, min_deal=0."""

        dates = pd.date_range("2024-01-01", periods=3)
        positions = pd.DataFrame({"MISSING": [0.0, 1.5, 2.5]}, index=dates)
        instruments = pd.DataFrame(
            {
                "instrument.lotSize": [2.0],
                "dealingRules.minDealSize.value": [1.0],
            },
            index=pd.Index(["OTHER"], name="Symbol"),
        )

        result = apply_dealing_rules(positions, instruments)

        # With lot_size=1 (default), positions stay as-is rounded to nearest int
        expected = pd.DataFrame({"MISSING": [0.0, 2.0, 2.0]}, index=dates)
        pd.testing.assert_frame_equal(result, expected)


class TestStopLoss:
    """Tests for stop_loss function.

    The stop_loss function calculates stop loss price levels and applies them
    to positions. When price moves adversely by more than max_percent from entry,
    the position is closed (set to 0).
    """

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Intraday timestamps for a single day."""
        return pd.DatetimeIndex(
            [
                "2024-01-01 09:00",
                "2024-01-01 10:00",
                "2024-01-01 11:00",
                "2024-01-01 12:00",
                "2024-01-01 13:00",
            ]
        )

    def test_long_position_stop_level_calculation(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Stop level for long position is entry * (1 - max_percent)."""
        from tradingo.portfolio import stop_loss

        # Open long position at t=1 when bid is 100
        position = pd.DataFrame({"SYM": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 98.0, 96.0, 94.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 100.0, 98.0, 96.0]}, index=dates)

        _, stop_levels = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Entry at bid=100, stop level should be 100 * (1 - 0.05) = 95
        assert stop_levels["SYM"].iloc[1] == 95.0
        # Stop level should persist (ffill within day)
        assert stop_levels["SYM"].iloc[4] == 95.0

    def test_short_position_stop_level_calculation(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Stop level for short position is entry * (1 + max_percent)."""
        from tradingo.portfolio import stop_loss

        # Open short position at t=1 when ask is 102
        position = pd.DataFrame({"SYM": [0.0, -1.0, -1.0, -1.0, -1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 102.0, 104.0, 106.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 104.0, 106.0, 108.0]}, index=dates)

        _, stop_levels = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Entry at ask=102, stop level should be 102 * (1 + 0.05) = 107.1
        assert stop_levels["SYM"].iloc[1] == pytest.approx(107.1)

    def test_long_position_not_stopped_when_price_above_stop(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Long position kept when bid stays above stop level."""
        from tradingo.portfolio import stop_loss

        # Entry at 100, stop at 95 (5%), bid stays at 96 (above stop)
        position = pd.DataFrame({"SYM": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 98.0, 96.0, 96.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 100.0, 98.0, 98.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Position should be maintained (bid 96 >= stop 95)
        pd.testing.assert_frame_equal(stop_lossed_position, position)

    def test_long_position_stopped_when_price_below_stop(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Long position closed when bid falls below stop level."""
        from tradingo.portfolio import stop_loss

        # Entry at 100, stop at 95 (5%), bid falls to 94 (below stop)
        position = pd.DataFrame({"SYM": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 98.0, 94.0, 92.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 100.0, 96.0, 94.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Position should be zeroed from t=3 onwards (bid 94 < stop 95)
        expected = pd.DataFrame({"SYM": [0.0, 1.0, 1.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(stop_lossed_position, expected)

    def test_short_position_not_stopped_when_price_below_stop(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Short position kept when ask stays below stop level."""
        from tradingo.portfolio import stop_loss

        # Entry at ask=102, stop at 107.1 (5%), ask stays at 106 (below stop)
        position = pd.DataFrame({"SYM": [0.0, -1.0, -1.0, -1.0, -1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 102.0, 104.0, 104.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 104.0, 106.0, 106.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Position should be maintained (ask 106 <= stop 107.1)
        pd.testing.assert_frame_equal(stop_lossed_position, position)

    def test_short_position_stopped_when_price_above_stop(
        self, dates: pd.DatetimeIndex
    ) -> None:
        """Short position closed when ask rises above stop level."""
        from tradingo.portfolio import stop_loss

        # Entry at ask=102, stop at 107.1 (5%), ask rises to 108 (above stop)
        position = pd.DataFrame({"SYM": [0.0, -1.0, -1.0, -1.0, -1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 102.0, 106.0, 106.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 104.0, 108.0, 108.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # Position should be zeroed from t=3 onwards (ask 108 > stop 107.1)
        expected = pd.DataFrame({"SYM": [0.0, -1.0, -1.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(stop_lossed_position, expected)

    def test_flat_position_unchanged(self, dates: pd.DatetimeIndex) -> None:
        """Flat positions (zero) are not affected by stop loss."""
        from tradingo.portfolio import stop_loss

        position = pd.DataFrame({"SYM": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [100.0, 50.0, 25.0, 10.0, 5.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [102.0, 52.0, 27.0, 12.0, 7.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        pd.testing.assert_frame_equal(stop_lossed_position, position)

    def test_multiple_instruments(self, dates: pd.DatetimeIndex) -> None:
        """Stop loss applied independently per instrument."""
        from tradingo.portfolio import stop_loss

        # SYM_A: long, gets stopped; SYM_B: short, not stopped
        position = pd.DataFrame(
            {
                "SYM_A": [0.0, 1.0, 1.0, 1.0, 1.0],
                "SYM_B": [0.0, -1.0, -1.0, -1.0, -1.0],
            },
            index=dates,
        )
        bid = pd.DataFrame(
            {
                "SYM_A": [99.0, 100.0, 98.0, 94.0, 92.0],
                "SYM_B": [99.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )
        ask = pd.DataFrame(
            {
                "SYM_A": [101.0, 102.0, 100.0, 96.0, 94.0],
                "SYM_B": [101.0, 102.0, 102.0, 102.0, 102.0],
            },
            index=dates,
        )

        stop_lossed_position, _ = stop_loss(
            position,
            bid,
            ask,
            aum=10000.0,
            max_percent=0.05,
            mode="entry-price",
        )

        # SYM_A: stopped at t=3 (bid 94 < stop 95)
        # SYM_B: not stopped (ask 102 <= stop 107.1)
        expected = pd.DataFrame(
            {
                "SYM_A": [0.0, 1.0, 1.0, 0.0, 0.0],
                "SYM_B": [0.0, -1.0, -1.0, -1.0, -1.0],
            },
            index=dates,
        )
        pd.testing.assert_frame_equal(stop_lossed_position, expected)

    def test_max_drawdown_long_not_stopped(self, dates: pd.DatetimeIndex) -> None:
        """Long position kept when drawdown stays within threshold."""
        from tradingo.portfolio import stop_loss

        # AUM=1000, max_percent=0.10 means max allowed loss is 100
        # Position of 1 unit, price drops from 100 to 95 = loss of 5 (within threshold)
        position = pd.DataFrame({"SYM": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 98.0, 96.0, 95.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 100.0, 98.0, 97.0]}, index=dates)

        stop_lossed_position, _ = stop_loss(
            position, bid, ask, aum=1000.0, max_percent=0.10, mode="max-drawdown"
        )

        # Loss is only 5 (mid drops from 101 to 96), threshold is 100
        pd.testing.assert_frame_equal(stop_lossed_position, position)

    def test_max_drawdown_long_stopped(self, dates: pd.DatetimeIndex) -> None:
        """Long position closed when drawdown exceeds threshold."""
        from tradingo.portfolio import stop_loss

        # AUM=1000, max_percent=0.05 means max allowed loss is 50
        # Position of 10 units, price drops from 100 to 90 = loss of 100 (exceeds threshold)
        position = pd.DataFrame({"SYM": [0.0, 10.0, 10.0, 10.0, 10.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 98.0, 92.0, 88.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 100.0, 94.0, 90.0]}, index=dates)
        # Mid prices: 100, 101, 99, 93, 89

        stop_lossed_position, _ = stop_loss(
            position, bid, ask, aum=1000.0, max_percent=0.05, mode="max-drawdown"
        )

        # At t=3: cumulative loss = (99-101)*10 + (93-99)*10 = -20 + -60 = -80 > -50 threshold
        # Position should be stopped
        expected = pd.DataFrame({"SYM": [0.0, 10.0, 10.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(stop_lossed_position, expected)

    def test_max_drawdown_short_stopped(self, dates: pd.DatetimeIndex) -> None:
        """Short position closed when drawdown exceeds threshold."""
        from tradingo.portfolio import stop_loss

        # Short position of -10 units, price rises = loss
        # AUM=1000, max_percent=0.05 means max allowed loss is 50
        position = pd.DataFrame({"SYM": [0.0, -10.0, -10.0, -10.0, -10.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 102.0, 108.0, 112.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 104.0, 110.0, 114.0]}, index=dates)
        # Mid prices: 100, 101, 103, 109, 113

        stop_lossed_position, _ = stop_loss(
            position, bid, ask, aum=1000.0, max_percent=0.05, mode="max-drawdown"
        )

        # For short, rising prices = loss
        # At t=3: cumulative loss = (103-101)*(-10) + (109-103)*(-10) = -20 + -60 = -80 > -50 threshold
        expected = pd.DataFrame({"SYM": [0.0, -10.0, -10.0, 0.0, 0.0]}, index=dates)
        pd.testing.assert_frame_equal(stop_lossed_position, expected)

    def test_max_drawdown_stop_level_calculation(self, dates: pd.DatetimeIndex) -> None:
        """Stop level is price at which drawdown threshold would be hit."""
        from tradingo.portfolio import stop_loss

        # Position of 10 units at mid price 101, AUM=1000, max_percent=0.05
        # Threshold = -50, current PnL = 0 at t=1
        # Stop level = 101 + (-50 - 0) / 10 = 101 - 5 = 96
        position = pd.DataFrame({"SYM": [0.0, 10.0, 10.0, 10.0, 10.0]}, index=dates)
        bid = pd.DataFrame({"SYM": [99.0, 100.0, 100.0, 100.0, 100.0]}, index=dates)
        ask = pd.DataFrame({"SYM": [101.0, 102.0, 102.0, 102.0, 102.0]}, index=dates)

        _, stop_levels = stop_loss(
            position, bid, ask, aum=1000.0, max_percent=0.05, mode="max-drawdown"
        )

        # At t=1: mid=101, pnl=0 (first period), stop = 101 + (-50-0)/10 = 96
        # Note: pnl at t=1 is 0 because diff() gives NaN for first row
        assert stop_levels["SYM"].iloc[1] == pytest.approx(96.0, abs=0.1)


class TestAggregatePortfolio:
    """Tests for aggregate_portfolio function.

    The aggregate_portfolio function combines multiple model positions into a single
    portfolio by applying model weights and gearing, aggregated to instrument level:

        result[instrument] = gearing * sum(model_weight[i] * model_positions[i][instrument])

    These tests use declarative inputs (prices, model_weights, gearing, models)
    and compare directly to expected positions at instrument level.
    """

    @pytest.fixture
    def simple_dates(self) -> pd.DatetimeIndex:
        """Simple date index for tests."""
        return pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            name="DateTime",
        )

    @pytest.mark.parametrize(
        "prices_data,model_weights,gearing,models_data,expected_data",
        [
            # Single model, unit weight, unit gearing - passthrough
            pytest.param(
                {"A": [100.0, 101.0, 102.0]},
                {"model1": 1.0},
                1.0,
                {"model1": {"A": [1.0, 2.0, 3.0]}},
                {"A": [1.0, 2.0, 3.0]},
                id="single_model_unit_weight_unit_gearing",
            ),
            # Single model with weight scaling
            pytest.param(
                {"A": [100.0, 101.0, 102.0]},
                {"model1": 0.5},
                1.0,
                {"model1": {"A": [2.0, 4.0, 6.0]}},
                {"A": [1.0, 2.0, 3.0]},
                id="single_model_half_weight",
            ),
            # Single model with gearing
            pytest.param(
                {"A": [100.0, 101.0, 102.0]},
                {"model1": 1.0},
                2.0,
                {"model1": {"A": [1.0, 2.0, 3.0]}},
                {"A": [2.0, 4.0, 6.0]},
                id="single_model_double_gearing",
            ),
            # Single model with weight and gearing combined
            # gearing=2.0, weight=0.5: result = 2.0 * 0.5 * [4,8,12] = [4,8,12]
            pytest.param(
                {"A": [100.0, 101.0, 102.0]},
                {"model1": 0.5},
                2.0,
                {"model1": {"A": [4.0, 8.0, 12.0]}},
                {"A": [4.0, 8.0, 12.0]},
                id="single_model_weight_and_gearing",
            ),
            # Two models with equal weights - positions are summed
            # model1: A=[2,4,6] * 0.5 = [1,2,3]
            # model2: A=[4,2,0] * 0.5 = [2,1,0]
            # sum: [3,3,3]
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"model1": 0.5, "model2": 0.5},
                1.0,
                {
                    "model1": {"A": [2.0, 4.0, 6.0]},
                    "model2": {"A": [4.0, 2.0, 0.0]},
                },
                {"A": [3.0, 3.0, 3.0]},
                id="two_models_equal_weights",
            ),
            # Two models with unequal weights
            # momentum: A=10 * 0.7 = 7
            # value: A=10 * 0.3 = 3
            # sum: 10
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"momentum": 0.7, "value": 0.3},
                1.0,
                {
                    "momentum": {"A": [10.0, 10.0, 10.0]},
                    "value": {"A": [10.0, 10.0, 10.0]},
                },
                {"A": [10.0, 10.0, 10.0]},
                id="two_models_70_30_split",
            ),
            # Multiple instruments
            pytest.param(
                {"A": [100.0, 100.0, 100.0], "B": [50.0, 50.0, 50.0]},
                {"model1": 1.0},
                1.0,
                {"model1": {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]}},
                {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]},
                id="multiple_instruments_single_model",
            ),
            # Multiple instruments, multiple models
            # A: 10*0.6 + 10*0.4 = 10
            # B: 5*0.6 + 5*0.4 = 5
            pytest.param(
                {"A": [100.0, 100.0], "B": [50.0, 50.0]},
                {"model1": 0.6, "model2": 0.4},
                1.0,
                {
                    "model1": {"A": [10.0, 10.0], "B": [5.0, 5.0]},
                    "model2": {"A": [10.0, 10.0], "B": [5.0, 5.0]},
                },
                {"A": [10.0, 10.0], "B": [5.0, 5.0]},
                id="multiple_instruments_multiple_models",
            ),
            # Zero weight model excluded
            # active: 5*1.0 = 5, inactive: 100*0.0 = 0, sum = 5
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"active": 1.0, "inactive": 0.0},
                1.0,
                {
                    "active": {"A": [5.0, 5.0, 5.0]},
                    "inactive": {"A": [100.0, 100.0, 100.0]},
                },
                {"A": [5.0, 5.0, 5.0]},
                id="zero_weight_model",
            ),
            # Negative positions (short)
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"model1": 1.0},
                1.0,
                {"model1": {"A": [-1.0, -2.0, -3.0]}},
                {"A": [-1.0, -2.0, -3.0]},
                id="negative_positions_short",
            ),
            # Zero gearing (no positions)
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"model1": 1.0},
                0.0,
                {"model1": {"A": [5.0, 10.0, 15.0]}},
                {"A": [0.0, 0.0, 0.0]},
                id="zero_gearing",
            ),
            # Negative gearing (inverse positions)
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"model1": 1.0},
                -1.0,
                {"model1": {"A": [1.0, 2.0, 3.0]}},
                {"A": [-1.0, -2.0, -3.0]},
                id="negative_gearing",
            ),
            # Fractional positions
            # 0.5 * 0.25 * [8,16,24] = [1,2,3]
            pytest.param(
                {"A": [100.0, 100.0, 100.0]},
                {"model1": 0.25},
                0.5,
                {"model1": {"A": [8.0, 16.0, 24.0]}},
                {"A": [1.0, 2.0, 3.0]},
                id="fractional_weight_and_gearing",
            ),
            # Three models
            # momentum: 10*0.5=5, value: 10*0.3=3, quality: 10*0.2=2, sum=10
            pytest.param(
                {"A": [100.0, 100.0]},
                {"momentum": 0.5, "value": 0.3, "quality": 0.2},
                1.0,
                {
                    "momentum": {"A": [10.0, 10.0]},
                    "value": {"A": [10.0, 10.0]},
                    "quality": {"A": [10.0, 10.0]},
                },
                {"A": [10.0, 10.0]},
                id="three_models",
            ),
            # Weights summing to more than 1 (leverage)
            # model1: 5*1.0=5, model2: 5*1.0=5, sum=10
            pytest.param(
                {"A": [100.0, 100.0]},
                {"model1": 1.0, "model2": 1.0},
                1.0,
                {
                    "model1": {"A": [5.0, 5.0]},
                    "model2": {"A": [5.0, 5.0]},
                },
                {"A": [10.0, 10.0]},
                id="leveraged_weights",
            ),
            # Opposing models cancel out
            # model1: 10*0.5=5, model2: -10*0.5=-5, sum=0
            pytest.param(
                {"A": [100.0, 100.0]},
                {"model1": 0.5, "model2": 0.5},
                1.0,
                {
                    "model1": {"A": [10.0, 10.0]},
                    "model2": {"A": [-10.0, -10.0]},
                },
                {"A": [0.0, 0.0]},
                id="opposing_models_cancel",
            ),
            # Gearing with multiple models
            # gearing=2: 2 * (10*0.6 + 10*0.4) = 2 * 10 = 20
            pytest.param(
                {"A": [100.0, 100.0]},
                {"model1": 0.6, "model2": 0.4},
                2.0,
                {
                    "model1": {"A": [10.0, 10.0]},
                    "model2": {"A": [10.0, 10.0]},
                },
                {"A": [20.0, 20.0]},
                id="gearing_with_multiple_models",
            ),
        ],
    )
    def test_aggregate_portfolio_parametrized(
        self,
        simple_dates: pd.DatetimeIndex,
        prices_data: dict[str, list[float]],
        model_weights: dict[str, float],
        gearing: float,
        models_data: dict[str, dict[str, list[float]]],
        expected_data: dict[str, list[float]],
    ) -> None:
        """Parametrized test for aggregate_portfolio with various inputs."""
        # Build input DataFrames
        dates = simple_dates[: len(next(iter(prices_data.values())))]
        prices = pd.DataFrame(prices_data, index=dates)

        models = {
            name: pd.DataFrame(data, index=dates) for name, data in models_data.items()
        }

        # Build expected DataFrame with instrument columns
        expected = pd.DataFrame(expected_data, index=dates)

        # Run function
        result = aggregate_portfolio(
            prices=prices,
            model_weights=model_weights,
            gearing=gearing,
            **models,
        )

        # Compare
        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected.sort_index(axis=1),
        )

    def test_aggregate_portfolio_preserves_index(
        self, simple_dates: pd.DatetimeIndex
    ) -> None:
        """Output preserves the input DataFrame index."""
        prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=simple_dates)
        model1 = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=simple_dates)

        result = aggregate_portfolio(
            prices=prices,
            model_weights={"model1": 1.0},
            gearing=1.0,
            model1=model1,
        )

        pd.testing.assert_index_equal(result.index, simple_dates)

    def test_aggregate_portfolio_column_structure(
        self, simple_dates: pd.DatetimeIndex
    ) -> None:
        """Output has instrument columns (not MultiIndex)."""
        prices = pd.DataFrame({"A": [100.0], "B": [50.0]}, index=simple_dates[:1])
        model1 = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=simple_dates[:1])
        model2 = pd.DataFrame({"A": [3.0], "B": [4.0]}, index=simple_dates[:1])

        result = aggregate_portfolio(
            prices=prices,
            model_weights={"model1": 1.0, "model2": 1.0},
            gearing=1.0,
            model1=model1,
            model2=model2,
        )

        # Columns should be instrument names, not MultiIndex
        assert not isinstance(result.columns, pd.MultiIndex)
        assert set(result.columns) == {"A", "B"}

    def test_aggregate_portfolio_empty_models(
        self, simple_dates: pd.DatetimeIndex
    ) -> None:
        """Handles case with no models passed."""
        prices = pd.DataFrame({"A": [100.0]}, index=simple_dates[:1])

        result = aggregate_portfolio(
            prices=prices,
            model_weights={},
            gearing=1.0,
        )

        assert result.empty

    def test_aggregate_portfolio_prices_not_used_in_calculation(
        self, simple_dates: pd.DatetimeIndex
    ) -> None:
        """Prices parameter doesn't affect position calculation.

        The prices parameter is available for potential future use but
        currently doesn't influence the output positions.
        """
        model1 = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=simple_dates)

        result_high_prices = aggregate_portfolio(
            prices=pd.DataFrame({"A": [1000.0, 1000.0, 1000.0]}, index=simple_dates),
            model_weights={"model1": 1.0},
            gearing=1.0,
            model1=model1,
        )

        result_low_prices = aggregate_portfolio(
            prices=pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=simple_dates),
            model_weights={"model1": 1.0},
            gearing=1.0,
            model1=model1,
        )

        pd.testing.assert_frame_equal(result_high_prices, result_low_prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
