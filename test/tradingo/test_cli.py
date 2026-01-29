"""Tests for the Tradingo CLI."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingo.api import Tradingo
from tradingo.cli import cli_app, handle_tasks, handle_universes, int_or_bool
from tradingo.dag import DAG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.bdate_range(start="2024-01-01", periods=10, tz="utc")
    return pd.DataFrame(
        {"AAPL": range(100, 110), "MSFT": range(200, 210)},
        index=dates,
    )


@pytest.fixture
def arctic_mem() -> Tradingo:
    """Create an in-memory ArcticDB instance for testing."""
    t = Tradingo(uri="mem://cli-test")
    for lib in ["prices", "signals", "instruments", "portfolio"]:
        t.create_library(lib)
    return t


@pytest.fixture
def simple_config() -> dict[str, Any]:
    """A minimal config with simple testable tasks."""
    return {
        "stage1": {
            "task_a": {
                "function": "test.tradingo.test_cli.dummy_task",
                "depends_on": [],
                "params": {"name": "task_a"},
                "symbols_in": {},
                "symbols_out": [],
            },
            "task_b": {
                "function": "test.tradingo.test_cli.dummy_task",
                "depends_on": ["task_a"],
                "params": {"name": "task_b"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
        "stage2": {
            "task_c": {
                "function": "test.tradingo.test_cli.dummy_task",
                "depends_on": ["task_b"],
                "params": {"name": "task_c"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
    }


@pytest.fixture
def config_with_symbols(
    arctic_mem: Tradingo, sample_prices: pd.DataFrame
) -> dict[str, Any]:
    """Config with symbol I/O for testing data flow."""
    # Populate test data
    arctic_mem.get_library("prices", create_if_missing=True).write(  # type: ignore
        "input_data", sample_prices
    )

    return {
        "pipeline": {
            "read_task": {
                "function": "test.tradingo.test_cli.transform_prices",
                "depends_on": [],
                "params": {"multiplier": 2},
                "symbols_in": {"prices": "prices/input_data"},
                "symbols_out": ["prices/output_data"],
            },
            "aggregate_task": {
                "function": "test.tradingo.test_cli.aggregate_prices",
                "depends_on": ["read_task"],
                "params": {},
                "symbols_in": {"prices": "prices/output_data"},
                "symbols_out": ["signals/aggregated"],
            },
        },
    }


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    """Create a temporary state file path."""
    state_dir = tmp_path / ".tradingo"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "dag-state.json"


# ---------------------------------------------------------------------------
# Dummy task functions for testing
# ---------------------------------------------------------------------------


# Track task executions for verification
_executed_tasks: list[dict[str, Any]] = []


def dummy_task(
    name: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    dry_run: bool = False,
    clean: bool = False,
    **kwargs: Any,
) -> None:
    """A dummy task that records its execution."""
    _executed_tasks.append(
        {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "dry_run": dry_run,
            "clean": clean,
            "kwargs": kwargs,
        }
    )


def transform_prices(
    prices: pd.DataFrame,
    multiplier: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transform prices by multiplying."""
    return prices * multiplier


def aggregate_prices(
    prices: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Aggregate prices to a single mean column."""
    return prices.mean(axis=1).to_frame(name="mean")


# ---------------------------------------------------------------------------
# Tests for int_or_bool helper
# ---------------------------------------------------------------------------


class TestIntOrBool:
    """Tests for the int_or_bool argument parser."""

    def test_true_values(self) -> None:
        assert int_or_bool("true") is True
        assert int_or_bool("True") is True
        assert int_or_bool("TRUE") is True
        assert int_or_bool("yes") is True
        assert int_or_bool("Yes") is True

    def test_false_values(self) -> None:
        assert int_or_bool("false") is False
        assert int_or_bool("False") is False
        assert int_or_bool("FALSE") is False
        assert int_or_bool("no") is False
        assert int_or_bool("No") is False

    def test_integer_values(self) -> None:
        assert int_or_bool("0") == 0
        assert int_or_bool("1") == 1
        assert int_or_bool("5") == 5
        assert int_or_bool("100") == 100

    def test_invalid_values(self) -> None:
        with pytest.raises(ValueError):
            int_or_bool("invalid")
        with pytest.raises(ValueError):
            int_or_bool("maybe")


# ---------------------------------------------------------------------------
# Tests for CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Tests for CLI argument parser structure."""

    def test_cli_app_requires_config(self) -> None:
        """Config argument is required."""
        parser = cli_app()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_cli_app_requires_entity(self) -> None:
        """Entity subcommand is required."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            with pytest.raises(SystemExit):
                parser.parse_args(["--config", "dummy.yaml"])

    def test_task_list_parsing(self) -> None:
        """Parse task list command."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(["--config", "dummy.yaml", "task", "list"])
        assert args.entity == "task"
        assert args.list_action == "list"

    def test_task_run_parsing(self) -> None:
        """Parse task run command with all options."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(
                [
                    "--config",
                    "dummy.yaml",
                    "task",
                    "run",
                    "my_task",
                    "--with-deps",
                    "true",
                    "--start-date",
                    "2024-01-01",
                    "--end-date",
                    "2024-01-31",
                    "--force-rerun",
                    "--dry-run",
                    "--clean",
                    "--skip-deps",
                    "task_.*",
                ]
            )

        assert args.entity == "task"
        assert args.list_action == "run"
        assert args.task == "my_task"
        assert args.with_deps is True
        assert args.start_date == pd.Timestamp("2024-01-01")
        assert args.end_date == pd.Timestamp("2024-01-31")
        assert args.force_rerun is True
        assert args.dry_run is True
        assert args.clean is True
        assert args.skip_deps.pattern == "task_.*"

    def test_task_run_with_deps_as_int(self) -> None:
        """Parse --with-deps as integer for depth control."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(
                [
                    "--config",
                    "dummy.yaml",
                    "task",
                    "run",
                    "my_task",
                    "--with-deps",
                    "3",
                ]
            )
        assert args.with_deps == 3

    def test_universe_list_parsing(self) -> None:
        """Parse universe list command."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(["--config", "dummy.yaml", "universe", "list"])
        assert args.entity == "universe"
        assert args.universe_action == "list"

    def test_universe_show_parsing(self) -> None:
        """Parse universe show command with name argument."""
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(
                ["--config", "dummy.yaml", "universe", "show", "etfs"]
            )
        assert args.entity == "universe"
        assert args.universe_action == "show"
        assert args.name == "etfs"


# ---------------------------------------------------------------------------
# Tests for task list command
# ---------------------------------------------------------------------------


class TestTaskListCommand:
    """Tests for the task list command."""

    def test_task_list_prints_all_tasks(
        self, simple_config: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Task list should print all tasks with their dependencies."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="list",
        )
        arctic = MagicMock()
        handle_tasks(args, arctic)

        captured = capsys.readouterr()
        output = captured.out

        # Verify all tasks are listed
        assert "task_a:" in output
        assert "task_b:" in output
        assert "task_c:" in output

        # Verify dependencies are shown
        assert "- task_a" in output  # task_b depends on task_a
        assert "- task_b" in output  # task_c depends on task_b

    def test_task_list_empty_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Task list with empty config produces no output."""
        args = argparse.Namespace(
            config={},
            entity="task",
            list_action="list",
        )
        arctic = MagicMock()
        handle_tasks(args, arctic)

        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Tests for task run command
# ---------------------------------------------------------------------------


class TestTaskRunCommand:
    """Tests for the task run command."""

    def setup_method(self) -> None:
        """Clear executed tasks before each test."""
        _executed_tasks.clear()

    def test_run_single_task(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a single task without dependencies."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        # Patch state file location
        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        assert len(_executed_tasks) == 1
        assert _executed_tasks[0]["name"] == "task_a"

    def test_run_task_with_deps(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task with its dependencies."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_c",
            with_deps=2,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        # All three tasks should execute in order
        assert len(_executed_tasks) == 3
        executed_names = [t["name"] for t in _executed_tasks]
        assert executed_names == ["task_a", "task_b", "task_c"]

    def test_run_task_with_deps_depth_limit(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task with limited dependency depth."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_c",
            with_deps=1,  # Only run direct dependencies (task_b)
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        # Only task_b (depth 1) and task_c should run
        assert len(_executed_tasks) == 2
        executed_names = [t["name"] for t in _executed_tasks]
        assert executed_names == ["task_b", "task_c"]

    def test_run_task_with_start_end_date(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task with start and end date parameters."""
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=start,
            end_date=end,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        assert len(_executed_tasks) == 1
        assert _executed_tasks[0]["start_date"] == start
        assert _executed_tasks[0]["end_date"] == end

    def test_run_task_with_dry_run(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task with dry_run flag."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=True,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        assert len(_executed_tasks) == 1
        assert _executed_tasks[0]["dry_run"] is True

    def test_run_task_with_clean(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task with clean flag."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=True,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        assert len(_executed_tasks) == 1
        assert _executed_tasks[0]["clean"] is True

    def test_run_task_with_skip_deps(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Run a task but skip certain dependencies."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_c",
            with_deps=True,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=re.compile("task_a"),  # Skip task_a
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        # task_a should be skipped
        executed_names = [t["name"] for t in _executed_tasks]
        assert "task_a" not in executed_names
        assert "task_b" in executed_names
        assert "task_c" in executed_names

    def test_run_nonexistent_task(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Running a nonexistent task should raise an error."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="nonexistent_task",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            with pytest.raises(ValueError, match="nonexistent_task is not a task"):
                handle_tasks(args, arctic)

    def test_run_task_respects_state(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Tasks with SUCCESS state should be skipped unless force_rerun."""
        state_file = tmp_path / "dag-state.json"
        state_file.write_text(json.dumps({"task_a": "SUCCESS"}))

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=False,  # Don't force rerun
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", state_file):
            handle_tasks(args, arctic)

        # Task should not execute because it was already successful
        assert len(_executed_tasks) == 0

    def test_state_is_serialized_after_run(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """DAG state should be saved after task execution."""
        state_file = tmp_path / "dag-state.json"

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="run",
            task="task_a",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", state_file):
            handle_tasks(args, arctic)

        # State file should exist and contain task_a as SUCCESS
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["task_a"] == "SUCCESS"

    def test_state_serialized_on_failure(self, tmp_path: Path) -> None:
        """DAG state should be saved even if task fails."""
        config = {
            "stage": {
                "failing_task": {
                    "function": "test.tradingo.test_cli.failing_task",
                    "depends_on": [],
                    "params": {},
                    "symbols_in": {},
                    "symbols_out": [],
                },
            },
        }
        state_file = tmp_path / "dag-state.json"

        args = argparse.Namespace(
            config=config,
            entity="task",
            list_action="run",
            task="failing_task",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", state_file):
            with pytest.raises(RuntimeError, match="Task failed"):
                handle_tasks(args, arctic)

        # State should still be serialized
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["failing_task"] == "FAILED"


def failing_task(**kwargs: Any) -> None:
    """A task that always fails."""
    raise RuntimeError("Task failed")


# ---------------------------------------------------------------------------
# Tests for task execution with symbol I/O
# ---------------------------------------------------------------------------


class TestTaskRunWithSymbols:
    """Tests for running tasks that read/write symbols."""

    def test_run_task_reads_symbols(
        self,
        config_with_symbols: dict[str, Any],
        arctic_mem: Tradingo,
        sample_prices: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Task should read input symbols and write output symbols."""
        args = argparse.Namespace(
            config=config_with_symbols,
            entity="task",
            list_action="run",
            task="read_task",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic_mem)

        # Verify output was written
        output = arctic_mem.prices.output_data()
        expected = sample_prices * 2
        pd.testing.assert_frame_equal(output, expected, check_freq=False)

    def test_run_pipeline_with_deps(
        self,
        config_with_symbols: dict[str, Any],
        arctic_mem: Tradingo,
        sample_prices: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Run full pipeline with dependencies."""
        args = argparse.Namespace(
            config=config_with_symbols,
            entity="task",
            list_action="run",
            task="aggregate_task",
            with_deps=True,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic_mem)

        # Verify intermediate output
        output = arctic_mem.prices.output_data()
        expected_intermediate = sample_prices * 2
        pd.testing.assert_frame_equal(output, expected_intermediate, check_freq=False)

        # Verify final aggregated output
        aggregated = arctic_mem.signals.aggregated()
        expected_aggregated = (sample_prices * 2).mean(axis=1).to_frame(name="mean")
        pd.testing.assert_frame_equal(aggregated, expected_aggregated, check_freq=False)

    def test_run_task_with_date_filter(
        self,
        config_with_symbols: dict[str, Any],
        arctic_mem: Tradingo,
        sample_prices: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Task should filter data by start/end date when reading symbols."""
        start = pd.Timestamp("2024-01-03").tz_localize("utc")
        end = pd.Timestamp("2024-01-08").tz_localize("utc")

        args = argparse.Namespace(
            config=config_with_symbols,
            entity="task",
            list_action="run",
            task="read_task",
            with_deps=False,
            start_date=start,
            end_date=end,
            force_rerun=True,
            dry_run=False,
            clean=True,  # Clean to ensure we only have filtered data
            skip_deps=None,
        )

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic_mem)

        # The output should only contain data within the date range
        output = arctic_mem.prices.output_data()
        assert output.index.min() >= start
        assert output.index.max() <= end


# ---------------------------------------------------------------------------
# Tests for universe commands
# ---------------------------------------------------------------------------


class TestUniverseCommands:
    """Tests for universe list and show commands."""

    def test_universe_list(
        self, arctic_mem: Tradingo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Universe list should print available instruments."""
        # Create some instrument data
        arctic_mem.instruments.etfs.update(
            pd.DataFrame({"name": ["ETF1", "ETF2"]}, index=["ETF1", "ETF2"]),
            upsert=True,
        )
        arctic_mem.instruments.stocks.update(
            pd.DataFrame({"name": ["AAPL", "MSFT"]}, index=["AAPL", "MSFT"]),
            upsert=True,
        )

        args = argparse.Namespace(
            entity="universe",
            universe_action="list",
        )
        handle_universes(args, arctic_mem)

        captured = capsys.readouterr()
        assert "etfs" in captured.out
        assert "stocks" in captured.out

    def test_universe_show(
        self, arctic_mem: Tradingo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Universe show should print specific instrument data."""
        instrument_data = pd.DataFrame(
            {"name": ["Apple", "Microsoft"], "sector": ["Tech", "Tech"]},
            index=["AAPL", "MSFT"],
        )
        arctic_mem.instruments.tech.update(instrument_data, upsert=True)

        args = argparse.Namespace(
            entity="universe",
            universe_action="show",
            name="tech",
        )
        handle_universes(args, arctic_mem)

        captured = capsys.readouterr()
        assert "Apple" in captured.out
        assert "Microsoft" in captured.out

    def test_universe_invalid_action(self, arctic_mem: Tradingo) -> None:
        """Invalid universe action should raise ValueError."""
        args = argparse.Namespace(
            entity="universe",
            universe_action="invalid",
        )
        with pytest.raises(ValueError, match="invalid"):
            handle_universes(args, arctic_mem)


# ---------------------------------------------------------------------------
# Integration test with config file
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Integration tests using actual config files."""

    def test_end_to_end_task_run(self, tmp_path: Path) -> None:
        """End-to-end test: create config file, run CLI, verify results."""
        # Create a config file
        config_content = """
stage1:
  compute:
    function: test.tradingo.test_cli.compute_sum
    depends_on: []
    params:
      a: 10
      b: 20
    symbols_in: {}
    symbols_out: []
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Reset tracking
        _computed_results.clear()

        # Create args as if parsed from CLI
        import os

        from tradingo.config import read_config_template

        config = read_config_template(config_file, os.environ)

        args = argparse.Namespace(
            config=config,
            entity="task",
            list_action="run",
            task="compute",
            with_deps=False,
            start_date=None,
            end_date=None,
            force_rerun=True,
            dry_run=False,
            clean=False,
            skip_deps=None,
        )
        arctic = MagicMock()

        with patch.object(DAG, "_state_filepath", tmp_path / "dag-state.json"):
            handle_tasks(args, arctic)

        # Verify the task was executed with correct params
        assert len(_computed_results) == 1
        assert _computed_results[0] == {"a": 10, "b": 20, "result": 30}


_computed_results: list[dict[str, Any]] = []


def compute_sum(a: int, b: int, **kwargs: Any) -> None:
    """Compute sum of a and b for testing."""
    _computed_results.append({"a": a, "b": b, "result": a + b})


# ---------------------------------------------------------------------------
# Tests for invalid configurations
# ---------------------------------------------------------------------------


class TestConfigurationErrors:
    """Tests for configuration error handling."""

    def test_invalid_list_action(self, simple_config: dict[str, Any]) -> None:
        """Invalid list_action should raise ValueError."""
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="invalid_action",
        )
        arctic = MagicMock()

        with pytest.raises(ValueError, match="invalid_action"):
            handle_tasks(args, arctic)

    def test_missing_dependency_in_config(self) -> None:
        """Config with missing dependency should raise ConfigLoadError."""
        from tradingo.config import ConfigLoadError

        config = {
            "stage": {
                "task_with_missing_dep": {
                    "function": "test.tradingo.test_cli.dummy_task",
                    "depends_on": ["nonexistent_dependency"],
                    "params": {},
                    "symbols_in": {},
                    "symbols_out": [],
                },
            },
        }

        with pytest.raises(ConfigLoadError, match="Missing task"):
            DAG.from_config(config)

    def test_missing_function_in_config(self) -> None:
        """Config with missing function key should raise ConfigLoadError."""
        from tradingo.config import ConfigLoadError

        config: dict[str, Any] = {
            "stage": {
                "task_missing_function": {
                    # "function" key is missing
                    "depends_on": [],
                    "params": {},
                },
            },
        }

        with pytest.raises(ConfigLoadError, match="missing setting function"):
            DAG.from_config(config)

    def test_disabled_tasks_excluded(self) -> None:
        """Tasks with enabled=False should be excluded from DAG."""
        config = {
            "stage": {
                "enabled_task": {
                    "function": "test.tradingo.test_cli.dummy_task",
                    "depends_on": [],
                    "params": {},
                    "enabled": True,
                },
                "disabled_task": {
                    "function": "test.tradingo.test_cli.dummy_task",
                    "depends_on": [],
                    "params": {},
                    "enabled": False,
                },
            },
        }

        dag = DAG.from_config(config)
        assert "enabled_task" in dag
        assert "disabled_task" not in dag


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
