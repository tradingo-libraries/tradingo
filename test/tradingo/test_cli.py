"""Tests for the Tradingo CLI.

This module provides comprehensive tests for the CLI including:
- Unit tests for argument parsing and helper functions
- Integration tests for task execution
- Parameterized tests for various CLI scenarios
- Data pipeline tests with symbol I/O
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import yaml
from arcticdb.arctic import Library

from tradingo.api import Tradingo
from tradingo.cli import (
    _print_active_plans,
    cli_app,
    handle_tasks,
    handle_universes,
    int_or_bool,
    parse_interval,
)
from tradingo.dag import DAG
from tradingo.execution_plan import ExecutionPlan

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.bdate_range(start="2024-01-01", periods=30, tz="UTC")
    return pd.DataFrame(
        {
            "AAPL": [100 + i * 0.5 for i in range(30)],
            "MSFT": [200 + i * 0.3 for i in range(30)],
            "GOOGL": [150 - i * 0.2 for i in range(30)],
        },
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
_computed_results: list[dict[str, Any]] = []


def reset_execution_log() -> None:
    """Clear the execution log before each test."""
    _executed_tasks.clear()


def get_execution_log() -> list[dict[str, Any]]:
    """Get the current execution log."""
    return _executed_tasks.copy()


@pytest.fixture(autouse=True)
def reset_logs() -> None:
    """Reset execution logs before each test."""
    _executed_tasks.clear()
    _computed_results.clear()


def dummy_task(
    name: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    **kwargs: Any,
) -> None:
    """A dummy task that records its execution.

    Intentionally does not declare ``dry_run`` / ``snapshot`` / ``clean``:
    those are task-envelope kwargs owned by ``symbol_publisher`` and must
    never reach the inner function. ``**kwargs`` captures any leaks.
    """
    _executed_tasks.append(
        {
            "name": name,
            "task_name": name,  # Alias for compatibility
            "start_date": start_date,
            "end_date": end_date,
            "kwargs": kwargs,
        }
    )


def logging_task(
    task_name: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> None:
    """A task that logs its execution for verification."""
    _executed_tasks.append(
        {
            "name": task_name,
            "task_name": task_name,
            "start_date": start_date,
            "end_date": end_date,
            "dry_run": dry_run,
            "kwargs": kwargs,
        }
    )


def transform_prices(
    prices: pd.DataFrame,
    multiplier: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transform prices by multiplying."""
    _executed_tasks.append(
        {
            "task_name": "transform_prices",
            "multiplier": multiplier,
            "input_shape": prices.shape,
        }
    )
    return prices * multiplier


def multiply_prices(
    prices: pd.DataFrame,
    factor: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Multiply prices by a factor."""
    _executed_tasks.append(
        {
            "task_name": "multiply_prices",
            "factor": factor,
            "input_shape": prices.shape,
        }
    )
    return prices * factor


def aggregate_prices(
    prices: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Aggregate prices to a single mean column."""
    _executed_tasks.append(
        {
            "task_name": "aggregate_prices",
            "input_shape": prices.shape,
        }
    )
    return prices.mean(axis=1).to_frame(name="mean")


def compute_returns(
    prices: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute returns from prices."""
    _executed_tasks.append(
        {
            "task_name": "compute_returns",
            "input_shape": prices.shape,
        }
    )
    return prices.pct_change().dropna()


def aggregate_to_signal(
    returns: pd.DataFrame,
    lookback: int = 5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Aggregate returns to a signal."""
    _executed_tasks.append(
        {
            "task_name": "aggregate_to_signal",
            "lookback": lookback,
            "input_shape": returns.shape,
        }
    )
    return returns.rolling(lookback).mean().dropna()


def failing_task(**kwargs: Any) -> None:
    """A task that always fails."""
    raise RuntimeError("Task failed")


def compute_sum(a: int, b: int, **kwargs: Any) -> None:
    """Compute sum of a and b for testing."""
    _computed_results.append({"a": a, "b": b, "result": a + b})


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


class TestParseInterval:
    """Tests for the parse_interval argument parser."""

    def test_days(self) -> None:
        assert parse_interval("3days") == pd.Timedelta(days=3)
        assert parse_interval("1day") == pd.Timedelta(days=1)

    def test_hours(self) -> None:
        assert parse_interval("2hours") == pd.Timedelta(hours=2)
        assert parse_interval("1h") == pd.Timedelta(hours=1)

    def test_minutes(self) -> None:
        assert parse_interval("30min") == pd.Timedelta(minutes=30)

    def test_weeks(self) -> None:
        assert parse_interval("1w") == pd.Timedelta(weeks=1)
        assert parse_interval("2weeks") == pd.Timedelta(weeks=2)

    def test_months_approximated(self) -> None:
        assert parse_interval("2months") == pd.Timedelta(days=60)
        assert parse_interval("1month") == pd.Timedelta(days=30)
        assert parse_interval("3mo") == pd.Timedelta(days=90)

    def test_with_spaces(self) -> None:
        assert parse_interval("3 days") == pd.Timedelta(days=3)

    def test_invalid_format(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid interval"):
            parse_interval("abc")

    def test_invalid_unit(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="Unrecognized"):
            parse_interval("3foobar")


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
# Tests for configuration errors
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


# ---------------------------------------------------------------------------
# CLI Runner Helper Class
# ---------------------------------------------------------------------------


@dataclass
class CLITestConfig:
    """Configuration for a CLI test run.

    Attributes:
        name: Descriptive name for this test case
        config: DAG configuration dict (will be written to YAML)
        task: Task name to run
        start_date: Optional start date for the task
        end_date: Optional end date for the task
        arctic_uri: Arctic connection URI (defaults to in-memory)
        with_deps: Whether to run dependencies
        dry_run: Whether to run in dry-run mode
        expected_tasks_run: List of task names expected to run
        setup_data: Optional dict of {library/symbol: DataFrame} to pre-populate
        verify_outputs: Optional dict of {library/symbol: callable} to verify outputs
    """

    name: str
    config: dict[str, Any]
    task: str
    start_date: pd.Timestamp | str | None = None
    end_date: pd.Timestamp | str | None = None
    arctic_uri: str = "mem://cli-runner-test"
    with_deps: bool | int = False
    dry_run: bool = False
    expected_tasks_run: list[str] = field(default_factory=list)
    setup_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    verify_outputs: dict[str, Any] = field(default_factory=dict)


class CLIRunner:
    """Helper class to run CLI tests with parameterized configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        arctic_uri: str,
        tmp_path: Path,
    ):
        self.config = config
        self.arctic_uri = arctic_uri
        self.tmp_path = tmp_path
        self.config_path = tmp_path / "config.yaml"
        self.state_path = tmp_path / ".tradingo" / "dag-state.json"
        self._arctic: Tradingo | None = None

    @property
    def arctic(self) -> Tradingo:
        """Get or create Arctic connection."""
        if self._arctic is None:
            self._arctic = Tradingo(uri=self.arctic_uri)
        return self._arctic

    def setup(self, data: dict[str, pd.DataFrame] | None = None) -> None:
        """Set up the test environment.

        Args:
            data: Dict mapping "library/symbol" to DataFrame to pre-populate
        """
        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config to YAML file
        self.config_path.write_text(yaml.dump(self.config))

        # Pre-populate data if provided
        if data:
            for symbol_path, df in data.items():
                lib_name, symbol = symbol_path.split("/")
                lib: Library = self.arctic.get_library(lib_name, create_if_missing=True)  # type: ignore
                lib.write(symbol, df)

    def run_task(
        self,
        task: str,
        start_date: pd.Timestamp | str | None = None,
        end_date: pd.Timestamp | str | None = None,
        with_deps: bool | int = False,
        dry_run: bool = False,
        force_rerun: bool = True,
        clean: bool = False,
        skip_deps: re.Pattern[str] | None = None,
    ) -> None:
        """Run a task via the CLI handler.

        Args:
            task: Task name to run
            start_date: Optional start date
            end_date: Optional end date
            with_deps: Whether to run dependencies (bool or int for depth)
            dry_run: Whether to run in dry-run mode
            force_rerun: Whether to force re-running already successful tasks
            clean: Whether to clean output before writing
            skip_deps: Optional regex pattern to skip dependencies
        """
        # Convert string dates to Timestamps
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)

        args = argparse.Namespace(
            config=self.config,
            entity="task",
            list_action="run",
            task=task,
            with_deps=with_deps,
            start_date=start_date,
            end_date=end_date,
            force_rerun=force_rerun,
            dry_run=dry_run,
            clean=clean,
            skip_deps=skip_deps,
        )

        # Patch the state file location
        with patch.object(
            DAG,
            "_state_filepath",
            new_callable=PropertyMock,
            return_value=self.state_path,
        ):
            handle_tasks(args, self.arctic)

    def read_output(self, symbol_path: str) -> pd.DataFrame:
        """Read output data from Arctic.

        Args:
            symbol_path: Path in format "library/symbol"

        Returns:
            DataFrame from the specified location
        """
        lib_name, symbol = symbol_path.split("/")
        lib: Library = self.arctic.get_library(lib_name)
        return cast(pd.DataFrame, lib.read(symbol).data)

    def list_symbols(self, library: str) -> list[str]:
        """List symbols in a library.

        Args:
            library: Library name

        Returns:
            List of symbol names
        """
        lib: Library = self.arctic.get_library(library)
        return list(lib.list_symbols())


@pytest.fixture
def cli_runner(tmp_path: Path) -> type[CLIRunner]:
    """Factory fixture for creating CLI runners."""

    def _create_runner(
        config: dict[str, Any],
        arctic_uri: str = "mem://cli-runner-test",
    ) -> CLIRunner:
        return CLIRunner(config, arctic_uri, tmp_path)

    return _create_runner  # type: ignore


# ---------------------------------------------------------------------------
# Sample Pipeline Configurations
# ---------------------------------------------------------------------------


def simple_pipeline_config() -> dict[str, Any]:
    """A simple pipeline with three sequential tasks."""
    return {
        "pipeline": {
            "task_1": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": [],
                "params": {"task_name": "task_1"},
                "symbols_in": {},
                "symbols_out": [],
            },
            "task_2": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": ["task_1"],
                "params": {"task_name": "task_2"},
                "symbols_in": {},
                "symbols_out": [],
            },
            "task_3": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": ["task_2"],
                "params": {"task_name": "task_3"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
    }


def data_pipeline_config() -> dict[str, Any]:
    """A data processing pipeline with symbol I/O."""
    return {
        "prices": {
            "multiply": {
                "function": "test.tradingo.test_cli.multiply_prices",
                "depends_on": [],
                "params": {"factor": 2.0},
                "symbols_in": {"prices": "prices/raw"},
                "symbols_out": ["prices/adjusted"],
            },
        },
        "signals": {
            "returns": {
                "function": "test.tradingo.test_cli.compute_returns",
                "depends_on": ["multiply"],
                "params": {},
                "symbols_in": {"prices": "prices/adjusted"},
                "symbols_out": ["signals/returns"],
            },
            "signal": {
                "function": "test.tradingo.test_cli.aggregate_to_signal",
                "depends_on": ["returns"],
                "params": {"lookback": 5},
                "symbols_in": {"returns": "signals/returns"},
                "symbols_out": ["signals/momentum"],
            },
        },
    }


def branching_pipeline_config() -> dict[str, Any]:
    """A pipeline with branching dependencies."""
    return {
        "stage1": {
            "root": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": [],
                "params": {"task_name": "root"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
        "stage2": {
            "branch_a": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": ["root"],
                "params": {"task_name": "branch_a"},
                "symbols_in": {},
                "symbols_out": [],
            },
            "branch_b": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": ["root"],
                "params": {"task_name": "branch_b"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
        "stage3": {
            "merge": {
                "function": "test.tradingo.test_cli.logging_task",
                "depends_on": ["branch_a", "branch_b"],
                "params": {"task_name": "merge"},
                "symbols_in": {},
                "symbols_out": [],
            },
        },
    }


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
# Tests for task run command (unit tests)
# ---------------------------------------------------------------------------


class TestTaskRunCommand:
    """Tests for the task run command."""

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
        """``dry_run`` is owned by the publisher envelope and must not leak
        into the inner function."""
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
        assert "dry_run" not in _executed_tasks[0]["kwargs"]
        assert "snapshot" not in _executed_tasks[0]["kwargs"]

    def test_run_task_with_clean(
        self, simple_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """``clean`` is owned by the publisher envelope and must not leak
        into the inner function."""
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
        assert "clean" not in _executed_tasks[0]["kwargs"]

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


# ---------------------------------------------------------------------------
# Tests for state management
# ---------------------------------------------------------------------------


class TestStateManagement:
    """Tests for DAG state serialization and management."""

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
        start = pd.Timestamp("2024-01-03").tz_localize("UTC")
        end = pd.Timestamp("2024-01-08").tz_localize("UTC")

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
# Parameterized CLI Test Cases
# ---------------------------------------------------------------------------

# NOTE: with_deps should use an integer (e.g., 100) instead of True because
# in Python, bool is a subclass of int, so True - 1 = 0 which is falsy.
# The DAG code decrements the int at each level, so True becomes 0 after one level.
PARAMETERIZED_TEST_CASES: list[CLITestConfig] = [
    CLITestConfig(
        name="simple_single_task",
        config=simple_pipeline_config(),
        task="task_1",
        expected_tasks_run=["task_1"],
    ),
    CLITestConfig(
        name="simple_with_deps",
        config=simple_pipeline_config(),
        task="task_3",
        with_deps=100,  # Use large int for "all deps"
        expected_tasks_run=["task_1", "task_2", "task_3"],
    ),
    CLITestConfig(
        name="simple_depth_limited",
        config=simple_pipeline_config(),
        task="task_3",
        with_deps=1,  # Only run direct dependencies
        expected_tasks_run=["task_2", "task_3"],
    ),
    CLITestConfig(
        name="with_date_range",
        config=simple_pipeline_config(),
        task="task_1",
        start_date="2024-01-01",
        end_date="2024-01-15",
        expected_tasks_run=["task_1"],
    ),
    CLITestConfig(
        name="branching_merge_with_deps",
        config=branching_pipeline_config(),
        task="merge",
        with_deps=100,
        # Note: root runs twice because branch_a and branch_b both depend on it,
        # and force_rerun=True causes each branch to re-run its dependencies
        expected_tasks_run=["root", "branch_a", "root", "branch_b", "merge"],
    ),
    CLITestConfig(
        name="branching_single_branch",
        config=branching_pipeline_config(),
        task="branch_a",
        with_deps=100,
        expected_tasks_run=["root", "branch_a"],
    ),
]


@pytest.mark.parametrize(
    "test_config",
    PARAMETERIZED_TEST_CASES,
    ids=[tc.name for tc in PARAMETERIZED_TEST_CASES],
)
def test_cli_runner_parameterized(
    test_config: CLITestConfig,
    tmp_path: Path,
) -> None:
    """Run parameterized CLI tests."""
    # Deep copy the config to avoid mutations affecting other tests
    config_copy = copy.deepcopy(test_config.config)

    runner = CLIRunner(
        config=config_copy,
        arctic_uri=test_config.arctic_uri,
        tmp_path=tmp_path,
    )

    # Setup
    runner.setup(test_config.setup_data or None)

    # Run task
    runner.run_task(
        task=test_config.task,
        start_date=test_config.start_date,
        end_date=test_config.end_date,
        with_deps=test_config.with_deps,
        dry_run=test_config.dry_run,
    )

    # Verify expected tasks were run
    log = get_execution_log()
    tasks_run = [entry["task_name"] for entry in log]

    assert (
        tasks_run == test_config.expected_tasks_run
    ), f"Expected tasks {test_config.expected_tasks_run}, got {tasks_run}"

    # Verify date parameters were passed correctly
    if test_config.start_date or test_config.end_date:
        for entry in log:
            if test_config.start_date:
                expected_start = pd.Timestamp(test_config.start_date)
                assert entry["start_date"] == expected_start
            if test_config.end_date:
                expected_end = pd.Timestamp(test_config.end_date)
                assert entry["end_date"] == expected_end


# ---------------------------------------------------------------------------
# Data Pipeline Tests
# ---------------------------------------------------------------------------


class TestDataPipeline:
    """Tests for data processing pipelines with symbol I/O."""

    def test_data_pipeline_full_run(
        self,
        tmp_path: Path,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test running a full data pipeline with dependencies."""
        runner = CLIRunner(
            config=copy.deepcopy(data_pipeline_config()),
            arctic_uri="mem://data-pipeline-test",
            tmp_path=tmp_path,
        )

        # Setup with input data
        runner.setup({"prices/raw": sample_prices})

        # Run the final task with dependencies
        runner.run_task(task="signal", with_deps=100)

        # Verify execution order
        log = get_execution_log()
        task_names = [entry["task_name"] for entry in log]
        assert task_names == [
            "multiply_prices",
            "compute_returns",
            "aggregate_to_signal",
        ]

        # Verify outputs exist
        assert "adjusted" in runner.list_symbols("prices")
        assert "returns" in runner.list_symbols("signals")
        assert "momentum" in runner.list_symbols("signals")

        # Verify data transformations
        adjusted = runner.read_output("prices/adjusted")
        pd.testing.assert_frame_equal(adjusted, sample_prices * 2.0, check_freq=False)

    def test_data_pipeline_single_task(
        self,
        tmp_path: Path,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test running a single task without dependencies."""
        runner = CLIRunner(
            config=data_pipeline_config(),
            arctic_uri="mem://single-task-test",
            tmp_path=tmp_path,
        )

        runner.setup({"prices/raw": sample_prices})

        # Run only the multiply task
        runner.run_task(task="multiply", with_deps=False)

        log = get_execution_log()
        assert len(log) == 1
        assert log[0]["task_name"] == "multiply_prices"

    def test_data_pipeline_with_date_filter(
        self,
        tmp_path: Path,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that date filters are passed through the pipeline."""
        runner = CLIRunner(
            config=data_pipeline_config(),
            arctic_uri="mem://date-filter-test",
            tmp_path=tmp_path,
        )

        runner.setup({"prices/raw": sample_prices})

        start = "2024-01-10"
        end = "2024-01-20"

        runner.run_task(
            task="multiply",
            start_date=start,
            end_date=end,
        )

        # Task executed successfully with date parameters
        log = get_execution_log()
        assert len(log) == 1


# ---------------------------------------------------------------------------
# Custom Arctic URI Tests
# ---------------------------------------------------------------------------


class TestCustomArcticURI:
    """Tests verifying custom Arctic URI handling."""

    @pytest.mark.parametrize(
        "arctic_uri",
        [
            "mem://test-1",
            "mem://test-2",
            "mem://custom-namespace",
        ],
        ids=["mem1", "mem2", "custom"],
    )
    def test_different_arctic_uris(
        self,
        arctic_uri: str,
        tmp_path: Path,
    ) -> None:
        """Test that different Arctic URIs work correctly."""
        runner = CLIRunner(
            config=simple_pipeline_config(),
            arctic_uri=arctic_uri,
            tmp_path=tmp_path,
        )

        runner.setup()
        runner.run_task(task="task_1")

        log = get_execution_log()
        assert len(log) == 1
        assert log[0]["task_name"] == "task_1"

    def test_isolated_arctic_instances(
        self,
        tmp_path: Path,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that different URIs have isolated data."""
        # Create two runners with different URIs and separate config copies
        runner1 = CLIRunner(
            copy.deepcopy(data_pipeline_config()),
            "mem://isolated-1",
            tmp_path / "r1",
        )
        runner2 = CLIRunner(
            copy.deepcopy(data_pipeline_config()),
            "mem://isolated-2",
            tmp_path / "r2",
        )

        # Setup data in runner1 only
        runner1.setup({"prices/raw": sample_prices})
        runner2.setup()  # No data

        # runner1 should succeed
        runner1.run_task(task="multiply")
        assert "adjusted" in runner1.list_symbols("prices")

        # runner2 should fail (no input data)
        with pytest.raises(Exception):
            runner2.run_task(task="multiply")


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestCLIEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_task(self, tmp_path: Path) -> None:
        """Test that running a nonexistent task raises an error."""
        runner = CLIRunner(
            config=simple_pipeline_config(),
            arctic_uri="mem://error-test",
            tmp_path=tmp_path,
        )
        runner.setup()

        with pytest.raises(ValueError, match="not a task"):
            runner.run_task(task="nonexistent_task")

    def test_empty_config(self, tmp_path: Path) -> None:
        """Test handling of empty configuration."""
        runner = CLIRunner(
            config={},
            arctic_uri="mem://empty-test",
            tmp_path=tmp_path,
        )
        runner.setup()

        with pytest.raises(ValueError):
            runner.run_task(task="any_task")

    def test_dry_run_mode(
        self,
        tmp_path: Path,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Test that dry_run mode executes tasks."""
        runner = CLIRunner(
            config=data_pipeline_config(),
            arctic_uri="mem://dry-run-test",
            tmp_path=tmp_path,
        )

        runner.setup({"prices/raw": sample_prices})

        # Run in dry-run mode
        runner.run_task(task="multiply", dry_run=True)

        # Task should have executed
        log = get_execution_log()
        assert len(log) == 1


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

        # Create args as if parsed from CLI
        import os

        from tradingo.config import read_config_template

        config = read_config_template(config_file, dict(os.environ))

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


# ---------------------------------------------------------------------------
# Fixtures shared by task-stop and active-plans tests
# ---------------------------------------------------------------------------

_STOP_PARAMS = {
    "config_path": "/cfg.yaml",
    "task_name": "task.a",
    "start_date": "2024-01-01",
    "end_date": "2024-01-15",
    "batch_interval": "7days",
    "batch_mode": "task",
    "with_deps": "False",
}
_STOP_SCHEDULE = [
    ("task.a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")),
    ("task.a", pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")),
]


@pytest.fixture()
def tmp_plans_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests for task stop command
# ---------------------------------------------------------------------------


class TestTaskStopParsing:
    def test_stop_parses_plan_key(self) -> None:
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(
                ["--config", "dummy.yaml", "task", "stop", "abc123"]
            )
        assert args.list_action == "stop"
        assert args.plan_key == "abc123"

    def test_stop_accepts_broker_url(self) -> None:
        parser = cli_app()
        with patch("tradingo.cli.read_config_template", return_value={}):
            args = parser.parse_args(
                [
                    "--config",
                    "dummy.yaml",
                    "task",
                    "stop",
                    "abc123",
                    "--broker-url",
                    "redis://myhost:6379/0",
                ]
            )
        assert args.broker_url == "redis://myhost:6379/0"


class TestTaskStopCommand:
    def test_stop_revokes_and_prints_count(
        self,
        simple_config: dict[str, Any],
        tmp_plans_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "tradingo.worker.app",
            MagicMock(control=MagicMock(revoke=MagicMock())),
        )
        plan = ExecutionPlan.from_schedule("plan-xyz", _STOP_PARAMS, _STOP_SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "celery-tid-001")

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="stop",
            plan_key="plan-xyz",
            broker_url=None,
        )
        handle_tasks(args, MagicMock())

        captured = capsys.readouterr()
        assert "Revoked 1" in captured.out

    def test_stop_missing_plan_prints_error(
        self,
        simple_config: dict[str, Any],
        tmp_plans_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="stop",
            plan_key="nonexistent-key",
            broker_url=None,
        )
        handle_tasks(args, MagicMock())

        captured = capsys.readouterr()
        assert "No execution plan found" in captured.out

    def test_stop_marks_steps_failed(
        self,
        simple_config: dict[str, Any],
        tmp_plans_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "tradingo.worker.app",
            MagicMock(control=MagicMock(revoke=MagicMock())),
        )
        plan = ExecutionPlan.from_schedule("plan-fail", _STOP_PARAMS, _STOP_SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "tid-x")
        plan.mark_step_submitted(1, "tid-y")

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="stop",
            plan_key="plan-fail",
            broker_url=None,
        )
        handle_tasks(args, MagicMock())

        loaded = ExecutionPlan.load("plan-fail")
        assert loaded is not None
        assert all(s.status == "FAILED" for s in loaded.steps)


# ---------------------------------------------------------------------------
# Tests for task list — active plans display
# ---------------------------------------------------------------------------


class TestTaskListActivePlans:
    def test_shows_active_plan(
        self,
        simple_config: dict[str, Any],
        tmp_plans_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = ExecutionPlan.from_schedule("active-plan", _STOP_PARAMS, _STOP_SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "tid-active")

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="list",
        )
        handle_tasks(args, MagicMock())

        captured = capsys.readouterr()
        assert "active-plan" in captured.out
        assert "task stop" in captured.out

    def test_no_active_plans_section_when_all_done(
        self,
        simple_config: dict[str, Any],
        tmp_plans_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = ExecutionPlan.from_schedule("done-plan", _STOP_PARAMS, _STOP_SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")
        plan.mark_step(1, "SUCCESS")

        args = argparse.Namespace(
            config=simple_config,
            entity="task",
            list_action="list",
        )
        handle_tasks(args, MagicMock())

        captured = capsys.readouterr()
        assert "Active Celery runs" not in captured.out

    def test_print_active_plans_silent_on_exception(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_print_active_plans must not propagate exceptions."""
        with patch(
            "tradingo.execution_plan.ExecutionPlan.list_plans",
            side_effect=RuntimeError("Redis down"),
        ):
            _print_active_plans()  # must not raise

        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Convenience function for running custom tests
# ---------------------------------------------------------------------------


def run_cli_test(
    config: dict[str, Any],
    task: str,
    tmp_path: Path,
    arctic_uri: str = "mem://test",
    start_date: str | None = None,
    end_date: str | None = None,
    with_deps: bool | int = False,
    setup_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[CLIRunner, list[dict[str, Any]]]:
    """Convenience function for running CLI tests.

    Args:
        config: DAG configuration dict
        task: Task name to run
        tmp_path: Temporary directory for test files
        arctic_uri: Arctic connection URI
        start_date: Optional start date
        end_date: Optional end date
        with_deps: Whether to run dependencies
        setup_data: Optional data to pre-populate

    Returns:
        Tuple of (runner, execution_log)
    """
    reset_execution_log()

    runner = CLIRunner(config, arctic_uri, tmp_path)
    runner.setup(setup_data)
    runner.run_task(
        task=task,
        start_date=start_date,
        end_date=end_date,
        with_deps=with_deps,
    )

    return runner, get_execution_log()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
