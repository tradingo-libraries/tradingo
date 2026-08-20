from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from tradingo.config import ConfigLoadError
from tradingo.dag import (
    DAG,
    BatchMode,
    Stage,
    generate_batch_schedule,
    generate_intervals,
)


def test_dag_configuration() -> None:
    nodes = {
        "raw_prices": {
            "MSFT.sample": {
                "name": "MSFT.sample",
                "function": "tradingo.sampling.ig.sample_instrument",
                "depends_on": [],
                "symbols_in": [],
                "symbols_out": [
                    "ig-trading/{symbol}.mid",
                    "ig-trading/{symbol}.bid",
                    "ig-trading/{symbol}.ask",
                ],
                "params": {
                    "symbol": "MSFT",
                },
            },
            "AAPL.sample": {
                "name": "AAPL.sample",
                "function": "tradingo.sampling.ig.sample_instrument",
                "symbols_in": [],
                "depends_on": [],
                "symbols_out": [
                    "ig-trading/mid",
                    "ig-trading/bid",
                    "ig-trading/ask",
                ],
                "params": {
                    "symbol": "AAPL",
                },
            },
        },
        "prices": {
            "universe.sample": {
                "function": "tradingo.sampling.ig.sample_universe",
                "depends_on": ["AAPL.sample", "MSFT.sample"],
                "symbols_in": [
                    "ig-trading/AAPL.mid",
                    "ig-trading/AAPL.bid",
                    "ig-trading/AAPL.ask",
                    "ig-trading/MSFT.mid",
                    "ig-trading/MSFT.bid",
                    "ig-trading/MSFT.ask",
                ],
                "symbols_out": [
                    "prices/{universe}.mid.open",
                    "prices/{universe}.mid.high",
                    "prices/{universe}.mid.low",
                    "prices/{universe}.mid.close",
                    "prices/{universe}.bid.open",
                    "prices/{universe}.bid.high",
                    "prices/{universe}.bid.low",
                    "prices/{universe}.bid.close",
                    "prices/{universe}.ask.open",
                    "prices/{universe}.ask.high",
                    "prices/{universe}.ask.low",
                    "prices/{universe}.ask.close",
                ],
                "params": {},
            },
        },
        "signals": {
            "signal.trend": {
                "function": "tradingo.signals.trend",
                "symbols_out": ["signals/{universe}.trend"],
                "depends_on": ["universe.sample"],
                "symbols_in": ["prices/{universe}.mid.close"],
                "params": {
                    "prices": "prices/mid.close",
                    "library": "signals",
                    "field": "trend",
                    "universe": "ig-trading",
                },
            },
        },
    }

    dag = DAG.from_config(nodes)

    assert dag.get_symbols() == [
        "ig-trading/{symbol}.mid",
        "ig-trading/{symbol}.bid",
        "ig-trading/{symbol}.ask",
        "ig-trading/mid",
        "ig-trading/bid",
        "ig-trading/ask",
        "prices/{universe}.mid.open",
        "prices/{universe}.mid.high",
        "prices/{universe}.mid.low",
        "prices/{universe}.mid.close",
        "prices/{universe}.bid.open",
        "prices/{universe}.bid.high",
        "prices/{universe}.bid.low",
        "prices/{universe}.bid.close",
        "prices/{universe}.ask.open",
        "prices/{universe}.ask.high",
        "prices/{universe}.ask.low",
        "prices/{universe}.ask.close",
        "signals/{universe}.trend",
    ]


def _make_simple_task(
    name: str,
    depends_on: list[str] | None = None,
    symbols_out: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to build a minimal task config dict."""
    return {
        "function": "builtins.print",
        "depends_on": depends_on or [],
        "symbols_in": {},
        "symbols_out": symbols_out or [],
        "params": params or {},
    }


class TestStage:
    """Tests for the Stage grouping feature."""

    def test_stage_creation(self) -> None:
        """Stage groups subtasks and computes external dependencies."""
        config = {
            "task.a": _make_simple_task("task.a"),
            "task.b": _make_simple_task(
                "task.b", depends_on=["task.a"], symbols_out=["lib/b"]
            ),
            "task.c": _make_simple_task(
                "task.c", depends_on=["task.b"], symbols_out=["lib/c"]
            ),
            "my.stage": {"stage": ["task.b", "task.c"]},
        }
        dag = DAG.from_config(config)

        # Stage replaces subtasks in the DAG
        assert "my.stage" in dag
        assert "task.b" not in dag
        assert "task.c" not in dag
        assert "task.a" in dag

        stage = dag["my.stage"]
        assert isinstance(stage, Stage)

    def test_stage_external_dependencies(self) -> None:
        """Stage auto-computes external deps from subtask deps."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        stage = dag["my.stage"]
        assert stage.dependency_names == ["upstream"]

    def test_stage_rewires_downstream_deps(self) -> None:
        """Tasks depending on stage members get re-wired to the stage."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task(
                "step.2",
                depends_on=["step.1"],
                symbols_out=["lib/result"],
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
            "downstream": _make_simple_task("downstream", depends_on=["step.2"]),
        }
        dag = DAG.from_config(config)

        # downstream should now depend on the stage, not step.2
        assert dag["downstream"].dependency_names == ["my.stage"]

    def test_stage_deduplicates_rewired_deps(self) -> None:
        """Multiple deps on stage members collapse to one dep on stage."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
            "downstream": _make_simple_task(
                "downstream", depends_on=["step.1", "step.2"]
            ),
        }
        dag = DAG.from_config(config)

        assert dag["downstream"].dependency_names == ["my.stage"]

    def test_stage_topological_ordering(self) -> None:
        """Subtasks are sorted by internal dependencies, not config order."""
        config = {
            "upstream": _make_simple_task("upstream"),
            # Listed in reverse order
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "my.stage": {"stage": ["step.2", "step.1"]},
        }
        dag = DAG.from_config(config)

        stage = cast(Stage, dag["my.stage"])
        subtask_names = [t.name for t in stage._tasks]
        assert subtask_names == ["step.1", "step.2"]

    def test_stage_aggregates_symbols(self) -> None:
        """Stage symbols_out is the union of all subtask symbols_out."""
        config = {
            "step.1": _make_simple_task("step.1", symbols_out=["lib/a", "lib/b"]),
            "step.2": _make_simple_task(
                "step.2", depends_on=["step.1"], symbols_out=["lib/c"]
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        assert dag["my.stage"].symbols_out == ["lib/a", "lib/b", "lib/c"]

    def test_stage_missing_subtask(self) -> None:
        """Stage referencing a non-existent task raises ConfigLoadError."""
        config = {
            "step.1": _make_simple_task("step.1"),
            "my.stage": {"stage": ["step.1", "step.missing"]},
        }
        with pytest.raises(ConfigLoadError, match="step.missing"):
            DAG.from_config(config)

    def test_stage_circular_dependency(self) -> None:
        """Circular deps within a stage raise ConfigLoadError."""
        config = {
            "step.1": _make_simple_task("step.1", depends_on=["step.2"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        with pytest.raises(ConfigLoadError, match="Circular"):
            DAG.from_config(config)

    def test_stage_get_symbols(self) -> None:
        """DAG.get_symbols includes stage subtask symbols."""
        config = {
            "task.a": _make_simple_task("task.a", symbols_out=["lib/a"]),
            "step.1": _make_simple_task(
                "step.1", depends_on=["task.a"], symbols_out=["lib/s1"]
            ),
            "step.2": _make_simple_task(
                "step.2", depends_on=["step.1"], symbols_out=["lib/s2"]
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        symbols = dag.get_symbols()
        assert "lib/a" in symbols
        assert "lib/s1" in symbols
        assert "lib/s2" in symbols

    def test_stage_with_explicit_depends_on(self) -> None:
        """Stage can have explicit depends_on in addition to auto-computed."""
        config = {
            "extra.dep": _make_simple_task("extra.dep"),
            "step.1": _make_simple_task("step.1"),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {
                "stage": ["step.1", "step.2"],
                "depends_on": ["extra.dep"],
            },
        }
        dag = DAG.from_config(config)

        assert "extra.dep" in dag["my.stage"].dependency_names

    def test_stage_repr(self) -> None:
        """Stage has informative repr."""
        config = {
            "step.1": _make_simple_task("step.1"),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        r = repr(dag["my.stage"])
        assert "my.stage" in r
        assert "step.1" in r
        assert "step.2" in r


class TestGenerateIntervals:
    """Tests for generate_intervals()."""

    def test_even_split(self) -> None:
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-10")
        interval = pd.Timedelta(days=3)
        result = generate_intervals(start, end, interval)
        assert result == [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
            (pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-10")),
        ]

    def test_uneven_split(self) -> None:
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-08")
        interval = pd.Timedelta(days=3)
        result = generate_intervals(start, end, interval)
        assert result == [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
            (pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-08")),
        ]

    def test_single_interval(self) -> None:
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-02")
        interval = pd.Timedelta(days=5)
        result = generate_intervals(start, end, interval)
        assert result == [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        ]

    def test_empty_when_start_equals_end(self) -> None:
        ts = pd.Timestamp("2024-01-01")
        result = generate_intervals(ts, ts, pd.Timedelta(days=1))
        assert result == []


class TestGenerateBatchSchedule:
    """Tests for generate_batch_schedule()."""

    def _intervals(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
        ]

    def test_stepped(self) -> None:
        intervals = self._intervals()
        result = generate_batch_schedule(
            task_names=["dep_a", "target"],
            intervals=intervals,
            batch_mode=BatchMode.STEPPED,
            full_start=pd.Timestamp("2024-01-01"),
            full_end=pd.Timestamp("2024-01-07"),
        )
        assert result == [
            ("dep_a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("target", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("dep_a", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
            ("target", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
        ]

    def test_task(self) -> None:
        intervals = self._intervals()
        result = generate_batch_schedule(
            task_names=["dep_a", "target"],
            intervals=intervals,
            batch_mode=BatchMode.TASK,
            full_start=pd.Timestamp("2024-01-01"),
            full_end=pd.Timestamp("2024-01-07"),
        )
        assert result == [
            ("dep_a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("dep_a", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
            ("target", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("target", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
        ]

    def test_deps_first(self) -> None:
        intervals = self._intervals()
        result = generate_batch_schedule(
            task_names=["dep_a", "dep_b", "target"],
            intervals=intervals,
            batch_mode=BatchMode.DEPS_FIRST,
            full_start=pd.Timestamp("2024-01-01"),
            full_end=pd.Timestamp("2024-01-07"),
        )
        assert result == [
            ("dep_a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-07")),
            ("dep_b", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-07")),
            ("target", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("target", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
        ]

    def test_single_task_no_deps(self) -> None:
        intervals = self._intervals()
        result = generate_batch_schedule(
            task_names=["target"],
            intervals=intervals,
            batch_mode=BatchMode.STEPPED,
            full_start=pd.Timestamp("2024-01-01"),
            full_end=pd.Timestamp("2024-01-07"),
        )
        assert result == [
            ("target", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("target", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")),
        ]


def _make_dated_task(
    name: str,
    function: str,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    """Helper to build a task config using dated recording helpers."""
    return {
        "function": function,
        "depends_on": depends_on or [],
        "symbols_in": {},
        "symbols_out": [],
        "params": {},
    }


class TestRunBatched:
    """Tests for DAG.run_batched() end-to-end execution."""

    def setup_method(self) -> None:
        from test.tradingo._dag_helpers import reset

        reset()

    def _make_dag(self) -> DAG:
        """Create a DAG: a -> b -> c (using dated recording helpers)."""
        config = {
            "task.a": _make_dated_task(
                "task.a",
                "test.tradingo._dag_helpers.record_a_dated",
            ),
            "task.b": _make_dated_task(
                "task.b",
                "test.tradingo._dag_helpers.record_b_dated",
                depends_on=["task.a"],
            ),
            "task.c": _make_dated_task(
                "task.c",
                "test.tradingo._dag_helpers.record_c_dated",
                depends_on=["task.b"],
            ),
        }
        return DAG.from_config(config)

    def test_stepped_sequential(self) -> None:
        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.c",
            run_dependencies=True,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="stepped",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        # 2 intervals x 3 tasks = 6 steps, interleaved by interval
        assert len(call_log_dated) == 6
        # First interval: a, b, c
        assert call_log_dated[0] == (
            "a",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-06"),
        )
        assert call_log_dated[1] == (
            "b",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-06"),
        )
        assert call_log_dated[2] == (
            "c",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-06"),
        )
        # Second interval: a, b, c
        assert call_log_dated[3] == (
            "a",
            pd.Timestamp("2024-01-06"),
            pd.Timestamp("2024-01-11"),
        )
        assert call_log_dated[4] == (
            "b",
            pd.Timestamp("2024-01-06"),
            pd.Timestamp("2024-01-11"),
        )
        assert call_log_dated[5] == (
            "c",
            pd.Timestamp("2024-01-06"),
            pd.Timestamp("2024-01-11"),
        )

    def test_task_mode_sequential(self) -> None:
        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.c",
            run_dependencies=True,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        assert len(call_log_dated) == 6
        # All of a, then all of b, then all of c
        assert call_log_dated[0][0] == "a"
        assert call_log_dated[1][0] == "a"
        assert call_log_dated[2][0] == "b"
        assert call_log_dated[3][0] == "b"
        assert call_log_dated[4][0] == "c"
        assert call_log_dated[5][0] == "c"

    def test_deps_first_sequential(self) -> None:
        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.c",
            run_dependencies=True,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="deps-first",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        assert len(call_log_dated) == 4
        # Deps run on full range
        assert call_log_dated[0] == (
            "a",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-11"),
        )
        assert call_log_dated[1] == (
            "b",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-11"),
        )
        # Target chunked
        assert call_log_dated[2] == (
            "c",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-06"),
        )
        assert call_log_dated[3] == (
            "c",
            pd.Timestamp("2024-01-06"),
            pd.Timestamp("2024-01-11"),
        )

    def test_requires_start_and_end_date(self) -> None:
        dag = self._make_dag()
        with pytest.raises(ValueError, match="start-date and --end-date"):
            dag.run(
                "task.c",
                force_rerun=True,
                batch_interval=pd.Timedelta(days=5),
                batch_mode="stepped",
            )

    def test_no_deps(self) -> None:
        """Batching works without run_dependencies."""
        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.c",
            run_dependencies=False,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="stepped",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        assert len(call_log_dated) == 2
        assert all(label == "c" for label, _, _ in call_log_dated)

    def test_parallel_stepped(self) -> None:
        """Parallel batched execution completes all steps."""
        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.c",
            run_dependencies=True,
            force_rerun=True,
            max_workers=2,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="stepped",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        # All 6 steps should complete (order may vary within interval)
        assert len(call_log_dated) == 6
        labels = [label for label, _, _ in call_log_dated]
        assert labels.count("a") == 2
        assert labels.count("b") == 2
        assert labels.count("c") == 2


class TestRunBatchedRecovery:
    """Tests for execution plan recovery in DAG.run_batched()."""

    def setup_method(self) -> None:
        from test.tradingo._dag_helpers import reset, reset_fail_counter

        reset()
        reset_fail_counter()

    def _make_dag(self) -> DAG:
        """Create a DAG: a -> b (using dated recording helpers)."""
        config = {
            "task.a": _make_dated_task(
                "task.a",
                "test.tradingo._dag_helpers.record_a_dated",
            ),
            "task.b": _make_dated_task(
                "task.b",
                "test.tradingo._dag_helpers.record_b_dated",
                depends_on=["task.a"],
            ),
        }
        return DAG.from_config(config)

    def test_plan_created_on_batched_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plan file is created when config_path is provided."""
        from tradingo.execution_plan import ExecutionPlan

        monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))

        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="stepped",
            config_path="/test/config.yaml",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        assert len(call_log_dated) == 4  # 2 intervals x 2 tasks
        plan_files = list(tmp_path.glob("*.json"))
        assert len(plan_files) == 1

        # All steps should be SUCCESS
        plan = ExecutionPlan.load(plan_files[0].stem)
        assert plan is not None
        assert all(s.status == "SUCCESS" for s in plan.steps)

    def test_recover_skips_completed_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--recover skips steps already marked SUCCESS."""
        from tradingo.execution_plan import ExecutionPlan

        monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))

        from test.tradingo._dag_helpers import call_log_dated

        dag = self._make_dag()

        # First run: complete all steps
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            config_path="/test/config.yaml",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )
        assert len(call_log_dated) == 4

        call_log_dated.clear()

        # Second run with recover=True, force_rerun=False: should skip everything
        dag2 = self._make_dag()
        dag2.run(
            "task.b",
            run_dependencies=True,
            force_rerun=False,
            recover=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            config_path="/test/config.yaml",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )
        # All steps were SUCCESS, so nothing should run
        assert len(call_log_dated) == 0

    def test_recover_resumes_from_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--recover resumes from the first non-SUCCESS step."""
        from tradingo.execution_plan import ExecutionPlan

        monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))

        from test.tradingo._dag_helpers import call_log_dated

        # Use a dag with task.a that fails on 2nd call
        config = {
            "task.a": _make_dated_task(
                "task.a",
                "test.tradingo._dag_helpers.record_a_fail_second",
            ),
            "task.b": _make_dated_task(
                "task.b",
                "test.tradingo._dag_helpers.record_b_dated",
                depends_on=["task.a"],
            ),
        }
        dag = DAG.from_config(config)

        # First run: a(interval1) succeeds, a(interval2) fails
        with pytest.raises(RuntimeError, match="task_a failed"):
            dag.run(
                "task.b",
                run_dependencies=True,
                force_rerun=True,
                batch_interval=pd.Timedelta(days=5),
                batch_mode="task",
                config_path="/test/config.yaml",
                start_date=pd.Timestamp("2024-01-01"),
                end_date=pd.Timestamp("2024-01-11"),
            )

        # Check the plan: first step SUCCESS, second FAILED
        plan_files = list(tmp_path.glob("*.json"))
        assert len(plan_files) == 1
        plan = ExecutionPlan.load(plan_files[0].stem)
        assert plan is not None
        assert plan.steps[0].status == "SUCCESS"
        assert plan.steps[1].status == "FAILED"

        # Recovery run: should skip the first step
        call_log_dated.clear()
        from test.tradingo._dag_helpers import reset_fail_counter

        reset_fail_counter()

        dag2 = DAG.from_config(
            {
                "task.a": _make_dated_task(
                    "task.a",
                    "test.tradingo._dag_helpers.record_a_dated",
                ),
                "task.b": _make_dated_task(
                    "task.b",
                    "test.tradingo._dag_helpers.record_b_dated",
                    depends_on=["task.a"],
                ),
            }
        )
        dag2.run(
            "task.b",
            run_dependencies=True,
            force_rerun=False,
            recover=True,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            config_path="/test/config.yaml",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        # Should have run the remaining 3 steps (a[interval2], b[interval1], b[interval2])
        assert len(call_log_dated) == 3


if __name__ == "__main__":
    pytest.main([__file__])
