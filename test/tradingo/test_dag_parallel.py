"""Tests for parallel DAG execution."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from tradingo.dag import DAG, TaskState
from tradingo.execution_plan import ExecutionPlan

from . import _dag_helpers as helpers

HELPERS = "test.tradingo._dag_helpers"


def _make_dag(task_specs: dict[str, dict[str, object]]) -> DAG:
    """Build a DAG from a simplified spec.

    Each key is a task name, value is a dict with optional keys:
      - function: str (default: HELPERS + ".noop")
      - depends_on: list[str] (default: [])
    """
    config: dict[str, dict[str, object]] = {}
    for name, spec in task_specs.items():
        config[name] = {
            "function": spec.get("function", f"{HELPERS}.noop"),
            "depends_on": spec.get("depends_on", []),
            "params": {},
        }
    return DAG.from_config(config)


class TestParallelExecution:
    """Tests for DAG.run_parallel and DAG.run with max_workers > 1."""

    def setup_method(self) -> None:
        helpers.reset()

    def test_parallel_independent_tasks_run_concurrently(self) -> None:
        """Two independent tasks + one downstream dependent all complete."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.slow_a"},
                "b": {"function": f"{HELPERS}.slow_b"},
                "c": {
                    "function": f"{HELPERS}.record_c",
                    "depends_on": ["a", "b"],
                },
            }
        )

        dag.run("c", run_dependencies=True, max_workers=4, force_rerun=True)

        assert dag["a"].state == TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS
        assert dag["c"].state == TaskState.SUCCESS
        # a and b should both appear before c
        assert "c" in helpers.call_log
        assert helpers.call_log.index("c") > helpers.call_log.index("a")
        assert helpers.call_log.index("c") > helpers.call_log.index("b")

    def test_parallel_preserves_dependency_ordering(self) -> None:
        """Chain A -> B -> C: ordering must be A, B, C."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.gated_a"},
                "b": {
                    "function": f"{HELPERS}.gated_b",
                    "depends_on": ["a"],
                },
                "c": {
                    "function": f"{HELPERS}.gated_c",
                    "depends_on": ["b"],
                },
            }
        )

        dag.run("c", run_dependencies=True, max_workers=4, force_rerun=True)

        assert helpers.call_log == ["a", "b", "c"]

    def test_parallel_collects_all_errors(self) -> None:
        """Two independent tasks both fail — ExceptionGroup with both errors."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.fail_a"},
                "b": {"function": f"{HELPERS}.fail_b"},
                "c": {
                    "function": f"{HELPERS}.record_c",
                    "depends_on": ["a", "b"],
                },
            }
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            dag.run("c", run_dependencies=True, max_workers=4, force_rerun=True)

        eg = exc_info.value
        assert len(eg.exceptions) >= 2
        error_types = {type(e) for e in eg.exceptions}
        assert ValueError in error_types
        assert RuntimeError in error_types

    def test_parallel_skip_deps_respected(self) -> None:
        """skip_deps pattern prevents matching tasks from running."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {
                    "function": f"{HELPERS}.record_b",
                    "depends_on": ["a"],
                },
            }
        )

        dag.run(
            "b",
            run_dependencies=True,
            max_workers=4,
            force_rerun=True,
            skip_deps=re.compile("^a$"),
        )

        # a should be skipped, only b runs
        assert "a" not in helpers.call_log
        assert "b" in helpers.call_log
        assert dag["b"].state == TaskState.SUCCESS

    def test_parallel_max_workers_1_matches_sequential(self) -> None:
        """max_workers=1 falls back to sequential execution."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {
                    "function": f"{HELPERS}.record_b",
                    "depends_on": ["a"],
                },
            }
        )

        # max_workers=1 uses sequential path
        dag.run("b", run_dependencies=True, max_workers=1, force_rerun=True)

        assert dag["a"].state == TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS
        assert helpers.call_log == ["a", "b"]

    def test_parallel_run_dependencies_depth_limit(self) -> None:
        """run_dependencies=1 only goes one level deep."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {
                    "function": f"{HELPERS}.record_b",
                    "depends_on": ["a"],
                },
                "c": {
                    "function": f"{HELPERS}.record_c",
                    "depends_on": ["b"],
                },
            }
        )

        dag.run("c", run_dependencies=1, max_workers=4, force_rerun=True)

        # Only b and c should run (depth=1 from c reaches b, but not a)
        assert "a" not in helpers.call_log
        assert "b" in helpers.call_log
        assert "c" in helpers.call_log


class TestBatchedParallelSafety:
    """Verify that same-task intervals run sequentially in TASK mode."""

    def setup_method(self) -> None:
        helpers.reset()
        helpers.reset_threaded()

    def test_task_mode_intervals_are_sequential_per_task(self) -> None:
        """In TASK mode with max_workers>1, each task's intervals run sequentially."""
        config = {
            "task.a": {
                "function": f"{HELPERS}.record_a_threaded",
                "depends_on": [],
                "params": {},
            },
            "task.b": {
                "function": f"{HELPERS}.record_b_threaded",
                "depends_on": ["task.a"],
                "params": {},
            },
        }
        dag = DAG.from_config(config)

        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            max_workers=4,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-16"),
        )

        # 3 intervals x 2 tasks = 6 entries
        assert len(helpers.call_log_threaded) == 6

        # All of task a's entries should appear before any of task b's
        labels = [label for label, _, _, _ in helpers.call_log_threaded]
        a_indices = [i for i, lbl in enumerate(labels) if lbl == "a"]
        b_indices = [i for i, lbl in enumerate(labels) if lbl == "b"]
        assert max(a_indices) < min(
            b_indices
        ), "task.a intervals should all complete before task.b starts"

        # Within each task, intervals should be in chronological order
        a_dates = [
            (s, e) for label, s, e, _ in helpers.call_log_threaded if label == "a"
        ]
        b_dates = [
            (s, e) for label, s, e, _ in helpers.call_log_threaded if label == "b"
        ]
        assert a_dates == sorted(a_dates)
        assert b_dates == sorted(b_dates)


@pytest.fixture()
def tmp_plans_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))
    return tmp_path


def _make_two_task_dag() -> DAG:
    config = {
        "task.a": {
            "function": f"{HELPERS}.record_a_threaded",
            "depends_on": [],
            "params": {},
        },
        "task.b": {
            "function": f"{HELPERS}.record_b_threaded",
            "depends_on": ["task.a"],
            "params": {},
        },
    }
    return DAG.from_config(config)


def _find_plan(tmp_plans_dir: Path) -> ExecutionPlan:
    plans = list(tmp_plans_dir.glob("*.json"))
    assert len(plans) == 1
    plan = ExecutionPlan.load(plans[0].stem)
    assert plan is not None
    return plan


class TestBatchedParallelPlanTracking:
    """Execution plan is updated incrementally during parallel batched runs."""

    def setup_method(self) -> None:
        helpers.reset()
        helpers.reset_threaded()
        helpers.reset_fail_counter()

    def test_task_mode_all_steps_marked_success(self, tmp_plans_dir: Path) -> None:
        """TASK mode with max_workers>1: every step is marked SUCCESS in the plan."""
        dag = _make_two_task_dag()
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            max_workers=4,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
            config_path="/fake/config.yaml",
        )

        plan = _find_plan(tmp_plans_dir)
        assert all(s.status == "SUCCESS" for s in plan.steps)

    def test_task_mode_failed_step_marked_failed(self, tmp_plans_dir: Path) -> None:
        """TASK mode: a failing step is marked FAILED in the plan."""
        config = {
            "task.a": {
                "function": f"{HELPERS}.record_a_fail_second",
                "depends_on": [],
                "params": {},
            }
        }
        dag = DAG.from_config(config)

        with pytest.raises(ExceptionGroup):
            dag.run(
                "task.a",
                run_dependencies=False,
                force_rerun=True,
                max_workers=4,
                batch_interval=pd.Timedelta(days=5),
                batch_mode="task",
                start_date=pd.Timestamp("2024-01-01"),
                end_date=pd.Timestamp("2024-01-11"),
                config_path="/fake/config.yaml",
            )

        plan = _find_plan(tmp_plans_dir)
        statuses = [s.status for s in plan.steps]
        assert "FAILED" in statuses

    def test_stepped_mode_all_steps_marked_success(self, tmp_plans_dir: Path) -> None:
        """STEPPED mode with max_workers>1: every step is marked SUCCESS in the plan."""
        dag = _make_two_task_dag()
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            max_workers=4,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="stepped",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
            config_path="/fake/config.yaml",
        )

        plan = _find_plan(tmp_plans_dir)
        assert all(s.status == "SUCCESS" for s in plan.steps)

    def test_deps_first_mode_all_steps_marked_success(
        self, tmp_plans_dir: Path
    ) -> None:
        """DEPS_FIRST mode with max_workers>1: every step is marked SUCCESS in the plan."""
        dag = _make_two_task_dag()
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            max_workers=4,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="deps-first",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
            config_path="/fake/config.yaml",
        )

        plan = _find_plan(tmp_plans_dir)
        assert all(s.status == "SUCCESS" for s in plan.steps)

    def test_no_plan_without_config_path(self, tmp_plans_dir: Path) -> None:
        """No plan file is created when config_path is not provided."""
        dag = _make_two_task_dag()
        dag.run(
            "task.b",
            run_dependencies=True,
            force_rerun=True,
            max_workers=4,
            batch_interval=pd.Timedelta(days=5),
            batch_mode="task",
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-11"),
        )

        assert list(tmp_plans_dir.glob("*.json")) == []
