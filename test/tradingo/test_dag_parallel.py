"""Tests for parallel DAG execution."""

from __future__ import annotations

import re

import pytest

from tradingo.dag import DAG, TaskState

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
