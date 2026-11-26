"""Tests for multiprocess DAG execution."""

from __future__ import annotations

import re

import pytest

from tradingo.dag import DAG, TaskState

from . import _dag_helpers as helpers

HELPERS = "test.tradingo._dag_helpers"


def _make_dag(task_specs: dict[str, dict[str, object]]) -> DAG:
    config: dict[str, dict[str, object]] = {}
    for name, spec in task_specs.items():
        config[name] = {
            "function": spec.get("function", f"{HELPERS}.noop"),
            "depends_on": spec.get("depends_on", []),
            "params": {},
        }
    return DAG.from_config(config)


class TestMultiprocessExecution:
    """Tests for DAG.run_multiprocess and DAG.run with executor='process'."""

    def setup_method(self) -> None:
        helpers.reset()

    def test_independent_tasks_all_succeed(self) -> None:
        """Independent tasks and their downstream dependent all reach SUCCESS."""
        dag = _make_dag(
            {
                "a": {},
                "b": {},
                "c": {"depends_on": ["a", "b"]},
            }
        )
        dag.run(
            "c",
            run_dependencies=True,
            max_workers=2,
            executor="process",
            force_rerun=True,
        )

        assert dag["a"].state == TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS
        assert dag["c"].state == TaskState.SUCCESS

    def test_chain_respects_dependency_order(self) -> None:
        """Linear chain A -> B -> C all complete; B and C cannot run before their deps."""
        dag = _make_dag(
            {
                "a": {},
                "b": {"depends_on": ["a"]},
                "c": {"depends_on": ["b"]},
            }
        )
        dag.run(
            "c",
            run_dependencies=True,
            max_workers=2,
            executor="process",
            force_rerun=True,
        )

        assert dag["a"].state == TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS
        assert dag["c"].state == TaskState.SUCCESS

    def test_failed_task_raises_exception_group(self) -> None:
        """Two independently failing tasks both appear in the ExceptionGroup."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.fail_a"},
                "b": {"function": f"{HELPERS}.fail_b"},
                "c": {"depends_on": ["a", "b"]},
            }
        )
        with pytest.raises(ExceptionGroup) as exc_info:
            dag.run(
                "c",
                run_dependencies=True,
                max_workers=2,
                executor="process",
                force_rerun=True,
            )

        assert len(exc_info.value.exceptions) >= 1

    def test_failed_task_state_is_failed(self) -> None:
        """A task that raises is marked FAILED in the DAG."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        with pytest.raises(ExceptionGroup):
            dag.run(
                "a",
                run_dependencies=False,
                max_workers=2,
                executor="process",
                force_rerun=True,
            )

        assert dag["a"].state == TaskState.FAILED

    def test_downstream_of_failed_task_not_run(self) -> None:
        """If a dependency fails its downstream task is never submitted."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.fail_a"},
                "b": {"depends_on": ["a"]},
            }
        )
        with pytest.raises(ExceptionGroup):
            dag.run(
                "b",
                run_dependencies=True,
                max_workers=2,
                executor="process",
                force_rerun=True,
            )

        assert dag["a"].state == TaskState.FAILED
        assert dag["b"].state != TaskState.SUCCESS

    def test_run_routes_to_multiprocess(self) -> None:
        """dag.run() with executor='process' dispatches via run_multiprocess."""
        dag = _make_dag({"a": {}})
        dag.run(
            "a",
            run_dependencies=False,
            max_workers=2,
            executor="process",
            force_rerun=True,
        )
        assert dag["a"].state == TaskState.SUCCESS

    def test_single_worker_completes(self) -> None:
        """max_workers=1 with executor='process' still executes correctly."""
        dag = _make_dag(
            {
                "a": {},
                "b": {"depends_on": ["a"]},
            }
        )
        dag.run(
            "b",
            run_dependencies=True,
            max_workers=1,
            executor="process",
            force_rerun=True,
        )

        assert dag["a"].state == TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS

    def test_skip_deps_excludes_matching_tasks(self) -> None:
        """skip_deps pattern prevents matching tasks from being submitted."""
        dag = _make_dag(
            {
                "a": {},
                "b": {"depends_on": ["a"]},
            }
        )
        dag.run(
            "b",
            run_dependencies=True,
            max_workers=2,
            executor="process",
            force_rerun=True,
            skip_deps=re.compile("^a$"),
        )

        assert dag["a"].state != TaskState.SUCCESS
        assert dag["b"].state == TaskState.SUCCESS
