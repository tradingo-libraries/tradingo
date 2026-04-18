"""Tests for DAG.run_batched_celery — batched Celery execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from tradingo.dag import DAG
from tradingo.execution_plan import ExecutionPlan

from . import _dag_helpers as helpers

HELPERS = "test.tradingo._dag_helpers"


# ---------------------------------------------------------------------------
# Fake Celery infrastructure
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """Synchronously-completed fake Celery AsyncResult."""

    _counter = 0

    def __init__(self, fn: Any, *args: Any) -> None:
        _FakeAsyncResult._counter += 1
        self.id = f"fake-task-{_FakeAsyncResult._counter}"
        self._exception: Exception | None = None
        try:
            fn(*args)
        except Exception as exc:
            self._exception = exc

    def ready(self) -> bool:
        return True

    def failed(self) -> bool:
        return self._exception is not None

    @property
    def result(self) -> Exception | None:
        return self._exception

    def get(self, propagate: bool = True) -> None:
        if propagate and self._exception is not None:
            raise self._exception


class _FakeCeleryApp:
    """Fake Celery app that executes tradingo.run_task synchronously in-process."""

    def send_task(
        self,
        task_name: str,
        args: list[Any] | None = None,
        queue: str | None = None,
    ) -> _FakeAsyncResult:
        assert task_name == "tradingo.run_task"
        task_spec, global_kwargs = args or [None, None]

        def _run() -> None:
            from tradingo.dag import Task
            from tradingo.worker import _deserialize_kwargs

            kwargs = _deserialize_kwargs(global_kwargs)
            kwargs.pop("arctic", None)

            task = Task(
                name=task_spec["name"],
                function=task_spec["function"],
                task_args=tuple(task_spec["task_args"]),
                task_kwargs=_deserialize_kwargs(task_spec["task_kwargs"]),
                symbols_out=task_spec["symbols_out"],
                symbols_in=task_spec["symbols_in"],
                load_args=dict(task_spec["load_args"]),
                publish_args=dict(task_spec["publish_args"]),
            )
            task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), kwargs)
            task.function(*task.task_args, **task_kwargs)

        return _FakeAsyncResult(_run)


@pytest.fixture(autouse=True)
def _fake_celery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch require_celery() to return a fake synchronous Celery app."""
    fake_app = _FakeCeleryApp()
    monkeypatch.setattr("tradingo.worker.app", fake_app)
    monkeypatch.setattr("tradingo.dag.time.sleep", lambda _: None)


@pytest.fixture(autouse=True)
def _reset_logs() -> None:
    helpers.reset_celery_batch_log()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = pd.Timestamp("2024-01-01")
_END = pd.Timestamp("2024-01-16")  # 15 days → 3 × 5-day intervals
_INTERVAL = pd.Timedelta(days=5)


def _make_dag(task_specs: dict[str, dict[str, object]]) -> DAG:
    config: dict[str, dict[str, object]] = {
        name: {
            "function": spec.get("function", f"{HELPERS}.noop"),
            "depends_on": spec.get("depends_on", []),
            "params": {},
        }
        for name, spec in task_specs.items()
    }
    return DAG.from_config(config)


def _run(dag: DAG, task: str, *, mode: str = "task", **kw: Any) -> None:
    dag.run(
        task,
        run_dependencies=True,
        force_rerun=True,
        executor="celery",
        batch_interval=_INTERVAL,
        batch_mode=mode,
        start_date=_START,
        end_date=_END,
        **kw,
    )


# ---------------------------------------------------------------------------
# TASK mode tests
# ---------------------------------------------------------------------------


class TestTaskMode:
    def test_all_steps_execute(self) -> None:
        """Every (task, chunk) combination runs exactly once."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b")

        labels = [lbl for lbl, *_ in helpers.celery_batch_log]
        assert labels.count("a") == 3
        assert labels.count("b") == 3

    def test_all_dep_chunks_before_first_downstream_chunk(self) -> None:
        """In TASK mode all task.a chunks complete before any task.b chunk runs."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b")

        labels = [lbl for lbl, *_ in helpers.celery_batch_log]
        a_indices = [i for i, lbl in enumerate(labels) if lbl == "a"]
        b_indices = [i for i, lbl in enumerate(labels) if lbl == "b"]
        assert max(a_indices) < min(b_indices)

    def test_same_task_chunks_are_chronological(self) -> None:
        """Chunks for each task arrive in start-date order."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b")

        a_starts = [s for lbl, s, *_ in helpers.celery_batch_log if lbl == "a"]
        b_starts = [s for lbl, s, *_ in helpers.celery_batch_log if lbl == "b"]
        assert a_starts == sorted(a_starts)
        assert b_starts == sorted(b_starts)

    def test_clean_only_on_first_chunk(self) -> None:
        """clean=True is forwarded only to the first chunk of each task."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})
        dag.run(
            "task.a",
            run_dependencies=False,
            force_rerun=True,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
            clean=True,
        )

        clean_flags = [clean for *_, clean in helpers.celery_batch_log]
        assert clean_flags[0] is True
        assert all(f is None for f in clean_flags[1:])

    def test_no_clean_kwarg_stays_absent(self) -> None:
        """When clean is not passed, subsequent chunks also omit it."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})
        dag.run(
            "task.a",
            run_dependencies=False,
            force_rerun=True,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
        )

        clean_flags = [clean for *_, clean in helpers.celery_batch_log]
        assert all(f is None for f in clean_flags)

    def test_error_raises_exception_group(self) -> None:
        """A failing step raises ExceptionGroup."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.fail_a_celery"}})

        with pytest.raises(ExceptionGroup) as exc_info:
            dag.run(
                "task.a",
                run_dependencies=False,
                force_rerun=True,
                executor="celery",
                batch_interval=_INTERVAL,
                batch_mode="task",
                start_date=_START,
                end_date=_END,
            )

        assert len(exc_info.value.exceptions) >= 1
        assert any(isinstance(e, RuntimeError) for e in exc_info.value.exceptions)


# ---------------------------------------------------------------------------
# STEPPED mode tests
# ---------------------------------------------------------------------------


class TestSteppedMode:
    def test_all_steps_execute(self) -> None:
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b", mode="stepped")

        labels = [lbl for lbl, *_ in helpers.celery_batch_log]
        assert labels.count("a") == 3
        assert labels.count("b") == 3

    def test_within_interval_dep_ordering(self) -> None:
        """For each interval, task.a runs before task.b."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b", mode="stepped")

        # Group by interval start
        by_start: dict[Any, list[str]] = {}
        for lbl, s, *_ in helpers.celery_batch_log:
            by_start.setdefault(s, []).append(lbl)

        for interval_labels in by_start.values():
            assert interval_labels.index("a") < interval_labels.index("b")


# ---------------------------------------------------------------------------
# DEPS_FIRST mode tests
# ---------------------------------------------------------------------------


class TestDepsFirstMode:
    def test_dep_full_range_before_target_chunks(self) -> None:
        """task.a runs once (full range) before any task.b chunk."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b", mode="deps-first")

        labels = [lbl for lbl, *_ in helpers.celery_batch_log]
        assert labels.count("a") == 1  # full range, single step
        assert labels.count("b") == 3  # chunked
        assert labels.index("a") < labels.index("b")

    def test_dep_receives_full_date_range(self) -> None:
        """The dep task step is dispatched with the original start/end dates."""
        dag = _make_dag(
            {
                "task.a": {"function": f"{HELPERS}.record_a_celery"},
                "task.b": {
                    "function": f"{HELPERS}.record_b_celery",
                    "depends_on": ["task.a"],
                },
            }
        )
        _run(dag, "task.b", mode="deps-first")

        a_entry = next(e for e in helpers.celery_batch_log if e[0] == "a")
        assert a_entry[1] == _START
        assert a_entry[2] == _END


# ---------------------------------------------------------------------------
# run() routing test
# ---------------------------------------------------------------------------


def test_run_routes_to_batched_celery_when_batch_interval_set() -> None:
    """run() calls run_batched_celery when executor=celery and batch_interval set."""
    dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})

    with (
        patch.object(dag, "run_batched_celery") as mock_batched,
        patch.object(dag, "run_celery") as mock_celery,
    ):
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
        )

    mock_batched.assert_called_once()
    mock_celery.assert_not_called()


def test_run_routes_to_run_celery_without_batch_interval() -> None:
    """run() calls run_celery when executor=celery and no batch_interval."""
    dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})

    with (
        patch.object(dag, "run_batched_celery") as mock_batched,
        patch.object(dag, "run_celery") as mock_celery,
    ):
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
        )

    mock_celery.assert_called_once()
    mock_batched.assert_not_called()


# ---------------------------------------------------------------------------
# Execution plan recovery tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_plans_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))
    return tmp_path


def _find_plan(tmp_plans_dir: Path) -> ExecutionPlan:
    plans = list(tmp_plans_dir.glob("*.json"))
    assert len(plans) == 1, f"Expected 1 plan file, found {len(plans)}"
    plan = ExecutionPlan.load(plans[0].stem)
    assert plan is not None
    return plan


class TestExecutionPlanRecovery:
    """Celery executor must create, update, and respect execution plans."""

    def setup_method(self) -> None:
        helpers.reset_celery_batch_log()

    def test_plan_created_when_config_path_given(self, tmp_plans_dir: Path) -> None:
        """Passing config_path causes a plan file to be written."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
            config_path="/fake/config.yaml",
        )
        assert list(tmp_plans_dir.glob("*.json")), "Expected a plan file to be created"

    def test_all_steps_marked_success_after_clean_run(
        self, tmp_plans_dir: Path
    ) -> None:
        """Every step is marked SUCCESS in the plan after a successful run."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
            config_path="/fake/config.yaml",
        )
        plan = _find_plan(tmp_plans_dir)
        assert all(s.status == "SUCCESS" for s in plan.steps)

    def test_failed_step_marked_failed_in_plan(self, tmp_plans_dir: Path) -> None:
        """A step that raises is marked FAILED in the plan."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.fail_a_celery"}})
        with pytest.raises(ExceptionGroup):
            dag.run(
                "task.a",
                run_dependencies=False,
                executor="celery",
                batch_interval=_INTERVAL,
                batch_mode="task",
                start_date=_START,
                end_date=_END,
                config_path="/fake/config.yaml",
            )
        plan = _find_plan(tmp_plans_dir)
        assert any(s.status == "FAILED" for s in plan.steps)

    def test_recover_skips_success_steps(self, tmp_plans_dir: Path) -> None:
        """With recover=True, steps already marked SUCCESS are not dispatched again."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})

        # First run — populates plan with all SUCCESS
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
            config_path="/fake/config.yaml",
        )
        plan = _find_plan(tmp_plans_dir)
        total_steps = len(plan.steps)
        assert total_steps > 1, "Need at least 2 steps to test partial recovery"

        # Rewind the last step to PENDING to simulate a mid-run failure
        plan.steps[-1].status = "PENDING"
        plan.save()

        helpers.reset_celery_batch_log()

        # Second run with recover=True — only the PENDING step should execute
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
            config_path="/fake/config.yaml",
            recover=True,
        )

        assert (
            len(helpers.celery_batch_log) == 1
        ), f"Expected only 1 step to execute on recovery, got {len(helpers.celery_batch_log)}"

    def test_no_plan_without_config_path(self, tmp_plans_dir: Path) -> None:
        """No plan file is created when config_path is omitted."""
        dag = _make_dag({"task.a": {"function": f"{HELPERS}.record_a_celery"}})
        dag.run(
            "task.a",
            run_dependencies=False,
            executor="celery",
            batch_interval=_INTERVAL,
            batch_mode="task",
            start_date=_START,
            end_date=_END,
        )
        assert list(tmp_plans_dir.glob("*.json")) == []
