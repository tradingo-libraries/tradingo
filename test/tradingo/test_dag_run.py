"""Tests for DAGRun — the unified run handle returned by all DAG run methods.

Every run method returns a DAGRun. The two fundamental usage patterns are:

Synchronous (default, background=False):
    dag_run = dag.run("task", ...)      # blocks until complete
    assert dag_run.is_done              # always True on return
    print(dag_run.summary())            # {"completed": N, "failed": 0, ...}

Background (background=True):
    dag_run = dag.run("task", ..., background=True)  # returns immediately
    dag_run.wait()                      # block + raise on failure
    pd.DataFrame(dag_run.steps())       # per-step status table

Both patterns share the same DAGRun API regardless of executor (thread pool,
process pool, Celery, batched).
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingo.dag import DAG, DAGRun

from . import _dag_helpers as helpers

HELPERS = "test.tradingo._dag_helpers"

# ---------------------------------------------------------------------------
# Module-level builders — reused across all test classes
# ---------------------------------------------------------------------------

_START = pd.Timestamp("2024-01-01")
_END = pd.Timestamp("2024-01-11")  # 10 days → 2 × 5-day intervals
_INTERVAL = pd.Timedelta(days=5)


def _make_dag(task_specs: dict[str, dict[str, object]]) -> DAG:
    """Build a minimal DAG from a compact spec dict.

    Each key is a task name; values support:
      function   — dotted path (default: helpers.noop)
      depends_on — list of upstream task names (default: [])
    """
    return DAG.from_config(
        {
            name: {
                "function": spec.get("function", f"{HELPERS}.noop"),
                "depends_on": spec.get("depends_on", []),
                "params": {},
            }
            for name, spec in task_specs.items()
        }
    )


@pytest.fixture(autouse=True)
def _reset() -> None:
    helpers.reset()


# ---------------------------------------------------------------------------
# Synchronous: all modes return a finished DAGRun immediately
# ---------------------------------------------------------------------------


class TestSynchronousReturn:
    """background=False (default): run blocks, returns a completed DAGRun."""

    def test_single_task(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a"}})

        dag_run = dag.run("a", force_rerun=True)

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done
        assert helpers.call_log == ["a"]

    def test_run_parallel_returns_dag_run(self) -> None:
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )

        dag_run = dag.run_parallel(
            "b", run_dependencies=True, max_workers=4, force_rerun=True
        )

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done
        assert set(helpers.call_log) == {"a", "b"}

    def test_run_batched_returns_dag_run(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a_dated"}})

        dag_run = dag.run_batched(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            start_date=_START,
            end_date=_END,
        )

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done
        assert len(helpers.call_log_dated) == 2

    def test_run_multiprocess_returns_dag_run(self) -> None:
        dag = _make_dag({"a": {}, "b": {"depends_on": ["a"]}})

        dag_run = dag.run(
            "b", run_dependencies=True, executor="process", force_rerun=True
        )

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done

    def test_dispatcher_parallel_path_returns_dag_run(self) -> None:
        """dag.run with max_workers>1 routes through run_parallel."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )

        dag_run = dag.run("b", run_dependencies=True, max_workers=4, force_rerun=True)

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done

    def test_dispatcher_batched_path_returns_dag_run(self) -> None:
        """dag.run with batch_interval routes through run_batched."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a_dated"}})

        dag_run = dag.run(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            start_date=_START,
            end_date=_END,
        )

        assert isinstance(dag_run, DAGRun)
        assert dag_run.is_done


# ---------------------------------------------------------------------------
# Background: returns immediately; wait() blocks until done
# ---------------------------------------------------------------------------


class TestBackground:
    """background=True: DAGRun is returned while the run is still in flight."""

    def test_single_task_returns_before_completion(self) -> None:
        """DAGRun is in hand before the task has finished."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.blocking_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)

        assert isinstance(dag_run, DAGRun)
        # Wait for the task to signal it has started
        assert helpers.gate_a.wait(timeout=5), "task never started"
        assert not dag_run.is_done

        helpers.gate_c.set()  # release the task
        dag_run.wait()
        assert dag_run.is_done

    def test_parallel_background_wait_completes(self) -> None:
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )

        dag_run = dag.run(
            "b",
            run_dependencies=True,
            max_workers=4,
            force_rerun=True,
            background=True,
        )
        dag_run.wait()

        assert dag_run.is_done
        assert helpers.call_log == ["a", "b"]

    def test_batched_background_wait_completes(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a_dated"}})

        dag_run = dag.run(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            background=True,
            start_date=_START,
            end_date=_END,
        )
        dag_run.wait()

        assert dag_run.is_done
        assert len(helpers.call_log_dated) == 2

    def test_background_is_done_transitions_false_then_true(self) -> None:
        """is_done moves from False (running) to True (finished)."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.blocking_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)
        helpers.gate_a.wait(timeout=5)

        assert not dag_run.is_done

        helpers.gate_c.set()
        dag_run.wait()

        assert dag_run.is_done

    def test_wait_raises_on_task_failure(self) -> None:
        """wait() raises ExceptionGroup when a task fails."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)

        with pytest.raises(ExceptionGroup) as exc_info:
            dag_run.wait()
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert dag_run.is_done

    def test_run_parallel_method_background(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.blocking_a"}})

        dag_run = dag.run_parallel(
            "a", run_dependencies=False, force_rerun=True, background=True
        )
        helpers.gate_a.wait(timeout=5)
        assert not dag_run.is_done

        helpers.gate_c.set()
        dag_run.wait()
        assert dag_run.is_done


# ---------------------------------------------------------------------------
# summary() — live count snapshot
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a dict with counts by state."""

    def test_all_completed_after_successful_run(self) -> None:
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )

        dag_run = dag.run("b", run_dependencies=True, max_workers=4, force_rerun=True)

        s = dag_run.summary()
        assert s == {
            "not_started": 0,
            "submitted": 0,
            "completed": 2,
            "failed": 0,
            "total": 2,
        }

    def test_counts_during_in_flight_run(self) -> None:
        """While a task is running, submitted > 0 and completed < total."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.blocking_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)
        helpers.gate_a.wait(timeout=5)

        s = dag_run.summary()
        assert s["total"] == 1
        assert s["completed"] == 0

        helpers.gate_c.set()
        dag_run.wait()

        s = dag_run.summary()
        assert s["completed"] == 1
        assert s["failed"] == 0

    def test_failed_tasks_reflected_in_summary(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)
        with pytest.raises(ExceptionGroup):
            dag_run.wait()

        s = dag_run.summary()
        assert s["failed"] == 1
        assert s["completed"] == 0


# ---------------------------------------------------------------------------
# steps() — per-step status table
# ---------------------------------------------------------------------------


class TestSteps:
    """steps() returns one row per step with status, timestamps, and task_id."""

    def test_column_names(self) -> None:
        dag = _make_dag({"a": {}})
        dag_run = dag.run("a", force_rerun=True)

        rows = dag_run.steps()

        assert len(rows) == 1
        assert set(rows[0].keys()) == {
            "index",
            "task_name",
            "start_date",
            "end_date",
            "status",
            "task_id",
        }

    def test_non_batched_timestamps_are_none(self) -> None:
        """Thread/process pool modes have no time dimension — timestamps are None."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a"}})
        dag_run = dag.run("a", force_rerun=True)

        rows = dag_run.steps()

        assert rows[0]["task_name"] == "a"
        assert rows[0]["start_date"] is None
        assert rows[0]["end_date"] is None

    def test_batched_steps_carry_timestamps(self) -> None:
        """Batched steps identify the time chunk they cover."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a_dated"}})
        dag_run = dag.run(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            start_date=_START,
            end_date=_END,
        )

        rows = dag_run.steps()

        assert len(rows) == 2
        assert all(row["start_date"] is not None for row in rows)
        assert all(row["end_date"] is not None for row in rows)

    def test_task_id_is_none_for_pool_executors(self) -> None:
        """Thread and process pool tasks have no external task id."""
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )
        dag_run = dag.run("b", run_dependencies=True, max_workers=4, force_rerun=True)

        rows = dag_run.steps()

        assert all(row["task_id"] is None for row in rows)

    def test_successful_steps_have_success_status(self) -> None:
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
            }
        )
        dag_run = dag.run("b", run_dependencies=True, max_workers=4, force_rerun=True)

        rows = dag_run.steps()

        assert all(row["status"] == "SUCCESS" for row in rows)

    def test_failed_step_has_failed_status(self) -> None:
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})
        dag_run = dag.run("a", force_rerun=True, background=True)
        with pytest.raises(ExceptionGroup):
            dag_run.wait()

        rows = dag_run.steps()

        assert rows[0]["status"] == "FAILED"

    def test_steps_ordered_by_index(self) -> None:
        """Rows are ordered by execution index, not insertion order."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.record_a_dated"}})
        dag_run = dag.run(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            start_date=_START,
            end_date=_END,
        )

        rows = dag_run.steps()
        indices = [row["index"] for row in rows]

        assert indices == sorted(indices)

    def test_multi_task_steps_cover_all_tasks(self) -> None:
        dag = _make_dag(
            {
                "a": {"function": f"{HELPERS}.record_a"},
                "b": {"function": f"{HELPERS}.record_b", "depends_on": ["a"]},
                "c": {"function": f"{HELPERS}.record_c", "depends_on": ["b"]},
            }
        )
        dag_run = dag.run("c", run_dependencies=True, max_workers=4, force_rerun=True)

        task_names = {row["task_name"] for row in dag_run.steps()}

        assert task_names == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Errors are captured in DAGRun._errors and raised by wait()."""

    def test_synchronous_failure_raises_directly(self) -> None:
        """background=False, single task: the raw exception propagates directly.

        Multi-task executors (run_parallel, run_batched) wrap errors in an
        ExceptionGroup because they can collect multiple failures; the single-task
        path re-raises the original exception unchanged.
        """
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        with pytest.raises(ValueError, match="task_a failed"):
            dag.run("a", force_rerun=True)

    def test_background_failure_raised_by_wait(self) -> None:
        """background=True: exception is deferred until wait() is called."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        dag_run = dag.run("a", force_rerun=True, background=True)

        with pytest.raises(ExceptionGroup) as exc_info:
            dag_run.wait()
        assert isinstance(exc_info.value.exceptions[0], ValueError)

    def test_two_independent_failures_both_collected(self) -> None:
        """Independent failing tasks: both errors appear in the ExceptionGroup."""
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

        error_types = {type(e) for e in exc_info.value.exceptions}
        assert ValueError in error_types
        assert RuntimeError in error_types

    def test_batched_failure_captured_in_dag_run(self) -> None:
        """A failing batched task: DAGRun reflects the failure after wait()."""
        dag = _make_dag({"a": {"function": f"{HELPERS}.fail_a"}})

        dag_run = dag.run(
            "a",
            batch_interval=_INTERVAL,
            batch_mode="stepped",
            run_dependencies=False,
            force_rerun=True,
            background=True,
            start_date=_START,
            end_date=_END,
        )

        with pytest.raises(ExceptionGroup):
            dag_run.wait()

        s = dag_run.summary()
        assert s["failed"] > 0
        assert s["completed"] == 0
