"""DAG generation logic."""

from __future__ import annotations

import importlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd

from . import symbols
from .config import ConfigLoadError
from .execution_plan import ExecutionPlan
from .worker import serialize_task

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State of a task."""

    PENDING = "PENDING"
    FAILED = "FAILED"
    SUCCESS = "SUCCESS"


class Task:
    """
    Tradingo base task.
    It wraps an arbitrary function via configs, handling args, kwargs,
    input, output and other dependencies.
    """

    def __init__(
        self,
        name: str,
        function: str,
        task_args: tuple[Any, ...],
        task_kwargs: dict[str, Any],
        symbols_out: list[str],
        symbols_in: dict[str, str],
        load_args: dict[str, Any],
        publish_args: dict[str, Any],
        dependencies: Iterable[str] = (),
    ):
        self._function = function
        self.name = name
        self.task_args = task_args
        self.task_kwargs = task_kwargs
        self._dependencies = list(dependencies)
        self._resolved_dependencies: list[Task] = []
        self._state = TaskState.PENDING
        self._state_lock = threading.Lock()
        self.symbols_out = symbols_out
        self.symbols_in = symbols_in
        self.load_args = load_args
        self.publish_args = publish_args
        self._cached_function: Callable[..., Any] | None = None

    @property
    def state(self) -> TaskState:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value: TaskState) -> None:
        with self._state_lock:
            self._state = value

    def __repr__(self) -> str:
        return (
            f"Task(name='{self.name}', function='{self._function}',"
            f" task_args={self.task_args},"
            f" task_kwargs={self.task_kwargs},"
            f" dependcies={self._dependencies}, "
            ")"
        )

    @property
    def function(self) -> Callable[..., Any]:
        """task function — built once and cached."""
        if self._cached_function is not None:
            return self._cached_function

        module, function_name = self._function.rsplit(".", maxsplit=1)
        function: Callable[..., Any] = getattr(
            importlib.import_module(module), function_name
        )

        function = symbols.symbol_publisher(
            *self.symbols_out,
            **self.publish_args,
            metadata=serialize_task(self),
        )(function)

        if self.symbols_in:
            function = symbols.symbol_provider(
                symbol_prefix=self.load_args.pop("symbol_prefix", ""),
                no_date=self.load_args.pop("no_date", None),
                **self.symbols_in,
                **self.load_args,
            )(function)

        self._cached_function = function
        return function

    @staticmethod
    def prepare_kwargs(
        task_kwargs: dict[str, Any], global_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """prepare kwargs to be passed to the function"""
        task_kwargs.update(global_kwargs)
        return task_kwargs

    def run(
        self,
        *args: object,
        run_dependencies: bool | int = False,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        **kwargs: object,
    ) -> None:
        """run this task. optinally run also dependency tasks"""
        if run_dependencies:
            if isinstance(run_dependencies, int):
                run_dependencies -= 1
            for dependency in self.dependencies:
                if skip_deps and skip_deps.match(dependency.name):
                    continue
                dependency.run(
                    *args,
                    run_dependencies=run_dependencies,
                    skip_deps=skip_deps,
                    force_rerun=force_rerun,
                    **kwargs,
                )

        state = self.state
        try:
            task_kwargs = self.prepare_kwargs(self.task_kwargs, kwargs)
            if self.state == TaskState.PENDING or force_rerun:
                state = TaskState.FAILED
                print(f"Running {self}")
                self.function(*self.task_args, *args, **task_kwargs)
                self.state = state = TaskState.SUCCESS
        finally:
            self.state = state

    def add_dependencies(self, *dependency: str) -> None:
        """add dependencies to this task"""
        self._dependencies.extend(dependency)

    def resolve_dependencies(self, tasks: dict[str, Task]) -> None:
        """resolve dependencies to this task"""
        self._resolved_dependencies.extend(
            tasks[dep_name] for dep_name in self._dependencies
        )

    @property
    def dependencies(self) -> list[Task]:
        """this tasks' other dependency tasks"""
        return self._resolved_dependencies

    @property
    def dependency_names(self) -> list[str]:
        """this tasks' other dependency task names"""
        return self._dependencies


class Stage(Task):
    """A group of tasks executed sequentially as a single unit.

    Eliminates inter-task overhead (e.g. Celery round-trips) by running
    multiple tasks in a single process. Each subtask still performs its
    own ArcticDB I/O via the standard symbol_provider/symbol_publisher
    decorators, so all intermediate results are persisted as normal.

    Config syntax::

        my.stage.name:
          stage:
            - task.a
            - task.b
            - task.c

    The listed tasks must be defined elsewhere in the config with their
    own ``function``, ``symbols_in``, ``symbols_out``, and ``params``.
    Dependencies between stage members are resolved internally; external
    dependencies are auto-computed. Any task outside the stage that depends
    on a stage member is automatically re-wired to depend on the stage.
    """

    def __init__(self, name: str, tasks: list[Task]):
        member_names = {t.name for t in tasks}
        external_deps = list(
            dict.fromkeys(
                dep for t in tasks for dep in t._dependencies if dep not in member_names
            )
        )
        super().__init__(
            name=name,
            function="",  # overridden by property
            task_args=(),
            # Store subtask kwargs in task_kwargs so Airflow can template-render
            # any Jinja2 variables (e.g. {{ data_interval_start }}) within them.
            task_kwargs={
                "_stage_steps": [
                    {"args": list(t.task_args), "kwargs": t.task_kwargs} for t in tasks
                ]
            },
            symbols_out=[s for t in tasks for s in t.symbols_out],
            symbols_in={},
            load_args={},
            publish_args={},
            dependencies=external_deps,
        )
        self._tasks = tasks

    def __repr__(self) -> str:
        subtask_names = [t.name for t in self._tasks]
        return f"Stage(name='{self.name}', tasks={subtask_names})"

    @property
    def function(self) -> Callable[..., Any]:
        """Return a callable that runs all subtasks sequentially."""
        subtasks = self._tasks

        def run_stage(
            arctic: Any,
            dry_run: bool = False,
            _stage_steps: list[dict[str, Any]] | None = None,
            **global_kwargs: Any,
        ) -> None:
            steps = _stage_steps or [{"args": (), "kwargs": {}} for _ in subtasks]
            for task, step in zip(subtasks, steps):
                kwargs = {**step["kwargs"], **global_kwargs}
                logger.info("Stage %s: running subtask %s", self.name, task.name)
                task.function(
                    *step["args"],
                    arctic=arctic,
                    dry_run=dry_run,
                    **kwargs,
                )

        return run_stage


class BatchMode(Enum):
    """Controls how dependencies are batched relative to time intervals."""

    STEPPED = "stepped"
    TASK = "task"
    DEPS_FIRST = "deps-first"


def generate_intervals(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start_date, end_date) into sub-intervals of size ``interval``.

    The last chunk may be shorter than ``interval`` if the range is not
    evenly divisible.
    """
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + interval, end_date)
        intervals.append((cursor, chunk_end))
        cursor = chunk_end
    return intervals


def generate_batch_schedule(
    task_names: list[str],
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    batch_mode: BatchMode,
    full_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Generate an ordered ``(task_name, start, end)`` execution schedule.

    Args:
        task_names: All tasks (deps + target) in topological order.
        intervals: The chunked time intervals.
        batch_mode: One of STEPPED, TASK, DEPS_FIRST.
        full_start: Original start_date (used in deps-first mode).
        full_end: Original end_date (used in deps-first mode).
    """
    if batch_mode == BatchMode.STEPPED:
        return [(t, start, end) for start, end in intervals for t in task_names]

    if batch_mode == BatchMode.TASK:
        return [(t, start, end) for t in task_names for start, end in intervals]

    if batch_mode == BatchMode.DEPS_FIRST:
        target = task_names[-1]
        deps = task_names[:-1]
        schedule: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
            (dep, full_start, full_end) for dep in deps
        ]
        schedule.extend((target, s, e) for s, e in intervals)
        return schedule

    raise ValueError(f"Unknown batch mode: {batch_mode}")


def collect_task_configs(
    config: dict[str, Any], _tasks: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """gather all task specifications from a config, accounting for dependencies."""
    tasks = _tasks or {}

    for key, value in config.items():
        if isinstance(value, dict) and ("depends_on" in value or "stage" in value):
            # its a task or stage, collect it
            tasks[key] = value
        elif isinstance(value, dict):
            # its a set of tasks collect them
            tasks.update(collect_task_configs(value, tasks))

    return tasks


def _topo_sort_tasks(tasks: list[Task], member_names: set[str]) -> list[Task]:
    """Topologically sort tasks within a stage based on internal dependencies."""
    by_name = {t.name: t for t in tasks}
    sorted_tasks: list[Task] = []
    sorted_names: set[str] = set()

    remaining = dict(by_name)
    while remaining:
        ready = [
            name
            for name, task in remaining.items()
            if all(
                dep not in member_names or dep in sorted_names
                for dep in task._dependencies
            )
        ]
        if not ready:
            raise ConfigLoadError(
                f"Circular dependency in stage among: {list(remaining.keys())}"
            )
        for name in ready:
            sorted_tasks.append(remaining.pop(name))
            sorted_names.add(name)

    return sorted_tasks


# Step identity: (task_name, chunk_start, chunk_end)
# Timestamps are None for non-batched (single-task) modes.
Step = tuple[str, pd.Timestamp | None, pd.Timestamp | None]


class DAGRun:
    """Live handle to a running DAG dispatch.

    Returned by all run methods.  When ``background=True`` the dispatch loop
    runs in a background daemon thread and this handle is returned
    immediately.  When ``background=False`` the object is returned after all
    work has finished.

    Batched modes use ``(task_name, chunk_start, chunk_end)`` as the step
    identity; non-batched modes use ``(task_name, None, None)``.

    Usage::

        dag_run = dag.run_parallel(..., background=True)
        print(dag_run.summary())       # check progress while running
        dag_run.wait()                 # block until done, raises on errors

    Or as a DataFrame in a notebook::

        import pandas as pd
        pd.DataFrame(dag_run.steps())
    """

    def __init__(
        self,
        plan: ExecutionPlan | None,
        step_index: dict[Step, int],
        already_completed: set[Step],
    ) -> None:
        self.plan = plan
        self._step_index = step_index
        self._lock = threading.Lock()
        self._not_started: set[Step] = set(step_index.keys()) - already_completed
        self._submitted: dict[Step, Any] = {}
        self._completed: set[Step] = set(already_completed)
        self._failed: dict[Step, BaseException] = {}
        self._done_event = threading.Event()
        self._errors: list[BaseException] = []

    # ------------------------------------------------------------------
    # Internal callbacks (called by the dispatch thread)
    # ------------------------------------------------------------------

    def _on_submitted(self, step: Step, result: Any) -> None:
        with self._lock:
            self._not_started.discard(step)
            self._submitted[step] = result

    def _on_completed(self, step: Step) -> None:
        with self._lock:
            self._submitted.pop(step, None)
            self._completed.add(step)

    def _on_failed(self, step: Step, exc: BaseException) -> None:
        with self._lock:
            self._submitted.pop(step, None)
            self._failed[step] = exc
        self._errors.append(exc)

    def _mark_done(self) -> None:
        self._done_event.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """True once the dispatch loop has finished (all steps terminal)."""
        return self._done_event.is_set()

    def summary(self) -> dict[str, int]:
        """Snapshot of step counts by state."""
        with self._lock:
            return {
                "not_started": len(self._not_started),
                "submitted": len(self._submitted),
                "completed": len(self._completed),
                "failed": len(self._failed),
                "total": len(self._step_index),
            }

    def steps(self) -> list[dict[str, Any]]:
        """Per-step status snapshot ordered by plan index — suitable for pd.DataFrame."""
        with self._lock:
            rows = []
            for step, idx in sorted(self._step_index.items(), key=lambda x: x[1]):
                task_name, start, end = step
                if step in self._failed:
                    status = "FAILED"
                elif step in self._completed:
                    status = "SUCCESS"
                elif step in self._submitted:
                    status = "SUBMITTED"
                else:
                    status = "PENDING"
                handle = self._submitted.get(step)
                row: dict[str, Any] = {
                    "index": idx,
                    "task_name": task_name,
                    "start_date": start,
                    "end_date": end,
                    "status": status,
                    "task_id": (
                        getattr(handle, "id", None) if handle is not None else None
                    ),
                }
                rows.append(row)
            return rows

    def wait(self) -> None:
        """Block until the run completes, then raise if any steps failed."""
        self._done_event.wait()
        if self._errors:
            raise ExceptionGroup(
                f"{len(self._errors)} step(s) failed",
                [
                    e if isinstance(e, Exception) else Exception(str(e))
                    for e in self._errors
                ],
            )

    def stop(self) -> int:
        """Cancel or revoke all in-flight tasks. Returns number of tasks stopped.

        For thread/process pool tasks calls ``Future.cancel()`` (only
        effective for tasks not yet started).  For Celery tasks revokes via
        the Celery control API.  Also marks stopped steps in the execution
        plan (if one exists) so ``--recover`` will retry them.
        """
        with self._lock:
            submitted = list(self._submitted.items())
        stopped = 0
        for step, handle in submitted:
            if handle is None:
                pass
            elif hasattr(handle, "cancel"):  # concurrent.futures.Future
                handle.cancel()
            else:  # Celery AsyncResult
                from tradingo.worker import require_celery

                celery_app = require_celery()
                celery_app.control.revoke(handle.id, terminate=True)
            if self.plan is not None:
                idx = self._step_index.get(step)
                if idx is not None:
                    self.plan.mark_step(idx, "FAILED")
            stopped += 1
        return stopped

    def __repr__(self) -> str:
        s = self.summary()
        status = "done" if self.is_done else "running"
        return (
            f"DAGRun({status}, completed={s['completed']}, "
            f"submitted={s['submitted']}, failed={s['failed']}, "
            f"total={s['total']})"
        )


class DAG(dict[str, Task]):
    """Tradingo DAG"""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DAG:
        """create a DAG from one config."""

        task_configs = collect_task_configs(config)

        # Separate stage definitions from normal tasks
        stage_configs = {k: v for k, v in task_configs.items() if "stage" in v}
        normal_configs = {k: v for k, v in task_configs.items() if "stage" not in v}

        tasks: dict[str, Task] = {}

        for task_name, task_config in normal_configs.items():
            if not task_config.get("enabled", True):
                continue
            params = task_config["params"]
            try:
                tasks[task_name] = Task(
                    name=task_name,
                    function=task_config["function"],
                    task_args=(),
                    task_kwargs=params,
                    dependencies=task_config["depends_on"],
                    symbols_in=task_config.get("symbols_in", {}),
                    load_args=task_config.get("load_args", {}),
                    publish_args=task_config.get("publish_args", {}),
                    symbols_out=task_config.get("symbols_out", []),
                )
            except KeyError as ex:
                raise ConfigLoadError(
                    f"{task_name} is missing setting {ex.args[0]}"
                ) from ex

        # Process stages: pull subtasks out, create Stage objects, re-wire deps
        for stage_name, stage_config in stage_configs.items():
            subtask_names = stage_config["stage"]
            member_names = set(subtask_names)

            subtasks: list[Task] = []
            for name in subtask_names:
                if name not in tasks:
                    raise ConfigLoadError(
                        f"Stage '{stage_name}': task '{name}' not found"
                    )
                subtasks.append(tasks.pop(name))

            sorted_subtasks = _topo_sort_tasks(subtasks, member_names)
            stage = Stage(stage_name, sorted_subtasks)

            # Add any explicit depends_on from the stage config
            if "depends_on" in stage_config:
                for dep in stage_config["depends_on"]:
                    if dep not in stage._dependencies:
                        stage._dependencies.append(dep)

            tasks[stage_name] = stage

            # Re-wire: any task depending on a stage member now depends
            # on the stage itself
            for task in tasks.values():
                rewired = []
                for dep in task._dependencies:
                    if dep in member_names:
                        rewired.append(stage_name)
                    else:
                        rewired.append(dep)
                task._dependencies = list(dict.fromkeys(rewired))

        for task_name, task in tasks.items():
            try:
                task.resolve_dependencies(tasks)
            except KeyError as ex:
                raise ConfigLoadError(
                    f"Missing task in dag '{ex.args[0]}' for '{task_name}'"
                ) from ex

        return cls(tasks)

    @property
    def _state_filepath(self) -> Path:
        return Path.home() / ".tradingo/dag-state.json"

    def print(self, include_dependencies: bool = True) -> None:
        """print the tasks of this DAG."""
        for task_name, task in self.items():
            print(f"{task_name}:")
            if not include_dependencies:
                continue
            if task:
                for dep in task.dependency_names:
                    print(f"  - {dep}")
            else:
                print("  No dependencies")
            print()

    def get_symbols(self) -> list[str]:
        """list all symbols produced by this DAG."""
        return [
            task
            for subl in (task.symbols_out for task in self.values())
            for task in subl
        ]

    def _collect_runnable_tasks(
        self,
        task_name: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
    ) -> set[str]:
        """Walk the dependency graph and return the set of task names to run."""
        collected: set[str] = set()

        def _walk(name: str, depth: bool | int) -> None:
            if name in collected:
                return
            collected.add(name)
            if not depth:
                return
            next_depth: bool | int = (
                depth - 1
                if isinstance(depth, int) and not isinstance(depth, bool)
                else depth
            )
            for dep in self[name].dependencies:
                if skip_deps and skip_deps.match(dep.name):
                    continue
                _walk(dep.name, next_depth)

        _walk(task_name, run_dependencies)
        return collected

    def _topo_sort_runnable(self, runnable: set[str]) -> list[str]:
        """Topologically sort a set of task names by their dependencies."""
        in_degree: dict[str, int] = {name: 0 for name in runnable}
        dependents: dict[str, list[str]] = {name: [] for name in runnable}

        for name in runnable:
            for dep in self[name].dependencies:
                if dep.name in runnable:
                    in_degree[name] += 1
                    dependents[dep.name].append(name)

        sorted_names: list[str] = []
        ready = [n for n, d in in_degree.items() if d == 0]

        while ready:
            name = ready.pop(0)
            sorted_names.append(name)
            for downstream in dependents[name]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    ready.append(downstream)

        return sorted_names

    def _run_intervals_sequential(
        self,
        task_name: str,
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        force_rerun: bool,
        plan: ExecutionPlan | None = None,
        step_index: dict[Step, int] | None = None,
        dag_run: DAGRun | None = None,
        **kwargs: object,
    ) -> None:
        """Run one task across all its intervals sequentially.

        ``clean`` is only applied to the first interval; subsequent
        intervals strip it so they append/update rather than overwrite.
        """
        task = self[task_name]
        for i, (start, end) in enumerate(intervals):
            step: Step = (task_name, start, end)
            step_kwargs = dict(kwargs)
            step_kwargs["start_date"] = start
            step_kwargs["end_date"] = end
            if i > 0:
                step_kwargs.pop("clean", None)
            task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), step_kwargs)
            if task.state == TaskState.PENDING or force_rerun:
                logger.info("Running %s [%s -> %s]", task_name, start, end)
                print(f"Running {task} [{start} -> {end}]")
                if dag_run is not None:
                    dag_run._on_submitted(step, None)
                try:
                    task.function(*task.task_args, **task_kwargs)
                    if plan is not None and step_index is not None:
                        idx = step_index.get(step)
                        if idx is not None:
                            plan.mark_step(idx, "SUCCESS")
                    if dag_run is not None:
                        dag_run._on_completed(step)
                except Exception as exc:
                    if plan is not None and step_index is not None:
                        idx = step_index.get(step)
                        if idx is not None:
                            plan.mark_step(idx, "FAILED")
                    if dag_run is not None:
                        dag_run._on_failed(step, exc)
                    raise

    def run_batched(
        self,
        task_name: str,
        batch_interval: pd.Timedelta,
        batch_mode: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        max_workers: int = 1,
        recover: bool = False,
        config_path: str | None = None,
        background: bool = False,
        **kwargs: object,
    ) -> DAGRun:
        """Run tasks in batched time intervals."""
        start_date = cast(pd.Timestamp | None, kwargs.get("start_date"))
        end_date = cast(pd.Timestamp | None, kwargs.get("end_date"))
        if start_date is None or end_date is None:
            raise ValueError(
                "--start-date and --end-date are required when using --batch-interval"
            )

        mode = BatchMode(batch_mode)
        intervals = generate_intervals(
            pd.Timestamp(start_date), pd.Timestamp(end_date), batch_interval
        )

        # Collect dependency names in topological order
        dep_names: list[str] = []
        if run_dependencies:
            runnable = self._collect_runnable_tasks(
                task_name, run_dependencies, skip_deps
            )
            runnable.discard(task_name)
            dep_names = self._topo_sort_runnable(runnable)

        all_task_names = dep_names + [task_name]

        schedule = generate_batch_schedule(
            task_names=all_task_names,
            intervals=intervals,
            batch_mode=mode,
            full_start=pd.Timestamp(start_date),
            full_end=pd.Timestamp(end_date),
        )

        logger.info(
            "Batch schedule: %d steps across %d intervals (mode=%s)",
            len(schedule),
            len(intervals),
            mode.value,
        )

        # Build or recover execution plan
        plan: ExecutionPlan | None = None
        if config_path is not None:
            plan_key = ExecutionPlan.make_key(
                config_path=config_path,
                task_name=task_name,
                start_date=str(start_date),
                end_date=str(end_date),
                batch_interval=str(batch_interval),
                batch_mode=batch_mode,
                with_deps=str(run_dependencies),
            )
            plan_params = {
                "config_path": config_path,
                "task_name": task_name,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "batch_interval": str(batch_interval),
                "batch_mode": batch_mode,
                "with_deps": str(run_dependencies),
            }
            if recover and not force_rerun:
                plan = ExecutionPlan.load(plan_key)
                if plan is not None:
                    resume_idx = plan.first_non_success()
                    if resume_idx is None:
                        logger.info("All steps already completed — nothing to do.")
                        done = DAGRun(
                            plan=plan,
                            step_index={
                                (t, s, e): i for i, (t, s, e) in enumerate(schedule)
                            },
                            already_completed={(t, s, e) for t, s, e in schedule},
                        )
                        done._mark_done()
                        return done
                    logger.info(
                        "Recovering execution plan: skipping %d completed steps",
                        resume_idx,
                    )
            if plan is None:
                plan = ExecutionPlan.from_schedule(plan_key, plan_params, schedule)
                plan.save()

        step_index: dict[Step, int] = {
            (t, s, e): i for i, (t, s, e) in enumerate(schedule)
        }
        already_completed: set[Step] = {
            (t, s, e)
            for (t, s, e), idx in step_index.items()
            if plan is not None and plan.steps[idx].status == "SUCCESS"
        }
        dag_run = DAGRun(
            plan=plan, step_index=step_index, already_completed=already_completed
        )

        def _execute() -> None:
            try:
                if max_workers > 1:
                    self._run_batched_parallel(
                        task_name=task_name,
                        dep_names=dep_names,
                        intervals=intervals,
                        mode=mode,
                        run_dependencies=run_dependencies,
                        skip_deps=skip_deps,
                        force_rerun=force_rerun,
                        max_workers=max_workers,
                        plan=plan,
                        step_index=step_index,
                        dag_run=dag_run,
                        **kwargs,
                    )
                    return

                # Sequential execution
                # Track which tasks have run at least once so --clean only applies to
                # the first interval per task (not every subsequent chunk).
                seen_tasks: set[str] = set()
                for step_idx, (step_task_name, chunk_start, chunk_end) in enumerate(
                    schedule
                ):
                    # Skip completed steps when recovering
                    if plan is not None and plan.steps[step_idx].status == "SUCCESS":
                        seen_tasks.add(step_task_name)
                        continue

                    bstep: Step = (step_task_name, chunk_start, chunk_end)
                    task = self[step_task_name]
                    # Reset state so the same task can run again for the next interval
                    task.state = TaskState.PENDING
                    step_kwargs = dict(kwargs)
                    step_kwargs["start_date"] = chunk_start
                    step_kwargs["end_date"] = chunk_end
                    if step_task_name in seen_tasks:
                        step_kwargs.pop("clean", None)
                    task_kwargs = Task.prepare_kwargs(
                        dict(task.task_kwargs), step_kwargs
                    )

                    if task.state == TaskState.PENDING or force_rerun:
                        state = TaskState.FAILED
                        dag_run._on_submitted(bstep, None)
                        try:
                            logger.info(
                                "Running %s [%s -> %s]",
                                step_task_name,
                                chunk_start,
                                chunk_end,
                            )
                            print(f"Running {task} [{chunk_start} -> {chunk_end}]")
                            task.function(*task.task_args, **task_kwargs)
                            state = TaskState.SUCCESS
                            if plan is not None:
                                plan.mark_step(step_idx, "SUCCESS")
                            dag_run._on_completed(bstep)
                        except Exception as exc:
                            if plan is not None:
                                plan.mark_step(step_idx, "FAILED")
                            dag_run._on_failed(bstep, exc)
                            raise
                        finally:
                            task.state = state

                    seen_tasks.add(step_task_name)
            finally:
                dag_run._mark_done()

        if background:

            def _bg() -> None:
                try:
                    _execute()
                except Exception:
                    pass  # errors already captured in dag_run._errors

            threading.Thread(target=_bg, daemon=True, name=f"dag-{task_name}").start()
        else:
            _execute()

        return dag_run

    def _run_batched_parallel(
        self,
        task_name: str,
        dep_names: list[str],
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        mode: BatchMode,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        max_workers: int = 4,
        plan: ExecutionPlan | None = None,
        step_index: dict[Step, int] | None = None,
        dag_run: DAGRun | None = None,
        **kwargs: object,
    ) -> None:
        """Parallel execution of a batched schedule.

        Independent tasks may run in parallel threads, but each task's
        intervals always run sequentially to avoid concurrent writes to the
        same ArcticDB symbol.
        """
        all_task_names = dep_names + [task_name]
        start_date = cast(pd.Timestamp, kwargs["start_date"])
        end_date = cast(pd.Timestamp, kwargs["end_date"])

        if mode == BatchMode.STEPPED:
            # Each interval is a barrier; within an interval, use run_parallel.
            # --clean only applies to the first interval; strip it thereafter.
            current_kwargs = dict(kwargs)
            for chunk_start, chunk_end in intervals:
                chunk_kwargs = {
                    **current_kwargs,
                    "start_date": chunk_start,
                    "end_date": chunk_end,
                }
                # Reset states so tasks can re-run for next interval
                for name in all_task_names:
                    self[name].state = TaskState.PENDING
                if dag_run is not None:
                    for name in all_task_names:
                        dag_run._on_submitted((name, chunk_start, chunk_end), None)
                try:
                    self.run_parallel(
                        task_name,
                        run_dependencies=run_dependencies,
                        skip_deps=skip_deps,
                        force_rerun=force_rerun,
                        max_workers=max_workers,
                        background=False,
                        **chunk_kwargs,
                    )
                    if plan is not None and step_index is not None:
                        for name in all_task_names:
                            idx = step_index.get((name, chunk_start, chunk_end))
                            if idx is not None:
                                plan.mark_step(idx, "SUCCESS")
                    if dag_run is not None:
                        for name in all_task_names:
                            dag_run._on_completed((name, chunk_start, chunk_end))
                except Exception as exc:
                    if plan is not None and step_index is not None:
                        for name in all_task_names:
                            idx = step_index.get((name, chunk_start, chunk_end))
                            if idx is not None:
                                plan.mark_step(idx, "FAILED")
                    if dag_run is not None:
                        for name in all_task_names:
                            dag_run._on_failed((name, chunk_start, chunk_end), exc)
                    raise
                current_kwargs.pop("clean", None)

        elif mode == BatchMode.TASK:
            # Each task runs its intervals sequentially; independent tasks
            # can run in parallel threads.
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def _run_task_intervals(t_name: str) -> None:
                try:
                    self._run_intervals_sequential(
                        t_name,
                        intervals,
                        force_rerun,
                        plan=plan,
                        step_index=step_index,
                        dag_run=dag_run,
                        **kwargs,
                    )
                except Exception as exc:
                    with errors_lock:
                        errors.append(exc)

            # Build dependency graph among the task names to respect ordering
            runnable_set = set(all_task_names)
            in_degree: dict[str, int] = {n: 0 for n in all_task_names}
            dependents: dict[str, list[str]] = {n: [] for n in all_task_names}
            for name in all_task_names:
                for dep in self[name].dependencies:
                    if dep.name in runnable_set:
                        in_degree[name] += 1
                        dependents[dep.name].append(name)

            ready = [n for n, d in in_degree.items() if d == 0]
            pending_futures: set[Future[str]] = set()
            in_degree_lock = threading.Lock()

            def _submit_task(name: str, executor: ThreadPoolExecutor) -> Future[str]:
                def _wrapped() -> str:
                    _run_task_intervals(name)
                    return name

                return executor.submit(_wrapped)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for name in ready:
                    pending_futures.add(_submit_task(name, executor))

                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        finished_name = future.result()
                        with in_degree_lock:
                            for dep_name in dependents[finished_name]:
                                in_degree[dep_name] -= 1
                                if in_degree[dep_name] == 0:
                                    pending_futures.add(
                                        _submit_task(dep_name, executor)
                                    )

            if errors:
                raise ExceptionGroup(
                    f"{len(errors)} task(s) failed in TASK mode", errors
                )

        elif mode == BatchMode.DEPS_FIRST:
            # Phase 1: deps on full range in parallel
            if dep_names:
                full_kwargs = {
                    **kwargs,
                    "start_date": start_date,
                    "end_date": end_date,
                }
                if dag_run is not None:
                    for dep_name in dep_names:
                        dag_run._on_submitted((dep_name, start_date, end_date), None)
                try:
                    self.run_parallel(
                        task_name,
                        run_dependencies=run_dependencies,
                        skip_deps=skip_deps,
                        force_rerun=force_rerun,
                        max_workers=max_workers,
                        background=False,
                        **full_kwargs,
                    )
                    if plan is not None and step_index is not None:
                        for dep_name in dep_names:
                            idx = step_index.get((dep_name, start_date, end_date))
                            if idx is not None:
                                plan.mark_step(idx, "SUCCESS")
                    if dag_run is not None:
                        for dep_name in dep_names:
                            dag_run._on_completed((dep_name, start_date, end_date))
                except Exception as exc:
                    if plan is not None and step_index is not None:
                        for dep_name in dep_names:
                            idx = step_index.get((dep_name, start_date, end_date))
                            if idx is not None:
                                plan.mark_step(idx, "FAILED")
                    if dag_run is not None:
                        for dep_name in dep_names:
                            dag_run._on_failed((dep_name, start_date, end_date), exc)
                    raise
            # Phase 2: target batched sequentially (same task, sequential intervals)
            self._run_intervals_sequential(
                task_name,
                intervals,
                force_rerun,
                plan=plan,
                step_index=step_index,
                dag_run=dag_run,
                **kwargs,
            )

    def _run_pool(
        self,
        make_future: Callable[[str], Future[str]],
        task_name: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        dag_run: DAGRun | None = None,
    ) -> None:
        """Topology-aware dispatch loop shared by thread and process pool executors.

        ``make_future`` must be a callable that submits a task by name to the
        pool and returns the resulting ``Future[str]`` (resolving to the task
        name on success, raising on failure).  All state updates happen in the
        calling thread — child workers must not mutate ``self``.
        """
        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)

        in_degree: dict[str, int] = {name: 0 for name in runnable}
        dependents: dict[str, list[str]] = {name: [] for name in runnable}

        for name in runnable:
            for dep in self[name].dependencies:
                if dep.name in runnable:
                    in_degree[name] += 1
                    dependents[dep.name].append(name)

        errors: list[Exception] = []
        future_to_name: dict[Future[str], str] = {}
        pending_futures: set[Future[str]] = set()

        def _submit(name: str) -> None:
            f = make_future(name)
            future_to_name[f] = name
            pending_futures.add(f)
            if dag_run is not None:
                dag_run._on_submitted((name, None, None), f)

        for name in [n for n, d in in_degree.items() if d == 0]:
            _submit(name)

        while pending_futures:
            done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
            for future in done:
                name = future_to_name.pop(future)
                try:
                    future.result()
                    self[name].state = TaskState.SUCCESS
                    if dag_run is not None:
                        dag_run._on_completed((name, None, None))
                    for dep_name in dependents[name]:
                        in_degree[dep_name] -= 1
                        if in_degree[dep_name] == 0:
                            _submit(dep_name)
                except Exception as exc:
                    self[name].state = TaskState.FAILED
                    if dag_run is not None:
                        dag_run._on_failed((name, None, None), exc)
                    errors.append(exc)

        if dag_run is not None:
            dag_run._mark_done()
        if errors:
            raise ExceptionGroup(f"{len(errors)} task(s) failed", errors)

    def run_parallel(
        self,
        task_name: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        max_workers: int = 4,
        background: bool = False,
        **kwargs: object,
    ) -> DAGRun:
        """Run tasks using a thread pool with dynamic dispatch."""
        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)
        step_index: dict[Step, int] = {
            (name, None, None): i
            for i, name in enumerate(self._topo_sort_runnable(runnable))
        }
        dag_run = DAGRun(plan=None, step_index=step_index, already_completed=set())

        def _run_task(name: str) -> str:
            task = self[name]
            task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), dict(kwargs))
            if task.state == TaskState.PENDING or force_rerun:
                print(f"Running {task}")
                task.function(*task.task_args, **task_kwargs)
            return name

        def _execute() -> None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                self._run_pool(
                    lambda name: pool.submit(_run_task, name),
                    task_name,
                    run_dependencies,
                    skip_deps,
                    dag_run=dag_run,
                )

        if background:

            def _bg() -> None:
                try:
                    _execute()
                except Exception:
                    pass  # errors already captured in dag_run._errors

            threading.Thread(target=_bg, daemon=True, name=f"dag-{task_name}").start()
        else:
            _execute()

        return dag_run

    def run_multiprocess(
        self,
        task_name: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        max_workers: int = 4,
        background: bool = False,
        **kwargs: object,
    ) -> DAGRun:
        """Run tasks using a process pool with dynamic dispatch.

        Each task is serialised and dispatched to a worker process via
        ``ProcessPoolExecutor``.  Workers reconstruct the task from its spec
        and connect to ArcticDB independently using ``TP_ARCTIC_URI``.
        """
        from tradingo.worker import (
            run_task_in_process,
            serialize_kwargs,
            serialize_task,
        )

        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)
        step_index: dict[Step, int] = {
            (name, None, None): i
            for i, name in enumerate(self._topo_sort_runnable(runnable))
        }
        dag_run = DAGRun(plan=None, step_index=step_index, already_completed=set())

        serialized_kwargs = serialize_kwargs(dict(kwargs))

        def _execute() -> None:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                self._run_pool(
                    lambda name: pool.submit(
                        run_task_in_process,
                        serialize_task(self[name]),
                        serialized_kwargs,
                    ),
                    task_name,
                    run_dependencies,
                    skip_deps,
                    dag_run=dag_run,
                )

        if background:

            def _bg() -> None:
                try:
                    _execute()
                except Exception:
                    pass  # errors already captured in dag_run._errors

            threading.Thread(target=_bg, daemon=True, name=f"dag-{task_name}").start()
        else:
            _execute()

        return dag_run

    def run_celery(
        self,
        task_name: str,
        run_dependencies: bool | int,
        broker_url: str | None = None,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        background: bool = False,
        **kwargs: object,
    ) -> DAGRun:
        """Dispatch tasks to Celery workers with topology-preserving dispatch.

        Mirrors ``run_parallel`` but sends each task to a remote Celery worker
        instead of a local thread.  Requires the ``[worker]`` extra and
        ``TP_CELERY_BROKER_URL`` / ``TP_ARCTIC_URI`` on all worker nodes.

        Workers are started separately::

            celery -A tradingo.worker worker -Q tradingo --concurrency=4
        """
        from tradingo.worker import (
            QUEUE,
            require_celery,
            serialize_kwargs,
            serialize_task,
        )

        celery_app = require_celery()
        if broker_url:
            celery_app.conf.broker_url = broker_url

        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)
        sorted_names = self._topo_sort_runnable(runnable)
        step_index: dict[Step, int] = {
            (name, None, None): i for i, name in enumerate(sorted_names)
        }
        dag_run = DAGRun(plan=None, step_index=step_index, already_completed=set())

        in_degree: dict[str, int] = {name: 0 for name in runnable}
        dependents: dict[str, list[str]] = {name: [] for name in runnable}
        for name in runnable:
            for dep in self[name].dependencies:
                if dep.name in runnable:
                    in_degree[name] += 1
                    dependents[dep.name].append(name)

        serialized_kwargs = serialize_kwargs(dict(kwargs))

        def _execute() -> None:
            # pending maps task name → Celery AsyncResult
            pending: dict[str, Any] = {}
            errors: list[Exception] = []

            def _submit(name: str) -> None:
                task_spec = serialize_task(self[name])
                result = celery_app.send_task(
                    "tradingo.run_task",
                    args=[task_spec, serialized_kwargs],
                    queue=QUEUE,
                )
                pending[name] = result
                dag_run._on_submitted((name, None, None), result)
                logger.info("Submitted %s to Celery queue '%s'", name, QUEUE)

            for name in [n for n, d in in_degree.items() if d == 0]:
                _submit(name)

            while pending:
                for name in list(pending):
                    result = pending[name]
                    if not result.ready():
                        continue
                    del pending[name]
                    if result.failed():
                        exc = result.result
                        logger.error("Task %s failed: %s", name, exc)
                        err = exc if isinstance(exc, Exception) else Exception(str(exc))
                        dag_run._on_failed((name, None, None), err)
                        errors.append(err)
                    else:
                        logger.info("Task %s completed", name)
                        dag_run._on_completed((name, None, None))
                        for dep_name in dependents[name]:
                            in_degree[dep_name] -= 1
                            if in_degree[dep_name] == 0:
                                _submit(dep_name)

                if pending:
                    time.sleep(0.5)

            dag_run._mark_done()
            if errors:
                raise ExceptionGroup(f"{len(errors)} task(s) failed on Celery", errors)

        if background:

            def _bg() -> None:
                try:
                    _execute()
                except Exception:
                    pass  # errors already captured in dag_run._errors

            threading.Thread(target=_bg, daemon=True, name=f"dag-{task_name}").start()
        else:
            _execute()

        return dag_run

    def run_batched_celery(
        self,
        task_name: str,
        run_dependencies: bool | int,
        batch_interval: pd.Timedelta,
        batch_mode: str = "stepped",
        broker_url: str | None = None,
        skip_deps: re.Pattern[str] | None = None,
        recover: bool = False,
        config_path: str | None = None,
        **kwargs: object,
    ) -> DAGRun:
        """Dispatch batched time-chunk tasks to Celery workers.

        Splits the date range into ``batch_interval`` chunks and dispatches
        each ``(task, chunk)`` pair as a separate Celery task.  Writer
        serialization is enforced: chunks of the same task are never
        dispatched concurrently (ArcticDB does not allow two writers to the
        same symbol).

        Cross-task dependency semantics follow ``batch_mode``:

        - TASK: all chunks of an upstream task must complete before the
          first chunk of a downstream task is dispatched.
        - STEPPED: within each interval, tasks run in topological order;
          the next interval only starts once all tasks for the current
          interval are done.
        - DEPS_FIRST: upstream tasks run across the full date range first,
          then the target task runs chunked.

        ``clean`` is forwarded only for the first chunk of each task;
        subsequent chunks strip it so they append rather than overwrite.
        """
        from tradingo.worker import (
            QUEUE,
            require_celery,
            serialize_kwargs,
            serialize_task,
        )

        celery_app = require_celery()
        if broker_url:
            celery_app.conf.broker_url = broker_url

        start_date = cast(pd.Timestamp | None, kwargs.get("start_date"))
        end_date = cast(pd.Timestamp | None, kwargs.get("end_date"))
        if start_date is None or end_date is None:
            raise ValueError(
                "--start-date and --end-date are required when using --batch-interval"
            )
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        mode = BatchMode(batch_mode)
        intervals = generate_intervals(start_date, end_date, batch_interval)

        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)
        runnable.discard(task_name)
        dep_names = self._topo_sort_runnable(runnable)
        all_task_names = dep_names + [task_name]

        runnable_set = set(all_task_names)
        task_upstream: dict[str, list[str]] = {
            name: [
                dep.name for dep in self[name].dependencies if dep.name in runnable_set
            ]
            for name in all_task_names
        }

        # Build step-level dependency graph.
        # Each step is (task_name, chunk_start, chunk_end).
        # Two constraint types:
        #   1. Writer-serial: (task, chunk_i) → (task, chunk_{i+1})
        #   2. Cross-task: determined by batch_mode
        step_deps: dict[Step, set[Step]] = {}

        if mode == BatchMode.TASK:
            for name in all_task_names:
                for i, (s, e) in enumerate(intervals):
                    deps: set[Step] = set()
                    if i > 0:
                        deps.add((name, intervals[i - 1][0], intervals[i - 1][1]))
                    else:
                        for up in task_upstream[name]:
                            for us, ue in intervals:
                                deps.add((up, us, ue))
                    step_deps[(name, s, e)] = deps

        elif mode == BatchMode.STEPPED:
            for name in all_task_names:
                for i, (s, e) in enumerate(intervals):
                    deps = set()
                    if i > 0:
                        deps.add((name, intervals[i - 1][0], intervals[i - 1][1]))
                    for up in task_upstream[name]:
                        deps.add((up, s, e))
                    step_deps[(name, s, e)] = deps

        elif mode == BatchMode.DEPS_FIRST:
            target = all_task_names[-1]
            for name in all_task_names[:-1]:
                deps = set()
                for up in task_upstream[name]:
                    deps.add((up, start_date, end_date))
                step_deps[(name, start_date, end_date)] = deps
            for i, (s, e) in enumerate(intervals):
                deps = set()
                if i > 0:
                    deps.add((target, intervals[i - 1][0], intervals[i - 1][1]))
                else:
                    for dep_name in all_task_names[:-1]:
                        deps.add((dep_name, start_date, end_date))
                step_deps[(target, s, e)] = deps

        # Build the flat schedule so we can index steps for plan tracking
        schedule = generate_batch_schedule(
            task_names=all_task_names,
            intervals=intervals,
            batch_mode=mode,
            full_start=start_date,
            full_end=end_date,
        )
        step_index: dict[Step, int] = {
            (t, s, e): i for i, (t, s, e) in enumerate(schedule)
        }

        # Build or recover execution plan
        plan: ExecutionPlan | None = None
        if config_path is not None:
            plan_key = ExecutionPlan.make_key(
                config_path=config_path,
                task_name=task_name,
                start_date=str(start_date),
                end_date=str(end_date),
                batch_interval=str(batch_interval),
                batch_mode=batch_mode,
                with_deps=str(run_dependencies),
            )
            plan_params = {
                "config_path": config_path,
                "task_name": task_name,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "batch_interval": str(batch_interval),
                "batch_mode": batch_mode,
                "with_deps": str(run_dependencies),
            }
            if recover:
                plan = ExecutionPlan.load(plan_key)
                if plan is not None:
                    resume_idx = plan.first_non_success()
                    if resume_idx is None:
                        logger.info("All steps already completed — nothing to do.")
                        done = DAGRun(
                            plan=plan,
                            step_index=step_index,
                            already_completed=set(step_index.keys()),
                        )
                        done._mark_done()
                        return done
                    logger.info(
                        "Recovering execution plan: skipping %d completed steps",
                        resume_idx,
                    )
            if plan is None:
                plan = ExecutionPlan.from_schedule(plan_key, plan_params, schedule)
                plan.save()

        # Seed completed with any steps already marked SUCCESS in a recovered plan.
        completed: set[Step] = {
            step
            for step, idx in step_index.items()
            if plan is not None and plan.steps[idx].status == "SUCCESS"
        }

        # Reverse dependency map and blocked-count for O(1) ready-check on completion.
        # When step S completes we decrement blocked_count for each entry in
        # reverse_deps[S]; a step becomes dispatchable when its count reaches 0.
        reverse_deps: dict[Step, list[Step]] = {s: [] for s in step_deps}
        blocked_count: dict[Step, int] = {}
        for step, deps in step_deps.items():
            active = deps - completed
            blocked_count[step] = len(active)
            for dep in active:
                reverse_deps[dep].append(step)

        dag_run = DAGRun(
            plan=plan,
            step_index=step_index,
            already_completed=completed,
        )

        import queue as _queue

        first_chunk_dispatched: set[str] = set()
        completion_queue: _queue.Queue[tuple[Step, bool, BaseException | None]] = (
            _queue.Queue()
        )

        # Each _await thread polls result.ready() (a single atomic Redis GET) rather
        # than calling result.get(), which blocks holding a Redis connection and
        # corrupts other threads' reads when many tasks are in flight concurrently.
        # After ready() returns True the result is locally cached — no further
        # Redis calls are needed to read failed()/result.
        await_pool = ThreadPoolExecutor(thread_name_prefix="celery-await")

        def _await(step: Step, async_result: Any) -> None:
            while not async_result.ready():
                time.sleep(0.05)
            completion_queue.put(
                (
                    step,
                    not async_result.failed(),
                    async_result.result if async_result.failed() else None,
                )
            )

        def _submit(step: Step) -> None:
            step_task_name, chunk_start, chunk_end = step
            raw: dict[str, Any] = dict(kwargs)
            raw["start_date"] = chunk_start
            raw["end_date"] = chunk_end
            if step_task_name in first_chunk_dispatched:
                raw.pop("clean", None)
            else:
                first_chunk_dispatched.add(step_task_name)
            step_kwargs = serialize_kwargs(raw)
            task_spec = serialize_task(self[step_task_name])
            result = celery_app.send_task(
                "tradingo.run_task",
                args=[task_spec, step_kwargs],
                queue=QUEUE,
            )
            dag_run._on_submitted(step, result)
            if plan is not None:
                plan.mark_step_submitted(step_index[step], result.id)
            await_pool.submit(_await, step, result)
            logger.info(
                "Submitted %s [%s -> %s] to Celery queue '%s'",
                step_task_name,
                chunk_start,
                chunk_end,
                QUEUE,
            )

        def _dispatch_loop() -> None:
            pending: set[Step] = set()
            try:
                for step in step_deps:
                    if blocked_count[step] == 0 and step not in completed:
                        _submit(step)
                        pending.add(step)

                while pending:
                    step, success, exc = completion_queue.get()
                    pending.discard(step)
                    if not success:
                        err = exc if isinstance(exc, Exception) else Exception(str(exc))
                        logger.error("Step %s [%s -> %s] failed: %s", *step, err)
                        dag_run._on_failed(step, err)
                        if plan is not None:
                            plan.mark_step(step_index[step], "FAILED")
                    else:
                        dag_run._on_completed(step)
                        completed.add(step)
                        logger.info("Step %s [%s -> %s] completed", *step)
                        if plan is not None:
                            plan.mark_step(step_index[step], "SUCCESS")
                        for downstream in reverse_deps[step]:
                            blocked_count[downstream] -= 1
                            if (
                                blocked_count[downstream] == 0
                                and downstream not in completed
                            ):
                                _submit(downstream)
                                pending.add(downstream)
            except Exception as exc:
                dag_run._errors.append(exc)
            finally:
                await_pool.shutdown(wait=False)
                dag_run._mark_done()

        dispatch_thread = threading.Thread(
            target=_dispatch_loop,
            daemon=True,
            name=f"dag-dispatch-{task_name}",
        )
        dispatch_thread.start()
        return dag_run

    def run(
        self,
        task_name: str,
        skip_deps: re.Pattern[str] | None = None,
        run_dependencies: bool | int = False,
        force_rerun: bool = False,
        max_workers: int = 1,
        batch_interval: pd.Timedelta | None = None,
        batch_mode: str = "stepped",
        recover: bool = False,
        config_path: str | None = None,
        executor: str = "thread",
        broker_url: str | None = None,
        *args: object,
        background: bool = False,
        **kwargs: object,
    ) -> DAGRun:
        """Run a specific task of this DAG.

        All run modes return a ``DAGRun`` handle.  Pass ``background=True``
        to any mode to start execution in a daemon thread and return
        immediately — call ``dag_run.wait()`` to block and re-raise failures.
        ``background=False`` (default) blocks until complete and still returns
        a finished ``DAGRun``.

        ``run_batched_celery`` is always non-blocking; ``background=False``
        causes ``run()`` to call ``dag_run.wait()`` before returning.
        """
        if task_name not in self:
            raise ValueError(f"{task_name} is not a task in the DAG.")

        if executor == "celery":
            if batch_interval is not None:
                dag_run = self.run_batched_celery(
                    task_name,
                    run_dependencies=run_dependencies,
                    batch_interval=batch_interval,
                    batch_mode=batch_mode,
                    broker_url=broker_url,
                    skip_deps=skip_deps,
                    recover=recover,
                    config_path=config_path,
                    **kwargs,
                )
                if not background:
                    dag_run.wait()
                return dag_run
            return self.run_celery(
                task_name,
                run_dependencies=run_dependencies,
                broker_url=broker_url,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                background=background,
                **kwargs,
            )

        if executor == "process":
            return self.run_multiprocess(
                task_name,
                run_dependencies=run_dependencies,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                max_workers=max_workers,
                background=background,
                **kwargs,
            )

        if batch_interval is not None:
            return self.run_batched(
                task_name,
                batch_interval=batch_interval,
                batch_mode=batch_mode,
                run_dependencies=run_dependencies,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                max_workers=max_workers,
                recover=recover,
                config_path=config_path,
                background=background,
                **kwargs,
            )

        if max_workers > 1 and run_dependencies:
            return self.run_parallel(
                task_name,
                run_dependencies=run_dependencies,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                max_workers=max_workers,
                background=background,
                **kwargs,
            )

        # Single-task path: wrap Task.run() in a DAGRun
        step: Step = (task_name, None, None)
        dag_run = DAGRun(
            plan=None,
            step_index={step: 0},
            already_completed=set(),
        )

        def _execute() -> None:
            try:
                dag_run._on_submitted(step, None)
                self[task_name].run(
                    *args,
                    skip_deps=skip_deps,
                    run_dependencies=run_dependencies,
                    force_rerun=force_rerun,
                    **kwargs,
                )
                dag_run._on_completed(step)
            except Exception as exc:
                dag_run._on_failed(step, exc)
                raise
            finally:
                dag_run._mark_done()

        if background:

            def _bg() -> None:
                try:
                    _execute()
                except Exception:
                    pass  # errors already captured in dag_run._errors

            threading.Thread(target=_bg, daemon=True, name=f"dag-{task_name}").start()
        else:
            _execute()

        return dag_run

    def update_state(self) -> None:
        """update the local json file which keeps the DAG state."""

        if not self._state_filepath.exists():
            return

        dag_state = json.loads(self._state_filepath.read_text())

        for k, v in dag_state.items():
            if k not in self:
                continue
            state = TaskState[v]

            self[k].state = state if state == TaskState.SUCCESS else TaskState.PENDING

    def serialise_state(self) -> None:
        """write the DAG state into a local json file."""

        self._state_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._state_filepath.write_text(
            json.dumps({k: v.state.value for k, v in self.items()}, indent=2)
        )
