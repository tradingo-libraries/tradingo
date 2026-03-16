"""DAG generation logic."""

from __future__ import annotations

import importlib
import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd

from . import symbols
from .config import ConfigLoadError
from .execution_plan import ExecutionPlan

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

        if self.symbols_out:
            function = symbols.symbol_publisher(
                *self.symbols_out,
                **self.publish_args,
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
        step_index: dict[tuple[str, pd.Timestamp, pd.Timestamp], int] | None = None,
        **kwargs: object,
    ) -> None:
        """Run one task across all its intervals sequentially.

        ``clean`` is only applied to the first interval; subsequent
        intervals strip it so they append/update rather than overwrite.
        """
        task = self[task_name]
        for i, (start, end) in enumerate(intervals):
            step_kwargs = dict(kwargs)
            step_kwargs["start_date"] = start
            step_kwargs["end_date"] = end
            if i > 0:
                step_kwargs.pop("clean", None)
            task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), step_kwargs)
            if task.state == TaskState.PENDING or force_rerun:
                logger.info("Running %s [%s -> %s]", task_name, start, end)
                print(f"Running {task} [{start} -> {end}]")
                try:
                    task.function(*task.task_args, **task_kwargs)
                    if plan is not None and step_index is not None:
                        idx = step_index.get((task_name, start, end))
                        if idx is not None:
                            plan.mark_step(idx, "SUCCESS")
                except Exception:
                    if plan is not None and step_index is not None:
                        idx = step_index.get((task_name, start, end))
                        if idx is not None:
                            plan.mark_step(idx, "FAILED")
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
        **kwargs: object,
    ) -> None:
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
                        return
                    logger.info(
                        "Recovering execution plan: skipping %d completed steps",
                        resume_idx,
                    )
            if plan is None:
                plan = ExecutionPlan.from_schedule(plan_key, plan_params, schedule)
                plan.save()

        step_index: dict[tuple[str, pd.Timestamp, pd.Timestamp], int] = {
            (t, s, e): i for i, (t, s, e) in enumerate(schedule)
        }

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
                **kwargs,
            )
            return

        # Sequential execution
        # Track which tasks have run at least once so --clean only applies to
        # the first interval per task (not every subsequent chunk).
        seen_tasks: set[str] = set()
        for step_idx, (step_task_name, chunk_start, chunk_end) in enumerate(schedule):
            # Skip completed steps when recovering
            if plan is not None and plan.steps[step_idx].status == "SUCCESS":
                seen_tasks.add(step_task_name)
                continue

            task = self[step_task_name]
            # Reset state so the same task can run again for the next interval
            task.state = TaskState.PENDING
            step_kwargs = dict(kwargs)
            step_kwargs["start_date"] = chunk_start
            step_kwargs["end_date"] = chunk_end
            if step_task_name in seen_tasks:
                step_kwargs.pop("clean", None)
            task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), step_kwargs)

            if task.state == TaskState.PENDING or force_rerun:
                state = TaskState.FAILED
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
                except Exception:
                    if plan is not None:
                        plan.mark_step(step_idx, "FAILED")
                    raise
                finally:
                    task.state = state

            seen_tasks.add(step_task_name)

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
        step_index: dict[tuple[str, pd.Timestamp, pd.Timestamp], int] | None = None,
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
                try:
                    self.run_parallel(
                        task_name,
                        run_dependencies=run_dependencies,
                        skip_deps=skip_deps,
                        force_rerun=force_rerun,
                        max_workers=max_workers,
                        **chunk_kwargs,
                    )
                    if plan is not None and step_index is not None:
                        for name in all_task_names:
                            idx = step_index.get((name, chunk_start, chunk_end))
                            if idx is not None:
                                plan.mark_step(idx, "SUCCESS")
                except Exception:
                    if plan is not None and step_index is not None:
                        for name in all_task_names:
                            idx = step_index.get((name, chunk_start, chunk_end))
                            if idx is not None:
                                plan.mark_step(idx, "FAILED")
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
                try:
                    self.run_parallel(
                        task_name,
                        run_dependencies=run_dependencies,
                        skip_deps=skip_deps,
                        force_rerun=force_rerun,
                        max_workers=max_workers,
                        **full_kwargs,
                    )
                    if plan is not None and step_index is not None:
                        for dep_name in dep_names:
                            idx = step_index.get((dep_name, start_date, end_date))
                            if idx is not None:
                                plan.mark_step(idx, "SUCCESS")
                except Exception:
                    if plan is not None and step_index is not None:
                        for dep_name in dep_names:
                            idx = step_index.get((dep_name, start_date, end_date))
                            if idx is not None:
                                plan.mark_step(idx, "FAILED")
                    raise
            # Phase 2: target batched sequentially (same task, sequential intervals)
            self._run_intervals_sequential(
                task_name,
                intervals,
                force_rerun,
                plan=plan,
                step_index=step_index,
                **kwargs,
            )

    def run_parallel(
        self,
        task_name: str,
        run_dependencies: bool | int,
        skip_deps: re.Pattern[str] | None = None,
        force_rerun: bool = False,
        max_workers: int = 4,
        **kwargs: object,
    ) -> None:
        """Run tasks using a thread pool with dynamic dispatch."""
        runnable = self._collect_runnable_tasks(task_name, run_dependencies, skip_deps)

        # Build in-degree and reverse-dependency maps (only among runnable tasks)
        in_degree: dict[str, int] = {name: 0 for name in runnable}
        dependents: dict[str, list[str]] = {name: [] for name in runnable}

        for name in runnable:
            task = self[name]
            for dep in task.dependencies:
                if dep.name in runnable:
                    in_degree[name] += 1
                    dependents[dep.name].append(name)

        errors: list[Exception] = []
        errors_lock = threading.Lock()
        in_degree_lock = threading.Lock()

        def _run_task(name: str) -> str:
            task = self[name]
            state = task.state
            try:
                task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), dict(kwargs))
                if task.state == TaskState.PENDING or force_rerun:
                    state = TaskState.FAILED
                    print(f"Running {task}")
                    task.function(*task.task_args, **task_kwargs)
                    state = TaskState.SUCCESS
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)
            finally:
                task.state = state
            return name

        ready = [name for name, deg in in_degree.items() if deg == 0]
        pending_futures: set[Future[str]] = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name in ready:
                pending_futures.add(executor.submit(_run_task, name))

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
                                    executor.submit(_run_task, dep_name)
                                )

        if errors:
            raise ExceptionGroup(f"{len(errors)} task(s) failed", errors)

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
        *args: object,
        **kwargs: object,
    ) -> None:
        """run a specific task of this DAG."""
        if task_name not in self:
            raise ValueError(f"{task_name} is not a task in the DAG.")

        if batch_interval is not None:
            self.run_batched(
                task_name,
                batch_interval=batch_interval,
                batch_mode=batch_mode,
                run_dependencies=run_dependencies,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                max_workers=max_workers,
                recover=recover,
                config_path=config_path,
                **kwargs,
            )
            return

        if max_workers > 1 and run_dependencies:
            self.run_parallel(
                task_name,
                run_dependencies=run_dependencies,
                skip_deps=skip_deps,
                force_rerun=force_rerun,
                max_workers=max_workers,
                **kwargs,
            )
            return

        self[task_name].run(
            *args,
            skip_deps=skip_deps,
            run_dependencies=run_dependencies,
            force_rerun=force_rerun,
            **kwargs,
        )

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
