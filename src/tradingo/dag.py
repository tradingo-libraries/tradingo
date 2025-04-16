"""DAG generation logic."""

from __future__ import annotations

import importlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import symbols
from .config import ConfigLoadError


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
        function,
        task_args,
        task_kwargs,
        symbols_out,
        symbols_in,
        load_args,
        publish_args,
        dependencies: Iterable[str] = (),
    ) -> None:
        self._function = function
        self.task_args = task_args
        self.task_kwargs = task_kwargs
        self._dependencies = list(dependencies)
        self._resolved_dependencies: list[Task] = []
        self.state = TaskState.PENDING
        self.symbols_out = symbols_out
        self.symbols_in = symbols_in
        self.load_args = load_args
        self.publish_args = publish_args

    def __repr__(self) -> str:
        return (
            f"Task(function='{self._function}',"
            f" task_args={self.task_args},"
            f" task_kwargs={self.task_kwargs},"
            f" dependcies={self._dependencies}, "
            ")"
        )

    @property
    def function(self) -> Callable:
        """task function"""
        module, function_name = self._function.rsplit(".", maxsplit=1)
        function = getattr(importlib.import_module(module), function_name)

        if self.symbols_out:
            function = symbols.symbol_publisher(
                *self.symbols_out,
                **self.publish_args,
            )(function)
        if self.symbols_in:
            function = symbols.symbol_provider(
                **self.symbols_in,
                **self.load_args,
            )(function)

        return function

    @staticmethod
    def prepare_kwargs(
        task_kwargs: dict[str, Any], global_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """prepare kwargs to be passed to the function"""
        task_kwargs.update(global_kwargs)
        return task_kwargs

    def run(self, *args, run_dependencies=False, force_rerun=False, **kwargs) -> None:
        """run this task. optinally run also dependency tasks"""
        if run_dependencies:
            for dependency in self.dependencies:
                dependency.run(
                    *args,
                    run_dependencies=run_dependencies,
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

    def add_dependencies(self, *dependency) -> None:
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


def collect_task_configs(
    config, _tasks: Optional[dict[str, Any]] = None
) -> dict[str, dict]:
    """gather all task specifications from a config, accounting for dependencies."""
    tasks = _tasks or {}

    for key, value in config.items():
        if isinstance(value, dict) and "depends_on" in value:
            # its a task, collect it
            tasks[key] = value
        elif isinstance(value, dict):
            # its a set of tasks collect them
            tasks.update(collect_task_configs(value, tasks))

    return tasks


class DAG(dict[str, Task]):
    """Tradingo DAG"""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DAG:
        """create a DAG from one config."""

        task_configs = collect_task_configs(config)

        tasks: dict[str, Task] = {}

        for task_name, task_config in task_configs.items():
            if not task_config.get("enabled", True):
                continue
            params = task_config["params"]
            try:
                tasks[task_name] = Task(
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

    def get_symbols(self) -> list[Task]:
        """list all symbols produced by this DAG."""
        return [
            task
            for subl in (task.symbols_out for task in self.values())
            for task in subl
        ]

    def run(self, task_name, **kwargs) -> None:
        """run a specific task of this DAG."""
        return self[task_name].run(**kwargs)

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
