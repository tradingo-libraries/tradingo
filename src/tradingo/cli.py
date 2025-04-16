from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import pathlib
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import jinja2
import pandas as pd
import yaml

from . import symbols
from .api import Tradingo
from .config import IGTradingConfig, TradingoConfig


class ConfigLoadError(Exception):
    """"""


def read_config_template(filepath: pathlib.Path, variables):
    filepath = pathlib.Path(filepath)

    renderedtext = jinja2.Template(filepath.read_text()).render(**variables)

    try:
        if filepath.suffix == ".json":
            return process_includes(json.loads(renderedtext), variables)

        if filepath.suffix == ".yaml":
            return process_includes(yaml.safe_load(renderedtext), variables)

    except jinja2.TemplateSyntaxError as ex:
        raise ConfigLoadError(f"Error reading config template: {filepath}") from ex

    raise ValueError(f"Unhandled file type: '{filepath.suffix}'")


def process_includes(config, variables):
    out = {}

    for key, value in config.items():
        if isinstance(value, dict) and "include" in value:
            value = process_includes(value, variables)

            protocol, path = value["include"].split("://")

            if protocol == "file":
                incvalue = read_config_template(
                    pathlib.Path(path),
                    variables={
                        **value.get("variables", {}),
                        **variables,
                    },
                )

            else:
                raise ValueError(
                    f"Unsupported protocol: '{protocol}' at '{key}' for '{path}'"
                )
            value = value.copy()
            value.update(incvalue)
            value.pop("include", None)
            value.pop("variables", None)

        elif isinstance(value, dict):
            value = process_includes(value, variables)

        out[key] = value

    return out


def cli_app():
    app = argparse.ArgumentParser("tradingo-tasks")

    app.add_argument(
        "--auth",
        type=lambda i: IGTradingConfig.from_env(
            env=read_config_template(pathlib.Path(i), os.environ)
        ).to_env(),
        required=True,
    )
    app.add_argument(
        "--config",
        type=lambda i: read_config_template(pathlib.Path(i), os.environ),
        required=True,
    )

    entity = app.add_subparsers(dest="entity", required=True)
    universe = entity.add_parser("universe")
    universe_subparsers = universe.add_subparsers(dest="universe_action", required=True)
    _ = universe_subparsers.add_parser("list")
    uni_show = universe_subparsers.add_parser("show")

    uni_show.add_argument("name")

    task = entity.add_parser("task")

    task_subparsers = task.add_subparsers(dest="list_action", required=True)
    run_tasks = task_subparsers.add_parser("run")
    run_tasks.add_argument("task")
    run_tasks.add_argument("--with-deps", action="store_true")
    run_tasks.add_argument("--start-date", type=pd.Timestamp, required=False)
    run_tasks.add_argument("--end-date", type=pd.Timestamp, required=False)
    run_tasks.add_argument("--force-rerun", action="store_true", default=True)
    run_tasks.add_argument("--dry-run", action="store_true")

    _ = task_subparsers.add_parser("list")
    return app


class TaskState(Enum):
    PENDING = "PENDING"
    FAILED = "FAILED"
    SUCCESS = "SUCCESS"


class Task:
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
    ):
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

    def __repr__(self):
        return (
            f"Task(function='{self._function}',"
            f" task_args={self.task_args},"
            f" task_kwargs={self.task_kwargs},"
            f" dependcies={self._dependencies}, "
            ")"
        )

    @property
    def function(self) -> Callable:
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
    def prepare_kwargs(task_kwargs, global_kwargs):
        task_kwargs.update(global_kwargs)
        return task_kwargs

    def run(self, *args, run_dependencies=False, force_rerun=False, **kwargs):
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

    def add_dependencies(self, *dependency):
        self._dependencies.extend(dependency)

    def resolve_dependencies(self, tasks: dict[str, Task]):
        self._resolved_dependencies.extend(
            tasks[dep_name] for dep_name in self._dependencies
        )

    @property
    def dependencies(self) -> list[Task]:
        return self._resolved_dependencies

    @property
    def dependency_names(self) -> list[str]:
        return self._dependencies


def collect_task_configs(config, _tasks: Optional[dict[str, Any]] = None):
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
    @classmethod
    def from_config(cls, config: dict[str, Any]):
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
                raise ConfigLoadError(f"{task_name} is missing setting {ex.args[0]}")

        for task_name, task in tasks.items():
            try:
                task.resolve_dependencies(tasks)
            except KeyError as ex:
                raise ConfigLoadError(
                    f"Missing task in dag '{ex.args[0]}' for '{task_name}'"
                ) from ex

        return cls(tasks)

    def print(self):
        for task_name, task in self.items():
            print(f"{task_name}:")
            if task:
                for dep in task.dependency_names:
                    print(f"  - {dep}")
            else:
                print("  No dependencies")
            print()

    def get_symbols(self):
        return [
            task
            for subl in (task.symbols_out for task in self.values())
            for task in subl
        ]

    def run(self, task_name, **kwargs):
        return self[task_name].run(**kwargs)

    def update_dag(self):
        dag_state = pathlib.Path.home() / ".tradingo/dag-state.json"

        if not dag_state.exists():
            return

        else:
            dag_state = json.loads(dag_state.read_text())

            for k, v in dag_state.items():
                if k not in self:
                    continue
                state = TaskState[v]

                self[k].state = (
                    state if state == TaskState.SUCCESS else TaskState.PENDING
                )

    def serialise_dag(self):
        dag_state = pathlib.Path.home() / ".tradingo/dag-state.json"

        dag_state.parent.mkdir(parents=True, exist_ok=True)
        dag_state.write_text(
            json.dumps({k: v.state.value for k, v in self.items()}, indent=2)
        )


def handle_tasks(args, arctic):
    if args.list_action == "list":
        graph = DAG.from_config(
            args.config,
        )

        graph.print()
        return

    elif args.list_action == "run":
        graph = DAG.from_config(
            args.config,
        )

        graph.update_dag()

        try:
            extra_kwargs = {}
            if args.start_date:
                extra_kwargs["start_date"] = args.start_date
            if args.end_date:
                extra_kwargs["end_date"] = args.end_date
            out = graph.run(
                args.task,
                run_dependencies=args.with_deps,
                force_rerun=args.force_rerun,
                arctic=arctic,
                dry_run=args.dry_run,
                **extra_kwargs,
            )
            if args.dry_run:
                print(out)
        finally:
            graph.serialise_dag()

    else:
        raise ValueError(args.list_action)


def handle_universes(args, api: Tradingo):
    if args.universe_action == "list":
        for item in api.instruments.list():
            print(item)

    elif args.universe_action == "show":
        print(api.instruments[args.name]())

    elif args.universe_action == "prices":
        print(api.instruments[args.name]())

    else:
        ValueError(args.universe_action)


def main():
    envconfig = TradingoConfig.from_env().to_env()
    args = cli_app().parse_args()
    IGTradingConfig.from_env().to_env()
    envconfig.to_env()

    arctic = Tradingo(envconfig.arctic_uri)

    if args.entity == "task":
        handle_tasks(args, arctic)

    elif args.entity == "universe":
        handle_universes(args, api=arctic)

    else:
        raise ValueError(args.entity)


if __name__ == "__main__":
    logging.getLogger("tradingo").setLevel(logging.INFO)
    main()
