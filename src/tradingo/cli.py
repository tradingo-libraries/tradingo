from __future__ import annotations
from enum import Enum
import importlib
import functools
import argparse
from json.decoder import JSONDecodeError
import pathlib
import json
from typing import Callable, Iterable
from airflow import DAG
from airflow.operators.python import PythonOperator
from arcticdb import Arctic

import pandas as pd

from tradingo.symbols import ARCTIC_URL

SIGNAL_KEY = "{0}.{1}".format


def resolve_config(
    config: str,
):
    try:
        return json.loads(config)
    except (JSONDecodeError, TypeError) as _:
        return json.loads(pathlib.Path(config).read_text())


def cli_app():
    app = argparse.ArgumentParser("tradingo")

    app.add_argument("--config", type=resolve_config)
    app.add_argument("--task", required=True)
    app.add_argument("--with-deps", action="store_true")
    app.add_argument("--start-date", type=pd.Timestamp)
    app.add_argument("--end-date", type=pd.Timestamp)
    app.add_argument("--force-rerun", action="store_true")
    app.add_argument("--arctic-uri", default=ARCTIC_URL)
    return app


def task_resolver(func):

    functools.wraps(func)

    def wrapper(*args, global_tasks, **kwargs):

        tasks_out = func(*args, **kwargs)
        global_tasks.update(tasks_out)

        for task in tasks_out.values():
            task.resolve_dependencies(global_tasks)

        return tasks_out

    return wrapper


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
        dependencies: Iterable[str] = (),
    ):
        self._function = function
        self.task_args = task_args
        self.task_kwargs = task_kwargs
        self._dependencies = list(dependencies)
        self._resolved_dependencies = []
        self.state = TaskState.PENDING

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
        return getattr(importlib.import_module(module), function_name)

    def run(self, *args, run_dependencies=False, force_rerun=False, **kwargs):

        if run_dependencies:

            for dependency in self.dependencies:

                dependency.run(
                    run_dependencies=run_dependencies,
                    force_rerun=force_rerun,
                )

        state = self.state
        try:
            if self.state == TaskState.PENDING or force_rerun:
                state = TaskState.FAILED
                print(f"Running {self}")
                self.function(*self.task_args, *args, **self.task_kwargs, **kwargs)
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


@task_resolver
def collect_signal_tasks(
    signals: dict[str, dict],
    universe: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):

    tasks = {
        SIGNAL_KEY(universe, signal): Task(
            config["function"],
            config.get("args", []),
            {
                **config.get("kwargs", {}),
                "start_date": start_date,
                "end_date": end_date,
            },
            [
                f"{universe}.sample",
                f"{universe}.vol",
                f"{universe}.ivol",
                *(SIGNAL_KEY(universe, k) for k in config.get("depends_on", [])),
            ],
        )
        for signal, config in signals.items()
    }

    return tasks


@task_resolver
def collect_sample_tasks(
    universes: dict[str, dict],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):

    tasks = {}

    for universe, config in universes.items():

        # TODO: start,end date
        #
        #
        provider = config["provider"]

        tasks[f"{universe}.instruments"] = Task(
            "tradingo.sampling.download_instruments",
            [],
            {
                "html": config.get("html"),
                "file": config.get("file"),
                "tickers": config.get("tickers"),
                "epics": config.get("epics"),
                "index_col": config["index_col"],
                "universe": universe,
            },
        )
        tasks[f"{universe}.sample"] = Task(
            config.get("function", "tradingo.sampling.sample_equity"),
            [],
            {
                "start_date": start_date,
                "end_date": end_date,
                "provider": provider,
                "interval": config.get("interval", "1d"),
                "universe": universe,
                "periods": config.get("periods"),
            },
            [f"{universe}.instruments"],
        )
        tasks[f"{universe}.vol"] = Task(
            "tradingo.signals.vol",
            [],
            {
                "start_date": start_date,
                "end_date": end_date,
                "provider": provider,
                # TODO: speeds
                "speeds": config["volatility"]["speeds"],
                "universe": universe,
            },
            [f"{universe}.sample"],
        )
        tasks[f"{universe}.ivol"] = Task(
            "tradingo.portfolio.instrument_ivol",
            [],
            {
                "start_date": None,
                "end_date": None,
                "provider": config["provider"],
                "interval": config.get("interval"),
                "universe": universe,
            },
            [f"{universe}.sample"],
        )

    return tasks


def build_graph(config, start_date, end_date) -> dict[str, Task]:

    global_tasks = {}

    sample_tasks = collect_sample_tasks(
        config["universe"],
        global_tasks=global_tasks,
        start_date=start_date,
        end_date=end_date,
    )

    for portfolio_name, portfolio_config in config["portfolio"].items():

        universe = portfolio_config["universe"]

        portfolio_config["kwargs"].setdefault("start_date", start_date)
        portfolio_config["kwargs"].setdefault("end_date", end_date)
        portfolio_config["kwargs"].setdefault("name", portfolio_name)

        portfolio_task = Task(
            portfolio_config["function"],
            portfolio_config.get("args", []),
            portfolio_config.get("kwargs", {}),
            dependencies=[
                f"{universe}.sample",
                f"{universe}.vol",
                f"{universe}.ivol",
                *(
                    SIGNAL_KEY(universe, sig)
                    for sig in portfolio_config["kwargs"].get("signal_weights", {})
                ),
            ],
        )

        signals = collect_signal_tasks(
            config["signal_configs"],
            portfolio_config["universe"],
            global_tasks=global_tasks,
            start_date=start_date,
            end_date=end_date,
        )

        portfolio_task.resolve_dependencies(global_tasks)

        global_tasks[portfolio_name] = portfolio_task

        backtest = Task(
            "tradingo.backtest.backtest",
            task_args=[],
            task_kwargs={
                "start_date": start_date,
                "end_date": end_date,
                "name": portfolio_name,
                "provider": portfolio_config["kwargs"]["provider"],
                "universe": portfolio_config["kwargs"]["universe"],
                "stage": "raw.shares",
            },
            dependencies=[portfolio_name],
        )

        global_tasks[f"{portfolio_name}.backtest"] = backtest
        backtest.resolve_dependencies(global_tasks)

    return global_tasks


def to_airflow_dag(graph: dict[str, Task], **kwargs) -> DAG:

    with DAG(**kwargs) as dag:

        tasks = {
            name: PythonOperator(
                task_id=name,
                python_callable=task.function,
                op_args=task.task_args,
                op_kwargs=task.task_kwargs,
            )
            for name, task in graph.items()
        }

        for (name, task), operator in zip(graph.items(), tasks.values()):

            for task_id in task.dependency_names:

                tasks[task_id] >> operator

    return dag


def make_airflow_dag(name, start_date, dag_start_date, end_date, config):
    graph = build_graph(resolve_config(config), start_date, end_date)

    return to_airflow_dag(
        graph,
        dag_id=name,
        start_date=dag_start_date,
    )


def serialise_dag(graph: dict[str, Task]):

    dag_state = pathlib.Path.home() / ".tradingo/dag-state.json"

    dag_state.parent.mkdir(parents=True, exist_ok=True)
    dag_state.write_text(
        json.dumps({k: v.state.value for k, v in graph.items()}, indent=2)
    )


def update_dag(graph: dict[str, Task]):

    dag_state = pathlib.Path.home() / ".tradingo/dag-state.json"

    if not dag_state.exists():

        return

    else:
        dag_state = json.loads(dag_state.read_text())

        for k, v in dag_state.items():

            if k not in graph:
                continue
            state = TaskState[v]

            graph[k].state = state if state == TaskState.SUCCESS else TaskState.PENDING


def main():

    args = cli_app().parse_args()

    graph = build_graph(args.config, args.start_date, args.end_date)
    arctic = Arctic(args.arctic_uri)

    update_dag(graph)

    task = graph[args.task]

    try:
        task.run(
            run_dependencies=args.with_deps, force_rerun=args.force_rerun, arctic=arctic
        )
    finally:
        serialise_dag(graph)


if __name__ == "__main__":
    main()
