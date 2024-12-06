from __future__ import annotations
import os
import logging
from enum import Enum
import importlib
import functools
import argparse
from json.decoder import JSONDecodeError
import pathlib
import json
from typing import Callable, Iterable
from arcticdb import Arctic

import pandas as pd

from tradingo.symbols import ARCTIC_URL

SIGNAL_KEY = "{0}.{1}".format

DEFAULT_STAGE = "raw.shares"


def resolve_config(
    config: str | pathlib.Path,
):
    try:
        return json.loads(config)
    except (JSONDecodeError, TypeError) as _:
        return json.loads(pathlib.Path(config).read_text())


def cli_app():
    app = argparse.ArgumentParser("tradingo")

    app.add_argument("--config", type=resolve_config)
    app.add_argument("--auth", type=resolve_config)
    app.add_argument("--task", required=True)
    app.add_argument("--with-deps", action="store_true")
    app.add_argument("--start-date", type=pd.Timestamp)
    app.add_argument("--end-date", type=pd.Timestamp)
    app.add_argument("--force-rerun", action="store_true")
    app.add_argument("--arctic-uri", default=ARCTIC_URL)
    app.add_argument("--dry-run", action="store_true")
    return app


def task_resolver(func):

    functools.wraps(func)

    def wrapper(*global_tasks, **kwargs):

        tasks_out = func(**kwargs)

        if isinstance(tasks_out, (tuple, list)):

            task_sets = tasks_out

            for tasks_out, all_tasks in zip(task_sets, global_tasks):

                all_tasks.update(tasks_out)

                for task in tasks_out.values():
                    task.resolve_dependencies(all_tasks)

            return task_sets

        else:

            global_tasks[0].update(tasks_out)

            for task in tasks_out.values():
                task.resolve_dependencies(global_tasks[0])

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
    provider: str,
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
                "universe": config.get("kwargs", {}).get("universe", universe),
                "provider": config.get("kwargs", {}).get("provider", provider),
            },
            [
                f"{universe}.sample",
                f"{universe}.vol",
                # f"{universe}.ivol",
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
    sample_start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_instruments: bool = True,
):

    tasks = {}
    instrument_tasks = {}

    for universe, config in universes.items():

        provider = config["provider"]

        instrument_task = Task(
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

        if include_instruments:
            tasks[f"{universe}.instruments"] = instrument_task
        instrument_tasks[f"{universe}.instruments"] = instrument_task

        tasks[f"{universe}.sample"] = Task(
            config.get("function", "tradingo.sampling.sample_equity"),
            [],
            {
                "start_date": sample_start_date,
                "end_date": end_date,
                "provider": provider,
                "interval": config.get("interval", "1d"),
                "universe": universe,
                "periods": config.get("periods"),
            },
            [f"{universe}.instruments"] if include_instruments else [],
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
                "close": config["volatility"].get("field", "adj_close"),
            },
            [f"{universe}.sample"],
        )

        if config["volatility"].get("ivol", True):

            tasks[f"{universe}.ivol"] = Task(
                "tradingo.portfolio.instrument_ivol",
                [],
                {
                    "start_date": None,
                    "end_date": None,
                    "provider": config["provider"],
                    "interval": config.get("interval"),
                    "universe": universe,
                    "close": config["volatility"].get("field", "adj_close"),
                },
                [f"{universe}.sample"],
            )

    return tasks, instrument_tasks


def build_graph(
    config,
    start_date,
    end_date,
    sample_start_date=None,
    snapshot_template=None,
    include_instruments: bool = True,
    include_live: bool = False,
) -> tuple[dict[str, Task], dict[str, Task]]:

    sample_start_date = sample_start_date or start_date

    global_tasks = {}
    eod_tasks = {}

    sample_tasks, instrument_tasks = collect_sample_tasks(
        global_tasks,
        eod_tasks,
        universes=config["universe"],
        start_date=start_date,
        sample_start_date=sample_start_date,
        end_date=end_date,
        include_instruments=include_instruments,
    )

    for portfolio_name, portfolio_config in config["portfolio"].items():

        universe = portfolio_config["universe"]
        provider = portfolio_config["provider"]

        portfolio_config["kwargs"].setdefault("start_date", start_date)
        portfolio_config["kwargs"].setdefault("end_date", end_date)
        portfolio_config["kwargs"].setdefault("name", portfolio_name)
        portfolio_config["kwargs"].setdefault("provider", provider)
        portfolio_config["kwargs"].setdefault("universe", universe)
        portfolio_config["kwargs"].setdefault("snapshot", snapshot_template)

        portfolio_task = Task(
            portfolio_config["function"],
            portfolio_config.get("args", []),
            portfolio_config.get("kwargs", {}),
            dependencies=[
                f"{universe}.sample",
                f"{universe}.vol",
                # f"{universe}.ivol",
                *(
                    SIGNAL_KEY(universe, sig)
                    for sig in portfolio_config["kwargs"].get("signal_weights", {})
                ),
            ],
        )

        signals = collect_signal_tasks(
            global_tasks,
            signals=config["signal_configs"],
            universe=portfolio_config["universe"],
            provider=portfolio_config["provider"],
            start_date=start_date,
            end_date=end_date,
        )

        portfolio_task.resolve_dependencies(global_tasks)

        global_tasks[portfolio_name] = portfolio_task

        backtest_kwargs = portfolio_config.get("backtest", {"stage": DEFAULT_STAGE})
        trade_kwargs = portfolio_config.get("trades", {"stage": DEFAULT_STAGE})

        backtest = global_tasks[f"{portfolio_name}.backtest"] = Task(
            "tradingo.backtest.backtest",
            task_args=[],
            task_kwargs={
                "start_date": start_date,
                "end_date": end_date,
                "name": portfolio_name,
                "provider": provider,
                "universe": universe,
                **backtest_kwargs,
            },
            dependencies=[portfolio_name],
        )

        trades = global_tasks[f"{portfolio_name}.trades"] = Task(
            "tradingo.portfolio.calculate_trades",
            task_args=[],
            task_kwargs={
                "start_date": start_date,
                "end_date": end_date,
                "name": portfolio_name,
                "provider": provider,
                "universe": universe,
                **trade_kwargs,
            },
            dependencies=[portfolio_name],
        )

        if "downstream" in portfolio_config:

            downstream = global_tasks[f"{portfolio_name}.downstream"] = Task(
                "tradingo.engine.adjust_position_sizes",
                task_args=[],
                task_kwargs={
                    "name": portfolio_name,
                    "provider": provider,
                    "universe": universe,
                    "stage": backtest_kwargs["stage"],
                },
                dependencies=[portfolio_name],
            )

            downstream_live = eod_tasks[f"{portfolio_name}.downstream.live"] = Task(
                "tradingo.live.get_activity_history",
                task_args=[],
                task_kwargs={
                    "name": portfolio_name,
                    "provider": provider,
                    "universe": universe,
                    "from_date": sample_start_date,
                    "to_date": end_date,
                },
                dependencies=(
                    [] if not include_live else [f"{portfolio_name}.downstream"]
                ),
            )

        if include_live:
            global_tasks[f"{portfolio_name}.downstream.live"] = downstream_live

        if "downstream" in portfolio_config:
            downstream.resolve_dependencies(global_tasks)
        backtest.resolve_dependencies(global_tasks)
        trades.resolve_dependencies(global_tasks)

    eod_tasks.update(instrument_tasks)

    return global_tasks, eod_tasks


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

    if args.auth:
        os.environ.update({k: str(v) for k, v in args.auth.items()})

    graph, _ = build_graph(
        args.config, args.start_date, args.end_date, include_live=True
    )
    arctic = Arctic(args.arctic_uri)

    update_dag(graph)

    task = graph[args.task]

    try:
        out = task.run(
            run_dependencies=args.with_deps,
            force_rerun=args.force_rerun,
            arctic=arctic,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(out)
    finally:
        serialise_dag(graph)


if __name__ == "__main__":

    logging.getLogger("tradingo").setLevel(logging.INFO)
    main()
