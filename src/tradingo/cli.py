"""Tradingo CLI"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib

import pandas as pd

from tradingo.api import Tradingo
from tradingo.config import read_config_template
from tradingo.dag import DAG
from tradingo.settings import IGTradingConfig, TradingoConfig


def cli_app() -> argparse.ArgumentParser:
    """Tradingo CLI app."""

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


def handle_tasks(args, arctic):
    """inspect or run Tradingo tasks."""

    if args.list_action == "list":
        graph = DAG.from_config(
            args.config,
        )

        graph.print()

    elif args.list_action == "run":
        graph = DAG.from_config(
            args.config,
        )

        graph.update_state()

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
            graph.serialise_state()

    else:
        raise ValueError(args.list_action)


def handle_universes(args, api: Tradingo):
    """inspect Tradingo's universe by inspecting the instruments in DB."""

    if args.universe_action == "list":
        for item in api.instruments.list():
            print(item)

    elif args.universe_action == "show":
        print(api.instruments[args.name]())

    elif args.universe_action == "prices":
        print(api.instruments[args.name]())

    else:
        raise ValueError(args.universe_action)


def main():
    """Tradingo CLI entrypoint"""

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
