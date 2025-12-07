"""Tradingo CLI"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re

import pandas as pd

from tradingo.api import Tradingo
from tradingo.config import read_config_template
from tradingo.dag import DAG
from tradingo.settings import IGTradingConfig, TradingoConfig

logger = logging.getLogger(__name__)


def int_or_bool(val: str) -> int | bool:
    if val.lower() in {"true", "yes"}:
        return True
    if val.lower() == {"false", "no"}:
        return False
    if not val.isnumeric():
        raise ValueError("Invalid value for int or bool: {val}")
    return int(val)


def cli_app() -> argparse.ArgumentParser:
    """Tradingo CLI app."""

    app = argparse.ArgumentParser("tradingo-tasks")

    app.add_argument(
        "--auth",
        type=lambda i: IGTradingConfig.from_env(
            env=read_config_template(pathlib.Path(i), os.environ),
            override_default_env=False,
        ).to_env(),
        required=False,
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
    run_tasks.add_argument("--with-deps", default=False, type=int_or_bool)
    run_tasks.add_argument("--start-date", type=pd.Timestamp, required=False)
    run_tasks.add_argument("--end-date", type=pd.Timestamp, required=False)
    run_tasks.add_argument("--force-rerun", action="store_true", default=True)
    run_tasks.add_argument("--dry-run", action="store_true")
    run_tasks.add_argument("--clean", action="store_true")
    run_tasks.add_argument("--skip-deps", type=re.compile)

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
            if args.clean:
                extra_kwargs["clean"] = args.clean
            out = graph.run(
                args.task,
                run_dependencies=args.with_deps,
                force_rerun=args.force_rerun,
                arctic=arctic,
                dry_run=args.dry_run,
                skip_deps=args.skip_deps,
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
