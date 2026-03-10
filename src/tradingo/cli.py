"""Tradingo CLI"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import logging.config
import os
import pathlib
import re
from typing import Any

import pandas as pd
import yaml
from arcticdb import Arctic

from tradingo.api import Tradingo
from tradingo.caching import CachingArctic
from tradingo.config import read_config_template
from tradingo.dag import DAG
from tradingo.runner import PipelineRunner
from tradingo.settings import IGTradingConfig, TradingoConfig

logger = logging.getLogger(__name__)


def setup_logging(config_path: str | None = None) -> None:
    """Configure logging from a YAML dictConfig file.

    Resolution order:
    1. ``config_path`` argument (from ``--log-config`` CLI flag)
    2. ``TRADINGO_LOG_CONFIG`` environment variable
    3. Bundled ``logging.yaml`` shipped with the package
    """
    path = config_path or os.environ.get("TRADINGO_LOG_CONFIG")

    if path is not None:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    else:
        ref = importlib.resources.files("tradingo").joinpath("logging.yaml")
        cfg = yaml.safe_load(ref.read_text(encoding="utf-8"))

    logging.config.dictConfig(cfg)


def parse_interval(value: str) -> pd.Timedelta:
    """Parse a human-readable interval string to pd.Timedelta.

    Accepted formats: ``"3days"``, ``"2months"``, ``"1hour"``, ``"30min"``,
    ``"1w"``.  Months are approximated as 30-day multiples.
    """
    m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", value.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid interval format: {value!r}")

    n, unit = int(m.group(1)), m.group(2).lower()

    if unit in {"month", "months", "mo"}:
        return pd.Timedelta(days=n * 30)

    if unit in {"week", "weeks"}:
        return pd.Timedelta(weeks=n)

    try:
        return pd.Timedelta(n, unit=unit)  # type: ignore
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Unrecognized time unit {unit!r} in interval {value!r}"
        ) from exc


def int_or_bool(val: str) -> int | bool:
    if val.lower() in {"true", "yes"}:
        return True
    if val.lower() in {"false", "no"}:
        return False
    if not val.isnumeric():
        raise ValueError("Invalid value for int or bool: {val}")
    return int(val)


def cli_app() -> argparse.ArgumentParser:
    """Tradingo CLI app."""

    app = argparse.ArgumentParser("tradingo-tasks")

    app.add_argument(
        "--log-config",
        default=None,
        help="Path to a logging YAML config file. Env: TRADINGO_LOG_CONFIG",
    )
    app.add_argument(
        "--auth",
        type=lambda i: IGTradingConfig.from_env(
            env=read_config_template(pathlib.Path(i), dict(os.environ)),
            override_default_env=False,
        ).to_env(),
        required=False,
    )
    app.add_argument(
        "--config",
        type=lambda i: read_config_template(pathlib.Path(i), dict(os.environ)),
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
    run_tasks.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution (default: 1 = sequential)",
    )
    run_tasks.add_argument(
        "--cache",
        action="store_true",
        help="Wrap Arctic in a write-through in-memory cache for faster reads",
    )
    run_tasks.add_argument(
        "--batch-interval",
        type=parse_interval,
        default=None,
        help="Split the date range into chunks of this size (e.g. '3days', '2months', '1hour')",
    )
    run_tasks.add_argument(
        "--batch-mode",
        choices=["stepped", "task", "deps-first"],
        default="stepped",
        help="How to order batched chunks vs dependencies (default: stepped)",
    )

    _ = task_subparsers.add_parser("list")

    pipeline = entity.add_parser("pipeline")
    pipeline_subparsers = pipeline.add_subparsers(dest="pipeline_action", required=True)
    run_pipeline = pipeline_subparsers.add_parser("run")
    run_pipeline.add_argument("task")
    run_pipeline.add_argument("--with-deps", default=True, type=int_or_bool)
    run_pipeline.add_argument("--start-date", type=pd.Timestamp, required=False)
    run_pipeline.add_argument("--end-date", type=pd.Timestamp, required=False)
    run_pipeline.add_argument("--warm-start", type=pd.Timestamp, required=False)
    run_pipeline.add_argument("--warm-end", type=pd.Timestamp, required=False)
    run_pipeline.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution (default: 1 = sequential)",
    )
    run_pipeline.add_argument(
        "--async-write",
        action="store_true",
        help="Use async write-through (faster, eventual consistency)",
    )

    serve_pipeline = pipeline_subparsers.add_parser("serve")
    serve_pipeline.add_argument("--host", default="0.0.0.0")
    serve_pipeline.add_argument("--port", type=int, default=8000)
    serve_pipeline.add_argument("--warm-start", type=pd.Timestamp, required=False)
    serve_pipeline.add_argument("--warm-end", type=pd.Timestamp, required=False)
    serve_pipeline.add_argument(
        "--async-write",
        action="store_true",
        help="Use async write-through (faster, eventual consistency)",
    )
    serve_pipeline.add_argument(
        "--task",
        required=False,
        help="Default task name for /run endpoint",
    )
    serve_pipeline.add_argument(
        "--schedule",
        default=os.environ.get("TP_SCHEDULE"),
        help="Cron expression (e.g. '*/5 * * * *'). Env: TP_SCHEDULE",
    )
    serve_pipeline.add_argument(
        "--webhook-url",
        default=os.environ.get("TP_WEBHOOK_URL"),
        help="URL to POST failure alerts to. Env: TP_WEBHOOK_URL",
    )
    serve_pipeline.add_argument(
        "--notify-on-success",
        action="store_true",
        default=os.environ.get("TP_NOTIFY_ON_SUCCESS", "").lower()
        in ("1", "true", "yes"),
        help="Also POST on successful ticks. Env: TP_NOTIFY_ON_SUCCESS",
    )
    serve_pipeline.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("TP_MAX_RETRIES", "3")),
        help="Retries per tick before alerting (default: 3). Env: TP_MAX_RETRIES",
    )
    serve_pipeline.add_argument(
        "--retry-base-seconds",
        type=int,
        default=int(os.environ.get("TP_RETRY_BASE_SECONDS", "10")),
        help="Base delay for exponential backoff (default: 10). Env: TP_RETRY_BASE_SECONDS",
    )

    return app


def handle_tasks(args: argparse.Namespace, arctic: Arctic | CachingArctic) -> None:
    """inspect or run Tradingo tasks."""

    if args.list_action == "list":
        graph = DAG.from_config(
            args.config,
        )

        graph.print()

    elif args.list_action == "run":
        if getattr(args, "cache", False):
            assert isinstance(arctic, Arctic)
            caching_arctic = CachingArctic(backing=arctic)
            caching_arctic.warm_cache()
            arctic = caching_arctic

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
            graph.run(
                args.task,
                run_dependencies=args.with_deps,
                force_rerun=args.force_rerun,
                max_workers=getattr(args, "n_workers", 1),
                batch_interval=getattr(args, "batch_interval", None),
                batch_mode=getattr(args, "batch_mode", "stepped"),
                arctic=arctic,
                dry_run=args.dry_run,
                skip_deps=args.skip_deps,
                **extra_kwargs,
            )
        finally:
            if isinstance(arctic, CachingArctic):
                arctic.flush()
            graph.serialise_state()

    else:
        raise ValueError(args.list_action)


def handle_pipeline(args: argparse.Namespace, arctic_uri: str) -> None:
    """Run pipeline with in-memory caching."""

    if args.pipeline_action == "run":
        runner = PipelineRunner(
            config=args.config,
            backing_uri=arctic_uri,
            async_write=getattr(args, "async_write", False),
        )

        warm_range = None
        if getattr(args, "warm_start", None) or getattr(args, "warm_end", None):
            warm_range = (
                getattr(args, "warm_start", None),
                getattr(args, "warm_end", None),
            )
        runner.warm(date_range=warm_range)

        extra_kwargs = {}
        if args.start_date:
            extra_kwargs["start_date"] = args.start_date
        if args.end_date:
            extra_kwargs["end_date"] = args.end_date

        runner.tick(
            args.task,
            run_dependencies=args.with_deps,
            max_workers=getattr(args, "n_workers", 1),
            **extra_kwargs,
        )

    elif args.pipeline_action == "serve":
        import uvicorn

        from tradingo.scheduler import ScheduleConfig
        from tradingo.service import create_app

        warm_range = None
        if getattr(args, "warm_start", None) or getattr(args, "warm_end", None):
            warm_range = (
                getattr(args, "warm_start", None),
                getattr(args, "warm_end", None),
            )

        schedule_config = None
        cron_expr = getattr(args, "schedule", None)
        if cron_expr:
            task = getattr(args, "task", None)
            if not task:
                raise SystemExit("--task is required when --schedule is specified")
            schedule_config = ScheduleConfig(
                cron=cron_expr,
                task=task,
                webhook_url=getattr(args, "webhook_url", None),
                notify_on_success=getattr(args, "notify_on_success", False),
                max_retries=getattr(args, "max_retries", 3),
                retry_base_seconds=getattr(args, "retry_base_seconds", 10),
            )

        app = create_app(
            config=args.config,
            backing_uri=arctic_uri,
            async_write=getattr(args, "async_write", False),
            warm_range=warm_range,
            default_task=getattr(args, "task", None),
            schedule_config=schedule_config,
        )
        uvicorn.run(app, host=args.host, port=args.port)

    else:
        raise ValueError(args.pipeline_action)


def handle_universes(args: Any, api: Tradingo) -> None:
    """inspect Tradingo's universe by inspecting the instruments in DB."""

    if args.universe_action == "list":
        for item in api.instruments.list_symbols():
            print(item)

    elif args.universe_action == "show":
        print(api.instruments[args.name]())

    elif args.universe_action == "prices":
        print(api.instruments[args.name]())

    else:
        raise ValueError(args.universe_action)


def main(
    _args: argparse.Namespace | None = None,
    _arctic: Arctic | None = None,
) -> None:
    """Tradingo CLI entrypoint"""

    args = _args or cli_app().parse_args()
    setup_logging(getattr(args, "log_config", None))

    envconfig = TradingoConfig.from_env().to_env()
    IGTradingConfig.from_env().to_env()
    envconfig.to_env()

    arctic = _arctic or Tradingo(envconfig.arctic_uri)
    if args.entity == "task":
        handle_tasks(args, arctic)

    elif args.entity == "universe":
        handle_universes(args, api=arctic)

    elif args.entity == "pipeline":
        handle_pipeline(args, arctic_uri=envconfig.arctic_uri)

    else:
        raise ValueError(args.entity)


if __name__ == "__main__":
    main()
