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
from tradingo.config import read_config_template
from tradingo.dag import DAG
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
        type=lambda i: (
            read_config_template(pathlib.Path(i), dict(os.environ)),
            str(pathlib.Path(i).resolve()),
        ),
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
    run_tasks.add_argument(
        "--recover",
        action="store_true",
        default=False,
        help="Resume a previously failed batched run, skipping completed steps",
    )
    run_tasks.add_argument(
        "--executor",
        choices=["thread", "process", "celery"],
        default="thread",
        help=(
            "Execution backend: 'thread' (default, local ThreadPoolExecutor), "
            "'process' (local ProcessPoolExecutor, bypasses GIL), or "
            "'celery' (distributed, requires tradingo[worker] and running workers)"
        ),
    )
    run_tasks.add_argument(
        "--broker-url",
        default=os.environ.get("TP_CELERY_BROKER_URL"),
        help="Celery broker URL (default: TP_CELERY_BROKER_URL env var)",
    )

    _ = task_subparsers.add_parser("list")

    stop_tasks = task_subparsers.add_parser(
        "stop", help="Revoke all SUBMITTED Celery tasks for an execution plan"
    )
    stop_tasks.add_argument(
        "plan_key", help="Execution plan key (shown by 'task list')"
    )
    stop_tasks.add_argument(
        "--broker-url",
        default=os.environ.get("TP_CELERY_BROKER_URL"),
        help="Celery broker URL (default: TP_CELERY_BROKER_URL env var)",
    )

    return app


def _unpack_config(raw: Any) -> tuple[Any, str | None]:
    """Unpack args.config which is either a (config, path) tuple or a plain dict."""
    if isinstance(raw, tuple):
        return raw
    return raw, None


def handle_tasks(args: argparse.Namespace, arctic: Arctic) -> None:
    """inspect or run Tradingo tasks."""
    config, config_path = _unpack_config(args.config)

    if args.list_action == "list":
        graph = DAG.from_config(config)
        graph.print()
        _print_active_plans()

    elif args.list_action == "stop":
        from tradingo.execution_plan import ExecutionPlan

        if getattr(args, "broker_url", None):
            os.environ["TP_CELERY_BROKER_URL"] = args.broker_url
        plan = ExecutionPlan.load(args.plan_key)
        if plan is None:
            print(f"No execution plan found for key: {args.plan_key}")
            return
        count = plan.revoke_submitted()
        short_key = args.plan_key[:16]
        print(f"Revoked {count} Celery task(s) from plan {short_key}...")

    elif args.list_action == "run":
        graph = DAG.from_config(config)
        graph.update_state()

        # --recover implies force_rerun=False unless explicitly overridden
        recover = getattr(args, "recover", False)
        force_rerun = args.force_rerun
        if recover and force_rerun:
            # force_rerun defaults to True, so unless the user explicitly
            # passed --force-rerun alongside --recover, disable it.
            force_rerun = False

        try:
            extra_kwargs: dict[str, Any] = {}
            if args.start_date:
                extra_kwargs["start_date"] = args.start_date
            if args.end_date:
                extra_kwargs["end_date"] = args.end_date
            if args.clean:
                extra_kwargs["clean"] = args.clean
            graph.run(
                args.task,
                run_dependencies=args.with_deps,
                force_rerun=force_rerun,
                max_workers=getattr(args, "n_workers", 1),
                batch_interval=getattr(args, "batch_interval", None),
                batch_mode=getattr(args, "batch_mode", "stepped"),
                recover=recover,
                config_path=config_path,
                executor=getattr(args, "executor", "thread"),
                broker_url=getattr(args, "broker_url", None),
                arctic=arctic,
                dry_run=args.dry_run,
                skip_deps=args.skip_deps,
                **extra_kwargs,
            )
        finally:
            graph.serialise_state()

    else:
        raise ValueError(args.list_action)


def _print_active_plans() -> None:
    """Print any execution plans that have SUBMITTED steps."""
    try:
        from tradingo.execution_plan import ExecutionPlan

        plans = ExecutionPlan.list_plans()
    except Exception:
        return
    active = [p for p in plans if p.get("SUBMITTED", 0) > 0]
    if not active:
        return
    print("Active Celery runs:")
    for p in active:
        total = p.get("total_steps", "?")
        submitted = p.get("SUBMITTED", 0)
        success = p.get("SUCCESS", 0)
        failed = p.get("FAILED", 0)
        key = p["key"]
        task_name = p.get("task_name", "")
        print(
            f"  {key[:16]}  task={task_name}"
            f"  submitted={submitted} success={success} failed={failed} total={total}"
            f"\n    stop: tradingo-cli --config <config> task stop {key}"
        )
    print()


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

    else:
        raise ValueError(args.entity)


if __name__ == "__main__":
    main()
