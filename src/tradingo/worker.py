"""Optional Celery worker for distributed task execution.

Install the ``[worker]`` extra to use this module::

    pip install tradingo[worker]
    # or with uv:
    uv sync --group worker

Starting a worker::

    celery -A tradingo.worker worker -Q tradingo --concurrency=4

Environment variables:

- ``TP_CELERY_BROKER_URL``  — Redis broker URL (default: ``redis://localhost:6379/0``)
- ``TP_CELERY_RESULT_BACKEND`` — Result backend URL (default: ``redis://localhost:6379/1``)
- ``TP_CELERY_QUEUE`` — Queue name (default: ``tradingo``)
- ``TP_ARCTIC_URI``  — ArcticDB connection string (required on worker nodes)
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from typing import TYPE_CHECKING, Any

import pandas as pd
from arcticdb import Arctic

if TYPE_CHECKING:
    from tradingo.dag import Task

logger = logging.getLogger(__name__)

BROKER_URL = os.environ.get("TP_CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("TP_CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
QUEUE = os.environ.get("TP_CELERY_QUEUE", "tradingo")

try:
    from celery import Celery

    app = Celery("tradingo", broker=BROKER_URL, backend=RESULT_BACKEND)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_default_queue=QUEUE,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
    )

    @app.task(name="tradingo.run_task")  # type: ignore[misc]
    def run_tradingo_task(
        task_spec: dict[str, Any], global_kwargs: dict[str, Any]
    ) -> str:
        """Execute a single tradingo Task on a Celery worker.

        Reconstructs the Task from its serialized spec, connects to ArcticDB
        using the worker's ``TP_ARCTIC_URI`` environment variable, and runs
        the task function with the provided kwargs.
        """
        from arcticdb import Arctic

        from tradingo.dag import Task

        arctic_uri = os.environ["TP_ARCTIC_URI"]
        arctic = Arctic(arctic_uri)

        task = Task(
            name=task_spec["name"],
            function=task_spec["function"],
            task_args=tuple(task_spec["task_args"]),
            task_kwargs=_deserialize_kwargs(task_spec["task_kwargs"]),
            symbols_out=task_spec["symbols_out"],
            symbols_in=task_spec["symbols_in"],
            load_args=dict(task_spec["load_args"]),
            publish_args=dict(task_spec["publish_args"]),
        )

        kwargs = _deserialize_kwargs(global_kwargs)
        kwargs["arctic"] = arctic

        task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), kwargs)
        logger.info("Worker executing task: %s", task.name)
        task.function(*task.task_args, **task_kwargs)
        return task.name

except ImportError:
    app = None
    logger.debug("celery not installed — distributed execution unavailable")


def require_celery() -> Celery:
    """Return the Celery app, raising ImportError if celery is not installed."""
    if app is None:
        raise ImportError(
            "Celery is not installed. "
            "Install the [worker] extra: uv sync --group worker"
        )
    return app


_TIMESTAMP_TAG = "__pd_timestamp__"
_PATTERN_TAG = "__re_pattern__"
_ARCTIC_TAG = "__arctic_Arctic__"


def _serialize_value(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return {_TIMESTAMP_TAG: v.isoformat()}
    elif isinstance(v, datetime.datetime):
        return {_TIMESTAMP_TAG: pd.Timestamp(v).isoformat()}
    elif isinstance(v, re.Pattern):
        return {_PATTERN_TAG: v.pattern}
    elif isinstance(v, Arctic):
        return {_ARCTIC_TAG: v.get_uri()}
    return v


def _deserialize_value(v: Any) -> Any:
    if isinstance(v, dict):
        if _TIMESTAMP_TAG in v:
            return pd.Timestamp(v[_TIMESTAMP_TAG])
        if _PATTERN_TAG in v:
            return re.compile(v[_PATTERN_TAG])
        if _ARCTIC_TAG in v:
            return Arctic(v[_ARCTIC_TAG])
    return v


def serialize_task(task: Task) -> dict[str, Any]:
    """Serialise a Task to a JSON-safe dict for transport to a Celery worker."""
    return {
        "name": task.name,
        "function": task._function,
        "task_args": list(task.task_args),
        "task_kwargs": serialize_kwargs(
            {k: v for k, v in task.task_kwargs.items() if k != "arctic"}
        ),
        "symbols_out": task.symbols_out,
        "symbols_in": task.symbols_in,
        "load_args": dict(task.load_args),
        "publish_args": dict(task.publish_args),
    }


def serialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert DAG global kwargs to a JSON-serialisable dict.

    - ``arctic`` is dropped (worker creates its own from ``TP_ARCTIC_URI``)
    - ``pd.Timestamp`` values are type-tagged for reliable round-trip
    - ``re.Pattern`` values are type-tagged for reliable round-trip
    """
    return {k: _serialize_value(v) for k, v in kwargs.items()}


def _deserialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reverse of ``serialize_kwargs`` — called on the worker side."""
    return {k: _deserialize_value(v) for k, v in kwargs.items()}


def run_task_in_process(
    task_spec: dict[str, Any], global_kwargs: dict[str, Any]
) -> str:
    """Execute a single Task in a worker process (``ProcessPoolExecutor`` target).

    Reconstructs the Task from its serialised spec and runs it.  If
    ``TP_ARCTIC_URI`` is set in the environment an ArcticDB connection is
    created and injected as ``arctic``; tasks without symbols work without it.
    """
    from tradingo.dag import Task

    kwargs = _deserialize_kwargs(global_kwargs)

    task = Task(
        name=task_spec["name"],
        function=task_spec["function"],
        task_args=tuple(task_spec["task_args"]),
        task_kwargs=_deserialize_kwargs(task_spec["task_kwargs"]),
        symbols_out=task_spec["symbols_out"],
        symbols_in=task_spec["symbols_in"],
        load_args=dict(task_spec["load_args"]),
        publish_args=dict(task_spec["publish_args"]),
    )

    task_kwargs = Task.prepare_kwargs(dict(task.task_kwargs), kwargs)
    logger.info("Process worker executing task: %s", task.name)
    task.function(*task.task_args, **task_kwargs)
    return task.name
