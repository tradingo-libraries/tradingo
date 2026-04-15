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

import logging
import os
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

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
            task_kwargs=task_spec["task_kwargs"],
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


def require_celery() -> Any:
    """Return the Celery app, raising ImportError if celery is not installed."""
    if app is None:
        raise ImportError(
            "Celery is not installed. "
            "Install the [worker] extra: uv sync --group worker"
        )
    return app


def serialize_task(task: Task) -> dict[str, Any]:
    """Serialise a Task to a JSON-safe dict for transport to a Celery worker."""
    return {
        "name": task.name,
        "function": task._function,
        "task_args": list(task.task_args),
        "task_kwargs": task.task_kwargs,
        "symbols_out": task.symbols_out,
        "symbols_in": task.symbols_in,
        "load_args": dict(task.load_args),
        "publish_args": dict(task.publish_args),
    }


def serialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert DAG global kwargs to a JSON-serialisable dict.

    - ``arctic`` is dropped (worker creates its own from ``TP_ARCTIC_URI``)
    - ``pd.Timestamp`` values are converted to ISO-8601 strings
    - ``re.Pattern`` values are converted to their pattern string
    """
    result: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "arctic":
            continue
        elif isinstance(v, pd.Timestamp):
            result[k] = v.isoformat()
        elif isinstance(v, re.Pattern):
            result[k] = v.pattern
        else:
            result[k] = v
    return result


def _deserialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reverse of ``serialize_kwargs`` — called on the worker side."""
    result: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in ("start_date", "end_date") and isinstance(v, str):
            result[k] = pd.Timestamp(v)
        else:
            result[k] = v
    return result
