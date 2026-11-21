"""Airflow plugin: one OpenTelemetry root span per task instance.

Registered through the ``airflow.plugins`` entry point, so it loads wherever ``tradingo`` is
installed next to Airflow. The listener hooks run inside the task process, so the span is made
current for the whole of ``execute()``: every child span (tradingo tasks, ArcticDB reads/writes,
Celery publishes, HTTP calls) hangs off it.

The root span carries ``airflow.task_id``; the process-wide sampler (``sitecustomize.py`` in
tradingo-plat) keeps a root span only if it has that attribute, which is what limits tracing to
Airflow task execution and drops scheduler/triggerer/broker polling.

Importing this module must not require Airflow or OpenTelemetry; only :data:`TradingoOtelPlugin`
(built lazily) does.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _HAVE_OTEL = True
except ImportError:  # pragma: no cover
    _HAVE_OTEL = False

logger = logging.getLogger(__name__)

TASK_ATTRIBUTE = "airflow.task_id"

_open: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_lock = threading.Lock()


def _key(ti: Any) -> tuple[Any, ...]:
    return (
        ti.dag_id,
        ti.run_id,
        ti.task_id,
        getattr(ti, "map_index", -1),
        getattr(ti, "try_number", 0),
    )


def start_task_span(ti: Any) -> None:
    """Open the root span for ``ti`` and make it current in this thread."""
    if not _HAVE_OTEL:
        return
    task = getattr(ti, "task", None)
    attributes: dict[str, Any] = {
        "airflow.dag_id": ti.dag_id,
        TASK_ATTRIBUTE: ti.task_id,
        "airflow.run_id": ti.run_id,
        "airflow.map_index": getattr(ti, "map_index", -1),
        "airflow.try_number": getattr(ti, "try_number", 0),
        "airflow.operator": (
            getattr(task, "task_type", None) or type(task).__name__
            if task is not None
            else None
        ),
    }
    span = trace.get_tracer("tradingo.airflow").start_span(
        f"airflow.task {ti.dag_id}.{ti.task_id}",
        context=otel_context.Context(),  # always a new root, never nested in a stray context
        attributes={k: v for k, v in attributes.items() if v is not None},
    )
    token = otel_context.attach(trace.set_span_in_context(span))
    with _lock:
        _open[_key(ti)] = (span, token)


def end_task_span(ti: Any, error: BaseException | str | None = None) -> None:
    """Close the root span opened by :func:`start_task_span`; a no-op if there is none."""
    if not _HAVE_OTEL:
        return
    with _lock:
        entry = _open.pop(_key(ti), None)
    if entry is None:
        return
    span, token = entry
    if error is not None:
        span.set_status(Status(StatusCode.ERROR, str(error)))
        if isinstance(error, BaseException):
            span.record_exception(error)
    try:
        otel_context.detach(token)
    except Exception:  # detached from a different context: harmless, span still ends
        logger.debug("could not detach task span context", exc_info=True)
    span.end()
    # The task process may leave via os._exit(), skipping the SDK's atexit flush, which would
    # drop the batch holding this task's spans. One flush per task is cheap.
    flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if flush is not None:
        try:
            flush(5000)
        except Exception:
            logger.debug("span flush failed", exc_info=True)


def build_plugin() -> Any:
    """Build the ``AirflowPlugin`` subclass; imports Airflow, so only called by the loader."""
    from airflow.listeners import hookimpl  # type: ignore[import-not-found]
    from airflow.plugins_manager import AirflowPlugin  # type: ignore[import-not-found]

    class _Listener:
        @hookimpl  # type: ignore[misc]
        def on_task_instance_running(
            self, previous_state: Any, task_instance: Any
        ) -> None:
            start_task_span(task_instance)

        @hookimpl  # type: ignore[misc]
        def on_task_instance_success(
            self, previous_state: Any, task_instance: Any
        ) -> None:
            end_task_span(task_instance)

        @hookimpl  # type: ignore[misc]
        def on_task_instance_failed(
            self, previous_state: Any, task_instance: Any, error: Any = None
        ) -> None:
            end_task_span(task_instance, error or "task failed")

    class TradingoOtelPlugin(AirflowPlugin):  # type: ignore[misc]
        name = "tradingo_otel"
        listeners = [_Listener()]

    return TradingoOtelPlugin


def __getattr__(name: str) -> Any:
    # Airflow's entry-point loader reads ``module:attr`` lazily; build on first access.
    if name == "TradingoOtelPlugin":
        plugin = build_plugin()
        globals()[name] = plugin
        return plugin
    raise AttributeError(name)
