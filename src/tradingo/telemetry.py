"""Optional OpenTelemetry tracing helpers.

Everything here is a no-op unless ``opentelemetry-api`` is importable *and* the host process has
configured a tracer provider (tradingo-plat does this in ``sitecustomize.py``). tradingo itself
takes no hard dependency on OpenTelemetry.

Spans created here only record when they sit inside a sampled trace; in Airflow that is the
per-task root span opened by :mod:`tradingo.airflow_plugin`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

try:
    from opentelemetry import trace as _trace

    _tracer: Any = _trace.get_tracer("tradingo")
except ImportError:  # pragma: no cover - exercised by the no-otel CI job
    _tracer = None

_PRIMITIVES = (str, bool, int, float)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values and stringify anything OTel cannot carry as an attribute."""
    return {
        k: v if isinstance(v, _PRIMITIVES) else str(v)
        for k, v in attributes.items()
        if v is not None
    }


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a child span of the current context; yields the span (``None`` without OTel)."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name, attributes=_clean(attributes)) as s:
        yield s


def annotate(s: Any, **attributes: Any) -> None:
    """Set attributes on a span yielded by :func:`span`; safe on ``None`` / non-recording."""
    if s is not None and s.is_recording():
        s.set_attributes(_clean(attributes))
