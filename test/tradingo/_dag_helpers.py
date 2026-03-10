"""Helper functions for DAG parallel execution tests."""

from __future__ import annotations

import threading
import time
from typing import Any

# Shared state for test assertions
call_log: list[str] = []
call_log_lock = threading.Lock()

# Dated call log: records (task_label, start_date, end_date) tuples
call_log_dated: list[tuple[str, Any, Any]] = []

# Events for controlling execution order in tests
gate_a = threading.Event()
gate_b = threading.Event()
gate_c = threading.Event()


def reset() -> None:
    """Reset shared state between tests."""
    call_log.clear()
    call_log_dated.clear()
    gate_a.clear()
    gate_b.clear()
    gate_c.clear()


def noop(**kwargs: object) -> None:
    """No-op task function."""


def record_a(**kwargs: object) -> None:
    """Record that task A ran."""
    with call_log_lock:
        call_log.append("a")


def record_b(**kwargs: object) -> None:
    """Record that task B ran."""
    with call_log_lock:
        call_log.append("b")


def record_c(**kwargs: object) -> None:
    """Record that task C ran."""
    with call_log_lock:
        call_log.append("c")


def slow_a(**kwargs: object) -> None:
    """Task A that signals and waits."""
    gate_a.set()
    time.sleep(0.05)
    with call_log_lock:
        call_log.append("a")


def slow_b(**kwargs: object) -> None:
    """Task B that signals and waits."""
    gate_b.set()
    time.sleep(0.05)
    with call_log_lock:
        call_log.append("b")


def gated_a(**kwargs: object) -> None:
    """Task A gated by event."""
    with call_log_lock:
        call_log.append("a")
    gate_a.set()


def gated_b(**kwargs: object) -> None:
    """Task B waits for A to finish first."""
    gate_a.wait(timeout=5)
    with call_log_lock:
        call_log.append("b")
    gate_b.set()


def gated_c(**kwargs: object) -> None:
    """Task C waits for B to finish first."""
    gate_b.wait(timeout=5)
    with call_log_lock:
        call_log.append("c")


def fail_a(**kwargs: object) -> None:
    """Task that always fails."""
    raise ValueError("task_a failed")


def fail_b(**kwargs: object) -> None:
    """Task that always fails."""
    raise RuntimeError("task_b failed")


def record_a_dated(**kwargs: Any) -> None:
    """Record task A with date range."""
    with call_log_lock:
        call_log_dated.append(("a", kwargs.get("start_date"), kwargs.get("end_date")))


def record_b_dated(**kwargs: Any) -> None:
    """Record task B with date range."""
    with call_log_lock:
        call_log_dated.append(("b", kwargs.get("start_date"), kwargs.get("end_date")))


def record_c_dated(**kwargs: Any) -> None:
    """Record task C with date range."""
    with call_log_lock:
        call_log_dated.append(("c", kwargs.get("start_date"), kwargs.get("end_date")))
