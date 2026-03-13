"""Execution plan persistence for batched DAG runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ExecutionStep:
    """A single step in an execution plan."""

    task_name: str
    start_date: str  # ISO format
    end_date: str  # ISO format
    status: str = "PENDING"  # PENDING | SUCCESS | FAILED


@dataclasses.dataclass
class ExecutionPlan:
    """Persisted execution plan for batched runs.

    Tracks per-step completion so a failed run can be resumed
    with ``--recover``.
    """

    key: str
    params: dict[str, str]
    steps: list[ExecutionStep]
    created_at: str

    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    PLANS_DIR: str = "~/.tradingo/execution-plans"

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        config_path: str,
        task_name: str,
        start_date: str,
        end_date: str,
        batch_interval: str,
        batch_mode: str,
        with_deps: str,
    ) -> str:
        """Deterministic SHA-256 key from canonical sorted parameters."""
        parts = sorted(
            {
                "config_path": config_path,
                "task_name": task_name,
                "start_date": start_date,
                "end_date": end_date,
                "batch_interval": batch_interval,
                "batch_mode": batch_mode,
                "with_deps": with_deps,
            }.items()
        )
        payload = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_schedule(
        cls,
        key: str,
        params: dict[str, str],
        schedule: list[tuple[str, Any, Any]],
    ) -> ExecutionPlan:
        """Create a plan from the output of ``generate_batch_schedule``."""
        steps = [
            ExecutionStep(
                task_name=task_name,
                start_date=str(start),
                end_date=str(end),
            )
            for task_name, start, end in schedule
        ]
        return cls(
            key=key,
            params=params,
            steps=steps,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def _plans_dir(cls) -> Path:
        return Path(cls.PLANS_DIR).expanduser()

    def _filepath(self) -> Path:
        return self._plans_dir() / f"{self.key}.json"

    def save(self) -> Path:
        """Write the plan to disk (thread-safe)."""
        with self._lock:
            path = self._filepath()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._to_dict(), indent=2))
            return path

    @classmethod
    def load(cls, key: str) -> ExecutionPlan | None:
        """Load a plan by key, or return None if it doesn't exist."""
        path = cls._plans_dir() / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls._from_dict(data)

    def _to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "params": self.params,
            "steps": [dataclasses.asdict(s) for s in self.steps],
            "created_at": self.created_at,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        return cls(
            key=data["key"],
            params=data["params"],
            steps=[ExecutionStep(**s) for s in data["steps"]],
            created_at=data["created_at"],
        )

    # ------------------------------------------------------------------
    # Step tracking
    # ------------------------------------------------------------------

    def mark_step(self, index: int, status: str) -> None:
        """Update a step's status and write-through to disk."""
        with self._lock:
            self.steps[index].status = status
            path = self._filepath()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._to_dict(), indent=2))

    def first_non_success(self) -> int | None:
        """Return the index of the first non-SUCCESS step, or None."""
        for i, step in enumerate(self.steps):
            if step.status != "SUCCESS":
                return i
        return None
