"""Execution plan persistence for batched DAG runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class _Backend(Protocol):
    def save(self, plan: ExecutionPlan) -> Path | None: ...
    def load(self, key: str) -> ExecutionPlan | None: ...
    def mark_step(self, plan: ExecutionPlan, index: int, status: str) -> None: ...


class _FileBackend:
    def __init__(self, plans_dir: Path) -> None:
        self._plans_dir = plans_dir

    def _filepath(self, key: str) -> Path:
        return self._plans_dir / f"{key}.json"

    def save(self, plan: ExecutionPlan) -> Path:
        path = self._filepath(plan.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan._to_dict(), indent=2))
        return path

    def load(self, key: str) -> ExecutionPlan | None:
        path = self._filepath(key)
        if not path.exists():
            return None
        return ExecutionPlan._from_dict(json.loads(path.read_text()))

    def mark_step(self, plan: ExecutionPlan, index: int, status: str) -> None:
        path = self._filepath(plan.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan._to_dict(), indent=2))


class _RedisBackend:
    """Stores plan metadata as a Redis string and step statuses as a hash.

    mark_step() is a single HSET — O(1) regardless of plan size.

    One instance is reused per URL (class-level cache) so the connection
    pool is created once, not on every save/load/mark_step call.
    """

    _META_SUFFIX = ""
    _STATUSES_SUFFIX = ":statuses"
    _KEY_PREFIX = "execution_plan"
    _TTL_SECONDS = 7 * 24 * 3600  # 7 days

    _instances: dict[str, _RedisBackend] = {}
    _instances_lock: threading.Lock = threading.Lock()

    def __new__(cls, redis_url: str) -> _RedisBackend:
        with cls._instances_lock:
            if redis_url not in cls._instances:
                instance = object.__new__(cls)
                instance._init(redis_url)
                cls._instances[redis_url] = instance
            return cls._instances[redis_url]

    def _init(self, redis_url: str) -> None:
        import redis

        self._client: redis.Redis[str] = redis.Redis.from_url(
            redis_url, decode_responses=True
        )

    def _meta_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}"

    def _statuses_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}:statuses"

    def save(self, plan: ExecutionPlan) -> None:
        meta_key = self._meta_key(plan.key)
        statuses_key = self._statuses_key(plan.key)
        statuses = {str(i): s.status for i, s in enumerate(plan.steps)}
        pipe = self._client.pipeline()
        # Only write immutable metadata if this is the first save.
        if not self._client.exists(meta_key):
            meta = {
                "key": plan.key,
                "params": plan.params,
                "created_at": plan.created_at,
                "steps_meta": [
                    {
                        "task_name": s.task_name,
                        "start_date": s.start_date,
                        "end_date": s.end_date,
                    }
                    for s in plan.steps
                ],
            }
            pipe.set(meta_key, json.dumps(meta), ex=self._TTL_SECONDS)
        else:
            pipe.expire(meta_key, self._TTL_SECONDS)
        pipe.hset(statuses_key, mapping=statuses)
        pipe.expire(statuses_key, self._TTL_SECONDS)
        pipe.execute()

    def load(self, key: str) -> ExecutionPlan | None:
        pipe = self._client.pipeline()
        pipe.get(self._meta_key(key))
        pipe.hgetall(self._statuses_key(key))
        raw, statuses = pipe.execute()
        if raw is None:
            return None
        meta = json.loads(raw)
        steps = [
            ExecutionStep(
                task_name=s["task_name"],
                start_date=s["start_date"],
                end_date=s["end_date"],
                status=statuses.get(str(i), "PENDING"),
            )
            for i, s in enumerate(meta["steps_meta"])
        ]
        return ExecutionPlan(
            key=meta["key"],
            params=meta["params"],
            steps=steps,
            created_at=meta["created_at"],
        )

    def mark_step(self, plan: ExecutionPlan, index: int, status: str) -> None:
        self._client.hset(self._statuses_key(plan.key), str(index), status)


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

    Backend selection (in priority order):
      1. ``REDIS_URL`` class attribute (for testing / programmatic override)
      2. ``TP_EXECUTION_PLAN_REDIS_URL`` env var
      3. File backend (default, ``~/.tradingo/execution-plans``)
    """

    key: str
    params: dict[str, str]
    steps: list[ExecutionStep]
    created_at: str

    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    PLANS_DIR: str = "~/.tradingo/execution-plans"
    REDIS_URL: str | None = None  # override in tests or via subclass

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
    # Backend selection
    # ------------------------------------------------------------------

    @classmethod
    def _get_backend(cls) -> _Backend:
        redis_url = cls.REDIS_URL or os.environ.get("TP_EXECUTION_PLAN_REDIS_URL")
        if redis_url:
            return _RedisBackend(redis_url)
        return _FileBackend(Path(cls.PLANS_DIR).expanduser())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def _plans_dir(cls) -> Path:
        return Path(cls.PLANS_DIR).expanduser()

    def save(self) -> Path | None:
        """Persist the plan (thread-safe). Returns Path for file backend, None for Redis."""
        with self._lock:
            return self._get_backend().save(self)

    @classmethod
    def load(cls, key: str) -> ExecutionPlan | None:
        """Load a plan by key, or return None if it doesn't exist."""
        return cls._get_backend().load(key)

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
        """Update a step's status and write-through to the backend."""
        with self._lock:
            self.steps[index].status = status
        self._get_backend().mark_step(self, index, status)

    def first_non_success(self) -> int | None:
        """Return the index of the first non-SUCCESS step, or None."""
        for i, step in enumerate(self.steps):
            if step.status != "SUCCESS":
                return i
        return None
