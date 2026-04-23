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
    def mark_step_submitted(
        self, plan: ExecutionPlan, index: int, task_id: str
    ) -> None: ...
    def list_plans(self, limit: int = 20) -> list[dict[str, Any]]: ...


class _FileBackend:
    def __init__(self, plans_dir: Path) -> None:
        self._plans_dir = plans_dir

    def _filepath(self, key: str) -> Path:
        return self._plans_dir / f"{key}.json"

    def _write(self, plan: ExecutionPlan) -> Path:
        path = self._filepath(plan.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan._to_dict(), indent=2))
        return path

    def save(self, plan: ExecutionPlan) -> Path:
        return self._write(plan)

    def load(self, key: str) -> ExecutionPlan | None:
        path = self._filepath(key)
        if not path.exists():
            return None
        return ExecutionPlan._from_dict(json.loads(path.read_text()))

    def mark_step(self, plan: ExecutionPlan, index: int, status: str) -> None:
        self._write(plan)

    def mark_step_submitted(
        self, plan: ExecutionPlan, index: int, task_id: str
    ) -> None:
        plan.steps[index].celery_task_id = task_id
        self._write(plan)

    def list_plans(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self._plans_dir.exists():
            return []
        plans = []
        paths = sorted(
            self._plans_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        for path in paths:
            try:
                plan = ExecutionPlan._from_dict(json.loads(path.read_text()))
            except Exception:
                continue
            counts: dict[str, int] = {}
            for s in plan.steps:
                counts[s.status] = counts.get(s.status, 0) + 1
            plans.append(
                {
                    "key": plan.key,
                    "task_name": plan.params.get("task_name", ""),
                    "start_date": plan.params.get("start_date", ""),
                    "end_date": plan.params.get("end_date", ""),
                    "created_at": plan.created_at,
                    "total_steps": str(len(plan.steps)),
                    **counts,
                }
            )
        return plans


class _RedisBackend:
    """Stores plan metadata as a Redis string and step statuses as a hash.

    mark_step() is a single HSET — O(1) regardless of plan size.

    One instance is reused per URL (class-level cache) so the connection
    pool is created once, not on every save/load/mark_step call.
    """

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

        self._client: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True
        )

    _INDEX_KEY = "execution_plans:index"
    _INDEX_META_PREFIX = "execution_plans:meta"

    def _meta_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}"

    def _statuses_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}:statuses"

    def _task_ids_key(self, key: str) -> str:
        return f"{self._KEY_PREFIX}:{key}:task_ids"

    def _index_meta_key(self, key: str) -> str:
        return f"{self._INDEX_META_PREFIX}:{key}"

    def save(self, plan: ExecutionPlan) -> None:
        import time as _time

        meta_key = self._meta_key(plan.key)
        statuses_key = self._statuses_key(plan.key)
        statuses = {str(i): s.status for i, s in enumerate(plan.steps)}
        is_new = not self._client.exists(meta_key)
        pipe = self._client.pipeline()
        # Only write immutable metadata if this is the first save.
        if is_new:
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
            # Add to the discoverable index with creation timestamp as score.
            created_ts = _time.time()
            pipe.zadd(self._INDEX_KEY, {plan.key: created_ts})
            # Human-readable display hash for Grafana / tooling.
            params = plan.params
            pipe.hset(
                self._index_meta_key(plan.key),
                mapping={
                    "task_name": params.get("task_name", ""),
                    "start_date": params.get("start_date", ""),
                    "end_date": params.get("end_date", ""),
                    "batch_mode": params.get("batch_mode", ""),
                    "batch_interval": params.get("batch_interval", ""),
                    "created_at": plan.created_at,
                    "total_steps": str(len(plan.steps)),
                },
            )
            pipe.expire(self._index_meta_key(plan.key), self._TTL_SECONDS)
            # Prune index entries whose plan keys have expired.
            cutoff = created_ts - self._TTL_SECONDS
            pipe.zremrangebyscore(self._INDEX_KEY, "-inf", cutoff)
        else:
            pipe.expire(meta_key, self._TTL_SECONDS)
        pipe.hset(statuses_key, mapping=statuses)
        pipe.expire(statuses_key, self._TTL_SECONDS)
        # Persist any task IDs already set on steps (e.g. after recovery).
        task_ids = {
            str(i): s.celery_task_id
            for i, s in enumerate(plan.steps)
            if s.celery_task_id is not None
        }
        if task_ids:
            task_ids_key = self._task_ids_key(plan.key)
            pipe.hset(task_ids_key, mapping=task_ids)
            pipe.expire(task_ids_key, self._TTL_SECONDS)
        pipe.execute()

    def load(self, key: str) -> ExecutionPlan | None:
        pipe = self._client.pipeline()
        pipe.get(self._meta_key(key))
        pipe.hgetall(self._statuses_key(key))
        pipe.hgetall(self._task_ids_key(key))
        raw, statuses, task_ids = pipe.execute()
        if raw is None:
            return None
        meta = json.loads(raw)
        steps = [
            ExecutionStep(
                task_name=s["task_name"],
                start_date=s["start_date"],
                end_date=s["end_date"],
                status=statuses.get(str(i), "PENDING"),
                celery_task_id=task_ids.get(str(i)),
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

    def mark_step_submitted(
        self, plan: ExecutionPlan, index: int, task_id: str
    ) -> None:
        pipe = self._client.pipeline()
        pipe.hset(self._statuses_key(plan.key), str(index), "SUBMITTED")
        pipe.hset(self._task_ids_key(plan.key), str(index), task_id)
        pipe.execute()

    def list_plans(self, limit: int = 20) -> list[dict[str, Any]]:
        keys: list[str] = self._client.zrevrange(self._INDEX_KEY, 0, limit - 1)  # type: ignore[assignment]
        if not keys:
            return []
        pipe = self._client.pipeline()
        for key in keys:
            pipe.hgetall(self._index_meta_key(key))
            pipe.hgetall(self._statuses_key(key))
        results: list[Any] = pipe.execute()
        plans = []
        for i, key in enumerate(keys):
            meta = results[i * 2]
            statuses = results[i * 2 + 1]
            counts: dict[str, int] = {}
            for s in statuses.values():
                counts[s] = counts.get(s, 0) + 1
            plans.append({"key": key, **meta, **counts})
        return plans


@dataclasses.dataclass
class ExecutionStep:
    """A single step in an execution plan."""

    task_name: str
    start_date: str  # ISO format
    end_date: str  # ISO format
    status: str = "PENDING"  # PENDING | SUBMITTED | SUCCESS | FAILED
    celery_task_id: str | None = None


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
            steps=[
                ExecutionStep(
                    task_name=s["task_name"],
                    start_date=s["start_date"],
                    end_date=s["end_date"],
                    status=s.get("status", "PENDING"),
                    celery_task_id=s.get("celery_task_id"),
                )
                for s in data["steps"]
            ],
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

    def mark_step_submitted(self, index: int, task_id: str) -> None:
        """Record that a step has been dispatched to Celery."""
        with self._lock:
            self.steps[index].status = "SUBMITTED"
            self.steps[index].celery_task_id = task_id
        self._get_backend().mark_step_submitted(self, index, task_id)

    def revoke_submitted(self) -> int:
        """Revoke all SUBMITTED Celery tasks and mark those steps FAILED.

        Returns the number of tasks revoked. Requires the ``[worker]`` extra.
        """
        from tradingo.worker import require_celery

        celery_app = require_celery()
        revoked = 0
        for i, step in enumerate(self.steps):
            if step.status == "SUBMITTED" and step.celery_task_id is not None:
                celery_app.control.revoke(step.celery_task_id, terminate=True)
                self.mark_step(i, "FAILED")
                revoked += 1
        return revoked

    @classmethod
    def list_plans(cls, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent execution plans from the backend, newest first."""
        return cls._get_backend().list_plans(limit=limit)

    def first_non_success(self) -> int | None:
        """Return the index of the first non-SUCCESS step, or None.

        SUBMITTED steps are treated as non-success — if the dispatcher died
        mid-run those tasks may not have completed.
        """
        for i, step in enumerate(self.steps):
            if step.status != "SUCCESS":
                return i
        return None
