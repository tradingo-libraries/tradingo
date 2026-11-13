"""Tests for ExecutionPlan persistence and recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, cast
from unittest.mock import MagicMock

import fakeredis
import pandas as pd
import pytest

from tradingo.execution_plan import ExecutionPlan, ExecutionStep, _RedisBackend


@pytest.fixture()
def tmp_plans_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ExecutionPlan storage to a temp directory."""
    monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path))
    return tmp_path


class TestMakeKey:
    def test_deterministic(self) -> None:
        args = dict(
            config_path="/a/b.yaml",
            task_name="task.c",
            start_date="2024-01-01",
            end_date="2024-07-01",
            batch_interval="30days",
            batch_mode="stepped",
            with_deps="True",
        )
        assert ExecutionPlan.make_key(**args) == ExecutionPlan.make_key(**args)

    def test_different_params_differ(self) -> None:
        base = dict(
            config_path="/a/b.yaml",
            task_name="task.c",
            start_date="2024-01-01",
            end_date="2024-07-01",
            batch_interval="30days",
            batch_mode="stepped",
            with_deps="True",
        )
        k1 = ExecutionPlan.make_key(**base)
        k2 = ExecutionPlan.make_key(**{**base, "batch_mode": "task"})
        assert k1 != k2


class TestFromSchedule:
    def test_creates_pending_steps(self) -> None:
        schedule = [
            ("a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            ("b", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
        ]
        plan = ExecutionPlan.from_schedule("key123", {"foo": "bar"}, schedule)
        assert len(plan.steps) == 2
        assert all(s.status == "PENDING" for s in plan.steps)
        assert plan.steps[0].task_name == "a"
        assert plan.steps[1].task_name == "b"


class TestSaveAndLoad:
    def test_round_trip(self, tmp_plans_dir: Path) -> None:
        plan = ExecutionPlan.from_schedule(
            "testkey",
            {"p": "1"},
            [("t", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))],
        )
        plan.save()

        loaded = ExecutionPlan.load("testkey")
        assert loaded is not None
        assert loaded.key == "testkey"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].task_name == "t"
        assert loaded.steps[0].status == "PENDING"

    def test_load_missing_returns_none(self, tmp_plans_dir: Path) -> None:
        assert ExecutionPlan.load("nonexistent") is None

    def test_save_creates_directory(self, tmp_plans_dir: Path) -> None:
        plan = ExecutionPlan.from_schedule(
            "dirtest",
            {},
            [("t", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))],
        )
        path = plan.save()
        assert path is not None
        assert path.exists()


class TestMarkStep:
    def test_mark_success(self, tmp_plans_dir: Path) -> None:
        plan = ExecutionPlan.from_schedule(
            "marktest",
            {},
            [
                ("a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
                ("b", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
            ],
        )
        plan.save()

        plan.mark_step(0, "SUCCESS")
        assert plan.steps[0].status == "SUCCESS"
        assert plan.steps[1].status == "PENDING"

        # Verify write-through
        loaded = ExecutionPlan.load("marktest")
        assert loaded is not None
        assert loaded.steps[0].status == "SUCCESS"
        assert loaded.steps[1].status == "PENDING"

    def test_mark_failed(self, tmp_plans_dir: Path) -> None:
        plan = ExecutionPlan.from_schedule(
            "failtest",
            {},
            [("a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))],
        )
        plan.save()
        plan.mark_step(0, "FAILED")
        assert plan.steps[0].status == "FAILED"


class TestFirstNonSuccess:
    def test_all_pending(self) -> None:
        plan = ExecutionPlan(
            key="k",
            params={},
            steps=[
                ExecutionStep("a", "2024-01-01", "2024-01-02"),
                ExecutionStep("b", "2024-01-02", "2024-01-03"),
            ],
            created_at="now",
        )
        assert plan.first_non_success() == 0

    def test_first_done(self) -> None:
        plan = ExecutionPlan(
            key="k",
            params={},
            steps=[
                ExecutionStep("a", "2024-01-01", "2024-01-02", status="SUCCESS"),
                ExecutionStep("b", "2024-01-02", "2024-01-03"),
            ],
            created_at="now",
        )
        assert plan.first_non_success() == 1

    def test_all_done(self) -> None:
        plan = ExecutionPlan(
            key="k",
            params={},
            steps=[
                ExecutionStep("a", "2024-01-01", "2024-01-02", status="SUCCESS"),
                ExecutionStep("b", "2024-01-02", "2024-01-03", status="SUCCESS"),
            ],
            created_at="now",
        )
        assert plan.first_non_success() is None

    def test_failed_step(self) -> None:
        plan = ExecutionPlan(
            key="k",
            params={},
            steps=[
                ExecutionStep("a", "2024-01-01", "2024-01-02", status="SUCCESS"),
                ExecutionStep("b", "2024-01-02", "2024-01-03", status="FAILED"),
                ExecutionStep("c", "2024-01-03", "2024-01-04"),
            ],
            created_at="now",
        )
        assert plan.first_non_success() == 1


# ---------------------------------------------------------------------------
# Redis backend fixtures
# ---------------------------------------------------------------------------

_SCHEDULE = [
    ("task.a", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")),
    ("task.a", pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")),
    ("task.b", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")),
]

_PARAMS = {
    "config_path": "/cfg.yaml",
    "task_name": "task.b",
    "start_date": "2024-01-01",
    "end_date": "2024-01-15",
    "batch_interval": "7days",
    "batch_mode": "task",
    "with_deps": "True",
}


@pytest.fixture()
def fake_redis_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[fakeredis.FakeRedis, None, None]:
    """Wire ExecutionPlan to a fakeredis in-memory server for the duration of the test."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)

    def _fake_init(self: _RedisBackend, redis_url: str) -> None:
        self._client = client

    _RedisBackend._instances.clear()
    monkeypatch.setattr(_RedisBackend, "_init", _fake_init)
    monkeypatch.setattr(ExecutionPlan, "REDIS_URL", "redis://fake:6379/0")
    yield client
    _RedisBackend._instances.clear()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRedisSaveAndLoad:
    def test_round_trip(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        plan = ExecutionPlan.from_schedule("key1", _PARAMS, _SCHEDULE)
        plan.save()

        loaded = ExecutionPlan.load("key1")
        assert loaded is not None
        assert loaded.key == "key1"
        assert loaded.params == _PARAMS
        assert len(loaded.steps) == 3
        assert all(s.status == "PENDING" for s in loaded.steps)
        assert loaded.steps[0].task_name == "task.a"
        assert loaded.steps[2].task_name == "task.b"

    def test_load_missing_returns_none(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        assert ExecutionPlan.load("nonexistent") is None

    def test_ttl_set_on_meta_and_statuses(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("ttlkey", _PARAMS, _SCHEDULE)
        plan.save()
        assert cast(int, fake_redis_backend.ttl("execution_plan:ttlkey")) > 0
        assert cast(int, fake_redis_backend.ttl("execution_plan:ttlkey:statuses")) > 0

    def test_second_save_does_not_overwrite_meta(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("key2", _PARAMS, _SCHEDULE)
        plan.save()
        raw_first = fake_redis_backend.get("execution_plan:key2")

        # Mutate in-memory and save again — meta blob should be unchanged.
        plan.steps[0].status = "SUCCESS"
        plan.save()
        assert fake_redis_backend.get("execution_plan:key2") == raw_first

    def test_celery_task_id_round_trip(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("tidkey", _PARAMS, _SCHEDULE)
        plan.steps[0].celery_task_id = "abc-123"
        plan.save()

        loaded = ExecutionPlan.load("tidkey")
        assert loaded is not None
        assert loaded.steps[0].celery_task_id == "abc-123"
        assert loaded.steps[1].celery_task_id is None


# ---------------------------------------------------------------------------
# Step tracking
# ---------------------------------------------------------------------------


class TestRedisMarkStep:
    def test_mark_success_write_through(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("ms1", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")

        loaded = ExecutionPlan.load("ms1")
        assert loaded is not None
        assert loaded.steps[0].status == "SUCCESS"
        assert loaded.steps[1].status == "PENDING"

    def test_mark_failed_write_through(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("ms2", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(1, "FAILED")

        loaded = ExecutionPlan.load("ms2")
        assert loaded is not None
        assert loaded.steps[1].status == "FAILED"

    def test_mark_step_submitted_sets_status_and_task_id(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("sub1", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "celery-task-xyz")

        assert plan.steps[0].status == "SUBMITTED"
        assert plan.steps[0].celery_task_id == "celery-task-xyz"

        loaded = ExecutionPlan.load("sub1")
        assert loaded is not None
        assert loaded.steps[0].status == "SUBMITTED"
        assert loaded.steps[0].celery_task_id == "celery-task-xyz"

    def test_submitted_treated_as_non_success_for_recovery(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("sub2", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")
        plan.mark_step_submitted(1, "celery-task-abc")

        loaded = ExecutionPlan.load("sub2")
        assert loaded is not None
        assert loaded.first_non_success() == 1


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TestRedisIndex:
    def test_index_populated_on_save(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("idx1", _PARAMS, _SCHEDULE)
        plan.save()

        keys = cast(
            list[str], fake_redis_backend.zrange("execution_plans:index", 0, -1)
        )
        assert "idx1" in keys

    def test_index_meta_has_expected_fields(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("idx2", _PARAMS, _SCHEDULE)
        plan.save()

        meta = cast(
            dict[str, str], fake_redis_backend.hgetall("execution_plans:meta:idx2")
        )
        assert meta["task_name"] == _PARAMS["task_name"]
        assert meta["start_date"] == _PARAMS["start_date"]
        assert meta["end_date"] == _PARAMS["end_date"]
        assert meta["batch_mode"] == _PARAMS["batch_mode"]
        assert meta["total_steps"] == str(len(_SCHEDULE))

    def test_index_meta_ttl_set(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        plan = ExecutionPlan.from_schedule("idx3", _PARAMS, _SCHEDULE)
        plan.save()
        assert cast(int, fake_redis_backend.ttl("execution_plans:meta:idx3")) > 0

    def test_multiple_plans_in_index(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        for key in ("p1", "p2", "p3"):
            ExecutionPlan.from_schedule(key, _PARAMS, _SCHEDULE).save()

        keys = cast(
            list[str], fake_redis_backend.zrange("execution_plans:index", 0, -1)
        )
        assert {"p1", "p2", "p3"}.issubset(set(keys))

    def test_second_save_does_not_duplicate_index_entry(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        plan = ExecutionPlan.from_schedule("dup1", _PARAMS, _SCHEDULE)
        plan.save()
        plan.steps[0].status = "SUCCESS"
        plan.save()  # second save does not re-add to index (is_new=False)

        keys = cast(
            list[str], fake_redis_backend.zrange("execution_plans:index", 0, -1)
        )
        assert keys.count("dup1") == 1


# ---------------------------------------------------------------------------
# list_plans — file backend
# ---------------------------------------------------------------------------


class TestListPlansFile:
    def test_empty_when_no_plans(self, tmp_plans_dir: Path) -> None:
        assert ExecutionPlan.list_plans() == []

    def test_returns_saved_plan(self, tmp_plans_dir: Path) -> None:
        ExecutionPlan.from_schedule("lp1", _PARAMS, _SCHEDULE).save()
        plans = ExecutionPlan.list_plans()
        assert len(plans) == 1
        assert plans[0]["key"] == "lp1"

    def test_status_counts(self, tmp_plans_dir: Path) -> None:
        plan = ExecutionPlan.from_schedule("lp2", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")
        plans = ExecutionPlan.list_plans()
        assert plans[0]["SUCCESS"] == 1
        assert plans[0]["PENDING"] == 2

    def test_limit_applied(self, tmp_plans_dir: Path) -> None:
        for key in ("a", "b", "c"):
            ExecutionPlan.from_schedule(key, _PARAMS, _SCHEDULE).save()
        assert len(ExecutionPlan.list_plans(limit=2)) == 2

    def test_returns_empty_for_missing_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(ExecutionPlan, "PLANS_DIR", str(tmp_path / "nonexistent"))
        assert ExecutionPlan.list_plans() == []


# ---------------------------------------------------------------------------
# list_plans — Redis backend
# ---------------------------------------------------------------------------


class TestListPlansRedis:
    def test_empty_when_no_plans(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        assert ExecutionPlan.list_plans() == []

    def test_returns_saved_plan(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        ExecutionPlan.from_schedule("rlp1", _PARAMS, _SCHEDULE).save()
        plans = ExecutionPlan.list_plans()
        assert len(plans) == 1
        assert plans[0]["key"] == "rlp1"

    def test_status_counts(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        plan = ExecutionPlan.from_schedule("rlp2", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")
        plan.mark_step_submitted(1, "tid-x")
        plans = ExecutionPlan.list_plans()
        assert plans[0]["SUCCESS"] == 1
        assert plans[0]["SUBMITTED"] == 1
        assert plans[0]["PENDING"] == 1

    def test_multiple_plans_newest_first(
        self, fake_redis_backend: fakeredis.FakeRedis
    ) -> None:
        for key in ("old", "mid", "new"):
            ExecutionPlan.from_schedule(key, _PARAMS, _SCHEDULE).save()
        plans = ExecutionPlan.list_plans()
        keys = [p["key"] for p in plans]
        # newest first — "new" was saved last
        assert keys.index("new") < keys.index("old")

    def test_limit_applied(self, fake_redis_backend: fakeredis.FakeRedis) -> None:
        for key in ("r1", "r2", "r3", "r4"):
            ExecutionPlan.from_schedule(key, _PARAMS, _SCHEDULE).save()
        assert len(ExecutionPlan.list_plans(limit=2)) == 2


# ---------------------------------------------------------------------------
# revoke_submitted
# ---------------------------------------------------------------------------


def _make_fake_celery(revoked: list[tuple[str, bool]]) -> MagicMock:
    """Return a fake Celery app that records revoke() calls."""
    fake_app = MagicMock()
    fake_app.control.revoke.side_effect = (
        lambda task_id, terminate=False: revoked.append((task_id, terminate))
    )
    return fake_app


class TestRevokeSubmittedFile:
    def test_revokes_submitted_step(
        self, tmp_plans_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        revoked: list[tuple[str, bool]] = []
        monkeypatch.setattr("tradingo.worker.app", _make_fake_celery(revoked))

        plan = ExecutionPlan.from_schedule("rv1", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "tid-001")
        plan.mark_step(1, "SUCCESS")

        count = plan.revoke_submitted()

        assert count == 1
        assert revoked == [("tid-001", True)]

    def test_marks_revoked_steps_failed(
        self, tmp_plans_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tradingo.worker.app", _make_fake_celery([]))

        plan = ExecutionPlan.from_schedule("rv2", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "tid-002")

        plan.revoke_submitted()

        loaded = ExecutionPlan.load("rv2")
        assert loaded is not None
        assert loaded.steps[0].status == "FAILED"

    def test_skips_non_submitted_steps(
        self, tmp_plans_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        revoked: list[tuple[str, bool]] = []
        monkeypatch.setattr("tradingo.worker.app", _make_fake_celery(revoked))

        plan = ExecutionPlan.from_schedule("rv3", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step(0, "SUCCESS")  # not submitted

        count = plan.revoke_submitted()
        assert count == 0
        assert revoked == []

    def test_returns_zero_when_no_submitted(
        self, tmp_plans_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tradingo.worker.app", _make_fake_celery([]))
        plan = ExecutionPlan.from_schedule("rv4", _PARAMS, _SCHEDULE[:1])
        plan.save()
        assert plan.revoke_submitted() == 0


class TestRevokeSubmittedRedis:
    def test_revokes_and_marks_failed(
        self,
        fake_redis_backend: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        revoked: list[tuple[str, bool]] = []
        monkeypatch.setattr("tradingo.worker.app", _make_fake_celery(revoked))

        plan = ExecutionPlan.from_schedule("rvr1", _PARAMS, _SCHEDULE)
        plan.save()
        plan.mark_step_submitted(0, "redis-tid-001")
        plan.mark_step_submitted(2, "redis-tid-003")

        count = plan.revoke_submitted()

        assert count == 2
        task_ids = [r[0] for r in revoked]
        assert "redis-tid-001" in task_ids
        assert "redis-tid-003" in task_ids

        loaded = ExecutionPlan.load("rvr1")
        assert loaded is not None
        assert loaded.steps[0].status == "FAILED"
        assert loaded.steps[2].status == "FAILED"
        assert loaded.steps[1].status == "PENDING"
