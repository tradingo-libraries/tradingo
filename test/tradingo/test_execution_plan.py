"""Tests for ExecutionPlan persistence and recovery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradingo.execution_plan import ExecutionPlan, ExecutionStep


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
