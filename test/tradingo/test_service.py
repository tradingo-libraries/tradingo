"""Tests for the FastAPI pipeline service."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import arcticdb as adb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tradingo.service import create_app

# ---------------------------------------------------------------------------
# Helpers — tiny DAG tasks
# ---------------------------------------------------------------------------

_run_log: list[str] = []


def scale_prices(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    _run_log.append("scale")
    return prices * 2


def aggregate(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    _run_log.append("aggregate")
    return pd.DataFrame({"total": prices.sum(axis=1)}, index=prices.index)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_log() -> None:
    _run_log.clear()


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=20, tz="UTC")
    return pd.DataFrame(
        {
            "AAPL": np.random.randn(20).cumsum() + 100,
            "MSFT": np.random.randn(20).cumsum() + 200,
        },
        index=dates,
    )


@pytest.fixture
def dag_config() -> dict[str, Any]:
    return {
        "stage1": {
            "scale_task": {
                "function": "test.tradingo.test_service.scale_prices",
                "depends_on": [],
                "params": {},
                "symbols_in": {"prices": "prices/close"},
                "symbols_out": ["prices/scaled"],
            },
        },
        "stage2": {
            "agg_task": {
                "function": "test.tradingo.test_service.aggregate",
                "depends_on": ["scale_task"],
                "params": {},
                "symbols_in": {"prices": "prices/scaled"},
                "symbols_out": ["signals/aggregated"],
            },
        },
    }


@pytest.fixture
def backing(sample_data: pd.DataFrame) -> adb.Arctic:
    arctic = adb.Arctic("mem://service-test-backing")
    arctic.create_library("prices")
    arctic.create_library("signals")
    arctic.get_library("prices").write("close", sample_data)
    return arctic


@pytest.fixture
def client(
    dag_config: dict[str, Any], backing: adb.Arctic
) -> Generator[TestClient, None, None]:
    app = create_app(
        config=dag_config,
        backing_uri=backing.get_uri(),
        async_write=False,
    )
    # Override runner to use our backing instance directly
    from tradingo.runner import PipelineRunner

    runner = PipelineRunner(
        config=dag_config,
        backing=backing,
        async_write=False,
    )
    runner.warm(libraries=["prices"])

    with TestClient(app) as tc:
        # Replace the runner created by lifespan with ours (shares backing)
        app.state.runner = runner
        yield tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestRunEndpoint:

    def test_run_executes_task(self, client: TestClient) -> None:
        resp = client.post(
            "/run",
            json={"task": "agg_task", "run_dependencies": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["task"] == "agg_task"
        assert body["elapsed_seconds"] >= 0
        assert "scale" in _run_log
        assert "aggregate" in _run_log

    def test_run_with_dates(self, client: TestClient) -> None:
        resp = client.post(
            "/run",
            json={
                "task": "scale_task",
                "start_date": "2024-01-01T00:00:00+00:00",
                "end_date": "2024-01-31T00:00:00+00:00",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["task"] == "scale_task"

    def test_run_invalid_task_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/run",
            json={"task": "nonexistent_task"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_run_no_task_and_no_default_returns_400(self, client: TestClient) -> None:
        resp = client.post("/run", json={"task": ""})
        assert resp.status_code == 400


class TestSubsequentRuns:

    def test_cache_reuse_across_runs(
        self,
        client: TestClient,
        sample_data: pd.DataFrame,
    ) -> None:
        """Second run reuses data written to mem cache by the first."""
        # First run — writes scaled prices to cache
        resp1 = client.post(
            "/run",
            json={"task": "scale_task", "run_dependencies": True},
        )
        assert resp1.status_code == 200

        # Verify data is in cache
        runner = client.app.state.runner  # type: ignore[attr-defined]
        scaled = runner.arctic.get_library("prices").read("scaled").data
        pd.testing.assert_frame_equal(scaled, sample_data * 2, check_freq=False)

        _run_log.clear()

        # Second run — agg_task depends on scale_task output already in mem
        resp2 = client.post(
            "/run",
            json={"task": "agg_task", "run_dependencies": True},
        )
        assert resp2.status_code == 200
        assert "aggregate" in _run_log


class TestListEndpoints:

    def test_list_libraries(self, client: TestClient) -> None:
        resp = client.get("/libraries")
        assert resp.status_code == 200
        libs = resp.json()
        assert "prices" in libs
        assert "signals" in libs

    def test_list_symbols(self, client: TestClient) -> None:
        resp = client.get("/libraries/prices/symbols")
        assert resp.status_code == 200
        symbols = resp.json()
        assert "close" in symbols

    def test_list_symbols_unknown_library(self, client: TestClient) -> None:
        resp = client.get("/libraries/nonexistent/symbols")
        assert resp.status_code == 404


class TestDefaultTask:

    def test_default_task_used_when_empty(
        self, dag_config: dict[str, Any], backing: adb.Arctic
    ) -> None:
        """When a default_task is configured, /run with empty task uses it."""
        from tradingo.runner import PipelineRunner

        app = create_app(
            config=dag_config,
            backing_uri=backing.get_uri(),
            default_task="scale_task",
        )
        runner = PipelineRunner(
            config=dag_config,
            backing=backing,
            async_write=False,
        )
        runner.warm(libraries=["prices"])

        with TestClient(app) as tc:
            app.state.runner = runner
            app.state.default_task = "scale_task"
            resp = tc.post("/run", json={"task": ""})
            assert resp.status_code == 200
            assert resp.json()["task"] == "scale_task"
