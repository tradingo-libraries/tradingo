"""Tests for the cron scheduler and /schedule endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import arcticdb as adb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tradingo.runner import PipelineRunner
from tradingo.scheduler import (
    ScheduleConfig,
    _make_tick_with_retry,
    _send_webhook,
    run_schedule,
)
from tradingo.service import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_task(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return prices * 2


@pytest.fixture
def dag_config() -> dict[str, Any]:
    return {
        "stage1": {
            "noop": {
                "function": "test.tradingo.test_scheduler._noop_task",
                "depends_on": [],
                "params": {},
                "symbols_in": {"prices": "prices/close"},
                "symbols_out": ["prices/scaled"],
            },
        },
    }


@pytest.fixture
def backing(dag_config: dict[str, Any]) -> adb.Arctic:
    arctic = adb.Arctic("mem://scheduler-test-backing")
    arctic.create_library("prices")
    dates = pd.bdate_range(start="2024-01-01", periods=5, tz="UTC")
    arctic.get_library("prices").write(
        "close",
        pd.DataFrame({"X": np.random.randn(5).cumsum() + 100}, index=dates),
    )
    return arctic


@pytest.fixture
def runner(dag_config: dict[str, Any], backing: adb.Arctic) -> PipelineRunner:
    r = PipelineRunner(config=dag_config, backing=backing, async_write=False)
    r.warm(libraries=["prices"])
    return r


@pytest.fixture
def base_config() -> ScheduleConfig:
    return ScheduleConfig(
        cron="*/5 * * * *",
        task="noop",
        max_retries=3,
        retry_base_seconds=0,
    )


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestTickWithRetry:

    def test_retries_on_failure_then_succeeds(
        self, runner: PipelineRunner, base_config: ScheduleConfig
    ) -> None:
        tick_fn = _make_tick_with_retry(base_config)
        call_count = 0
        original_tick = runner.tick

        def flaky_tick(*a: Any, **kw: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            original_tick(*a, **kw)

        runner.tick = flaky_tick  # type: ignore[method-assign]
        tick_fn(runner)
        assert call_count == 3

    def test_raises_after_retries_exhausted(
        self, runner: PipelineRunner, base_config: ScheduleConfig
    ) -> None:
        tick_fn = _make_tick_with_retry(base_config)

        def always_fail(*a: Any, **kw: Any) -> None:
            raise RuntimeError("permanent")

        runner.tick = always_fail  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="permanent"):
            tick_fn(runner)


# ---------------------------------------------------------------------------
# Webhook tests
# ---------------------------------------------------------------------------


class TestSendWebhook:

    @patch("tradingo.scheduler.urllib.request.urlopen")
    def test_posts_json(self, mock_urlopen: MagicMock) -> None:
        _send_webhook("http://example.com/hook", {"status": "ok"})
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        assert json.loads(req.data) == {"status": "ok"}

    @patch("tradingo.scheduler.urllib.request.urlopen", side_effect=OSError("fail"))
    def test_failure_is_non_fatal(self, mock_urlopen: MagicMock) -> None:
        # Should not raise
        _send_webhook("http://example.com/hook", {"status": "ok"})
        mock_urlopen.assert_called_once()


# ---------------------------------------------------------------------------
# run_schedule async tests
# ---------------------------------------------------------------------------


class TestRunSchedule:

    def test_tick_executes_on_schedule(
        self, runner: PipelineRunner, base_config: ScheduleConfig
    ) -> None:
        """Scheduler calls tick after sleeping until the next fire time."""
        tick_called = asyncio.Event()
        original_tick = runner.tick

        def capture_tick(*a: Any, **kw: Any) -> None:
            original_tick(*a, **kw)
            tick_called.set()

        runner.tick = capture_tick  # type: ignore[method-assign]

        async def _run() -> None:
            with patch("tradingo.scheduler.asyncio.sleep", return_value=None):
                task = asyncio.create_task(run_schedule(runner, base_config))
                try:
                    await asyncio.wait_for(tick_called.wait(), timeout=5)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        asyncio.run(_run())

    def test_webhook_on_failure(self, runner: PipelineRunner) -> None:
        """Webhook is sent after retries exhausted."""
        config = ScheduleConfig(
            cron="*/5 * * * *",
            task="noop",
            webhook_url="http://example.com/hook",
            max_retries=1,
            retry_base_seconds=0,
        )
        runner.tick = MagicMock(side_effect=RuntimeError("fail"))  # type: ignore[method-assign]

        async def _run() -> None:
            with (
                patch("tradingo.scheduler.asyncio.sleep", return_value=None),
                patch("tradingo.scheduler._send_webhook") as mock_wh,
            ):
                tick_done = asyncio.Event()
                original_send = mock_wh.side_effect

                def capture_wh(*a: Any, **kw: Any) -> None:
                    if original_send:
                        original_send(*a, **kw)
                    tick_done.set()

                mock_wh.side_effect = capture_wh

                task = asyncio.create_task(run_schedule(runner, config))
                try:
                    await asyncio.wait_for(tick_done.wait(), timeout=5)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                assert mock_wh.call_count >= 1
                payload = mock_wh.call_args_list[0][0][1]
                assert payload["status"] == "error"

        asyncio.run(_run())

    def test_success_webhook(
        self, runner: PipelineRunner, base_config: ScheduleConfig
    ) -> None:
        """Webhook is sent on success when notify_on_success=True."""
        config = ScheduleConfig(
            cron=base_config.cron,
            task=base_config.task,
            webhook_url="http://example.com/hook",
            notify_on_success=True,
            max_retries=base_config.max_retries,
            retry_base_seconds=0,
        )

        async def _run() -> None:
            with (
                patch("tradingo.scheduler.asyncio.sleep", return_value=None),
                patch("tradingo.scheduler._send_webhook") as mock_wh,
            ):
                tick_done = asyncio.Event()

                def capture_wh(*a: Any, **kw: Any) -> None:
                    tick_done.set()

                mock_wh.side_effect = capture_wh

                task = asyncio.create_task(run_schedule(runner, config))
                try:
                    await asyncio.wait_for(tick_done.wait(), timeout=5)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                mock_wh.assert_called_once()
                payload = mock_wh.call_args[0][1]
                assert payload["status"] == "ok"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# /schedule endpoint tests
# ---------------------------------------------------------------------------


class TestScheduleEndpoint:

    def test_schedule_enabled(
        self, dag_config: dict[str, Any], backing: adb.Arctic
    ) -> None:
        config = ScheduleConfig(cron="*/5 * * * *", task="noop")
        app = create_app(
            config=dag_config,
            backing_uri=backing.get_uri(),
            schedule_config=config,
        )
        runner = PipelineRunner(config=dag_config, backing=backing, async_write=False)
        runner.warm()

        with TestClient(app) as tc:
            app.state.runner = runner
            app.state.schedule_config = config
            resp = tc.get("/schedule")

        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["cron"] == "*/5 * * * *"
        assert body["task"] == "noop"
        assert "next_fire" in body

    def test_schedule_disabled(
        self, dag_config: dict[str, Any], backing: adb.Arctic
    ) -> None:
        app = create_app(
            config=dag_config,
            backing_uri=backing.get_uri(),
        )
        runner = PipelineRunner(config=dag_config, backing=backing, async_write=False)
        runner.warm()

        with TestClient(app) as tc:
            app.state.runner = runner
            resp = tc.get("/schedule")

        assert resp.status_code == 200
        assert resp.json() == {"enabled": False}
