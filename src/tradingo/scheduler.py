"""Cron-based pipeline scheduler.

Runs ``PipelineRunner.tick()`` on a cron schedule with retry logic
and optional webhook alerting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from croniter import croniter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tradingo.runner import PipelineRunner

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    cron: str
    task: str
    webhook_url: str | None = None
    notify_on_success: bool = False
    max_retries: int = 3
    retry_base_seconds: int = 10
    tick_kwargs: dict[str, Any] = field(default_factory=dict)


def _make_tick_with_retry(config: ScheduleConfig) -> Any:
    """Build a retry-wrapped tick callable from config."""

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=config.retry_base_seconds, max=300),
        reraise=True,
    )
    def _tick(runner: PipelineRunner, **kwargs: Any) -> None:
        runner.tick(config.task, **kwargs)

    return _tick


def _send_webhook(url: str, payload: dict[str, str]) -> None:
    """POST JSON to *url*. Failures are logged but never propagated."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        logger.exception("Webhook POST to %s failed", url)


async def run_schedule(runner: PipelineRunner, config: ScheduleConfig) -> None:
    """Async loop that sleeps until the next cron fire time, then ticks."""

    loop = asyncio.get_running_loop()
    tick_fn = _make_tick_with_retry(config)

    while True:
        now = datetime.now(timezone.utc)
        cron = croniter(config.cron, now)
        next_fire = cron.get_next(datetime)
        delay = (next_fire - now).total_seconds()

        logger.info("Next tick at %s (in %.1fs)", next_fire.isoformat(), delay)
        await asyncio.sleep(delay)

        t0 = time.monotonic()
        try:
            await loop.run_in_executor(
                None, lambda: tick_fn(runner, **config.tick_kwargs)
            )
            elapsed = time.monotonic() - t0
            logger.info("Tick completed in %.2fs", elapsed)
            if config.notify_on_success and config.webhook_url:
                _send_webhook(
                    config.webhook_url,
                    {
                        "status": "ok",
                        "task": config.task,
                        "elapsed_seconds": str(round(elapsed, 4)),
                        "fired_at": next_fire.isoformat(),
                    },
                )
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception("Tick failed after retries (%.2fs)", elapsed)
            if config.webhook_url:
                _send_webhook(
                    config.webhook_url,
                    {
                        "status": "error",
                        "task": config.task,
                        "elapsed_seconds": str(round(elapsed, 4)),
                        "fired_at": next_fire.isoformat(),
                    },
                )
