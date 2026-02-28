"""FastAPI service wrapping PipelineRunner.

Keeps a ``CachingArctic`` alive across HTTP requests so that
intermediate DAG data stays in memory between ticks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tradingo.runner import PipelineRunner
from tradingo.scheduler import ScheduleConfig, run_schedule

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    task: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    run_dependencies: bool | int = True
    force_rerun: bool = True


class RunResponse(BaseModel):
    status: str
    elapsed_seconds: float
    task: str


def create_app(
    config: dict[str, Any],
    backing_uri: str,
    async_write: bool = False,
    warm_range: tuple[Any, Any] | None = None,
    default_task: str | None = None,
    schedule_config: ScheduleConfig | None = None,
) -> FastAPI:
    """Build a FastAPI app with an attached PipelineRunner.

    Parameters
    ----------
    config
        DAG configuration dict (parsed YAML).
    backing_uri
        URI for the durable Arctic backing store (e.g. S3).
    async_write
        Use async write-through for the caching layer.
    warm_range
        Optional ``(start, end)`` date range for cache warm-up at startup.
    default_task
        Default task name used when ``/run`` is called without a task.
    schedule_config
        Optional cron schedule configuration.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runner = PipelineRunner(
            config=config,
            backing_uri=backing_uri,
            async_write=async_write,
        )
        logger.info("Warming cache...")
        runner.warm(date_range=warm_range, libraries=[])
        logger.info("Cache warm-up complete")
        app.state.runner = runner
        app.state.default_task = default_task
        app.state.schedule_config = schedule_config

        scheduler_task = None
        if schedule_config is not None:
            logger.info(
                "Starting scheduler: cron=%s task=%s",
                schedule_config.cron,
                schedule_config.task,
            )
            scheduler_task = asyncio.create_task(run_schedule(runner, schedule_config))

        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                try:
                    await scheduler_task
                except asyncio.CancelledError:
                    pass
            logger.info("Flushing pending writes...")
            runner.flush()
            logger.info("Shutdown complete")

    app = FastAPI(title="Tradingo Pipeline Service", lifespan=lifespan)

    @app.post("/run")
    def run_pipeline(req: RunRequest) -> RunResponse:
        runner: PipelineRunner = app.state.runner
        task = req.task or app.state.default_task
        if not task:
            raise HTTPException(
                status_code=400,
                detail="No task specified and no default task configured",
            )

        kwargs: dict[str, Any] = {}
        if req.start_date is not None:
            kwargs["start_date"] = req.start_date
        if req.end_date is not None:
            kwargs["end_date"] = req.end_date

        t0 = time.monotonic()
        try:
            runner.tick(
                task,
                run_dependencies=req.run_dependencies,
                force_rerun=req.force_rerun,
                **kwargs,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {exc}",
            ) from exc
        elapsed = time.monotonic() - t0

        return RunResponse(status="ok", elapsed_seconds=round(elapsed, 4), task=task)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/libraries")
    def list_libraries() -> list[str]:
        runner: PipelineRunner = app.state.runner
        return runner.arctic.list_libraries()

    @app.get("/libraries/{name}/symbols")
    def list_symbols(name: str) -> list[str]:
        runner: PipelineRunner = app.state.runner
        try:
            lib = runner.arctic.get_library(name)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Library not found: {name}",
            ) from exc
        return lib.list_symbols()

    @app.get("/schedule")
    def schedule_status() -> dict[str, Any]:
        cfg: ScheduleConfig | None = getattr(app.state, "schedule_config", None)
        if cfg is None:
            return {"enabled": False}
        from croniter import croniter

        now = datetime.now(timezone.utc)
        next_fire = croniter(cfg.cron, now).get_next(datetime)
        return {
            "enabled": True,
            "cron": cfg.cron,
            "task": cfg.task,
            "next_fire": next_fire.isoformat(),
        }

    return app
