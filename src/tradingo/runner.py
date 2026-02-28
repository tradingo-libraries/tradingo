"""Pipeline runner with in-memory state machine.

Keeps a ``CachingArctic`` alive across ticks so that intermediate data
stays in memory between pipeline intervals.
"""

from __future__ import annotations

import logging
from typing import Any

import arcticdb as adb

from tradingo.caching import CachingArctic
from tradingo.dag import DAG

logger = logging.getLogger(__name__)


class PipelineRunner:
    """State-machine runner for pipeline execution.

    On each tick the DAG is re-executed against the same ``CachingArctic``
    instance so that reads hit the in-memory cache rather than the remote
    backing store.
    """

    def __init__(
        self,
        config: dict[str, Any],
        backing_uri: str | None = None,
        *,
        backing: adb.Arctic | None = None,
        async_write: bool = True,
    ):
        self.dag = DAG.from_config(config)
        self.arctic = CachingArctic(
            backing_uri=backing_uri,
            backing=backing,
            async_write=async_write,
        )

    def warm(
        self,
        date_range: tuple[Any, Any] | None = None,
        libraries: list[str] | None = None,
    ) -> None:
        """Cold start: load historical data from the backing store."""
        libraries = libraries or []
        self.arctic.warm_cache(date_range=date_range, libraries=libraries)

    def tick(
        self,
        task_name: str,
        *,
        run_dependencies: bool | int = True,
        force_rerun: bool = True,
        **kwargs: Any,
    ) -> None:
        """Run one pipeline iteration for a new interval."""
        self.dag.run(
            task_name,
            run_dependencies=run_dependencies,
            force_rerun=force_rerun,
            arctic=self.arctic,
            dry_run=False,
            skip_deps=None,
            **kwargs,
        )
        self.arctic.flush()

    def flush(self) -> None:
        """Ensure all async writes have completed."""
        self.arctic.flush()
