"""Write-through in-memory cache for ArcticDB.

Uses ArcticDB's ``mem://`` backend as a fast primary store, with a real
S3-backed (or any other) Arctic instance as the durable backing store.
All reads come from memory; writes go to both mem and backing store
(synchronously or asynchronously).
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import arcticdb as adb

logger = logging.getLogger(__name__)


class CachingLibrary:
    """Wraps two ``Library`` instances (mem + backing).

    Reads from the mem library.  Writes go to mem immediately and are
    forwarded to the backing library (sync or async depending on the
    ``async_write`` flag).
    """

    def __init__(
        self,
        mem: adb.library.Library,
        backing: adb.library.Library,
        async_write: bool = False,
        executor: ThreadPoolExecutor | None = None,
    ):
        self._mem = mem
        self._backing = backing
        self._async_write = async_write
        self._executor = executor
        self._futures: list[Future[None]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self._backing.name)

    # ------------------------------------------------------------------
    # Read operations — always from mem
    # ------------------------------------------------------------------

    def read(self, *args: Any, **kwargs: Any) -> adb.VersionedItem | adb.LazyDataFrame:
        return self._mem.read(*args, **kwargs)

    def head(self, *args: Any, **kwargs: Any) -> adb.VersionedItem | adb.LazyDataFrame:
        return self._mem.head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> adb.VersionedItem | adb.LazyDataFrame:
        return self._mem.tail(*args, **kwargs)

    def list_symbols(self, *args: Any, **kwargs: Any) -> list[str]:
        return list(self._mem.list_symbols(*args, **kwargs))

    def list_snapshots(self, *args: Any, **kwargs: Any) -> Any:
        return self._mem.list_snapshots(*args, **kwargs)

    # ------------------------------------------------------------------
    # Write operations — write-through to both stores
    # ------------------------------------------------------------------

    def _forward_to_backing(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self._backing, method_name)
        if self._async_write and self._executor is not None:
            future = self._executor.submit(fn, *args, **kwargs)
            self._futures.append(future)
            return future
        return fn(*args, **kwargs)

    def write(self, *args: Any, **kwargs: Any) -> adb.VersionedItem:
        result = self._mem.write(*args, **kwargs)
        self._forward_to_backing("write", *args, **kwargs)
        return result

    def update(self, *args: Any, **kwargs: Any) -> adb.VersionedItem:
        result = self._mem.update(*args, **kwargs)
        self._forward_to_backing("update", *args, **kwargs)
        return result

    def write_pickle(self, *args: Any, **kwargs: Any) -> adb.VersionedItem:
        result = self._mem.write_pickle(*args, **kwargs)
        self._forward_to_backing("write_pickle", *args, **kwargs)
        return result

    def delete(self, *args: Any, **kwargs: Any) -> None:
        self._mem.delete(*args, **kwargs)
        self._forward_to_backing("delete", *args, **kwargs)

    def snapshot(self, *args: Any, **kwargs: Any) -> None:
        self._mem.snapshot(*args, **kwargs)
        self._forward_to_backing("snapshot", *args, **kwargs)

    def delete_snapshot(self, *args: Any, **kwargs: Any) -> None:
        self._mem.delete_snapshot(*args, **kwargs)
        self._forward_to_backing("delete_snapshot", *args, **kwargs)

    # ------------------------------------------------------------------
    # Flush pending async writes
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for all pending async writes to complete."""
        exceptions = []
        for future in self._futures:
            try:
                future.result()
            except Exception as exc:
                logger.error("Async write failed: %s", exc)
                exceptions.append(exc)
        self._futures.clear()
        if exceptions:
            raise RuntimeError(
                f"{len(exceptions)} async write(s) failed"
            ) from exceptions[0]

    # ------------------------------------------------------------------
    # Fallback for any other attribute
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mem, name)


class CachingArctic:
    """Drop-in replacement for ``adb.Arctic`` that caches via ``mem://``.

    The ``mem://`` store is the fast primary store for reads.  The backing
    Arctic (e.g. S3) is the durable store that receives write-through
    copies.

    Because ``adb.Arctic`` doesn't support subclassing cleanly, we use
    composition and forward ``get_library`` / ``list_libraries`` and
    friends explicitly.

    Accepts either a *backing_uri* string (creates a new Arctic) or an
    existing *backing* Arctic instance.  The latter is necessary because
    ArcticDB ``mem://`` namespaces are not shared across instances.
    """

    def __init__(
        self,
        backing_uri: str | None = None,
        *,
        backing: adb.Arctic | None = None,
        async_write: bool = False,
        max_workers: int = 4,
    ):
        if backing is not None:
            self._backing = backing
        elif backing_uri is not None:
            self._backing = adb.Arctic(backing_uri)
        else:
            raise ValueError("Either backing_uri or backing must be provided")

        self._cache_uri = f"mem://cache-{uuid.uuid4().hex[:12]}"
        self._mem = adb.Arctic(self._cache_uri)
        self._async_write = async_write
        self._executor = (
            ThreadPoolExecutor(max_workers=max_workers) if async_write else None
        )
        self._libraries: dict[str, CachingLibrary] = {}

    # ------------------------------------------------------------------
    # Library access
    # ------------------------------------------------------------------

    def get_library(
        self,
        name: str,
        create_if_missing: bool = False,
        **kwargs: Any,
    ) -> CachingLibrary:
        if name in self._libraries:
            return self._libraries[name]

        mem_lib = self._mem.get_library(name, create_if_missing=True, **kwargs)
        backing_lib = self._backing.get_library(
            name, create_if_missing=create_if_missing, **kwargs
        )

        lib = CachingLibrary(
            mem=mem_lib,
            backing=backing_lib,
            async_write=self._async_write,
            executor=self._executor,
        )
        self._libraries[name] = lib
        return lib

    def get_uri(self) -> str:
        return str(self._backing.get_uri())

    def list_libraries(self) -> list[str]:
        return list(self._backing.list_libraries())

    def create_library(self, name: str, **kwargs: Any) -> CachingLibrary:
        self._mem.create_library(name, **kwargs)
        self._backing.create_library(name, **kwargs)
        return self.get_library(name)

    # ------------------------------------------------------------------
    # Cache warming
    # ------------------------------------------------------------------

    def warm_cache(
        self,
        libraries: list[str] | None = None,
        date_range: tuple[Any, Any] | None = None,
    ) -> None:
        """Bulk-load data from the backing store into the mem cache.

        If *libraries* is ``None``, all libraries in the backing store
        are warmed.
        """
        lib_names = (
            libraries if libraries is not None else self._backing.list_libraries()
        )

        for lib_name in lib_names:
            backing_lib = self._backing.get_library(lib_name, create_if_missing=True)
            mem_lib = self._mem.get_library(lib_name, create_if_missing=True)

            for symbol in backing_lib.list_symbols():
                try:
                    item = backing_lib.read(symbol, date_range=date_range)
                    mem_lib.write(symbol, item.data, metadata=item.metadata)
                except Exception:
                    # Try pickle path
                    try:
                        item = backing_lib.read(symbol)
                        mem_lib.write_pickle(symbol, item.data, metadata=item.metadata)
                    except Exception as exc:
                        logger.warning(
                            "Failed to warm %s/%s: %s", lib_name, symbol, exc
                        )

            logger.info(
                "Warmed %s: %d symbols", lib_name, len(backing_lib.list_symbols())
            )

    # ------------------------------------------------------------------
    # Flush & cleanup
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for all pending async writes across all libraries."""
        for lib in self._libraries.values():
            lib.flush()

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
