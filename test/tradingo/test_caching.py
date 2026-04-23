"""Tests for the write-through in-memory caching layer."""

from __future__ import annotations

from typing import Any

import arcticdb as adb
import numpy as np
import pandas as pd
import pytest

from tradingo.caching import CachingArctic, CachingLibrary
from tradingo.runner import PipelineRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=30, tz="UTC")
    return pd.DataFrame(
        {
            "AAPL": np.random.randn(30).cumsum() + 100,
            "MSFT": np.random.randn(30).cumsum() + 200,
        },
        index=dates,
    )


@pytest.fixture
def backing_arctic() -> adb.Arctic:
    arctic = adb.Arctic("mem://backing-test")
    arctic.create_library("prices")
    arctic.create_library("signals")
    return arctic


@pytest.fixture
def caching_arctic(backing_arctic: adb.Arctic) -> CachingArctic:
    return CachingArctic(
        backing=backing_arctic,
        async_write=False,
    )


@pytest.fixture
def caching_arctic_async(backing_arctic: adb.Arctic) -> CachingArctic:
    return CachingArctic(
        backing=backing_arctic,
        async_write=True,
    )


# ---------------------------------------------------------------------------
# CachingLibrary unit tests
# ---------------------------------------------------------------------------


class TestCachingLibrary:

    def test_write_propagates_to_both_stores(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)

        # Read from cache
        result = lib.read("close").data
        pd.testing.assert_frame_equal(result, sample_data, check_freq=False)

        # Also written to backing
        backing_data = backing_arctic.get_library("prices").read("close").data
        pd.testing.assert_frame_equal(backing_data, sample_data, check_freq=False)

    def test_update_propagates_to_both_stores(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)

        new_dates = pd.bdate_range(start="2024-02-12", periods=5, tz="UTC")
        update_data = pd.DataFrame(
            {"AAPL": [150.0] * 5, "MSFT": [250.0] * 5},
            index=new_dates,
        )
        lib.update("close", update_data, upsert=True)

        mem_result = lib.read("close").data
        assert len(mem_result) == 35

        backing_data = backing_arctic.get_library("prices").read("close").data
        assert len(backing_data) == 35

    def test_read_comes_from_mem(
        self, caching_arctic: CachingArctic, sample_data: pd.DataFrame
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)

        result = lib.read("close").data
        pd.testing.assert_frame_equal(result, sample_data, check_freq=False)

    def test_list_symbols(
        self, caching_arctic: CachingArctic, sample_data: pd.DataFrame
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)
        lib.write("open", sample_data)

        symbols = lib.list_symbols()
        assert set(symbols) == {"close", "open"}

    def test_delete_propagates(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)
        lib.delete("close")

        assert "close" not in lib.list_symbols()
        assert "close" not in backing_arctic.get_library("prices").list_symbols()

    def test_name_property(self, caching_arctic: CachingArctic) -> None:
        lib = caching_arctic.get_library("prices")
        assert lib.name == "prices"

    def test_head_and_tail(
        self, caching_arctic: CachingArctic, sample_data: pd.DataFrame
    ) -> None:
        lib = caching_arctic.get_library("prices")
        lib.write("close", sample_data)

        head = lib.head("close", 5)
        assert len(head.data) == 5

        tail = lib.tail("close", 5)
        assert len(tail.data) == 5

    def test_write_pickle_propagates(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        data = pd.DataFrame({"a": [1, 2, 3]})
        lib.write_pickle("pickled", data)

        assert "pickled" in lib.list_symbols()
        assert "pickled" in backing_arctic.get_library("prices").list_symbols()

    def test_snapshot_propagates(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        result = lib.write("close", sample_data)
        lib.snapshot("snap1", versions={"close": result.version})

        assert "snap1" in lib.list_snapshots()
        assert "snap1" in backing_arctic.get_library("prices").list_snapshots()

    def test_delete_snapshot_propagates(
        self,
        caching_arctic: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic.get_library("prices")
        result = lib.write("close", sample_data)
        lib.snapshot("snap1", versions={"close": result.version})
        lib.delete_snapshot("snap1")

        assert "snap1" not in lib.list_snapshots()
        assert "snap1" not in backing_arctic.get_library("prices").list_snapshots()

    def test_getattr_fallback(self, caching_arctic: CachingArctic) -> None:
        """Unknown attributes fall through to the mem library."""
        lib = caching_arctic.get_library("prices")
        assert hasattr(lib, "list_symbols")


# ---------------------------------------------------------------------------
# CachingArctic tests
# ---------------------------------------------------------------------------


class TestCachingArctic:

    def test_get_library_returns_caching_library(
        self, caching_arctic: CachingArctic
    ) -> None:
        lib = caching_arctic.get_library("prices")
        assert isinstance(lib, CachingLibrary)

    def test_get_library_caches_instance(self, caching_arctic: CachingArctic) -> None:
        lib1 = caching_arctic.get_library("prices")
        lib2 = caching_arctic.get_library("prices")
        assert lib1 is lib2

    def test_get_library_create_if_missing(self, caching_arctic: CachingArctic) -> None:
        lib = caching_arctic.get_library("new_lib", create_if_missing=True)
        assert isinstance(lib, CachingLibrary)

    def test_list_libraries(self, caching_arctic: CachingArctic) -> None:
        libs = caching_arctic.list_libraries()
        assert "prices" in libs
        assert "signals" in libs

    def test_create_library(self, caching_arctic: CachingArctic) -> None:
        lib = caching_arctic.create_library("new_lib_2")
        assert isinstance(lib, CachingLibrary)
        assert "new_lib_2" in caching_arctic.list_libraries()

    def test_warm_cache(
        self,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        backing_arctic.get_library("prices").write("close", sample_data)
        backing_arctic.get_library("prices").write("open", sample_data * 0.99)

        ca = CachingArctic(backing=backing_arctic, async_write=False)
        ca.warm_cache(libraries=["prices"])

        lib = ca.get_library("prices")
        result = lib.read("close").data
        pd.testing.assert_frame_equal(result, sample_data, check_freq=False)

        result_open = lib.read("open").data
        pd.testing.assert_frame_equal(result_open, sample_data * 0.99, check_freq=False)

    def test_warm_cache_all_libraries(
        self,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        backing_arctic.get_library("prices").write("close", sample_data)
        backing_arctic.get_library("signals").write("signal1", sample_data)

        ca = CachingArctic(backing=backing_arctic, async_write=False)
        ca.warm_cache()

        assert set(ca.get_library("prices").list_symbols()) == {"close"}
        assert set(ca.get_library("signals").list_symbols()) == {"signal1"}

    def test_warm_cache_with_date_range(
        self,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        backing_arctic.get_library("prices").write("close", sample_data)

        ca = CachingArctic(backing=backing_arctic, async_write=False)
        start = pd.Timestamp("2024-01-10", tz="UTC")
        end = pd.Timestamp("2024-01-20", tz="UTC")
        ca.warm_cache(libraries=["prices"], date_range=(start, end))

        result = ca.get_library("prices").read("close").data
        assert result.index[0] >= start
        assert result.index[-1] <= end

    def test_flush_sync_is_noop(self, caching_arctic: CachingArctic) -> None:
        """Flush on sync mode should complete without error."""
        caching_arctic.flush()

    def test_requires_backing_uri_or_instance(self) -> None:
        with pytest.raises(ValueError, match="Either backing_uri or backing"):
            CachingArctic()


# ---------------------------------------------------------------------------
# Async write-through tests
# ---------------------------------------------------------------------------


class TestAsyncWriteThrough:

    def test_async_write_eventual_consistency(
        self,
        caching_arctic_async: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic_async.get_library("prices")
        lib.write("close", sample_data)

        # Mem should have it immediately
        result = lib.read("close").data
        pd.testing.assert_frame_equal(result, sample_data, check_freq=False)

        # Flush to ensure backing has it
        caching_arctic_async.flush()

        backing_data = backing_arctic.get_library("prices").read("close").data
        pd.testing.assert_frame_equal(backing_data, sample_data, check_freq=False)

    def test_flush_waits_for_all_writes(
        self,
        caching_arctic_async: CachingArctic,
        backing_arctic: adb.Arctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic_async.get_library("prices")

        for i in range(5):
            lib.write(f"sym_{i}", sample_data)

        caching_arctic_async.flush()

        backing_lib = backing_arctic.get_library("prices")
        for i in range(5):
            assert f"sym_{i}" in backing_lib.list_symbols()

    def test_flush_clears_futures(
        self,
        caching_arctic_async: CachingArctic,
        sample_data: pd.DataFrame,
    ) -> None:
        lib = caching_arctic_async.get_library("prices")
        lib.write("close", sample_data)
        caching_arctic_async.flush()

        # Second flush should be a no-op
        caching_arctic_async.flush()


# ---------------------------------------------------------------------------
# Integration: DAG with CachingArctic
# ---------------------------------------------------------------------------


def double_prices(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return prices * 2


def sum_columns(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame({"total": prices.sum(axis=1)}, index=prices.index)


class TestDagIntegration:

    @pytest.fixture
    def dag_config(self) -> dict[str, Any]:
        return {
            "stage1": {
                "double_task": {
                    "function": "test.tradingo.test_caching.double_prices",
                    "depends_on": [],
                    "params": {},
                    "symbols_in": {"prices": "prices/close"},
                    "symbols_out": ["prices/doubled"],
                },
            },
            "stage2": {
                "sum_task": {
                    "function": "test.tradingo.test_caching.sum_columns",
                    "depends_on": ["double_task"],
                    "params": {},
                    "symbols_in": {"prices": "prices/doubled"},
                    "symbols_out": ["signals/summed"],
                },
            },
        }

    def test_dag_with_caching_arctic(
        self,
        dag_config: dict[str, Any],
        sample_data: pd.DataFrame,
    ) -> None:
        """Full DAG run using CachingArctic backed by two mem:// instances."""
        from tradingo.dag import DAG

        backing = adb.Arctic("mem://dag-integration-backing")
        backing.create_library("prices")
        backing.create_library("signals")
        backing.get_library("prices").write("close", sample_data)

        ca = CachingArctic(backing=backing, async_write=False)
        ca.warm_cache(libraries=["prices"])

        dag = DAG.from_config(dag_config)
        dag.run(
            "sum_task",
            run_dependencies=True,
            force_rerun=True,
            arctic=ca,
            dry_run=False,
            skip_deps=None,
        )

        # Verify results in cache
        doubled = ca.get_library("prices").read("doubled").data
        pd.testing.assert_frame_equal(doubled, sample_data * 2, check_freq=False)

        summed = ca.get_library("signals").read("summed").data
        expected_sum = (sample_data * 2).sum(axis=1)
        pd.testing.assert_series_equal(
            summed["total"], expected_sum, check_names=False, check_freq=False
        )

        # Verify results also in backing store
        backing_doubled = backing.get_library("prices").read("doubled").data
        pd.testing.assert_frame_equal(
            backing_doubled, sample_data * 2, check_freq=False
        )

    def test_dag_results_match_plain_arctic(
        self,
        dag_config: dict[str, Any],
        sample_data: pd.DataFrame,
    ) -> None:
        """Verify CachingArctic DAG output matches a plain Arctic run."""
        from tradingo.dag import DAG

        # Run with plain Arctic
        plain_arctic = adb.Arctic("mem://plain-run")
        plain_arctic.create_library("prices")
        plain_arctic.create_library("signals")
        plain_arctic.get_library("prices").write("close", sample_data)

        dag1 = DAG.from_config(dag_config)
        dag1.run(
            "sum_task",
            run_dependencies=True,
            force_rerun=True,
            arctic=plain_arctic,
            dry_run=False,
            skip_deps=None,
        )

        # Run with CachingArctic
        cached_backing = adb.Arctic("mem://cached-run-backing")
        cached_backing.create_library("prices")
        cached_backing.create_library("signals")
        cached_backing.get_library("prices").write("close", sample_data)

        ca = CachingArctic(backing=cached_backing, async_write=False)
        ca.warm_cache(libraries=["prices"])

        dag2 = DAG.from_config(dag_config)
        dag2.run(
            "sum_task",
            run_dependencies=True,
            force_rerun=True,
            arctic=ca,
            dry_run=False,
            skip_deps=None,
        )

        # Compare outputs
        plain_doubled = plain_arctic.get_library("prices").read("doubled").data
        cached_doubled = ca.get_library("prices").read("doubled").data
        pd.testing.assert_frame_equal(plain_doubled, cached_doubled, check_freq=False)

        plain_summed = plain_arctic.get_library("signals").read("summed").data
        cached_summed = ca.get_library("signals").read("summed").data
        pd.testing.assert_frame_equal(plain_summed, cached_summed, check_freq=False)


# ---------------------------------------------------------------------------
# PipelineRunner tests
# ---------------------------------------------------------------------------


_tick_log: list[str] = []


def tick_task(prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    _tick_log.append("tick")
    return prices * 1.01


class TestPipelineRunner:

    @pytest.fixture(autouse=True)
    def reset_tick_log(self) -> None:
        _tick_log.clear()

    @pytest.fixture
    def pipeline_config(self) -> dict[str, Any]:
        return {
            "compute": {
                "compute_task": {
                    "function": "test.tradingo.test_caching.tick_task",
                    "depends_on": [],
                    "params": {},
                    "symbols_in": {"prices": "prices/close"},
                    "symbols_out": ["prices/output"],
                },
            },
        }

    def test_pipeline_runner_single_tick(
        self,
        pipeline_config: dict[str, Any],
        sample_data: pd.DataFrame,
    ) -> None:
        backing = adb.Arctic("mem://pipeline-backing")
        backing.create_library("prices")
        backing.get_library("prices").write("close", sample_data)

        runner = PipelineRunner(
            config=pipeline_config,
            backing=backing,
            async_write=False,
        )
        runner.warm(libraries=["prices"])

        runner.tick(
            "compute_task",
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-02-28", tz="UTC"),
        )

        assert len(_tick_log) == 1

        # Result should be in backing store
        output = backing.get_library("prices").read("output").data
        pd.testing.assert_frame_equal(output, sample_data * 1.01, check_freq=False)

    def test_pipeline_runner_multi_tick(
        self,
        pipeline_config: dict[str, Any],
        sample_data: pd.DataFrame,
    ) -> None:
        backing = adb.Arctic("mem://pipeline-multi-backing")
        backing.create_library("prices")
        backing.get_library("prices").write("close", sample_data)

        runner = PipelineRunner(
            config=pipeline_config,
            backing=backing,
            async_write=False,
        )
        runner.warm(libraries=["prices"])

        for _ in range(3):
            runner.tick(
                "compute_task",
                start_date=pd.Timestamp("2024-01-01", tz="UTC"),
                end_date=pd.Timestamp("2024-02-28", tz="UTC"),
            )

        assert len(_tick_log) == 3

    def test_pipeline_runner_data_persists_across_ticks(
        self,
        sample_data: pd.DataFrame,
    ) -> None:
        """Data written in one tick is readable in the next from mem."""
        config = {
            "write": {
                "write_task": {
                    "function": "test.tradingo.test_caching.tick_task",
                    "depends_on": [],
                    "params": {},
                    "symbols_in": {"prices": "prices/close"},
                    "symbols_out": ["prices/output"],
                },
            },
            "read": {
                "read_task": {
                    "function": "test.tradingo.test_caching.tick_task",
                    "depends_on": ["write_task"],
                    "params": {},
                    "symbols_in": {"prices": "prices/output"},
                    "symbols_out": ["signals/final"],
                },
            },
        }

        backing = adb.Arctic("mem://pipeline-persist-backing")
        backing.create_library("prices")
        backing.create_library("signals")
        backing.get_library("prices").write("close", sample_data)

        runner = PipelineRunner(
            config=config,
            backing=backing,
            async_write=False,
        )
        runner.warm(libraries=["prices"])

        runner.tick(
            "read_task",
            run_dependencies=True,
            start_date=pd.Timestamp("2024-01-01", tz="UTC"),
            end_date=pd.Timestamp("2024-02-28", tz="UTC"),
        )

        # The final output should be prices * 1.01 * 1.01
        final = runner.arctic.get_library("signals").read("final").data
        expected = sample_data * 1.01 * 1.01
        pd.testing.assert_frame_equal(final, expected, check_freq=False)
