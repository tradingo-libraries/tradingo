from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from arcticdb import Arctic

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from tradingo import airflow_plugin, telemetry  # noqa: E402
from tradingo.symbols import symbol_provider, symbol_publisher  # noqa: E402


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    # the module-level tracer was bound before this provider existed; point it at ours
    monkeypatch.setattr(telemetry, "_tracer", tracer)
    monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_k: tracer)
    yield exporter
    provider.shutdown()


def _ti(**kw: Any) -> Any:
    base = dict(
        dag_id="d", task_id="t", run_id="r", map_index=-1, try_number=1, task=None
    )
    return SimpleNamespace(**{**base, **kw})


def test_span_cleans_attributes(spans: InMemorySpanExporter) -> None:
    with telemetry.span("x", a=1, b=None, c=object()) as s:
        telemetry.annotate(s, d="y", e=None)
    (finished,) = spans.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["a"] == 1
    assert "b" not in finished.attributes and "e" not in finished.attributes
    assert isinstance(finished.attributes["c"], str)
    assert finished.attributes["d"] == "y"


def test_span_noop_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_tracer", None)
    with telemetry.span("x", a=1) as s:
        telemetry.annotate(s, b=2)
    assert s is None


def test_airflow_root_span_parents_children(spans: InMemorySpanExporter) -> None:
    ti = _ti()
    airflow_plugin.start_task_span(ti)
    with telemetry.span("child"):
        pass
    airflow_plugin.end_task_span(ti)

    by_name = {s.name: s for s in spans.get_finished_spans()}
    root, child = by_name["airflow.task d.t"], by_name["child"]
    assert root.parent is None
    assert child.parent is not None and child.parent.span_id == root.context.span_id
    assert root.attributes is not None
    assert root.attributes[airflow_plugin.TASK_ATTRIBUTE] == "t"
    assert root.attributes["airflow.dag_id"] == "d"
    assert root.status.is_ok or root.status.status_code.name == "UNSET"


def test_airflow_root_span_records_failure(spans: InMemorySpanExporter) -> None:
    ti = _ti(map_index=3)
    airflow_plugin.start_task_span(ti)
    airflow_plugin.end_task_span(ti, ValueError("boom"))
    (root,) = spans.get_finished_spans()
    assert root.status.status_code.name == "ERROR"
    assert root.attributes is not None and root.attributes["airflow.map_index"] == 3
    assert any(e.name == "exception" for e in root.events)


def test_end_without_start_is_a_noop(spans: InMemorySpanExporter) -> None:
    airflow_plugin.end_task_span(_ti(task_id="never-started"))
    assert not spans.get_finished_spans()


@pytest.fixture
def metric_reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # the global meter provider can only be set once per process; patch the lookups instead
    monkeypatch.setattr(metrics, "get_meter", provider.get_meter)
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)
    monkeypatch.setattr(airflow_plugin, "_peak_rss_gauge", None)
    yield reader
    provider.shutdown()


def _peak_rss_points(reader: Any) -> list[Any]:
    data = reader.get_metrics_data()
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == airflow_plugin.PEAK_RSS_METRIC
        for point in metric.data.data_points
    ]


def test_peak_rss_recorded_on_metric_and_span(
    spans: InMemorySpanExporter, metric_reader: Any
) -> None:
    ti = _ti(dag_id="dag", task_id="big")
    airflow_plugin.start_task_span(ti)
    airflow_plugin.end_task_span(ti)

    (point,) = _peak_rss_points(metric_reader)
    assert dict(point.attributes) == {"airflow.dag_id": "dag", "airflow.task_id": "big"}
    rss = airflow_plugin.peak_rss_bytes()
    assert rss is not None
    # this test process is at least tens of MB, so a unit slip (KiB vs bytes) would show
    assert point.value >= 10 * 2**20
    assert point.value <= max(rss)

    (root,) = spans.get_finished_spans()
    assert root.attributes is not None
    own = root.attributes["process.peak_rss_bytes"]
    assert isinstance(own, int) and own >= 10 * 2**20
    assert "process.children_peak_rss_bytes" in root.attributes


def test_peak_rss_recorded_without_a_span(metric_reader: Any) -> None:
    airflow_plugin.end_task_span(_ti(task_id="no-span"))
    (point,) = _peak_rss_points(metric_reader)
    assert point.attributes["airflow.task_id"] == "no-span"


def test_arcticdb_read_and_write_spans(spans: InMemorySpanExporter) -> None:
    arctic = Arctic("mem://telemetry")

    @symbol_publisher("my-lib/symbol")
    def publisher() -> pd.DataFrame:
        return pd.DataFrame(
            1, index=pd.date_range("2025-01-01", periods=5), columns=["A"]
        )

    @symbol_provider(df="my-lib/symbol")
    def provider(df: pd.DataFrame) -> pd.DataFrame:
        return df

    publisher(arctic=arctic, dry_run=False)
    provider(arctic=arctic, df="my-lib/symbol")  # type: ignore[arg-type]

    finished: dict[str, ReadableSpan] = {s.name: s for s in spans.get_finished_spans()}
    write, read = finished["arcticdb.write"], finished["arcticdb.read"]
    assert write.attributes is not None and read.attributes is not None
    assert write.attributes["arcticdb.library"] == "my-lib"
    assert write.attributes["arcticdb.symbol"] == "symbol"
    assert write.attributes["arcticdb.rows"] == 5
    assert read.attributes["db.system"] == "arcticdb"
    assert "arcticdb.version" in read.attributes


def test_sampler_keeps_only_airflow_task_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.trace import SpanKind

    from tradingo.sampler import task_root_sampler

    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=task_root_sampler())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(telemetry, "_tracer", tracer)
    monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_k: tracer)

    # background work with no task: root and descendants are all dropped
    with tracer.start_as_current_span("PUBLISH", kind=SpanKind.PRODUCER):
        with telemetry.span("arcticdb.read"):
            pass

    # inside a task root span everything is kept
    ti = _ti()
    airflow_plugin.start_task_span(ti)
    with telemetry.span("arcticdb.write"):
        pass
    airflow_plugin.end_task_span(ti)

    names = sorted(s.name for s in exporter.get_finished_spans())
    assert names == ["airflow.task d.t", "arcticdb.write"]
    provider.shutdown()


def test_sampler_entry_point_is_registered() -> None:
    from importlib.metadata import entry_points

    eps = entry_points(group="opentelemetry_traces_sampler", name="tradingo_task")
    assert [e.value for e in eps] == ["tradingo.sampler:task_root_sampler"]
