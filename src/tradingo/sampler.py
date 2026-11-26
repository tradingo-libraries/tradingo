"""OpenTelemetry sampler that records only traces rooted in an Airflow task.

Select it with ``OTEL_TRACES_SAMPLER=tradingo_task`` (registered as an
``opentelemetry_traces_sampler`` entry point). It is a ``ParentBased`` sampler whose *root* rule
keeps a span only if it carries ``airflow.task_id`` (set by :mod:`tradingo.airflow_plugin`):

- background work with no task (scheduler loops, Celery broker polling, DB heartbeats) is dropped
  at the root, and so is everything below it;
- anything inside a task root span follows that span's decision;
- Celery workers continue a task's trace through the propagated context (remote parent sampled).
"""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes

from tradingo.airflow_plugin import TASK_ATTRIBUTE


class AirflowTaskRootSampler(Sampler):
    """Sample a root span iff it is an Airflow task span."""

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        if attributes and TASK_ATTRIBUTE in attributes:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, trace_state)
        return SamplingResult(Decision.DROP, None, trace_state)

    def get_description(self) -> str:
        return "AirflowTaskRootSampler"


def task_root_sampler(arg: str | None = None) -> Sampler:
    """Entry-point factory for ``OTEL_TRACES_SAMPLER=tradingo_task``."""
    return ParentBased(root=AirflowTaskRootSampler())
