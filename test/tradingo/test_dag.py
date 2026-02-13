from typing import Any, cast

import pytest

from tradingo.config import ConfigLoadError
from tradingo.dag import DAG, Stage


def test_dag_configuration() -> None:
    nodes = {
        "raw_prices": {
            "MSFT.sample": {
                "name": "MSFT.sample",
                "function": "tradingo.sampling.ig.sample_instrument",
                "depends_on": [],
                "symbols_in": [],
                "symbols_out": [
                    "ig-trading/{symbol}.mid",
                    "ig-trading/{symbol}.bid",
                    "ig-trading/{symbol}.ask",
                ],
                "params": {
                    "symbol": "MSFT",
                },
            },
            "AAPL.sample": {
                "name": "AAPL.sample",
                "function": "tradingo.sampling.ig.sample_instrument",
                "symbols_in": [],
                "depends_on": [],
                "symbols_out": [
                    "ig-trading/mid",
                    "ig-trading/bid",
                    "ig-trading/ask",
                ],
                "params": {
                    "symbol": "AAPL",
                },
            },
        },
        "prices": {
            "universe.sample": {
                "function": "tradingo.sampling.ig.sample_universe",
                "depends_on": ["AAPL.sample", "MSFT.sample"],
                "symbols_in": [
                    "ig-trading/AAPL.mid",
                    "ig-trading/AAPL.bid",
                    "ig-trading/AAPL.ask",
                    "ig-trading/MSFT.mid",
                    "ig-trading/MSFT.bid",
                    "ig-trading/MSFT.ask",
                ],
                "symbols_out": [
                    "prices/{universe}.mid.open",
                    "prices/{universe}.mid.high",
                    "prices/{universe}.mid.low",
                    "prices/{universe}.mid.close",
                    "prices/{universe}.bid.open",
                    "prices/{universe}.bid.high",
                    "prices/{universe}.bid.low",
                    "prices/{universe}.bid.close",
                    "prices/{universe}.ask.open",
                    "prices/{universe}.ask.high",
                    "prices/{universe}.ask.low",
                    "prices/{universe}.ask.close",
                ],
                "params": {},
            },
        },
        "signals": {
            "signal.trend": {
                "function": "tradingo.signals.trend",
                "symbols_out": ["signals/{universe}.trend"],
                "depends_on": ["universe.sample"],
                "symbols_in": ["prices/{universe}.mid.close"],
                "params": {
                    "prices": "prices/mid.close",
                    "library": "signals",
                    "field": "trend",
                    "universe": "ig-trading",
                },
            },
        },
    }

    dag = DAG.from_config(nodes)

    assert dag.get_symbols() == [
        "ig-trading/{symbol}.mid",
        "ig-trading/{symbol}.bid",
        "ig-trading/{symbol}.ask",
        "ig-trading/mid",
        "ig-trading/bid",
        "ig-trading/ask",
        "prices/{universe}.mid.open",
        "prices/{universe}.mid.high",
        "prices/{universe}.mid.low",
        "prices/{universe}.mid.close",
        "prices/{universe}.bid.open",
        "prices/{universe}.bid.high",
        "prices/{universe}.bid.low",
        "prices/{universe}.bid.close",
        "prices/{universe}.ask.open",
        "prices/{universe}.ask.high",
        "prices/{universe}.ask.low",
        "prices/{universe}.ask.close",
        "signals/{universe}.trend",
    ]


def _make_simple_task(
    name: str,
    depends_on: list[str] | None = None,
    symbols_out: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to build a minimal task config dict."""
    return {
        "function": "builtins.print",
        "depends_on": depends_on or [],
        "symbols_in": {},
        "symbols_out": symbols_out or [],
        "params": params or {},
    }


class TestStage:
    """Tests for the Stage grouping feature."""

    def test_stage_creation(self) -> None:
        """Stage groups subtasks and computes external dependencies."""
        config = {
            "task.a": _make_simple_task("task.a"),
            "task.b": _make_simple_task(
                "task.b", depends_on=["task.a"], symbols_out=["lib/b"]
            ),
            "task.c": _make_simple_task(
                "task.c", depends_on=["task.b"], symbols_out=["lib/c"]
            ),
            "my.stage": {"stage": ["task.b", "task.c"]},
        }
        dag = DAG.from_config(config)

        # Stage replaces subtasks in the DAG
        assert "my.stage" in dag
        assert "task.b" not in dag
        assert "task.c" not in dag
        assert "task.a" in dag

        stage = dag["my.stage"]
        assert isinstance(stage, Stage)

    def test_stage_external_dependencies(self) -> None:
        """Stage auto-computes external deps from subtask deps."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        stage = dag["my.stage"]
        assert stage.dependency_names == ["upstream"]

    def test_stage_rewires_downstream_deps(self) -> None:
        """Tasks depending on stage members get re-wired to the stage."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task(
                "step.2",
                depends_on=["step.1"],
                symbols_out=["lib/result"],
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
            "downstream": _make_simple_task("downstream", depends_on=["step.2"]),
        }
        dag = DAG.from_config(config)

        # downstream should now depend on the stage, not step.2
        assert dag["downstream"].dependency_names == ["my.stage"]

    def test_stage_deduplicates_rewired_deps(self) -> None:
        """Multiple deps on stage members collapse to one dep on stage."""
        config = {
            "upstream": _make_simple_task("upstream"),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
            "downstream": _make_simple_task(
                "downstream", depends_on=["step.1", "step.2"]
            ),
        }
        dag = DAG.from_config(config)

        assert dag["downstream"].dependency_names == ["my.stage"]

    def test_stage_topological_ordering(self) -> None:
        """Subtasks are sorted by internal dependencies, not config order."""
        config = {
            "upstream": _make_simple_task("upstream"),
            # Listed in reverse order
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "step.1": _make_simple_task("step.1", depends_on=["upstream"]),
            "my.stage": {"stage": ["step.2", "step.1"]},
        }
        dag = DAG.from_config(config)

        stage = cast(Stage, dag["my.stage"])
        subtask_names = [t.name for t in stage._tasks]
        assert subtask_names == ["step.1", "step.2"]

    def test_stage_aggregates_symbols(self) -> None:
        """Stage symbols_out is the union of all subtask symbols_out."""
        config = {
            "step.1": _make_simple_task("step.1", symbols_out=["lib/a", "lib/b"]),
            "step.2": _make_simple_task(
                "step.2", depends_on=["step.1"], symbols_out=["lib/c"]
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        assert dag["my.stage"].symbols_out == ["lib/a", "lib/b", "lib/c"]

    def test_stage_missing_subtask(self) -> None:
        """Stage referencing a non-existent task raises ConfigLoadError."""
        config = {
            "step.1": _make_simple_task("step.1"),
            "my.stage": {"stage": ["step.1", "step.missing"]},
        }
        with pytest.raises(ConfigLoadError, match="step.missing"):
            DAG.from_config(config)

    def test_stage_circular_dependency(self) -> None:
        """Circular deps within a stage raise ConfigLoadError."""
        config = {
            "step.1": _make_simple_task("step.1", depends_on=["step.2"]),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        with pytest.raises(ConfigLoadError, match="Circular"):
            DAG.from_config(config)

    def test_stage_get_symbols(self) -> None:
        """DAG.get_symbols includes stage subtask symbols."""
        config = {
            "task.a": _make_simple_task("task.a", symbols_out=["lib/a"]),
            "step.1": _make_simple_task(
                "step.1", depends_on=["task.a"], symbols_out=["lib/s1"]
            ),
            "step.2": _make_simple_task(
                "step.2", depends_on=["step.1"], symbols_out=["lib/s2"]
            ),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        symbols = dag.get_symbols()
        assert "lib/a" in symbols
        assert "lib/s1" in symbols
        assert "lib/s2" in symbols

    def test_stage_with_explicit_depends_on(self) -> None:
        """Stage can have explicit depends_on in addition to auto-computed."""
        config = {
            "extra.dep": _make_simple_task("extra.dep"),
            "step.1": _make_simple_task("step.1"),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {
                "stage": ["step.1", "step.2"],
                "depends_on": ["extra.dep"],
            },
        }
        dag = DAG.from_config(config)

        assert "extra.dep" in dag["my.stage"].dependency_names

    def test_stage_repr(self) -> None:
        """Stage has informative repr."""
        config = {
            "step.1": _make_simple_task("step.1"),
            "step.2": _make_simple_task("step.2", depends_on=["step.1"]),
            "my.stage": {"stage": ["step.1", "step.2"]},
        }
        dag = DAG.from_config(config)

        r = repr(dag["my.stage"])
        assert "my.stage" in r
        assert "step.1" in r
        assert "step.2" in r


if __name__ == "__main__":
    pytest.main([__file__])
