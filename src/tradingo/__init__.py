"""Tradingo framework.

The public names are imported lazily (PEP 562): ``import tradingo`` must stay cheap
because the OpenTelemetry sampler entry point (``tradingo.sampler``) is resolved in
the ``sitecustomize`` of every Python process in the Airflow image, healthchecks and
CLI calls included. Eagerly importing ``backtest``/``dag`` pulled in pandas, numpy and
sqlalchemy, ~1.1s CPU per process start on the cluster nodes.
"""

import importlib
from typing import TYPE_CHECKING, Any

__all__ = ["Tradingo", "dag", "config", "backtest"]

if TYPE_CHECKING:
    from . import backtest, config, dag
    from .api import Tradingo
else:
    # Runtime only: a module __getattr__ visible to mypy would type every unknown
    # attribute of the package (``from . import _backtest`` included) as Any.
    _SUBMODULES = {"backtest", "config", "dag"}

    def __getattr__(name: str) -> Any:
        if name in _SUBMODULES:
            value = importlib.import_module(f".{name}", __name__)
        elif name == "Tradingo":
            value = importlib.import_module(".api", __name__).Tradingo
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(globals()) | set(__all__))
