"""trading_ig's REST client, without its streaming client.

``trading_ig/__init__.py`` imports ``.stream``, which imports Lightstreamer, and
importing Lightstreamer starts executor threads that poll every 10ms for the
life of the process (~0.05 core per process on the cluster, ~100ms of import).
Tradingo only uses the REST client (``trading_ig.rest`` needs only requests),
so a stub ``trading_ig.stream`` is registered before trading_ig is imported.

Import ``IGService`` / ``ApiExceededException`` from here, not from trading_ig.
A process that needs streaming must import ``trading_ig.stream`` before tradingo.
"""

import sys
import types
from typing import Any

if "trading_ig" not in sys.modules and "trading_ig.stream" not in sys.modules:

    class IGStreamService:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "trading_ig streaming is stubbed out by tradingo.ig_rest: import"
                " trading_ig.stream before tradingo in a process that streams"
            )

    _stream = types.ModuleType("trading_ig.stream")
    _stream.IGStreamService = IGStreamService  # type: ignore[attr-defined]
    sys.modules["trading_ig.stream"] = _stream

from trading_ig.rest import ApiExceededException, IGService  # noqa: E402

__all__ = ["ApiExceededException", "IGService"]
