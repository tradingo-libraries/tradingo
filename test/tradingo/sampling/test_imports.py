"""Importing tradingo's IG code must not load Lightstreamer.

``import trading_ig`` also imports its streaming client, and importing
Lightstreamer starts threads that poll every 10ms for the life of the process
(~0.05 core each in the sampler's Celery pool, even with no jobs).
``tradingo.ig_rest`` stubs ``trading_ig.stream`` so only the REST client loads.

Each check runs in a fresh interpreter: another test may already have imported
trading_ig here.
"""

import subprocess
import sys

import pytest

MODULES = [
    "tradingo.sampling",
    "tradingo.sampling.dukascopy",
    "tradingo.sampling.yf",
    "tradingo.sampling.ig",
    "tradingo.sampling.instruments",
    "tradingo.engine.ig",
]


def run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("module", MODULES)
def test_import_does_not_load_lightstreamer(module: str) -> None:
    run(
        f"import sys, threading, {module}\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'lightstreamer')\n"
        "assert not loaded, loaded\n"
        "assert threading.active_count() == 1, threading.enumerate()\n"
    )


def test_rest_client_available_and_stream_stub_fails_loudly() -> None:
    run(
        "import pytest\n"
        "from tradingo.sampling.ig import IGService\n"
        "import trading_ig\n"
        "assert IGService is trading_ig.IGService\n"
        "with pytest.raises(RuntimeError, match='stubbed out'):\n"
        "    trading_ig.IGStreamService(None)\n"
    )


def test_streaming_process_keeps_the_real_stream_client() -> None:
    run(
        "import trading_ig.stream, tradingo.sampling.ig, trading_ig\n"
        "assert trading_ig.IGStreamService.__module__ == 'trading_ig.stream'\n"
        "assert hasattr(trading_ig.stream, 'LightstreamerClient')\n"
    )
