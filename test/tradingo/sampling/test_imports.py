"""Importing the samplers must not load trading_ig.

``import trading_ig`` also imports its streaming client, and importing
Lightstreamer starts threads that poll every 10ms for the life of the process
(~0.05 core each in the sampler's Celery pool, even with no jobs).
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
]


@pytest.mark.parametrize("module", MODULES)
def test_import_does_not_load_trading_ig(module: str) -> None:
    # A fresh interpreter: another test may already have imported trading_ig here.
    code = (
        f"import sys, threading, {module}\n"
        "loaded = sorted(m for m in sys.modules"
        " if m.split('.')[0] in ('trading_ig', 'lightstreamer'))\n"
        "assert not loaded, loaded\n"
        "assert threading.active_count() == 1, threading.enumerate()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
