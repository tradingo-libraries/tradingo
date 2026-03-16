"""Smoke tests for config templates shipped with tradingo.

Verifies each template in tradingo/templates/ renders without error and
produces a valid task graph when the documented template variables are provided.
"""

import pathlib
from typing import Any

import pytest

import tradingo
from tradingo import config
from tradingo.dag import collect_task_configs

TEMPLATES_DIR = pathlib.Path(tradingo.__file__).parent / "templates"

_DATES = {
    "data_interval_start": "2025-01-01T00:00:00+00:00",
    "data_interval_end": "2025-01-02T00:00:00+00:00",
}

TEMPLATE_CASES = [
    pytest.param(
        "instruments/ig-trading.yaml",
        {
            **_DATES,
            "universe_name": "test-universe",
            "epics": ["EPIC1", "EPIC2"],
            "raw_prices_lib": "prices_test",
            "interval": "15min",
            "samplingFunction": "tradingo.sampling.ig.sample_instrument",
            "TP_INCLUDE_INSTRUMENTS": True,
        },
        id="ig-trading",
    ),
    pytest.param(
        "instruments/yfinance.yaml",
        {
            **_DATES,
            "universe_name": "test-universe",
            "tickers": ["AAPL", "MSFT"],
            "raw_prices_lib": "prices_yf",
            "interval": "1d",
            "index_col": "",
            "currency": None,
            "fx_universe_name": None,
            "TP_INCLUDE_INSTRUMENTS": True,
        },
        id="yfinance",
    ),
    pytest.param(
        "instruments/dukascopy.yaml",
        {
            **_DATES,
            "universe_name": "test-universe",
            "epics": ["EURUSD", "GBPUSD"],
            "raw_prices_lib": "prices_dukascopy",
            "interval": "15min",
            "samplingFunction": "tradingo.sampling.dukascopy.sample_instrument",
            "permit_missing": False,
            "TP_INCLUDE_INSTRUMENTS": True,
        },
        id="dukascopy",
    ),
    pytest.param(
        "instruments/forexsb.yaml",
        {
            **_DATES,
            "universe_name": "test-universe",
            "epics": ["EURUSD", "GBPUSD"],
            "raw_prices_lib": "prices_forexsb",
            "interval": 15,
            "data_dir": "/tmp/forexsb-data",
            "TP_INCLUDE_INSTRUMENTS": True,
        },
        id="forexsb",
    ),
    pytest.param(
        "portfolio_construction.yaml",
        {
            **_DATES,
            "name": "my-portfolio",
            "model_weights": {"my-signal.test-universe": 1.0},
            "portfolio": {"aum": 10000, "multiplier": 0.5},
            "position": "portfolio/my-portfolio.rounded.shares",
            "bid_close": "prices/test-universe.mid.close",
            "ask_close": "prices/test-universe.mid.close",
            "backtest": {"price_ffill_limit": 5},
        },
        id="portfolio-construction",
    ),
    pytest.param(
        "downstream_tasks.yaml",
        {
            "portfolio_name": "my-portfolio",
            "universe_name": "test-universe",
            "tradingEnabled": True,
        },
        id="downstream-tasks",
    ),
]


@pytest.mark.parametrize("template_name,variables", TEMPLATE_CASES)
def test_template_parses(template_name: str, variables: dict[str, Any]) -> None:
    """Each template renders and produces a valid task graph.

    Templates are fragment configs designed to be composed with other configs,
    so we validate rendering and task structure without resolving cross-template
    dependencies.
    """
    template_path = TEMPLATES_DIR / template_name
    result = config.read_config_template(template_path, variables)
    assert result, f"Template {template_name} rendered to an empty config"
    tasks = collect_task_configs(result)
    assert tasks, f"Template {template_name} produced no task configs"
