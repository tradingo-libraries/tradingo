import pandas as pd
from tradingo.cli import build_graph


PROVIDER = "ig-trading"
UNIVERSE = "test-universe"
PORTFOLIO = "test-portfolio"
NAME = "test-name"


def test_build_graph():

    config = {
        "name": NAME,
        "universe": {
            UNIVERSE: {
                "provider": PROVIDER,
                "epics": ["A", "B"],
                "index_col": None,
                "volatility": {
                    "speeds": [32, 64],
                },
            }
        },
        "volatility": {"speeds": []},
        "signal_configs": {
            "signal1": {"function": "module.function", "args": [], "kwargs": {}},
            "signal1.capped": {
                "depends_on": ["signal1"],
                "function": "module.function",
                "args": [],
                "kwargs": {},
            },
        },
        "portfolio": {
            PORTFOLIO: {
                "function": "module.portfolio_function",
                "args": [],
                "kwargs": {
                    "signal_weights": {"signal1.capped": 1},
                },
                "universe": UNIVERSE,
                "provider": PROVIDER,
            }
        },
    }

    tasks, eod_tasks = build_graph(
        config, pd.Timestamp("2018-01-01"), pd.Timestamp("2024-09-18")
    )

    assert tasks[PORTFOLIO].dependencies == [
        tasks[f"{UNIVERSE}.sample"],
        tasks[f"{UNIVERSE}.vol"],
        tasks[f"{UNIVERSE}.signal1.capped"],
    ]

    assert "A.sample" in tasks
    assert "B.sample" in tasks
    assert f"{UNIVERSE}.sample" in tasks

    assert tasks[f"{PORTFOLIO}.backtest"].dependencies == [tasks[PORTFOLIO]]

    assert tasks[f"{UNIVERSE}.sample"].dependencies == [
        tasks[f"{UNIVERSE}.instruments"]
    ]
    assert tasks[f"{UNIVERSE}.signal1"].dependencies == [
        tasks[f"{UNIVERSE}.sample"],
        tasks[f"{UNIVERSE}.vol"],
    ]
