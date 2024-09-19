from tradingo.cli import build_graph


PROVIDER = "test-provider"
UNIVERSE = "test-universe"
PORTFOLIO = "test-portfolio"


def test_build_graph():

    config = {
        "universe": {
            UNIVERSE: {
                "provider": PROVIDER,
                "index_col": None,
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
                "kwargs": {},
                "signal_weights": {"signal1.capped": 1},
                "universe": UNIVERSE,
            }
        },
    }

    tasks = build_graph(config)

    print(tasks)
