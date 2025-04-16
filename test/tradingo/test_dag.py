import pytest

from tradingo.dag import DAG


def test_dag_configuration():
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


if __name__ == "__main__":
    pytest.main([__file__])
