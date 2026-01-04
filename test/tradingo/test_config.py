import os
import textwrap

import pytest

from tradingo import config, dag
from tradingo.settings import TradingoConfig


@pytest.fixture
def config_home(tmp_path):
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "signals").mkdir(exist_ok=True)
    (tmp_path / "configs" / "universes").mkdir(exist_ok=True)

    (tmp_path / "configs" / "myconfig.yaml").write_text(
        textwrap.dedent(
            """\
        {% set universe_name = 'myuniverse' %}
        prices:
            include: "file://{{ TP_TEMPLATES }}/instruments/ig-trading.yaml"
            variables:
                include: "file://{{ TP_CONFIG_HOME }}/universes/{{ universe_name }}.yaml"
                start_date: "{{ data_interval_start }}"
                end_date: "{{ data_interval_end }}"
        signals:
            include: "file://{{ TP_CONFIG_HOME }}/signals/mysignals.yaml"
            variables:
                include: "file://{{ TP_CONFIG_HOME }}/universes/{{ universe_name }}.yaml"
                universe_name: "{{ universe_name }}"
                bid_close: "prices/{{ universe_name }}.bid.close"
                ask_close: "prices/{{ universe_name }}.ask.close"
                frequency: "15min"
        portfolio:
            include: "file://{{ TP_TEMPLATES }}/portfolio_construction.yaml"
            variables:
                close: "prices/{{ universe_name }}.mid.close"
                ask_close: "prices/{{ universe_name }}.ask.close"
                bid_close: "prices/{{ universe_name }}.bid.close"
                backtest:
                price_ffill_limit: 5
                name: intraday
                model_weights:
                    "intraday_momentum.{{ universe_name }}": 1.0
                portfolio:
                aum: 10000
                multiplier: 0.5
        """
        )
    )
    (tmp_path / "configs" / "signals" / "mysignals.yaml").write_text(
        textwrap.dedent(
            """\
            "signals.intraday_momentum.myuniverse": 
                function: func
                symbols_in: []
                symbols_out: []
                depends_on: ["sample.{{ universe_name }}"]
                params: {}
            """
        )
    )
    (tmp_path / "configs" / "universes" / "myuniverse.yaml").write_text(
        textwrap.dedent(
            """\
            universe_name: im-multi-asset-3
            raw_prices_lib: prices_igtrading
            epics:
                - "CC.D.NG.UMP.IP"
                - "IX.D.SPTRD.IFS.IP"
                - "CC.D.LCO.UMP.IP"
                - "IR.D.10YEAR100.FWM2.IP"
                - "CC.D.CC.UMP.IP"
                - "CC.D.S.UMP.IP"
                - "IX.D.NASDAQ.IFS.IP"
                - "CS.D.CFPGOLD.CFP.IP"
            interval: 15min
            start_date: "2017-01-01 00:00:00+00:00"
            end_date: "{{ data_interval_end }}"
            """
        )
    )

    return tmp_path / "configs"


@pytest.fixture
def config_home_two_models(tmp_path):
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "signals").mkdir(exist_ok=True)
    (tmp_path / "configs" / "universes").mkdir(exist_ok=True)

    (tmp_path / "configs" / "balancedconfig.yaml").write_text(
        textwrap.dedent(
            """\
        prices:
            prices.bonds:
                include: "file://{{ TP_TEMPLATES }}/instruments/yfinance.yaml"
                variables:
                    include: "file://{{ TP_CONFIG_HOME }}/universes/bonds-universe.yaml"
                    start_date: "2017-01-01 00:00:00+00:00"
                    end_date: "{{ data_interval_end }}"
            prices.equities:
                include: "file://{{ TP_TEMPLATES }}/instruments/yfinance.yaml"
                variables:
                    include: "file://{{ TP_CONFIG_HOME }}/universes/equities-universe.yaml"
                    start_date: "2017-01-01 00:00:00+00:00"
                    end_date: "{{ data_interval_end }}"
            prices.fx:
                include: "file://{{ TP_TEMPLATES }}/instruments/yfinance.yaml"
                variables:
                    include: "file://{{ TP_CONFIG_HOME }}/universes/fx-universe.yaml"
                    start_date: "2017-01-01 00:00:00+00:00"
                    end_date: "{{ data_interval_end }}"

        signals:
            signals.equities:
                depends_on: ["sample.equities-universe.GBP"]
                symbols_out:
                    - "signals/equities.pct_returns"
                    - "signals/equities.vol"
                    - "signals/equities.sharpe"
                    - "signals/equities"
                symbols_in:
                    instruments: "instruments/equities-universe"
                    prices: "prices/equities-universe.mid.close.GBP"
                function: "func"
                params: {}
            signals.bonds:
                depends_on: ["sample.bonds-universe.GBP"]
                symbols_out:
                    - "signals/bonds.pct_returns"
                    - "signals/bonds.vol"
                    - "signals/bonds.sharpe"
                    - "signals/bonds"
                symbols_in:
                    instruments: "instruments/bonds-universe"
                    prices: "prices/bonds-universe.mid.close.GBP"
                function: "func"
                params: {}

        portfolio:
            include: "file://{{ TP_TEMPLATES }}/portfolio_construction.yaml"
            variables:
                backtest:
                    price_ffill_limit: 5
                name: balanced_portfolio
                bid_field: mid
                ask_field: mid
                model_weights:
                    equities: 0.6
                    bonds: 0.4
                portfolio:
                    currency: GBP
                    aum: 10000
                    multiplier: 1.0
        """
        )
    )
    (tmp_path / "configs" / "universes" / "equities-universe.yaml").write_text(
        textwrap.dedent(
            """\
            universe_name: equities-universe
            raw_prices_lib: equities_prices_lib
            tickers:
                - SPX
                - DAX
                - FTSE
            interval: 30min
            start_date: "2017-01-01 00:00:00+00:00"
            end_date: "{{ data_interval_end }}"
            currency: GBP
            fx_universe_name: fx-universe
            """
        )
    )
    (tmp_path / "configs" / "universes" / "bonds-universe.yaml").write_text(
        textwrap.dedent(
            """\
            universe_name: bonds-universe
            raw_prices_lib: bonds_prices_lib
            tickers:
                - Treasury10Y
                - Bund10Y
                - Gilt10Y
            interval: 30min
            start_date: "2017-01-01 00:00:00+00:00"
            end_date: "{{ data_interval_end }}"
            currency: GBP
            fx_universe_name: fx-universe
            """
        )
    )
    (tmp_path / "configs" / "universes" / "fx-universe.yaml").write_text(
        textwrap.dedent(
            """\
            universe_name: fx-universe
            raw_prices_lib: fx_prices_lib
            tickers:
                - EURUSD
                - GBPUSD
            interval: 30min
            start_date: "2017-01-01 00:00:00+00:00"
            end_date: "{{ data_interval_end }}"
            """
        )
    )

    return tmp_path / "configs"


def test_config(config_home):
    env = TradingoConfig.from_env(
        env={
            "TP_CONFIG_HOME": str(config_home),
            "TP_ARCTIC_URI": "mem://",
        }
    )
    env.to_env()

    out = config.read_config_template(
        config_home / "myconfig.yaml",
        variables=os.environ,
    )

    dag.DAG.from_config(out)


def test_multiple_models(config_home_two_models):
    env = TradingoConfig.from_env(
        env={
            "TP_CONFIG_HOME": str(config_home_two_models),
            "TP_ARCTIC_URI": "mem://",
        }
    )
    env.to_env()

    out = config.read_config_template(
        config_home_two_models / "balancedconfig.yaml",
        variables=os.environ,
    )

    dag.DAG.from_config(out)

    portfolio_construction_config = out["portfolio"]["portfolio.balanced_portfolio"]

    # positions from 2 models
    assert portfolio_construction_config["depends_on"] == [
        "signals.equities",
        "signals.bonds",
    ]
    assert portfolio_construction_config["params"]["model_weights"] == {
        "equities": 0.6,
        "bonds": 0.4,
    }

    # universe from 2 models
    assert portfolio_construction_config["symbols_in"]["instruments"] == [
        "instruments/equities",
        "instruments/bonds",
    ]

    # prices in reference currency (GBP)
    assert portfolio_construction_config["symbols_in"]["close"] == [
        "prices/equities.mid.close.GBP",
        "prices/bonds.mid.close.GBP",
    ]
