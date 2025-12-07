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
                universe_name: "{{ universe_name }}"
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
