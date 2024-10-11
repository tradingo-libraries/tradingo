import logging
import importlib
import os
import pathlib
import re
from datetime import datetime
from airflow.exceptions import DuplicateTaskIdFound

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Operator

import arcticdb as adb
import pandas as pd
from sqlalchemy.orm import strategy_options
from tradingo.cli import make_airflow_dag

from tradingo.symbols import ARCTIC_URL, symbol_provider, symbol_publisher
from tradingo import signals
from tradingo.backtest import backtest
from tradingo.portfolio import (
    instrument_ivol,
    portfolio_construction,
    portfolio_optimization,
    position_from_trades,
)
from tradingo import sampling
from tradingo.utils import get_config
from airflow.models import Variable

logger = logging.getLogger(__name__)


HOME_DIR = pathlib.Path(os.environ["AIRFLOW_HOME"]) / "trading"
START_DATE = "2018-01-01 00:00:00+00:00"
ARCTIC = adb.Arctic(ARCTIC_URL)


def make_task_id(task, symbol=None) -> str:
    if symbol:
        symbol = re.sub(r"\^|/", ".", str(symbol).strip())
        return f"{task}.{symbol}"
    return task


def get_function(function):

    module, name = function.rsplit(".", maxsplit=1)

    return getattr(importlib.import_module(module), name)


def get_or_create_signal(
    name,
    universe,
    provider,
    signal,
    function,
    config,
    depends_on,
    dag: DAG,
    prices,
) -> Operator:
    signal = signal.copy()
    task_id = make_task_id("signal", ".".join((universe, provider, name)))
    try:
        signal = PythonOperator(
            task_id=task_id,
            python_callable=get_function(function),
            op_kwargs={
                "model_name": name,
                "start_date": START_DATE,
                "end_date": "{{ data_interval_end }}",
                "signal_name": name,
                "config_name": dag.dag_id,
                "provider": provider,
                "universe": universe,
                **signal,
            },
        )
        _ = prices >> signal

    except DuplicateTaskIdFound:
        signal = dag.get_task(task_id)

    for sig in depends_on:
        signal_config = config["signal_configs"][sig]
        _ = (
            get_or_create_signal(
                sig,
                universe,
                provider,
                signal_config["kwargs"],
                signal_config["function"],
                config,
                signal_config.get("depends_on", []),
                dag,
                prices,
            )
            >> signal
        )
    return signal


DAG_DEFAULT_ARGS = {
    "depends_on_past": False,
}


def get_or_create_universe(universe, provider, dag, vol_speeds, **kwargs):
    insts = _get_or_create_task(
        task_id=f"{universe}.instruments",
        callable=sampling.download_instruments,
        dag=dag,
        universe=universe,
        **kwargs,
    )
    prices = _get_or_create_task(
        task_id=f"{universe}.sample",
        callable=sampling.sample_equity,
        dag=dag,
        universe=universe,
        start_date=START_DATE,
        end_date="{{ data_interval_end }}",
        provider=provider,
        **kwargs,
    )

    ivol = _get_or_create_task(
        task_id=f"{universe}.ivol",
        callable=instrument_ivol,
        dag=dag,
        start_date=START_DATE,
        end_date="{{ data_interval_end }}",
        provider=provider,
        universe=universe,
        **kwargs,
    )
    vol = _get_or_create_task(
        speeds=vol_speeds,
        task_id=f"{universe}.vol",
        callable=signals.vol,
        dag=dag,
        start_date=START_DATE,
        end_date="{{ data_interval_end }}",
        provider=provider,
        universe=universe,
        **kwargs,
    )

    _ = insts >> prices >> ivol
    _ = prices >> vol
    return insts, prices, ivol, vol


def _get_or_create_task(*args, task_id, callable, dag, **kwargs):
    try:
        return PythonOperator(
            task_id=task_id,
            python_callable=callable,
            op_args=args,
            op_kwargs=kwargs,
        )
    except DuplicateTaskIdFound:
        return dag.get_task(task_id)


def trading_dag(
    dag_id,
    config_path=str(HOME_DIR / "config.json"),
    start_date=datetime.fromisoformat("2024-04-25 00:00:00+00:00"),
    schedule="0 0 * * 1-5",
    catchup=True,
):
    with DAG(
        dag_id=dag_id,
        start_date=start_date,
        schedule=schedule,
        catchup=catchup,
        default_args=DAG_DEFAULT_ARGS,
        max_active_runs=1,
    ) as dag:
        config = get_config(config_path)

        if "futures" in config:

            _ = PythonOperator(
                task_id="sample_futures",
                python_callable=sampling.sample_futures,
                op_kwargs=dict(
                    universe=config["futures"],
                    provider="cboe",
                    start_date="{{ data_interval_start }}",
                    end_date="{{ data_interval_end }}",
                    config_name=config["name"],
                ),
            )

        if "options" in config:

            _ = PythonOperator(
                task_id="sample_options",
                python_callable=sampling.sample_options,
                op_kwargs=dict(
                    universe=config["options"],
                    provider="cboe",
                    start_date="{{ data_interval_start }}",
                    end_date="{{ data_interval_end }}",
                    config_name=config["name"],
                ),
            )

        for name, strategy in config["portfolio"].items():
            insts, prices, ivol, vol = get_or_create_universe(
                universe=strategy["universe"],
                config_name=config["name"],
                dag=dag,
                **config["universe"][strategy["universe"]],
                vol_speeds=config["volatility"]["speeds"],
            )

            pos = PythonOperator(
                task_id=make_task_id("portfolio.raw.shares", name),
                python_callable=get_function(strategy["function"]),
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    **strategy["kwargs"],
                },
            )
            buffered_pos = PythonOperator(
                task_id=make_task_id("portfolio.raw.shares.buffered", name),
                python_callable=signals.buffered,
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    "config_name": config["name"],
                    "provider": strategy["provider"],
                    "signal": "raw.shares",
                    "library": "portfolio",
                    "model_name": name,
                    "buffer_width": strategy["buffer_width"],
                    "universe": strategy["universe"],
                },
            )
            _ = pos >> buffered_pos
            _ = buffered_pos >> PythonOperator(
                task_id=make_task_id("backtest", f"{name}.buffered"),
                python_callable=backtest,
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    "stage": "raw.shares.buffered",
                    "config_name": config["name"],
                    "provider": strategy["provider"],
                    "universe": strategy["universe"],
                },
            )

            _ = prices >> pos
            _ = vol >> pos
            _ = pos >> PythonOperator(
                task_id=make_task_id("backtest", name),
                python_callable=backtest,
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    "stage": "raw.shares",
                    "config_name": config["name"],
                    "provider": strategy["provider"],
                    "universe": strategy["universe"],
                },
            )

            # _ = pos >> PythonOperator(
            #    task_id=make_task_id("calculate_trades", name),
            #    python_callable=calculate_trades,
            #    op_kwargs={
            #        "start_date": START_DATE,
            #        "end_date": "{{ data_interval_end }}",
            #        "previous_refdate": "{{ data_interval_start }}",
            #        "name": name,
            #        "stage": "raw",
            #        "provider": PROVIDER,
            #        "config_name": config["name"],
            #    },
            # )

            for signal in strategy["signal_weights"]:
                signal_config = config["signal_configs"][signal].copy()
                _ = (
                    prices
                    >> get_or_create_signal(
                        signal,
                        strategy["universe"],
                        strategy["provider"],
                        signal_config["kwargs"],
                        function=signal_config["function"],
                        config=config,
                        depends_on=signal_config.get("depends_on", []),
                        dag=dag,
                        prices=prices,
                    )
                    >> pos
                )

    return dag


os.environ["IG_SERVICE_ACC_TYPE"] = Variable.get("IG_SERVICE_ACC_TYPE")
os.environ["IG_SERVICE_PASSWORD"] = Variable.get("IG_SERVICE_PASSWORD")
os.environ["IG_SERVICE_USERNAME"] = Variable.get("IG_SERVICE_USERNAME")
os.environ["IG_SERVICE_API_KEY"] = Variable.get("IG_SERVICE_API_KEY")


# trading_dag("etft", start_date=pd.Timestamp("2024-08-29 00:00:00+00:00"))
etft = make_airflow_dag(
    name="etft",
    config=HOME_DIR / "config.json",
    dag_start_date=pd.Timestamp("2024-10-08 00:00:00+00:00"),
    start_date=pd.Timestamp("2023-11-07 00:00:00+00"),
)

igtrading = make_airflow_dag(
    name="igtrading",
    config=HOME_DIR / "ig-trading.json",
    dag_start_date=pd.Timestamp("2024-10-10 00:00:00+00:00"),
    start_date=pd.Timestamp("2017-01-01 00:00:00+00"),
    schedule="*/15 5-22 * * MON-FRI",
)
