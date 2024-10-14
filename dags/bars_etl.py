import logging
import os
import pathlib

from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
from tradingo.cli import (
    build_graph,
    resolve_config,
    Task,
)

from tradingo.symbols import ARCTIC_URL
from airflow.models import Variable

logger = logging.getLogger(__name__)


HOME_DIR = pathlib.Path(os.environ["AIRFLOW_HOME"]) / "trading"
START_DATE = "2018-01-01 00:00:00+00:00"


def to_airflow_dag(graph: dict[str, Task], **kwargs) -> DAG:

    with DAG(**kwargs) as dag:

        tasks = {
            name: PythonOperator(
                task_id=name,
                python_callable=task.function,
                op_args=task.task_args,
                op_kwargs=task.task_kwargs,
            )
            for name, task in graph.items()
        }

        for task, operator in zip(graph.values(), tasks.values()):

            for task_id in task.dependency_names:

                _ = tasks[task_id] >> operator

    return dag


def make_airflow_dag(
    name,
    start_date,
    dag_start_date,
    config,
    **kwargs,
):
    graph = build_graph(
        resolve_config(config),
        start_date=start_date,
        end_date="{{ data_interval_end }}",
        sample_start_date="{{ data_interval_start }}",
        snapshot_template=f"{name}_{{{{ run_id }}}}_{{{{ task_instance.task_id }}}}",
    )

    return to_airflow_dag(
        graph,
        dag_id=name,
        start_date=dag_start_date,
        **kwargs,
    )


os.environ["IG_SERVICE_ACC_TYPE"] = Variable.get("IG_SERVICE_ACC_TYPE")
os.environ["IG_SERVICE_PASSWORD"] = Variable.get("IG_SERVICE_PASSWORD")
os.environ["IG_SERVICE_USERNAME"] = Variable.get("IG_SERVICE_USERNAME")
os.environ["IG_SERVICE_API_KEY"] = Variable.get("IG_SERVICE_API_KEY")


etft = make_airflow_dag(
    name="etft",
    config=HOME_DIR / "config.json",
    dag_start_date=pd.Timestamp("2024-10-08 00:00:00+00:00"),
    start_date=pd.Timestamp("2023-11-07 00:00:00+00"),
)

igtrading = make_airflow_dag(
    name="igtrading",
    config=HOME_DIR / "ig-trading.json",
    dag_start_date=pd.Timestamp("2024-10-13 00:00:00+00:00"),
    start_date=pd.Timestamp("2017-01-01 00:00:00+00"),
    schedule="*/15 5-21 * * MON-FRI",
)
