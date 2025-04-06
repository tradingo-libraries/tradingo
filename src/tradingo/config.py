import dataclasses
from typing import Any
import pathlib
import pandas as pd

from .env_provider import EnvProvider


@dataclasses.dataclass
class IGTradingConfig(EnvProvider):
    password: str
    username: str
    api_key: str
    acc_type: str
    app_prefix = "IG_SERVICE"


@dataclasses.dataclass
class TradingoConfig(EnvProvider):
    include_instruments: bool
    dag_start_date: pd.Timestamp
    start_date: pd.Timestamp
    config_home: pathlib.Path
    arctic_uri: str
    graph: dict[str, Any]
    app_prefix = "TP"
