import os
import pathlib

import pytest

from tradingo import cli
from tradingo.config import TradingoConfig


@pytest.mark.skip
def test_config():
    env = TradingoConfig.from_env(
        env={
            "TP_CONFIG_HOME": "/home/rory/dev/tradingo-plat/config/tradingo/",
            "TP_ARCTIC_URI": "lmdb:///home/rory/dev/tradingo-plat/data/prod/tradingo.db",
        }
    )
    env.to_env()

    out = cli.read_config_template(
        pathlib.Path("/home/rory/dev/tradingo-plat/config/tradingo/yfinance.yaml"),
        variables=os.environ,
    )

    cli.DAG.from_config(out)
