import pathlib
from tradingo import cli


def test_config():

    out = cli.read_config_template(
        pathlib.Path("/home/rory/dev/tradingo-plat/config/tradingo/root.yaml"),
        variables={"TP_CONFIG_HOME": "/home/rory/dev/tradingo-plat/config/tradingo/"},
    )
