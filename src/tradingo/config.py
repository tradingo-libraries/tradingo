import dataclasses
import pathlib

from . import templates
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
    config_home: pathlib.Path
    arctic_uri: str
    templates: pathlib.Path = pathlib.Path(templates.__file__).parent
    include_instruments: bool = False
    app_prefix = "TP"
