"""Module for handling environment variables"""

import dataclasses
import json
import os
import pathlib
import typing
from typing import Any, Optional

import jinja2
import pandas as pd
import yaml
from jinja2 import Environment, select_autoescape

from tradingo import templates

env = Environment(autoescape=select_autoescape())


class EnvProviderError(Exception):
    """"""


def getbool_(val):
    if isinstance(val, (int, bool)):
        return bool(val)
    if val.lower() in {"true", "yes", "1"}:
        return True
    if val.lower() in {"false", "no", "0"}:
        return False
    raise ValueError(val)


def get_cls(cls):
    if cls is int:
        return int
    if cls is float:
        return float
    if cls is bool:
        return getbool_
    if cls is str:
        return str
    if cls is pd.Timestamp:
        return cls
    if cls is pathlib.Path:
        return cls
    if isinstance(cls, type) and issubclass(cls, EnvProvider):
        return cls
    if isinstance(cls(), typing.Mapping):
        return (
            cls,
            get_cls(typing.get_args(cls)[0]),
            get_cls(typing.get_args(cls)[1]),
        )
    if isinstance(cls(), typing.Iterable):
        return (
            cls,
            get_cls(typing.get_args(cls)[0]),
        )

    raise EnvProviderError(f"Unhandled type '{cls}'")


def _read_dict(cls, val):
    containercls, kcls, vcls = cls
    if isinstance(val, str):
        return containercls(json.loads(val))
    elif isinstance(val, typing.Mapping):
        return val
    else:
        raise EnvProviderError(f"Cannot read mapping '{cls}': '{val}'")


def type_shed(field: dataclasses.Field, val, prefix, env) -> Any:
    cls = get_cls(field.type)

    if isinstance(cls, tuple):
        if len(cls) == 2:
            containercls, vcls = cls
            if isinstance(val, str):
                if isinstance(vcls, tuple):
                    return containercls(
                        _read_dict((vcls[0], *typing.get_args(vcls[0])), v.strip())
                        for v in val.split(";")
                    )
                return containercls(vcls(v) for v in val.split(";"))
            return containercls(val)
        if len(cls) == 3:
            return _read_dict(cls, val)

    if isinstance(cls, type) and issubclass(cls, EnvProvider):
        if isinstance(val, dict):
            return cls(**val, app_prefix=f"{prefix}{field.name}__")
        return cls.from_env(
            app_prefix=f"{prefix}{field.name}_",
            env=env,
        )

    if callable(cls):
        return cls(val)

    raise EnvProviderError(f"Unhandled type {cls} in config")


@dataclasses.dataclass
class EnvProvider:
    app_prefix: Optional[str]

    @classmethod
    def _resolve_args(cls, env, app_prefix):
        out = {}
        for k, v in env.items():
            if not (k.lower().startswith(app_prefix)):
                continue
            k_ = k.lower().replace(app_prefix, "")
            k__ = k_.split("__")[0] if "__" in k_ else k_
            try:
                field = cls.__dataclass_fields__[k__]
            except KeyError:
                raise EnvProviderError(
                    f"Unused config field '{k__}' with value '{v}' for prefix {app_prefix}"
                )
            v_ = type_shed(
                field,
                v,
                app_prefix,
                env,
            )
            out[k__] = v_

        return out

    def to_env(self, env=None):
        env = env if isinstance(env, dict) else os.environ
        for k, v in self.__dict__.items():
            if k == "app_prefix":
                continue
            if isinstance(v, EnvProvider):
                v.to_env(env=env)
                continue
            env[f"{self.app_prefix}{k}".upper()] = str(v)
        return self

    def clean(self, env=None):
        env = env if isinstance(env, dict) else os.environ
        for k, v in self.__dict__.items():
            if k == "app_prefix":
                continue
            if isinstance(v, EnvProvider):
                v.clean(env=env)
                continue
            env.pop(f"{self.app_prefix}{k}".upper(), None)
        return self

    @classmethod
    def from_env(
        cls,
        *,
        app_prefix: str | None = None,
        env: dict[str, Any] | None = None,
    ):
        try:
            app_prefix = app_prefix or getattr(cls, "app_prefix")
        except AttributeError as ex:
            raise EnvProviderError(
                "app_prefix must be passed or set on the config object"
            ) from ex

        app_prefix = (app_prefix + "_").lower()

        env = env or dict(os.environ)

        resolved_args = cls._resolve_args(env, app_prefix)

        try:
            return cls(**resolved_args, app_prefix=app_prefix)
        except TypeError as ex:
            if "missing" in ex.args[0]:
                raise EnvProviderError(
                    f"Missing settings:{ex.args[0].split(':')[-1]} for {app_prefix}"
                ) from ex

            if "unexpected" in ex.args[0]:
                raise EnvProviderError(
                    f"Unknown settings:{ex.args[0].split('argument')[-1]}"
                ) from ex
            raise ex

    @classmethod
    def from_file(cls, variables: str | pathlib.Path):
        variables = pathlib.Path(variables)
        rendered = jinja2.Template(variables.read_text()).render(**os.environ)
        if variables.suffix == ".json":
            return cls.from_env(env=json.loads(rendered))
        if variables.suffix == ".yaml":
            return cls.from_env(env=yaml.safe_load(rendered))
        raise ValueError(variables.suffix)


@dataclasses.dataclass
class IGTradingConfig(EnvProvider):
    """IG Trading configuration"""

    password: str
    username: str
    api_key: str
    acc_type: str
    app_prefix = "IG_SERVICE"


@dataclasses.dataclass
class TradingoConfig(EnvProvider):
    """Tradingo configuration"""

    config_home: pathlib.Path
    arctic_uri: str
    templates: pathlib.Path = pathlib.Path(templates.__file__).parent
    include_instruments: bool = False
    app_prefix = "TP"


@dataclasses.dataclass
class SMTPConfig(EnvProvider):
    """SMTP server configuration"""

    server_uri: str
    port: int
    username: str
    password: str
    app_prefix = "SMTP"
