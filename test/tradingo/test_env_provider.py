import dataclasses
import json
import os
import pathlib
from unittest import mock

import pytest

from tradingo.settings import EnvProvider, EnvProviderError


def test_env_provider(tmp_path):
    @dataclasses.dataclass
    class Settings1(EnvProvider):
        parameter: int

    env = {"TRADINGO_PARAMETER": 1}
    prefix = "TRADINGO"

    settings = Settings1.from_env(app_prefix=prefix, env=env)
    assert settings.parameter == 1

    env = {"TRADINGO_PARAMETER": "1"}
    with mock.patch.dict(os.environ, env):
        s = Settings1.from_env(app_prefix=prefix, env=env)
    assert s.parameter == 1

    with pytest.raises(EnvProviderError):
        s = Settings1.from_env(env=env)

    @dataclasses.dataclass
    class Settings2(EnvProvider):
        parameter: bool
        app_prefix = "TRADINGO"

    env = {"TRADINGO_PARAMETER": "true"}
    s = Settings2.from_env(env=env)
    assert s.parameter

    env = {"TRADINGO_PARAMETER": "false"}
    s = Settings2.from_env(env=env)
    assert not s.parameter

    @dataclasses.dataclass
    class Settings3(EnvProvider):
        param1: int
        param2: float
        param3: bool
        param4: str
        app_prefix = "APP"

    env = {"TRADINGO_PARAMETER": "1"}

    with pytest.raises(
        EnvProviderError,
        match="Missing settings: 'param1', 'param2', 'param3', and 'param4'",
    ):
        s = Settings3.from_env(env=env)

    s3 = Settings3.from_env(
        env={
            "app_param1": 1,
            "app_param2": 1.5,
            "app_param3": False,
            "app_param4": "value",
        }
    )

    assert s3.param1 == 1
    assert s3.param2 == 1.5
    assert not s3.param3
    assert s3.param4 == "value"

    with pytest.raises(
        EnvProviderError,
        match="Unused config field 'unknown' with value 'value' for prefix app_",
    ):
        s3 = Settings3.from_env(
            env={
                "app_param1": 1,
                "app_param2": 1.5,
                "app_param3": False,
                "app_param4": "value",
                "app_unknown": "value",
            }
        )

    @dataclasses.dataclass
    class Settings4(EnvProvider):
        param1: Settings3
        param2: str
        app_prefix = "APP"

    env_flat = {
        "app_param1__param1": 1,
        "app_param1__param2": 1.5,
        "app_param1__param3": False,
        "app_param1__param4": "value",
        "app_param2": "value",
    }

    s4 = Settings4.from_env(env=env_flat)

    assert s4.param1 == Settings3(
        param1=1,
        param2=1.5,
        param3=False,
        param4="value",
        app_prefix="app_param1__",
    )

    env = {
        "app_param1": {
            "param1": 1,
            "param2": 1.5,
            "param3": False,
            "param4": "value",
        },
        "app_param2": "value",
    }

    s4 = Settings4.from_env(env=env)

    assert s4.param1 == Settings3(
        param1=1,
        param2=1.5,
        param3=False,
        param4="value",
        app_prefix="app_param1__",
    )

    vars = pathlib.Path(tmp_path) / "variables.json"
    vars.write_text(json.dumps(env_flat))
    s4_file = Settings4.from_file(vars)

    assert s4_file == s4

    vars.write_text(json.dumps(env))
    s4_file = Settings4.from_file(vars)
    assert s4_file == s4

    @dataclasses.dataclass
    class Settings5(EnvProvider):
        param1: list[str]
        param2: tuple[str]
        param3: list[dict[str, str]]
        param4: dict[str, str]
        app_prefix = "APP"

    env = {
        "app_param1": ["val", "val2"],
        "app_param2": ("val",),
        "app_param3": [{"key": "val"}],
        "app_param4": {"key": "val"},
    }

    vars.write_text(json.dumps(env))
    s5_file = Settings5.from_file(vars)

    env = {
        "app_param1": "val;val2",
        "app_param2": "val",
        "app_param3": '{"key": "val"}',
        "app_param4": '{"key": "val"}',
    }
    s5_env = Settings5.from_env(env=env)
    assert s5_file == s5_env
    assert s5_env == Settings5(
        param1=["val", "val2"],
        param2=("val",),
        param3=[{"key": "val"}],
        param4={"key": "val"},
        app_prefix="app_",
    )

    s5 = Settings5.from_env(env=env)
    assert s5 == s5_env

    env = {}
    s3.to_env(env)
    assert env == {
        "APP_PARAM1": "1",
        "APP_PARAM2": "1.5",
        "APP_PARAM3": "False",
        "APP_PARAM4": "value",
    }

    env = {}
    s4.to_env(env)
    assert env == {
        "APP_PARAM1__PARAM1": "1",
        "APP_PARAM1__PARAM2": "1.5",
        "APP_PARAM1__PARAM3": "False",
        "APP_PARAM1__PARAM4": "value",
        "APP_PARAM2": "value",
    }
