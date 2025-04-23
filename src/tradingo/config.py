"""Tradingo config."""

import json
import pathlib
from typing import Any

import jinja2
import yaml


class ConfigLoadError(Exception):
    """raised when loading config incorrectly."""


def read_config_template(filepath: pathlib.Path, variables) -> dict[str, Any]:
    """read a config template and populate it with values"""
    filepath = pathlib.Path(filepath)

    renderedtext = jinja2.Template(filepath.read_text(encoding="utf-8")).render(
        **variables
    )

    try:
        if filepath.suffix == ".json":
            return process_includes(json.loads(renderedtext), variables)

        if filepath.suffix == ".yaml":
            return process_includes(yaml.safe_load(renderedtext), variables)

    except jinja2.TemplateSyntaxError as ex:
        raise ConfigLoadError(f"Error reading config template: {filepath}") from ex

    raise ValueError(f"Unhandled file type: '{filepath.suffix}'")


def process_includes(config: dict[str, Any], variables) -> dict[str, Any]:
    """bake nested configs via 'include' pattern."""

    out = {}

    for key, value in config.items():
        if isinstance(value, dict) and "include" in value:
            value = process_includes(value, variables)

            protocol, path = value["include"].split("://")

            if protocol == "file":
                incvalue = read_config_template(
                    pathlib.Path(path),
                    variables={
                        **value.get("variables", {}),
                        **variables,
                    },
                )

            else:
                raise ValueError(
                    f"Unsupported protocol: '{protocol}' at '{key}' for '{path}'"
                )
            value = value.copy()
            value.update(incvalue)
            value.pop("include", None)
            value.pop("variables", None)

        elif isinstance(value, dict):
            value = process_includes(value, variables)

        out[key] = value

    return out
