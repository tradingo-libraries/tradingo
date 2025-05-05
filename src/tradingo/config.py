"""Tradingo config."""

import json
import pathlib
from typing import Any, Optional

import jinja2
import yaml


class ConfigLoadError(Exception):
    """raised when loading config incorrectly."""


def read_config_template(
    filepath: pathlib.Path,
    variables: dict[str, Any],
    path_so_far: Optional[list[str]] = None,
) -> dict[str, Any]:
    """read a config template and populate it with values"""
    filepath = pathlib.Path(filepath)

    path_so_far = path_so_far or []

    # TODO:  undefined=jinja2.StrictUndefined
    renderedtext = jinja2.Template(filepath.read_text(encoding="utf-8")).render(
        **variables
    )

    try:
        if filepath.suffix == ".json":
            return process_includes(
                json.loads(renderedtext),
                variables,
                path_so_far=path_so_far,
            )

        if filepath.suffix == ".yaml":
            return process_includes(
                yaml.safe_load(renderedtext),
                variables,
                path_so_far=path_so_far,
            )

    except jinja2.TemplateSyntaxError as ex:
        raise ConfigLoadError(f"Error reading config template: {filepath}") from ex
    except FileNotFoundError as ex:
        raise ConfigLoadError(
            f"Error rendering include: {filepath=} {path_so_far=}"
        ) from ex

    raise ValueError(f"Unhandled file type: '{filepath.suffix}'")


def process_includes(
    config: dict[str, Any],
    variables,
    path_so_far: Optional[list[str]] = None,
) -> dict[str, Any]:
    """bake nested configs via 'include' pattern."""

    out = {}

    path_so_far = path_so_far or []
    for key, value in config.items():
        path_so_far.append(key)
        if isinstance(value, dict) and "include" in value:
            value = process_includes(value, variables, path_so_far)

            protocol, path = value["include"].split("://")

            if protocol == "file":
                try:
                    incvalue = read_config_template(
                        pathlib.Path(path),
                        variables={
                            **value.get("variables", {}),
                            **variables,
                        },
                        path_so_far=path_so_far,
                    )
                except ConfigLoadError as ex:
                    raise ConfigLoadError(
                        f"Error processing include at '{value}' {path_so_far=}"
                    ) from ex

            else:
                raise ValueError(
                    f"Unsupported protocol: '{protocol}' at '{key}' for '{path}'"
                )
            value = value.copy()
            value.update(incvalue)
            value.pop("include", None)
            value.pop("variables", None)

        elif isinstance(value, dict):
            value = process_includes(value, variables, path_so_far)

        out[key] = value

    return out
