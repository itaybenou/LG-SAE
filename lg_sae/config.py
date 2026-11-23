"""
Configuration from a YAML file plus command-line overrides.

Every script declares its options as a list of ``Option``. Values are read
from ``--config <file.yaml>`` (optional) and then overridden by
``--<name> <value>`` flags. There are no defaults in code: every option must
be given in the YAML file or on the command line. Options marked
``nullable`` accept ``null`` (YAML) / ``none`` (CLI).
"""

import argparse
import json
from types import SimpleNamespace
from typing import NamedTuple, Optional, Sequence

import yaml

_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}
_NULL = {"none", "null"}


class Option(NamedTuple):
    name: str
    type: type
    help: str
    nullable: bool = False
    choices: Optional[Sequence] = None


class ConfigError(ValueError):
    pass


def _cast(option, value):
    if value is None or (isinstance(value, str) and value.strip().lower() in _NULL):
        if option.nullable:
            return None
        raise ConfigError(f"'{option.name}' must be set (null is not allowed)")
    if option.type is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"'{option.name}' expects a boolean, got {value!r}")
    if option.type is int and isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"'{option.name}' expects an integer, got {value!r}")
    try:
        cast = option.type(value)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"'{option.name}' expects {option.type.__name__}, got {value!r}") from err
    if option.choices is not None and cast not in option.choices:
        raise ConfigError(f"'{option.name}' must be one of {list(option.choices)}, got {cast!r}")
    return cast


def load_config(options, description, argv=None):
    """Parse ``--config`` + CLI overrides into a namespace with exactly ``options``."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with option values (CLI flags override it).")
    for opt in options:
        extra = f" [choices: {', '.join(map(str, opt.choices))}]" if opt.choices else ""
        extra += " [nullable]" if opt.nullable else ""
        parser.add_argument(f"--{opt.name}", type=str, default=None, help=opt.help + extra)
    args = parser.parse_args(argv)

    values = {}
    if args.config is not None:
        with open(args.config) as f:
            values = yaml.safe_load(f) or {}
        known = {opt.name for opt in options}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigError(f"Unknown keys in {args.config}: {unknown}")
    for opt in options:
        cli_value = getattr(args, opt.name)
        if cli_value is not None:
            values[opt.name] = cli_value

    missing = [opt.name for opt in options if opt.name not in values]
    if missing:
        raise ConfigError(f"Missing options (set them in --config or via --<name>): {missing}")
    return SimpleNamespace(**{opt.name: _cast(opt, values[opt.name]) for opt in options})


def config_to_json(cfg):
    return json.dumps(vars(cfg), indent=2)
