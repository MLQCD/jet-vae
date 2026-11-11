"""YAML config support for the command-line scripts in this directory.

``--config file.yaml`` is expanded into ordinary command-line flags placed *before* the real command
line, so explicit flags still override the file, and argparse still checks types and choices.

YAML keys are flag names with dashes or underscores (``latent-dim`` or ``latent_dim``). A list becomes
a comma-separated value; ``true`` switches a flag on and ``false`` / ``null`` leave it off.
"""

import argparse

import yaml


def add_config_argument(parser):
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file of flag -> value, applied before the command-line flags "
                             "(command-line flags override it).")


def config_to_argv(config_path, parser):
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        parser.error(f"--config {config_path} must contain a YAML mapping")
    tokens, unknown = [], []
    for key, value in raw.items():
        flag = "--" + str(key).replace("_", "-")
        if flag == "--config" or flag not in parser._option_string_actions:
            unknown.append(str(key))
            continue
        if value is None or value is False:
            continue
        if value is True:
            tokens.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        tokens += [flag, str(value)]
    if unknown:
        parser.error(f"Unknown key(s) in --config {config_path}: {sorted(unknown)}")
    return tokens


def parse_args_with_config(parser, argv=None):
    """Like ``parser.parse_args(argv)`` but honours ``--config``."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        argv = config_to_argv(known.config, parser) + argv
    return parser.parse_args(argv)
