"""
Config loading utilities with YAML inheritance (_base_ support) and CLI overrides.
"""

import yaml
import copy
import argparse
from pathlib import Path
from typing import Any, Dict, Optional


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge override into base. Override values take precedence.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load a YAML config file with _base_ inheritance support.

    If the config contains a '_base_' key, it loads the base config first
    and merges the current config on top.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of overrides to apply on top.

    Returns:
        Merged config dict.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Handle _base_ inheritance
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)

    # Apply overrides
    if overrides:
        config = deep_merge(config, overrides)

    return config


def parse_override(s: str) -> tuple:
    """
    Parse a CLI override string like 'attribution.method=repsim'.

    Returns:
        (key_path_list, value)
    """
    key, value = s.split("=", 1)
    keys = key.split(".")

    # Try to parse value as int, float, bool
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.lower() == "none":
        value = None
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string

    return keys, value


def overrides_to_dict(override_strs: list) -> Dict:
    """
    Convert list of CLI override strings to a nested dict.

    Example:
        ['attribution.method=repsim', 'evaluation.n_seeds=3']
        -> {'attribution': {'method': 'repsim'}, 'evaluation': {'n_seeds': 3}}
    """
    result = {}
    for s in override_strs:
        keys, value = parse_override(s)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common CLI arguments to an argument parser."""
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal steps to verify pipeline")
    parser.add_argument("--max-steps", type=int, default=2, help="Max steps for dry run")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key.path=value")
    return parser


def get_config_from_args(args) -> Dict[str, Any]:
    """Load config from parsed CLI args with overrides."""
    overrides = overrides_to_dict(args.override) if args.override else {}
    if args.seed is not None:
        overrides.setdefault("reproducibility", {})["seed"] = args.seed
    config = load_config(args.config, overrides)
    return config


def expand_path(path_str: str) -> str:
    """Expand ~ and resolve path."""
    return str(Path(path_str).expanduser().resolve())
