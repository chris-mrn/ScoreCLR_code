from typing import Iterable

import yaml


class Config(dict):
    """Nested dict with attribute access."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def parse_value(text: str):
    """YAML-parses a CLI value; '1e-4' (a string in YAML 1.1) becomes a float."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Applies `a.b.c=value` overrides in place."""
    for item in overrides:
        key, sep, value = item.partition("=")
        assert sep, f"override must be key=value, got {item!r}"
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = parse_value(value)
    return cfg


def load_config(path: str, overrides: Iterable[str] = ()) -> Config:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return Config(apply_overrides(cfg, overrides))
