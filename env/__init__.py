"""Component A: the imitation environment (see env/imitation.py for the overview).

    from env import make_env, default_config
    env = make_env({"motion_path": "data/motions/cmu_16_35.npz"})   # partial dict = overrides

`make_env` accepts a full ConfigDict, or a (possibly nested, possibly partial) dict such as the
`env:` section of an experiment YAML, which is merged over `default_config()`. Unknown keys raise
an error, so a typo in a YAML file fails loudly instead of being ignored.
"""

from __future__ import annotations

from typing import Any

from ml_collections import config_dict

from env.imitation import RunImitation, default_config

__all__ = ["RunImitation", "default_config", "make_env", "config_from_dict"]


def config_from_dict(d: dict[str, Any] | None) -> config_dict.ConfigDict:
    cfg = default_config()
    if d:
        _merge(cfg, d, prefix="")
    return cfg


def _merge(cfg: config_dict.ConfigDict, d: dict, prefix: str) -> None:
    for k, v in d.items():
        if k not in cfg:
            raise KeyError(f"unknown env config key: {prefix}{k}")
        if isinstance(v, dict):
            _merge(cfg[k], v, prefix=f"{prefix}{k}.")
        else:
            cfg[k] = tuple(v) if isinstance(v, list) and k == "future_steps" else v


def make_env(config: config_dict.ConfigDict | dict[str, Any] | None = None) -> RunImitation:
    if not isinstance(config, config_dict.ConfigDict):
        config = config_from_dict(config)
    return RunImitation(config)
