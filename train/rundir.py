"""Experiment configs, run directories and reproducibility metadata (docs/interfaces.md, section 3).

Every run records exactly what produced it (config, git commit, motion file hash, library versions),
so any result in the README can be traced back and rerun.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Load an experiment YAML. A config may say `base: other.yaml` to inherit and override."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def make_run_dir(cfg: dict, runs_root: str | Path = "runs") -> Path:
    name = cfg.get("name", "exp")
    run = Path(runs_root) / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_meta(run, cfg)
    return run


def write_meta(run: Path, cfg: dict) -> None:
    meta = {
        "git": git_info(),
        "seed": cfg.get("seed"),
        "versions": {p: _version(p) for p in
                     ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground")},
        "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    motion = cfg.get("env", {}).get("motion_path")
    if motion and (ROOT / motion).exists():
        meta["motion_sha256"] = file_sha256(ROOT / motion)
    try:
        import jax
        meta["device"] = str(jax.devices()[0])
        meta["backend"] = jax.default_backend()
    except Exception:
        pass
    (run / "meta.json").write_text(json.dumps(meta, indent=2))


def latest_checkpoint(run: str | Path) -> Path | None:
    ck = Path(run) / "checkpoints"
    steps = [p for p in ck.iterdir() if p.is_dir() and p.name.isdigit()] if ck.exists() else []
    return max(steps, key=lambda p: int(p.name)) if steps else None


def log_metrics(run: Path, row: dict) -> None:
    with open(run / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
