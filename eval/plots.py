"""Plots for training runs and evaluations (static PNGs via matplotlib, Agg backend).

  training_curves(run_dir)   reward, episode length and MPJPE vs training step, from
                             runs/<run>/metrics.jsonl. Three separate panels with their own
                             y-axes (no dual axes): they have different units.
  push_curve({label: push.json})   survival share vs push force, one line per policy, shaded
                             band = 95% Wilson interval. Levels where no push was valid (the policy
                             fell before the push every time) are left out, because there's
                             nothing to measure there.
  episode_curves({label: metrics.json})   share of episodes still alive vs episode time, and
                             the median MPJPE of the alive episodes vs time.

How to read them (also in docs/notes_eval.md):
  * reward: Brax's mean *eval episode return* = sum of the per-step reward (each in [0, 1]) over
    the episode. It rises both when tracking improves *and* when episodes get longer, so read it
    together with ep_len. Max = episode_length (500 by default).
  * ep_len: mean eval episode length in policy steps. At the training cap (500) the policy no
    longer falls within 10 s.
  * MPJPE: per-step mean body position error (m), length-weighted over eval episodes. It should
    fall as training goes on. Early on it can *rise* while ep_len rises, because longer episodes
    include later, harder-to-track moments.
  * push curve: a curve that stays near 1 and then drops tells you the push strength at which
    the policy starts failing. Read the x position where it crosses 0.5 (F50). Overlapping
    bands mean the difference between two policies isn't reliable at this sample size.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots (fixed order) from the dataviz reference palette, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _dedupe(rows: list[dict]) -> list[dict]:
    """Sort by step, keeping the *last* logged row per step. A resumed run logs the resume step
    twice (end of the first call and the eval at the start of the resumed call); without this
    the curve zig-zags back in time."""
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[k] for k in sorted(by_step)]


def training_curves(run_dirs: dict[str, str | Path] | str | Path, out_path: str | Path) -> Path | None:
    """Reward / ep_len / MPJPE vs step for one or more runs (label -> run dir)."""
    if not isinstance(run_dirs, dict):
        run_dirs = {Path(run_dirs).name: run_dirs}
    rows = {k: _dedupe([r for r in read_jsonl(Path(v) / "metrics.jsonl") if "reward" in r])
            for k, v in run_dirs.items() if (Path(v) / "metrics.jsonl").exists()}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [("reward", "Eval episode return", "sum of per-step reward"),
             ("ep_len", "Eval episode length", "policy steps"),
             ("mpjpe", "Tracking error (MPJPE)", "metres")]
    for ax, (key, title, yl) in zip(axes, specs):
        for i, (label, rr) in enumerate(rows.items()):
            xs = [r["step"] for r in rr if key in r]
            ys = [r[key] for r in rr if key in r]
            ax.plot(xs, ys, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=4, label=label)
        _style(ax, title, "environment steps", yl)
    if len(rows) > 1:
        axes[0].legend(frameon=False, fontsize=8)
    else:
        fig.suptitle(next(iter(rows)), x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def push_curve(push_jsons: dict[str, str | Path | dict], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    notes = []
    dur = None
    for i, (label, src) in enumerate(push_jsons.items()):
        res = src if isinstance(src, dict) else json.loads(Path(src).read_text())
        dur = res["config"]["duration_s"]
        lv = [x for x in res["levels"] if x["share"] is not None]
        skipped = [x["force_N"] for x in res["levels"] if x["share"] is None]
        if skipped:
            notes.append(f"{label}: no valid pushes at {len(skipped)} of {len(res['levels'])} levels "
                         f"(fell before the push)")
        if not lv:
            continue
        c = SERIES[i % len(SERIES)]
        f = [x["force_N"] for x in lv]
        ax.fill_between(f, [x["ci95"][0] for x in lv], [x["ci95"][1] for x in lv], color=c,
                        alpha=0.15, lw=0)
        ax.plot(f, [x["share"] for x in lv], color=c, lw=2, marker="o", ms=5,
                label=f"{label} (n={res['config']['n_per_level']}/level)")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    _style(ax, "Push survival", f"push force (N), applied for {dur} s  [impulse = force x {dur} s]",
           "share of pushes survived")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    if notes:
        ax.text(0.99, 0.98, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=INK2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def episode_curves(metrics_jsons: dict[str, str | Path | dict], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (label, src) in enumerate(metrics_jsons.items()):
        res = src if isinstance(src, dict) else json.loads(Path(src).read_text())
        ot = res["over_time"]
        c = SERIES[i % len(SERIES)]
        axes[0].step(ot["t_s"], ot["alive_share"], where="post", color=c, lw=2, label=label)
        pts = [(t, y) for t, y in zip(ot["t_s"], ot["mpjpe_median"]) if y is not None]
        if pts:
            axes[1].plot(*zip(*pts), color=c, lw=2, marker="o", ms=3, label=label)
    axes[0].set_ylim(-0.03, 1.03)
    _style(axes[0], "Episodes still running", "episode time (s)", "share alive")
    _style(axes[1], "Tracking error of running episodes", "episode time (s)", "median MPJPE (m)")
    if len(metrics_jsons) > 1:
        axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
