# Run Like Me

A physics-simulated humanoid that learns to run with **my own stride**, taken from a phone video.
It is a reinforcement-learning policy in MuJoCo (MJX on GPU), trained DeepMimic-style to track a
reference motion. It is not animation playback: it has to balance, push off and recover from shoves.

> Status: **work in progress**. Phase 0 (foundations) is done; nothing has been trained yet.
> Results below are filled in only from logged runs.

## Pipeline
```
phone video ─GVHMR─► SMPL poses ─┐
CMU mocap (AMC) ─────────────────┴─► retarget (motion/) ─► reference .npz ─► imitation env (env/) ─► PPO (train/) ─► eval + videos (eval/)
```

## Repo map
| dir | what |
|---|---|
| `env/` | MJX imitation environment and DeepMimic reward; `assets/` holds the humanoid model |
| `motion/` | reference format, mocap loading, retargeting, kinematic playback |
| `video/` | video → SMPL (GVHMR) → cleaned stride |
| `train/` | Brax PPO launcher, smoke test, our own PPO (`train/ppo/`) |
| `eval/` | tracking metrics, push tests, renderers, plots |
| `configs/` | one YAML per experiment |
| `notebooks/` | Kaggle launch notebooks |
| `docs/` | `interfaces.md` (contracts), `decisions.md` (why), `experiment_log.md` (what broke) |

## Setup
Local (macOS/Linux, CPU): `uv sync`, then `uv run pytest`.
GPU (Kaggle T4): open `notebooks/00_smoke.ipynb`; it runs `scripts/kaggle_setup.sh` (`uv sync --extra cuda`).

The humanoid model is generated from dm_control's CMU humanoid: `uv run python scripts/build_humanoid.py`.

## Built vs. reused
Reused: MuJoCo/MJX, MuJoCo Playground, Brax PPO (pinned to a git commit; see `pyproject.toml`), dm_control CMU
humanoid, CMU mocap, GVHMR, mink.
Built here: imitation env and reward, retargeting, video cleanup, evaluation, push tests, renderers, own PPO.

## Licenses and data
- Code in this repo: see `LICENSE` (to be added).
- `env/assets/dm_control/`: dm_control files, Apache-2.0.
- CMU mocap: "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with
  funding from NSF EIA-0196217."
- SMPL/SMPL-X body models and GVHMR checkpoints are **not** included and must not be committed; download them
  yourself under their own (non-commercial) licenses. GVHMR is licensed for non-commercial/educational use, so this
  project is too.

## Results
_None yet._
