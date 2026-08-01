# Getting started

This guide covers installation, the main entry point, common workflows, and development
checks. For the game itself, continue to [environment and physics](environment.md); for the
learning setup, see [training](training.md).

## Requirements

- Python 3.13 or newer;
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management;
- Git LFS for the included reference checkpoints (`*.pt`) — not needed to train from
  scratch;
- `ffmpeg` only when generating MP4/GIF replay assets;
- a CUDA-capable GPU for practical training and large evaluations.

The project selects CUDA automatically when PyTorch reports it as available and otherwise
falls back to CPU. CPU is useful for tests and small inspections, but the production
profiles are sized for GPU execution.

## Install

With Git LFS installed before cloning, the checkpoints materialize automatically at
clone time. In an existing clone:

```bash
git lfs install
git lfs pull
uv sync
```

All Python commands use the single repository entry point:

```bash
uv run main.py --help
```

`uv run` keeps the environment aligned with the project's lockfile. The available modes
and their arguments are defined in [`main.py`](../main.py).

## Verify the checkout

```bash
uv run pytest -q
uv run ruff check .
```

The suite passes on CPU; a handful of hardware-specific tests skip when no CUDA device
is visible. There is no CI workflow yet, so these local commands are the verification
path.

## Watch or play

```bash
# Human team 0 vs the newest available checkpoint
uv run main.py --mode watch

# Learned policy vs scripted controller
uv run main.py --mode watch --team0 latest --team1 scripted

# Two views of the same learned weights in self-play
uv run main.py --mode watch --team0 latest --team1 latest
```

Human controls are WASD for flight, Shift for sharp turns, and Space to shoot. Agent specs
accepted by `--team0` and `--team1` include `null` (human in watch mode), `random`,
`scripted`, `latest`, a checkpoint path, and the named scripted controllers listed by
`main.py --help`.

## Train

Start with the smoke path after changing code or configuration:

```bash
uv run main.py --mode rl --smoke
```

Smoke mode uses four environments, disables W&B and compilation, and stops after a few
updates. It is a crash test, not a meaningful experiment.

Production entry points:

```bash
# Recurrent PPO from scratch
uv run main.py --mode rl

# Full-size run without W&B
uv run main.py --mode rl --no-wandb

# Warm-start policy/scaler weights; optimizer starts fresh
uv run main.py --mode rl \
  --pretrain_from checkpoints/<run>/best_training.pt

# Restore a complete training state
uv run main.py --mode rl --resume checkpoints/<run>/step_<N>.pt

# Behavior cloning, or cloning followed by RL in one process
uv run main.py --mode bc
uv run main.py --mode bc_warmstart
```

Hyperparameters live in [`runs/`](../runs/). The main combat profile is
[`runs/rl.py`](../runs/rl.py), shared model/physics/reward definitions are in
[`runs/shared.py`](../runs/shared.py), and configuration types are frozen dataclasses in
[`src/boost_and_broadside/config/`](../src/boost_and_broadside/config/).

## Evaluate

```bash
# Direct parallel matchup
uv run main.py --mode collect_stats \
  --team0 latest --team1 scripted --matchups 4v4 8v11

# Post-hoc Elo calibration for a completed run
uv run main.py --mode elo_calibrate \
  --run resilient-resonance-682

# Refit the stored calibration matrices without replaying (CPU-only)
uv run main.py --mode elo_calibrate \
  --run resilient-resonance-682 --refit

# Rate frozen checkpoints across symmetric fleet sizes (resumable)
uv run main.py --mode elo_scale \
  --run resilient-resonance-682 --team-counts 1,2,4,8,16,32,64

# Build the random-to-scripted reference ladder used to condition scale ratings
uv run main.py --mode semi_random \
  --run resilient-resonance-682 --team-counts 1,2,4,8,16,32,64

# Search the scripted-team crossover for selected learned-team sizes
uv run main.py --mode crossover \
  --run resilient-resonance-682 --trained-counts 4,8,16,32,64 --eval-envs 256
```

Calibration writes to `checkpoints/<run>/elo_calibrated.json` and `elo_calibration/`.
Scale calibration writes its resumable match matrices to `checkpoints/<run>/elo_scale.json`;
the reference ladder writes `checkpoints/<run>/semi_random_tournament.json`; crossover
writes `docs/crossover/crossover.json`. These evaluations can require substantial GPU
time, and the reference-run artifacts are already included. Methodology and
interpretation are in [evaluation and results](evaluation.md).

## Capture replays

```bash
uv run main.py --mode capture \
  --run resilient-resonance-682 \
  --scenarios vs_scripted \
  --sizes 8v11 \
  --seeds 3 \
  --gif
```

Capture mode uses the run's final `step_*.pt` checkpoint, writes seeded MP4 files, and can
also emit downscaled GIFs. Files go to `gameplay_clips/` by default; the curated subset
lives under `docs/results/replays/`. See the [replay guide](replays.md).

## Checkpoint and result artifacts

Full `step_<N>.pt` checkpoints contain the policy, optimizer, scalers, running-average
state, ratings, counters, and serialized configuration needed by `--resume`. Training
also writes scheduled average/best snapshots and unpruned ladder checkpoints; the
[checkpoint implementation](../src/boost_and_broadside/train/rl/checkpoint.py) defines
the current filenames. (The included reference-run directory retains files from an
earlier naming convention, such as `recent_avg.pt`.)

## Development notes

- Follow [STYLE_GUIDE.md](../STYLE_GUIDE.md).
- Use `uv run` for project commands.
- Keep physical behavior covered by real tensor tests rather than mocks; environment tests
  live in [`tests/env/`](../tests/env/).
- Memory measurements and the host-backed rollout design are documented in
  [memory optimization](engineering/memory-optimization.md).

The project is [MIT-licensed](../LICENSE). There is no contribution guide or CI workflow
yet.
