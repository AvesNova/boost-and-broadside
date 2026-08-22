# Getting started

This guide covers installation, the installed CLI, common workflows, and development
checks. For the game itself, continue to [environment and physics](environment.md); for the
learning setup, see [training](training.md).

## Requirements

- Python 3.13 or newer;
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management;
- Git LFS for historical reference checkpoints (`*.pt`), not needed to train from
  scratch. These weights predate the refractive-field observation schema and are retained
  as result artifacts, not as loadable weights for the current encoder;
- `ffmpeg` only when generating MP4/GIF replay assets;
- a CUDA-capable GPU for practical training and large evaluations.

The project selects CUDA automatically when PyTorch reports it as available and otherwise
falls back to CPU. CPU is useful for tests and small inspections, but the production
profiles are sized for GPU execution.

## Install

With Git LFS installed before cloning, the historical checkpoints materialize
automatically at clone time. In an existing clone:

```bash
git lfs install
git lfs pull
uv sync
```

The project installs one `bnb` executable:

```bash
uv run bnb --help
```

`uv run` keeps the environment aligned with the project's lockfile. The available modes
and their command-owned arguments are defined by
[`boost_and_broadside.cli`](../src/boost_and_broadside/cli.py). Calling `bnb` without a
subcommand prints help and performs no simulation or training.

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
# Fixed play mode: one player ship vs one null ship, with four fields
uv run bnb play

# Human team 0 vs a newly trained current-schema checkpoint
uv run bnb watch --team0 null --team1 checkpoints/<run>/<checkpoint>.pt

# Learned policy vs scripted controller
uv run bnb watch --team0 checkpoints/<run>/<checkpoint>.pt --team1 scripted

# Two views of the same learned weights in self-play
uv run bnb watch \
  --team0 checkpoints/<run>/<checkpoint>.pt \
  --team1 checkpoints/<run>/<checkpoint>.pt
```

Play mode has no match timer and starts a new match as soon as either ship dies. The
`Unlimited HP/PW` button in the upper-right corner toggles full health and power for both
ships. Human controls are WASD for flight, Shift for sharp turns, and Space to shoot. Agent specs
accepted by `--team0` and `--team1` include `null` (human in watch mode), `random`,
`scripted`, an explicit checkpoint path, and the named scripted controllers listed by
`bnb watch --help`. A checkpoint is always named explicitly.

## Train

Resolve a launch before constructing the trainer:

```bash
uv run bnb train --profile rl --print-config
```

The printed document includes the complete configuration, its semantic and launch
fingerprints, and the source of every resolved value. `--num-envs` and
`--microbatch-tokens` are explicit launch overrides and are validated before printing.

Production entry points:

```bash
# Recurrent PPO from scratch
uv run bnb train --profile rl

# Full-size run without W&B
uv run bnb train --profile rl --no-wandb

# Warm-start policy/scaler weights; optimizer starts fresh
uv run bnb train --profile rl \
  --pretrain-from checkpoints/<run>/best_training.pt

# Restore a complete training state
uv run bnb train --profile rl --resume checkpoints/<run>/step_<N>.pt

# Restore the most recently written run of this profile
uv run bnb train --profile rl --resume-last

# Behavior cloning, then an explicit RL warm-start when desired
uv run bnb train --profile bc
uv run bnb train --profile rl \
  --pretrain-from checkpoints/<bc-run>/best_training.pt
```

`--resume`, `--resume-last`, and `--pretrain-from` are mutually exclusive. `--resume`
always takes a value: either an explicit `.pt` path or an exact run name whose greatest
numeric `step_*.pt` checkpoint should be selected. `--resume-last` takes none, and
selects the most recently written run recorded as this profile; runs from before
`run.json` existed have no recorded profile and are never selected that way.

`bnb runs` lists the ten most recently written runs with their profile, status, progress
and newest resumable checkpoint. `--limit`, `--all`, `--profile` and `--resumable` narrow
it:

```bash
uv run bnb runs
uv run bnb runs --profile rl --resumable --limit 5
```

### Fitting the launch to a GPU

`--vram` chooses how much of the fixed logical batch stays resident and how finely the
backward pass is chunked. It never changes the batch itself:

```bash
uv run bnb train --profile rl --vram probe    # measure this card once, then train
uv run bnb train --profile rl                 # --vram auto: reuse that measurement
uv run bnb train --profile rl --vram 16       # a provisional preset for a 16 GB card
uv run bnb train --profile rl --vram off      # the profile's own sizing, nothing else
```

`auto` is the default. It uses a stored measurement only when that measurement was taken
on this exact GPU, software stack, compile mode, and profile; otherwise it leaves the
profile's derived sizing alone and says so. `probe` measures the machine if it has not
already, `reprobe` measures it again, and both write `.vram.json`, a gitignored local
cache rather than an artifact. Probing needs a CUDA device and takes minutes, because it
runs one real training update per candidate in its own subprocess.

A numeric preset (`8|16|24|32`, in GB) is a starting point rather than a measurement of
your card: only the 8 GB row was measured, and applying any row is reported as
`provisional`. `--print-config` shows the whole decision: which knobs moved, which
equivalence tier each one belongs to, and the resolved shard count.

Explicit `--num-envs` and `--microbatch-tokens` outrank whatever `--vram` proposes, and
the printed source map records which value came from where. They cannot be combined with
`--vram probe` or `--vram reprobe`, which exist to determine exactly those two values.
`docs/engineering/memory-optimization.md` describes what each knob costs.

Training profiles live in
[`src/boost_and_broadside/profiles/`](../src/boost_and_broadside/profiles/). Global ship, field,
and projectile defaults are defined on `ShipConfig` in
[`src/boost_and_broadside/config/core.py`](../src/boost_and_broadside/config/core.py).
The most relevant field/projectile controls are:

| Setting | Default | Meaning |
|---|---:|---|
| `field_index_step` | `sqrt(2)` | Four sampled levels span index 0.5 through 2 |
| `field_interface_damage` | `10` | Base health exposure of a standard interface |
| `field_integrator` | `midpoint` | Ship passive-field integrator |
| `field_integration_substeps` | `2` | Ship field substeps per 60 Hz tick |
| `bullet_field_integrator` | `two_step` | Projectile passive-field integrator |
| `bullet_field_integration_substeps` | `2` | Projectile field substeps per tick |
| `bullet_drag_coeff` | `8e-4` | Quadratic projectile drag coefficient |
| `bullet_field_damage_scale` | `0.1` | Projectile potential lost per interface-damage point |

Field geometry must satisfy
`field_radius_max + field_transition_width_max/2 < min(world_size)/2`. With the default
1024×1024 world and 40-pixel transition width, `field_radius_max` must be below 492.
The main combat profile is
[`profiles/rl.py`](../src/boost_and_broadside/profiles/rl.py), shared
model/physics/reward definitions are in
[`config/defaults.py`](../src/boost_and_broadside/config/defaults.py), and configuration types
are frozen dataclasses in
[`src/boost_and_broadside/config/`](../src/boost_and_broadside/config/).

The production projectile pool is
`DEFAULT_MAX_BULLETS_PER_SHIP=10` in
[`src/boost_and_broadside/constants.py`](../src/boost_and_broadside/constants.py). The
fixed pool keeps GPU shapes static; changing lifetime or firing cooldown may require
rechecking capacity with [`benchmarks/bullet_throughput.py`](../benchmarks/bullet_throughput.py).

## Evaluate

```bash
# Direct parallel matchup
uv run bnb collect-stats \
  --team0 checkpoints/<run>/<checkpoint>.pt --team1 scripted --sizes 4v4 8v11

# Post-hoc Elo calibration for a completed run
uv run bnb elo-calibrate \
  --run resilient-resonance-682

# Bradley-Terry calibration for any explicit stationary agent field
uv run bnb elo-calibrate \
  --agents scripted random checkpoints/<run>/<checkpoint>.pt

# Rate frozen checkpoints across symmetric fleet sizes (resumable)
uv run bnb elo-scale \
  --run resilient-resonance-682 --sizes 1,2,4,8,16,32,64

# Measure the random-to-scripted reference ladder used to condition scale ratings
uv run bnb semi-random \
  --run resilient-resonance-682 --sizes 1,2,4,8,16,32,64

# Measure it under a training profile's environment instead, with no run involved
uv run bnb semi-random --profile rl --sizes 4

# Search the scripted-team crossover for selected learned-team sizes
uv run bnb crossover \
  --run resilient-resonance-682 --sizes 4,8,16,32,64 --games-per-matchup 256
```

Every evaluation writes a versioned artifact rather than a file of its own choosing.
A measurement about one exact run lands in `checkpoints/<run>/artifacts/<type>/<id>/`.
Measurements with no single owning run land in `artifacts/<type>/<id>/` instead: an
explicit `--agents` field, say, or a ladder fitted for a training profile. Each directory holds `result.json`
(the aggregates every report is built from) beside `artifact.json` (the recipe, the exact
subjects and their hashes, and the code, dependency, and device provenance behind them).
Resumable sweeps (`elo-scale`, `semi-random`, `crossover`) continue the artifact for
their exact recipe and start a new one for any other. These evaluations can require
substantial GPU time, and the reference-run measurements are already included.
Methodology and interpretation are in [evaluation and results](evaluation.md).

The `--profile` form of `semi-random` measures the ladder under a training profile's
environment rather than a finished run's. It is a check, not a prerequisite: training
assigns each rung `1000·p` outright, so nothing has to be fitted before a profile can run.
The artifact adds a `live_gauge_error` per rung: how far the ratings training uses sit
from the ones the tournament measures in that exact environment. Fitted ratings are a
property of the environment they play in, so re-measure whenever the tick rate, field
count, ship config, fleet size or scripted controller moves; see
[the reference ladder](training.md#the-reference-ladder).

## Capture replays

```bash
uv run bnb capture \
  --run resilient-resonance-682 \
  --scenarios vs_scripted \
  --sizes 8v11 \
  --seeds 3 \
  --gif
```

Capture mode uses the run's final `step_*.pt` checkpoint, writes seeded MP4 files, and can
also emit downscaled GIFs. Scratch files go to `out/` by default; the curated subset
lives under `docs/results/replays/`. See the [replay guide](replays.md).

## Checkpoint and result artifacts

Full `step_<N>.pt` checkpoints contain the policy, optimizer, scalers, running-average
state, ratings, counters, and the complete resolved configuration needed by `--resume`. Training
also writes scheduled average/best snapshots and unpruned ladder checkpoints; the
[checkpoint implementation](../src/boost_and_broadside/train/rl/checkpoint.py) defines
the current filenames. (The included reference-run directory retains files from an
earlier naming convention, such as `recent_avg.pt`.)

A full checkpoint is written every update, and the newest few of each family are kept.

`config.json` beside them records what the run trained under, as a list of segments
each keyed by the step it took effect at. A run resumed with changed settings appends
a segment rather than overwriting, so `config_at(step)` answers what was in force when
a given checkpoint was written, and the newest segment is what a final checkpoint was
produced under. Runs from before this file existed do not have one.

## Changing settings for one launch

Any profile value can be changed positionally, before anything is derived from it:

```bash
uv run bnb train --profile rl clip_coef=0.2 elo_eval.window_size=64
uv run bnb train --profile rl num_fields=0 field_map=none   # re-derives the shard width
```

An unknown key is refused with the nearest real one rather than ignored. Overrides are
recorded in the run's config segment alongside the values they produced.

`--resume RUN` continues the same run and logs to the same W&B run, so
`--resume RUN key=value` is how a run is extended with different settings.
`--from RUN [--at STEP]` is the other thing: a fork, taking only weights into a new run
with its own history. Use `--from` when the change is to the task itself — ship count,
field count — because the ratings either side of such a change are not one series.

Each run that writes a checkpoint also writes `run.json` beside it: the profile, the
status, the update and step reached, elapsed training time, and the latest live rating.
It exists so a run can be identified without loading a checkpoint. Runs from before it
existed do not have one.

## Development notes

- Follow [STYLE_GUIDE.md](../STYLE_GUIDE.md).
- Use `uv run` for project commands.
- Keep physical behavior covered by real tensor tests rather than mocks; environment tests
  live in [`tests/env/`](../tests/env/).
- Memory measurements and the host-backed rollout design are documented in
  [memory optimization](engineering/memory-optimization.md).

The project is [MIT-licensed](../LICENSE). There is no contribution guide or CI workflow
yet.
