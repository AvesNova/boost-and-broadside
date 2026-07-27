# Boost and Broadside

### One recurrent team policy, from 4-vs-4 training to 64-vs-87 combat

![Eight policy-controlled blue ships defeating eleven scripted-controlled red ships](docs/results/replays/vs_scripted_8v11_seed03.gif)

*One policy jointly controls all eight blue ships; the 11 red ships use the scripted
controller. This seeded replay ends with three blue ships surviving.*

Boost and Broadside is a tensorized 2D team-dogfighting environment and reinforcement
learning system. Its central experiment asks whether a policy trained at one small team
size can coordinate much larger teams without retraining.

The landmark policy trained only in **4-vs-4** combat. At evaluation time, the same saved
weights control teams from one to 64 ships, including asymmetric battles against more
numerous scripted opponents. “Eight vs eleven” therefore means **one neural policy
controlling eight ships**, not eight independently instantiated neural agents.

[Explore the evaluation](docs/evaluation.md) · [See more replays](docs/replays.md) ·
[Understand the architecture](docs/architecture.md) · [Get started](docs/getting-started.md)

## Zero-shot team-size transfer

Every ship is an entity token. A spatial Transformer lets ships exchange information
within a timestep; per-entity Griffin recurrence carries state through time. The policy
then emits one factored action for every ship in a single forward pass. Because the
learned parameters are not sized by the number of ships, the controller can be evaluated
at team sizes it never encountered during training.

![Empirical boundary between policy-controlled and scripted-controlled team wins](docs/results/crossover_phase.png)

*The existing plot labels the axes “trained agents” and “scripted agents”; each unit is a
**ship**. One recurrent policy controls the entire learned team. The line lies halfway
between the largest scripted team still beaten in at least 50% of games and the first
adjacent count below 50%.*

Selected results from the [recorded crossover sweep](docs/crossover/crossover.json):

| Learned team | Scripted team | Learned-team win rate |
|---:|---:|---:|
| 8 policy-controlled ships | 11 ships | **69.5%** |
| 16 policy-controlled ships | 24 ships | **52.7%** |
| 32 policy-controlled ships | 47 ships | **55.9%** |
| 64 policy-controlled ships | 87 ships | **53.1%** |

These are empirical boundary measurements from one checkpoint, not a claim of a universal
scaling law. The [evaluation guide](docs/evaluation.md) documents the search procedure,
sample sizes, uncertainty limitations, and raw artifacts.

## Learning progression

The landmark run targeted one billion environment steps and logged 999,424,000 in 7.50
hours on one RTX 5090. Post-hoc calibration estimates the final live policy at about
**2053 ELO**, versus **1240** for the scripted controller on the same random-anchored scale.

![Post-hoc calibrated ELO over training](docs/results/elo_curve.png)

This curve is reconstructed from each update's recorded match outcomes after a stationary
tournament calibrates its opponents. It is distinct from the drifting online rating shown
during training. See [results and methodology](docs/evaluation.md#post-hoc-elo-calibration)
for the exact values and uncertainty.

## How the system fits together

1. The [tensorized simulator](src/boost_and_broadside/env/env.py) advances thousands of
   battles in parallel on the GPU, with toroidal movement, projectiles, resources, and
   optional orbital obstacles.
2. The [feature coordinator](src/boost_and_broadside/train/rl/features.py) converts global
   ship and obstacle state into topology-aware entity features and auxiliary prediction
   targets.
3. [YemongPolicy](src/boost_and_broadside/models/yemong/policy.py) combines spatial
   attention with temporal recurrence, producing per-ship actions, decomposed values, and
   next-state predictions.
4. The [recurrent PPO trainer](src/boost_and_broadside/train/rl/ppo.py) learns from
   self-play, scripted opponents, a running-average policy, and frozen historical
   checkpoints.
5. [Crossover evaluation](src/boost_and_broadside/modes/crossover.py), continuous match
   records, and [post-hoc calibration](src/boost_and_broadside/modes/elo_calibrate.py)
   turn those policies into reproducible quantitative and qualitative results.

The deeper design is split by concern:

- [Environment and physics](docs/environment.md) explains the game, state, actions, and
  simulation engine.
- [Policy architecture](docs/architecture.md) covers entity encoding, spatial attention,
  Griffin recurrence, and output heads.
- [Training system](docs/training.md) covers PPO, reward decomposition, `ego_pass`, policy
  averaging, league opponents, and checkpoints.
- [Evaluation and results](docs/evaluation.md) gives the crossover and ELO methodology,
  exact result provenance, and limitations.
- [Replays](docs/replays.md) curates qualitative examples and explains how to regenerate
  them.
- The maintainer-facing [evidence map](docs/evidence.md) traces claims to code and raw
  artifacts and records deferred plot work.

## Quick start

The included landmark checkpoints use Git LFS. Install Git LFS before cloning, then:

```bash
git lfs pull
uv sync

# Confirm the available modes
uv run --no-sync main.py --help

# Human vs the latest checkpoint (WASD, Shift, Space)
uv run --no-sync main.py --mode watch

# Small no-W&B training crash test
uv run --no-sync main.py --mode rl --smoke
```

Training is designed for CUDA hardware; the simulator and test suite can also exercise
many paths on CPU. The [setup and usage guide](docs/getting-started.md) covers checkpoints,
training, evaluation, capture commands, and development checks.

## Repository map

Experiment profiles live in [`runs/`](runs/), implementation code in
[`src/boost_and_broadside/`](src/boost_and_broadside/), tests in [`tests/`](tests/), and
curated results under [`docs/results/`](docs/results/). Landmark-run configuration,
checkpoints, raw match records, W&B exports, and calibration outputs are preserved under
[`checkpoints/resilient-resonance-682/`](checkpoints/resilient-resonance-682/).

For code conventions and engineering expectations, see the [project style guide](STYLE_GUIDE.md).
