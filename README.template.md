# Boost and Broadside

### Zero-shot fleet coordination from a 4-vs-4 policy

{{figure: clip_hero}}

<sub>More replays: <a href="docs/replays.md">replays</a>.</sub>

Boost and Broadside is a tensorized 2D dogfighting environment and reinforcement
learning system built to study coordination at scale. The central question is simple:
can a policy trained with a small team command much larger fleets without retraining?

The reference policy trained exclusively in **4-vs-4** combat, then transferred
zero-shot to fleets of one to {{crossover_largest_size}} ships. A single recurrent network
commands the whole fleet, producing an action for every ship on each forward pass.

[Explore the results](docs/evaluation.md) · [Watch more replays](docs/replays.md) ·
[Understand the architecture](docs/architecture.md) · [Get started](docs/getting-started.md) ·
[Training runs](docs/training-runs.md)

## Zero-shot team-size transfer

Ships and optional refractive fields are represented as entity tokens. Spatial attention coordinates the
fleet within each timestep, while per-entity [Griffin-style](https://arxiv.org/abs/2402.19427)
recurrence carries information through time. Because the network operates over a
variable-length token sequence, the same weights can run at fleet sizes never seen
during training.

{{figure: fan_4_ships}}

Rated against the same scripted controller at every size
([chart](docs/evaluation.md#at-a-glance)), the final live checkpoint of the 4-vs-4 policy
is stronger at 8-vs-8 and 16-vs-16 than at the size it trained at: about {{scale_16_ships_elo:,.0f}} Elo
at 16-vs-16 against {{scale_4_ships_elo:,.0f}} at 4-vs-4. Past that it weakens, to
{{scale_32_ships_elo:,.0f}} at 32-vs-32 and {{scale_64_ships_elo:,.0f}} at 64-vs-64, still far
above the scripted controller's 1000.

Selected results from the [recorded crossover sweep](checkpoints/{{run_name}}/artifacts/figures/crossover.json)
of the final live checkpoint, each row the largest scripted fleet it still beats, from
{{crossover_games_per_matchup}} games per point ({{crossover_64_ships_beats_up_to_games}} at
64 ships); ties count as non-wins:

| Learned ships | Scripted ships | Win rate (live) |
|---:|---:|---:|
| 4 | {{crossover_4_ships_beats_up_to}} | **{{crossover_4_ships_beats_up_to_win_rate:.1%}}** |
| 8 | {{crossover_8_ships_beats_up_to}} | **{{crossover_8_ships_beats_up_to_win_rate:.1%}}** |
| 16 | {{crossover_16_ships_beats_up_to}} | **{{crossover_16_ships_beats_up_to_win_rate:.1%}}** |
| 32 | {{crossover_32_ships_beats_up_to}} | **{{crossover_32_ships_beats_up_to_win_rate:.1%}}** |
| 64 | {{crossover_64_ships_beats_up_to}} | **{{crossover_64_ships_beats_up_to_win_rate:.1%}}** |

The averaged policy, the running average of the weights that the run keeps beside the live
ones, holds more when outnumbered: four averaged ships win
{{fan_4_ships_vs_7_averaged_win_rate:.1%}} of {{fan_4_ships_vs_7_averaged_games:,}} games
against seven scripted ships, where the final live weights win
{{fan_4_ships_vs_7_live_win_rate:.1%}} of {{fan_4_ships_vs_7_live_games:,}}
([over training](docs/evaluation.md#outnumbered-over-training)); ties again count as
non-wins. That {{fan_4_ships_vs_7_live_win_rate:.1%}} and the table's
{{crossover_4_ships_vs_7_win_rate:.1%}} measure the same checkpoint at 4-vs-7, over
{{fan_4_ships_vs_7_live_games:,}} and {{crossover_4_ships_vs_7_games}} games. The
[evaluation guide](docs/evaluation.md#zero-shot-crossover) covers the search method,
sample sizes, raw artifacts, and limitations behind these measurements.

## Learning progression

The reference run, `{{run_name}}`, trained for {{run_steps_millions:.0f}}M of a planned
{{run_target_steps_millions:,.0f}}M environment steps in about {{run_days:.1f}} days on a
single {{run_gpu}}, and was stopped there. Post-hoc calibration places its final live
checkpoint at about **{{calibration_final_live_elo:,.0f}} Elo** on a scale that fixes the
scripted controller at 1000, a lead of roughly
**{{calibration_final_live_lead_over_scripted:,.0f}} points**, where 400 points already
means ten-to-one odds. Its best averaged checkpoint rates
{{calibration_best_averaged_elo:,.0f}} in the same tournament.

{{figure: elo_curve}}

The live policy's calibrated rating keeps rising long after wins against the scripted
controller stop being informative: the live policy first wins 99% of evaluation games
against it at {{scripted_win_rate_first_99_millions:.0f}}M steps, while its refit rating
climbs from about
{{calibration_curve_elo_at_100m:,.0f}} at 100M to about
{{calibration_curve_elo_at_250m:,.0f}} at 250M before levelling off.

See [results and methodology](docs/evaluation.md#post-hoc-elo-calibration) for the rating
procedure, exact values, and uncertainty.

## Under the hood

![YemongPolicy architecture: entity tokens through spatial attention and temporal recurrence to per-ship heads](docs/policy_architecture.png)

One trunk processes the whole fleet as entity tokens, with attention mixing across ships
within a timestep and Griffin recurrence carrying each ship through time. Every head
emits one output per ship, however many there are.

- The [environment and physics engine](docs/environment.md) runs thousands of tensorized
  battles in parallel, with toroidal movement, projectiles, resources, and optional
  static refractive fields that continuously affect both ships and projectiles. The core
  simulator lives in
  [`env.py`](src/boost_and_broadside/env/env.py).
- The [policy architecture](docs/architecture.md) combines spatial attention, Griffin
  recurrence, factored action heads, decomposed value estimates, and auxiliary dynamics
  prediction. See
  [`YemongPolicy`](src/boost_and_broadside/models/yemong/policy.py).
- The [training system](docs/training.md) uses recurrent PPO with scripted, self-play,
  running-average, and historical opponents. The update logic is in
  [`ppo.py`](src/boost_and_broadside/train/rl/ppo.py).
- [Evaluation](docs/evaluation.md) and [seeded replays](docs/replays.md) connect aggregate
  measurements with qualitative behavior. The crossover evaluator is
  [`crossover.py`](src/boost_and_broadside/modes/crossover.py).

The run's own charts are linked from
[its artifacts](checkpoints/{{run_name}}/artifacts/figures/), whose records name the
measurement behind each chart and that measurement's command, seeds and commit. The fan,
survivor and `{{run_name}}` clip results come from the scripts in
[`scripts/studies/`](scripts/studies/), whose [README](scripts/studies/README.md) records
their commands, seeds, games per point and commit.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Git LFS is only needed for the reference
checkpoints: `{{run_name}}`, and the earlier `good-leaf-719` and
`resilient-resonance-682`, all of which load in this version. After cloning:

```bash
git lfs pull   # fetch reference checkpoints (skip if training from scratch)
uv sync

# Resolve and inspect the RL launch without allocating the trainer
uv run bnb train --profile rl --print-config

# Play a 1v1 match against a null ship in four refractive fields
uv run bnb play

# Human vs a trained checkpoint (WASD, Shift, Space)
uv run bnb watch --team0 null --team1 checkpoints/<run>/<checkpoint>.pt
```

Training is designed for CUDA hardware; the simulator and test suite also run on CPU.
The [setup and usage guide](docs/getting-started.md) covers checkpoints, training,
evaluation, replay capture, and development checks.

## Where the numbers come from

The README and the pages in `docs/` are rendered from `*.template.md` files beside them.
Every run-specific number is a named value in [`docs/values.json`](docs/values.json),
built by `scripts/docs_values.py` from the reference run's artifacts and the study results
in `docs/results/`, and every chart is
named in [`docs/figures.yml`](docs/figures.yml). Edit a template, then run
`uv run python scripts/render_docs.py`; with `--check` it fails on an unknown name or a
page out of date.

## License

[MIT](LICENSE).
