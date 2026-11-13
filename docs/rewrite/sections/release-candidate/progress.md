# Release candidate: progress

Branch `rel/candidate` from `origin/integrate/next-runs` 40cbd425, worktree
`/home/vizia/avesnova/bnb-release`, 2026-11-12 20:35-. CPU only (`CUDA_VISIBLE_DEVICES=""`):
the laptop GPU is another agent's. For the 5090 dry run (BC -> RL mainline).

## 1. Merges

- `origin/feat/launch` (4b09b70b: the launch kit, position codes, the BC -> RL chain).
  Two conflicts, both docstrings only: `bench/spawn_boundary_deaths.py::measure` and
  `bench/spawn_turn_excursion.py::measure`. Kept integrate's (the bug sweep's) wording; the
  code was the same on both sides.
- `origin/perf/compile-split` (8c6d09e3, itself built on feat/launch 4b09b70b, so only its
  own 20 commits came in): clean merge. Checked in the merged tree: `Learner.head_sums` is
  `self._head_sums` (eager); micro-batches of one size where they divide evenly;
  `lanes.RUN_SLACK = 1.05` (BC's lanes one run); `cache.sh` archives `BNB_CACHE`;
  `compile_regions` (B) is gone from `src`, `tests`, `bench`; `compile_split` (C) is a
  `TrainConfig` switch, off by default.
- Lint after the merge (its own commit): `ruff` flagged two lines from compile-split
  (`Learner._loss`'s return line; `bench/compile_baseline.py`'s docstring, whose example was
  the removed `compile_regions=true`, now `compile_split=true`) and `ruff format` the
  `Trunk` docstring's closing quotes in `yemong/network.py`. `_loss`'s docstring said the
  head sums were compiled; it says eager now.

## 2. Lineup

`launch.toml` `lineup = ["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]` (both names in
`api.type_names("real-scale")`). `launch.py check` on the CPU (scratch `--runs`, the shared
venv's `bnb` with this tree on `PYTHONPATH`): every buildable arm resolves (about 75 s);
`self-play` is NOT BUILDABLE by its own `needs` line, as before. The resolved configs carry
the lineup in both stages. The runbook's table no longer calls the lineup a placeholder.

## 3. The champion's model settings as the code's defaults

Each was a real change (champion `2026-11-09-rl-deadreckon-split-hiddenphysics`
`step_180355072/live/checkpoint.json`, `model.build.config`; the learner coefficient from
the 2026-11-12 runs' `config.json`, 0.0033, as the checkpoint has no learner config):

| setting | old default | new default |
| --- | --- | --- |
| `yemong.next_state` | `"absolute"` | `"delta"` |
| `yemong.delta_baseline` | `"persistence"` | `"extrapolation"` |
| `yemong.delta_split` | `false` | `true` |
| `yemong.physics_hidden_decay` | `null` | `0.5` |
| `learner.next_state` | 0.0041 | 0.0033 |

Where: the four model settings go in `TrainConfig.yemong` (a new run's build settings,
beside the position codes), not `YemongConfig`'s field defaults: those are the record
defaults a build record without the field reads (`YemongConfig` docstring), so changing
them would silently re-read older policy files as delta models. `LearnerConfig.next_state`
itself is 0.0033 (no record reads it by default: a run's `config.json` has every field).
The four lines and the learner line are gone from `launch.toml`'s `set`; `launch.py check`
afterwards resolves byte-identical configs (and the same model keys) for every arm.

Consequence: from the defaults, an arm without the split now needs `--set
yemong.delta_split=false --set yemong.physics_hidden_decay=null`, and the absolute head
also `yemong.next_state=absolute` (`delta_split` needs delta, so `next_state=absolute`
alone is refused at config time). No arm in `arms.toml` turns these off. Documented in
`docs/explanation/yemong.md` (the next-state head; `physics_hidden_decay`) and
`docs/explanation/learner.md` (the coefficient).

Tests touched:

- `tests/sections/test_yemong_delta.py::test_the_arms_set_line_builds_the_arm` (arm 2):
  the line adds `delta_split=false`, `physics_hidden_decay=null` so it still builds arm 2
  from the new defaults; asserts both.
- `tests/sections/test_yemong_error_head.py` `ARM_SETS` (arm 3,
  `test_a_vector_and_a_covariance_round_trip_through_the_three_axes`): adds
  `physics_hidden_decay=null`, asserted. New
  `test_a_new_runs_defaults_are_the_champions_model`: the defaults are the champion's, the
  champion's five lines change nothing, `YemongConfig()` keeps the record defaults.
- `tests/sections/test_yemong_extrapolation.py` `ARM_SETS` (arm 4,
  `test_the_arms_set_line_builds_the_arm`, `test_the_arm_4c_set_line_builds_the_arm`): adds
  `physics_hidden_decay=null`, asserted in the first.

## 4. Checks (CPU, quick tier `-m "not slow"`, this tree on `PYTHONPATH`)

- The three touched test files, `tests/sections/test_game_version.py`, `tests/docs`, and
  the files that build a `TrainConfig` or `LearnerConfig` (learner, BC, chain, mixed,
  lanes, lineup, position codes, compile split, delta / error head / extrapolation /
  accelerations / spread fit, nearest, dashboards, elimination, ship sets,
  `tests/loop/test_learning.py`, `tests/runs/*`): all pass but one.
- `tests/runs/test_compatibility.py::...[yemong]` fails, and fails the same at the base
  (`origin/integrate/next-runs` 40cbd425) and at the merged tree before the default
  change: one element of 120 of `.carry.belief.spreads`, relative 1.30e-4 against 1e-4.
  The stored policy loads with the record defaults, which this branch does not change.
  Known since the elimination branch (its progress notes). Not fixed here.
- `tests/docs` flagged my own `learner.next_state` in yemong.md (now
  `LearnerConfig.next_state`) and two older references (learner.md's clip test name,
  physics.md's bare `D`): fixed; 585 pass.
- `ruff check` and `ruff format --check` on `src tests bench`: clean.
- `launch.py check`: passes (section 2), identical configs with and without the removed
  `--set` lines.

## Risks for the dry run

- Nothing here ran on a GPU. The merged compile-split changes (eager head sums, equal
  micro-batches, `RUN_SLACK` 1.05 stacking BC's lanes into one run) were measured on the
  compile-split box, not on this tree; the dry run is their first run together with the
  launch kit.
- A run started without `launch.toml` now builds the champion's model; an ablation that
  wants the absolute head or no split must turn off `delta_split` and
  `physics_hidden_decay` too, or the config is refused at start.
- The compatibility test above is red on the base: worth a look before the release merge.
