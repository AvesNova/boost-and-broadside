---
covers: [src/bnb/league/unified]
---

# Unified evaluation

`bnb unified SPEC --out DIR` plays every evaluation target of a spec in one continuously
running engine per size bucket: Elo per condition, break-even per mode, win rates per
team-size lineup and pairwise decisions. Each environment starts its next game the
moment its game ends, the game the planner values most, so there are no batch
boundaries and no waiting for a batch's longest game. The design is in
`docs/rewrite/sections/unified-eval/plan.md`.

## A spec

A TOML file names the players and the targets (the full format is in
`bnb.league.unified.run`'s docstring):

```toml
lineup_types = ["Yongma", "Jujak"]

[players]                      # "scripted" (Imugi, the 1000 anchor) is always in
champion = "RUN/policies/step_N/exponential_average"
early = "RUN/policies/step_M/exponential_average"

[[elo]]
preset = "real_scale_elimination"
players = ["scripted", "champion", "early"]
stderr = 60.0

[[win_rate]]
preset = "real_scale_deathmatch"
first = "champion"
second = "scripted"
lineups = [[1, 1], [2, 3], [5, 5]]
stderr = 0.05

[[break_even]]
preset = "real_scale_frontline"
policy = "champion"
stderr = 0.1

[[decision]]
preset = "real_scale_deathmatch"
first = "champion"
second = "early"
delta = 0.02
```

Every target stops at its goal; tighten a goal and rerun with the same `--out` to continue
from the recorded games. A `budget` (games) and `--hours` are safety bounds only: a target
they stop short of its goal is named in a `WARNING` line of `progress.log`{.plain}.

## Elo fields

An Elo rating is finished only at its standard-error goal. A player whose games leave it
without a rating (it won or lost every game against the rest, a one-sided bound, or met
only players without one) is never a result: the planner takes it as the least precise
player of the field and gives it games first, against the opponents nearest it. Nearness
starts from the field's training neighbours (each checkpoint and the next one of its run;
other runs' checkpoints at the nearest steps; a chain's first RL checkpoint and its last
BC one) and moves to the fitted estimates as games come in, so a learning curve's ratings
chain along the run to the anchor through pairs of uncertain result.

Earlier games count: `--prior POOL` (repeatable) loads a rating pool's games
(`bnb pool rate`'s records) of each Elo target's condition before the first game, players
matched by their weights, games of another game version left out (the log says how many).
A target's `prior_keys` names the pool's preset keys for its condition (default: the
condition's own key; the evaluation queue's pools use `frontline_5v5` and the like). Pass
the same `--prior` again on a resume.

## Running

<!-- not-runnable: needs a spec and trained policies -->
```bash
uv run --no-sync bnb unified spec.toml --out runs/eval-x --runs-directory runs
```

- `--environments N`: each bucket's batch (default: sized from the free device memory, a
  multiple of 64, the same on every run of one machine, so the compiled graphs repeat);
- `--memory-share 0.7`: the share of the free memory a batch may take;
- `--concurrent auto|yes|no`: buckets with work share the device when each gets 512
  environments, else they run in turn;
- `--hours 24`: the safety bound on the run's time (reported when reached);
- `--prior POOL`: a rating pool whose games the Elo targets start from.

Each bucket compiles once (about 4 minutes cold on the laptop, seconds warm). On a box,
`box/unified_check.sh CHAIN` builds the spec of the dry run's evaluation queue and runs
it, importing a compile-cache archive first when the script's archive variable names one.

## What it writes

- `progress.log`{.plain}: a line every 30 s with the games, games per hour, the GPU's
  utilisation, each bucket's environments and the ETA as a clock time;
- `games.jsonl`{.plain}: every finished game (flushed every 10 s, synced), which a rerun resumes
  from;
- `report.json`{.plain}: each target's estimate (ratings and standard errors, break-even ratio,
  scores, a decision's looks and outcome);
- `sizing.json`{.plain}: each bucket's measured memory footprint and chosen size.

## Decisions

A `[[decision]]` is the group-sequential test of `bnb evaluate` (four looks, O'Brien-Fleming
spending), its looks taken when the games reach a quarter, half, three quarters and all of
the planned maximum, the boundary spent at the fraction actually reached. The planner
gives it games by their count alone, so the allocation cannot bias the test.
