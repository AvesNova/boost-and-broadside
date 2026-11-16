---
covers: [src/bnb/figures/results/, src/bnb/figures/ladder.py]
reviewed: 2026-11-16 23:30
---

# Results charts

`bnb figures results CHART SOURCE ...` draws one chart of the write-up's results from
records the runs and the measurements already write. Each chart is a PNG, with its
provenance (the render's commit, command line and time, the sources read and the numbers
shown) in its PNG text chunks, plus a PDF for the paper (`--formats png svg` for SVG
instead). The output goes to `--out` (`docs/results/`{.plain} by default). Each chart's numbers are
merged into the values file (below). Needs the tools extra (matplotlib).

<!-- not-runnable: needs a run's records -->
```bash
bnb figures results fan $OUT/break-even                  # per-snapshot bnb crossover results
bnb figures results break-even $OUT/break-even --chain r0
bnb figures results arms docs/results/arms.json
bnb figures results team-size $OUT/team-sizes            # *.batches.jsonl below it
bnb figures results health runs/r0-bc runs/r0-rl         # or wandb:ENTITY/PROJECT/ID
bnb figures results game-length games/*.jsonl
```

The house style: lines and densities, not bars; the modes in a fixed order and colour
(frontline blue, deathmatch orange, elimination green); ratios on the viridis ramp, lowest
darkest; no second y axis; up to four lines labelled at their ends, a legend beyond that.

## The charts and their inputs

**`fan`**, the headline: the policy's win rate against Imugi over training, one line per
lineup (`--policy-count` 5 against each of `--opponent-counts` 5 6 8 10), one panel per mode
(`--modes` picks some). The source is the directory the dry run's loop writes `bnb crossover`
results into, `<run>/step_<n>/artifacts/crossover/<time>/result.json`
(`bnb.figures.ladder.crossover_results`). A line's point at a snapshot is that snapshot's
logistic fit (`bnb.league.crossover.fit`) at the lineup's log ratio, with a band of the 95%
interval of the fit's linear predictor. The games played at a line's ratio (within 0.03 in
log ratio) are open points. A snapshot whose games are separated (no crossing) has no line
point; the panel counts them. A chain's `-bc` and `-rl` runs share one axis, the hand-off
dashed (`bnb.figures.ladder.chain_offsets`).

**`break-even`**: Imugi's ships per one of ours at even odds, against the chain's steps, one
line per mode with standard-error bars, on a log axis. The source is the same directory; the
points are the ladder's break-even row (`bnb.figures.ladder.break_even_points`), drawn in
one panel. `--chain NAME` picks the chain when the directory holds several.

**`arms`**: the forest plot of the arms' decisions. One row per decision with its 95%
interval, the zero line, the indifference zone (plus or minus the suite's delta) shaded, the
decision marked, and a last row for the seed-to-seed noise. The score is the decision
suite's (`bnb.league.evaluation`): the arm's expected score against its champion less one
half. The source is a JSON file, or a CSV file of the rows alone:

<!-- not-runnable: an input file's shape -->
```json
{
  "title": "Arm decisions",
  "delta": 0.02,
  "rows": [
    {"name": "A1 map sizes", "score": 0.041, "stderr": 0.012, "decision": "kept"},
    {"name": "A2 team sizes", "key": "a2", "evaluation": "../evaluate/a2", "arm_side": "B"},
    {"name": "seed to seed", "kind": "seed_noise", "score": 0.004, "stderr": 0.009}
  ]
}
```

A row has `name`, optionally `key` (its values' name; default `name` in lower case with
underscores) and `kind` (`arm` or `seed_noise`), and either `score` and `stderr` given or an
`evaluation` directory (relative to the file) read as `bnb evaluate decide --peek` reads it:
the last look's score and standard error, its decision, and the suite's delta. `arm_side` is
the side the arm played (`A` by default). `decision` is `kept`, `rejected` or `pending`; from
an evaluation, the arm's side winning is kept, the champion's or a tie rejected, an open test
pending. A row without a score shows "no estimate yet".

**`team-size`**: the win rate against Imugi at each team size (the x axis is our ship
count), one line per `--ratios` (Imugi's ships per ours: 1, 1.5, 2), one panel per mode, the
size it trained at marked (`--trained-team-size`). The sources are JSONL files, or
directories of `*.batches.jsonl`, of game tallies: the lines `bnb crossover` records (each
line's `blocks`: `policy_count`, `opponent_count`, `wins`, `draws`, `losses`), or lines that
are tallies themselves, with a `mode` or `preset` (else the file's `<preset>` name). A
lineup counts for a ratio within 0.035 in log ratio, so 1:1.5 has points at even sizes only,
and no game is larger than 20 a side, so 1:2 stops at 10. The band is Wilson's 95% interval
(a draw half a win).

**`health`**: small multiples against steps: PPO's clip fraction, the approximate KL,
explained variance and value loss per reward part, entropy per action factor, BC's KL to its
teacher, and one panel per auxiliary head's loss (every `<head>/loss` and `<head>/loss/<part>`
key). Each line is faint raw and a centred moving mean. The sources are run directories or
`metrics.jsonl`{.plain} files, or W&B runs as `wandb:ENTITY/PROJECT/ID`, read only through
`run.history(samples=N, keys=[...])` (`--samples`, one request per key). Several sources are
a chain in order, each starting at the previous one's last steps.

**`game-length`**: one density per mode (a Gaussian kernel, reflected at zero and at the
game clock), the medians dotted and the clocks dashed. Games that ran the clock out are a
point mass at the clock: they are left out of the curve, which then integrates to the share
decided before it, and their share is in the legend. The sources are JSONL files, one game
per line, `{"mode": "deathmatch", "seconds": 212.4, "clock_seconds": 360}` (`preset` in
place of `mode` works too), or one line per mode with `"lengths": [...]`.

## The values file

Each chart merges the numbers it shows into one flat JSON object, `docs/results/values.json`{.plain}
(`--values`; `--values ""` writes none), {name: number}: a name it writes replaces the same
name, every other name stays, and a number that is not finite is written as `null`. A docs
page quotes a value as `{{name}}`, or `{{pending: name}}` before the number exists. The
names, with `<mode>` one of `frontline`, `deathmatch`, `elimination`:

| Chart | Names |
|---|---|
| `fan` | `fan_<mode>_<p>v<o>_win_rate`, `_low`, `_high`: the last snapshot's win rate at p against o and its 95% interval |
| `break-even` | `crossover_ratio_<mode>`, `_stderr`, `_steps`: the last snapshot's ratio, its standard error and the chain's steps |
| `arms` | `arm_<key>_score`, `_low`, `_high`; `seed_noise_score`{.plain}, `_low`, `_high` |
| `team-size` | `team_size_<mode>_ratio_<k>_size_<n>_win_rate`, k written 1, 1_5, 2 |
| `health` | `health_<key>_final`: the key's mean over the last twentieth of the last source, the key in lower case with underscores (`health_policy_clip_fraction_final`{.plain}) |
| `game-length` | `game_length_<mode>_median_seconds`, `_lower_quartile_seconds`, `_upper_quartile_seconds`, `_games`, `_clock_out_share` |
