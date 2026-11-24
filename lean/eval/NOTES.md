# lean/eval

Evaluation: play agents against each other in the core game, keep every game in one SQLite
store, rate with Bradley-Terry Elo, and run v1's crossover (how many scripted ships a network
beats). Decisions: `docs/rewrite/core-decisions.md` ("Evaluation runner and scripted search");
judgement calls: `docs/rewrite/morning-notes.md` ("V"); history: `docs/rewrite/core-worklog.md`.

## Entry points

    python -m lean.eval play STORE yardstick net:DIR/rl.pt:ema --games 512
    python -m lean.eval play STORE yardstick --sobol screen --count 256 --games 64
    python -m lean.eval play STORE A B C --target-error 0.02 --budget 100000
    python -m lean.eval report STORE yardstick net:DIR/rl.pt
    python -m lean.eval elo STORE --sizes 4 4
    python -m lean.eval crossover STORE DIR/rl.pt --sizes 4 8 --games 64

Agents: `yardstick`, `scripted:FILE.json` (parameter rows over the defaults),
`net:CHECKPOINT[:live|ema|uniform]` (ema by default), or an agent id in the store. Game options
(`--clock-ticks`, `--arena-radius`, `--spawn`, `--fields`, `--no-fog`) change the conditions;
evaluation always uses the standard spawn. In Python: `Evaluator(Store(path), GameConfig())`,
then `ensure([(a, b, (4, 4), games)])`, `adaptive(pairings, target_error=0.02)`, `report`, `elo`.

## Files

- `runner.py`: `play(game_config, queue, scripted_table, policy_a, policy_b)`. E environments
  are slots; a slot whose game ends takes the next game of the `Queue` at once. At the cutoff
  (the queue's end, or `seconds`) no game starts and every started game plays to its end (the
  drain), so the results are a prefix of the queue with each game whole. A seat is one network
  for every game, or a scripted agent per game. Deals are drawn once per 32-tick segment; each
  network match draws its own position offset, as in training. A network acts through `lean.model.agent.network_step` (the agent step training's
  live policy and the league share) and the game steps through `lean.game.tick` (shared with
  the rollout and the sandbox).
- `store.py`: the store (agents, conditions, games, a `counts` view, a `styles` view, Sobol
  streams), `ScriptedAgent`, `NetworkAgent`, `game_version`.
- `style.py`: style counters, summed per slot and seat inside the compiled tick (no host read),
  zeroed at each deal, written to the game's record. Per game and seat (columns `a_...`, `b_...`
  of `games`): ship-minutes; share of time in a turning fight (the ship and its nearest living
  enemy both turning at least 0.25 sustained rates) and the share of that in two-circle flow
  (same turn sense; the rest one-circle), the flow of lean/scripted/situation.py with a hard
  floor; mean store fraction; mean speed in corner speeds; real shots per ship-minute; hits and
  mean range at hits (shooter to target at the hit tick's start, gun ranges); share of time
  climbing and diving (power applied). `python -m lean.eval styles STORE [AGENT ...]` prints the
  view: per agent and conditions, means weighted by ship-minutes (two-circle by turning-fight
  time, hit range by hits). Games stored before the counters have NULL style.
- `rating.py`: `standard_error`, `allocate`, `bradley_terry` (Newton, a 1000-Elo Gaussian prior
  about the anchor). `evaluator.py`: `Evaluator` (`ensure` plays only the missing games with
  sides balanced; `adaptive` rounds to a target error or a budget; `sobol_candidates`).
- `matches.py`: v1's crossover walk (`walk`, `crossover`, `evaluator_round`), every network size
  at once in rounds; a tie is a non-win.

## Identity and reuse

- A scripted agent is its full parameter vector; a network is its checkpoint file's hash and
  the weights (live, ema, uniform).
- Conditions: the whole `GameConfig`, the seats' ship counts and the game version (a hash of
  lean/game, lean/scripted and lean/contract.py). An edit there starts new conditions; old games
  stay in the file. The model's code is not in the version.
- A pairing is read in both orientations; A's games as team 0 are kept at half of the total.

## What compiles

One graph per seat layout (scripted vs scripted, a network in seat A, networks in both). No new
graph for: new scripted vectors, another network of the same architecture, team sizes 1 to 6 in
one queue, E and the queue length, a new arena radius, spawn rule or clock. Fog off compiles a
new graph. Not measured: field count, lineup, another architecture.

## Numbers (2026-11-24, laptop RTX 4070, performance profile, core game 4v4, then a 5,400-tick clock)

| | games/s | game ticks/s |
|---|---|---|
| scripted vs scripted, E 4,096, 16 games per slot | 810 | 1.13M |
| scripted vs scripted, E 4,096, 4 games per slot | 539 | 0.75M |
| untrained network vs scripted, E 4,096, 16 games per slot, compile included | 107 | 87k |
| the same, steady | | 122k env ticks/s |

The style counters' cost (2026-11-24, same laptop and profile, scripted vs scripted 4v4, E
4,096, 65,536 games, compile excluded; one clean pair only, later runs overlapped another GPU
job): before 793 and 782 games/s, after 775 games/s, about 1-2% slower.

The drain is unbiased: the yardstick against a variant with fight_speed 1.3 scored 0.4656 +- 0.0019 over a whole queue
of 65,536 games and 0.4654 +- 0.0034 over the 22,043 started before a 20 s cutoff. A drain lasts
up to the longest game, so use at least 16 games per slot. Compile: the scripted tick 66 s cold,
23 s warm; the network tick about 110 s cold. `functional_call` costs 9% against calling the
policy directly.

## Tests

`lean/eval/tests`: the rolling queue, the drain, store reuse and identity, Bradley-Terry and
allocation, the crossover walk.

## Open

- Unequal team sizes with a network are untested in a full run.
- The batched crossover's GPU gain over v1's walk is not shown.
