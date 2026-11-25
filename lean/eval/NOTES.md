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

- `runner.py`: `play(game_config, queue, scripted_table, policies, bf16=...)`. E environments
  are slots; a slot whose game ends takes the next game of the `Queue` at once. At the cutoff
  (the queue's end, or `seconds`) no game starts and every started game plays to its end (the
  drain), so the results are a prefix of the queue with each game whole. Each game's seat A
  and seat B is a scripted vector (`a`, `b`) or a network (`net_a`, `net_b`, an index into
  `policies`, all one architecture), so one queue holds many pairings. Deals are drawn once
  per 32-tick segment (at most ceil(E / 8) games start in a segment); each seat draws its own
  position offset per game, as in training. A network acts through
  `lean.model.agent.network_step` (the agent step training's live policy and the league
  share) and the game steps through `lean.game.tick` (shared with the rollout and the
  sandbox). See "Routing".
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
  With unequal sizes the other orientation is stored under the mirrored conditions (A on 4
  ships against B on 8 is B on 8 against A on 4): `counts`, `elo` (`pairing_totals`) and
  `styles` read both conditions ids (2026-11-25; before, the reverse orientation was looked
  up under the same id, and the old rule that seated a network in seat A stored a
  scripted-first unequal pairing where `ensure` could not find it, so it replayed every call).

## Shared game counts

Every command reads and writes the same rows: a game is (agent A, agent B, side, conditions),
the conditions being the game config, the seats' sizes and the game version. `play` and
`adaptive` (through `ensure`) write; `report` and `crossover` (through `ensure`, which plays
only what is missing) read the `counts` view; `elo` reads every pairing under the conditions;
`styles` reads each agent's games from both seats. So a crossover match (network on 8 against
the yardstick on 6) is the `elo --sizes 8 6` pairing and the `report yardstick net:... --sizes
6 8` pairing turned round, and its games' styles are the agent's `styles --sizes 8 6`. A Sobol
screen against the yardstick and a ladder share their yardstick games likewise. A network's id
covers its inference code (lean/model, lean/eval/runner.py): a runner edit starts new network
ids, so network games from before it are not reused (scripted games are).

## Routing

The training league routes its environments to its members (L2, "mixture-of-experts style"):
`lean.model.agent.route_rows` gives each network the rows it plays, in order, padded to a
width W; `routed_step` (the league's `slot_actions`) runs the network once on its W rows and
writes back with a `where` over all rows. The runner routes the 2E seats (seat A of every
environment, then seat B) the same way, with `route_rows` and `network_step`, but writes in
place: each network gathers its seats' view, carry and offset, runs on W rows, and scatters
its moves and carry back to its own seats only (`index_copy_`, padding rows to a spare row
2E); a seat's carry is zeroed when its game is at tick 0. W per network and segment is its
seats in play plus those of the games the segment may start, at most E per seat side it
plays (host arithmetic on the queue and one read per segment; nothing is read in a tick).
Per tick: one `seat_moves` call per network with seats, then `advance_decide` (the game tick,
the starts, and the next tick's scripted moves and views in one graph). Without networks the
fused `tick` runs (scripted moves and the game in one graph).

`Evaluator.ensure` puts every game with a network into one shuffled queue (one runner call per
architecture), and the scripted-only games into a call of their own: inside a network call
they would pay the network tick's per-environment costs (views, routing). `minimum_envs`
(default 2,048) lets a short job fill the GPU rather than keep 16 games per slot.

Why the gain is smaller than training's 2.5x: before L2 the league ran every member on every
environment (a vmap over the slots); the old runner never ran a network on rows it did not
play. What one batch saves here is calls: E is bounded by a call's games, and each call pays a
drain, so many small pairings (adaptive rounds) ran at a small E. With one call the whole
round shares the GPU.


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
