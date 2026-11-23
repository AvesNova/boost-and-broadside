# lean/eval: notes

Decisions: docs/rewrite/core-decisions.md "Evaluation runner and scripted search". Judgement
calls of the build: docs/rewrite/morning-notes.md ("V").

## Files

- `runner.py`: `play(game_config, queue, scripted_table, policy_a, policy_b)`. E environments
  are slots; a slot whose game ends takes the next game of the `Queue` at once (rolling). At
  the cutoff (the queue's end, or `seconds`) no game starts and every started game plays to
  its end (the drain): the results are a prefix of the queue, each game whole. Seat A and
  seat B are each one network for every game, or a scripted agent per game (an index into
  `scripted_table`, full parameter vectors). Deals are drawn once per segment (32 ticks);
  the host reads two numbers per segment.
- `store.py`: the SQLite store (agents, conditions, games; a `counts` view; Sobol streams),
  `ScriptedAgent`, `NetworkAgent`, `game_version`, `SobolStream`.
- `rating.py`: `standard_error`, `allocate` (lots of two to the largest error),
  `bradley_terry` (Newton, a 1000-Elo Gaussian prior about the anchor).
- `evaluator.py`: `Evaluator`: `ensure` (play only the missing games, sides balanced),
  `adaptive` (rounds to a target error or a budget), `report`, `elo`, `sobol_candidates`.
- `matches.py`: v1's crossover walk; `evaluator_round` plays its rounds on an evaluator.

## Commands

    python -m lean.eval play STORE yardstick net:DIR/rl.pt:ema --games 512
    python -m lean.eval play STORE yardstick --sobol screen --count 256 --games 64
    python -m lean.eval play STORE A B C --target-error 0.02 --budget 100000
    python -m lean.eval report STORE yardstick net:DIR/rl.pt
    python -m lean.eval elo STORE --sizes 4 4
    python -m lean.eval crossover STORE DIR/rl.pt --sizes 4 8 --games 64

Python: `Evaluator(Store(path), GameConfig())`, then `ensure([(a, b, (4, 4), games)])`,
`adaptive(pairings, target_error=0.02)`, `report(pairings)`, `elo((4, 4))`.

## Identity and reuse

- A scripted agent is its full vector (`lean.scripted.parameters.vector`, every row of the
  table with its name); a network is its checkpoint file's hash and the weights (live, ema,
  uniform).
- Conditions: the whole GameConfig, the seats' ship counts and the game version (a hash of
  lean/game, lean/scripted and lean/contract.py). Any edit there starts new conditions; old
  games stay in the file. The model's code is not in the version (a network's identity is its
  file).
- A pairing is read in both orientations. Sides alternate: A's games as team 0 are kept at
  half of the total as new games are added.

## What compiles

One graph per seat layout (scripted against scripted, a network in seat A, networks in both
seats). Measured with no new graph: new scripted vectors, another network of the same
architecture, team sizes from 1 to 6 per game in one queue, E and the queue's length, and a
new arena radius, spawn rule or clock (PyTorch 2.15 traces numbers as symbols:
`specialize_float` and `specialize_int` are off). Fog off compiles a new graph (a branch on
a flag). Not measured: field count (0 is a branch), lineup, another architecture.

## Measured (2026-11-24, laptop RTX 4070, performance profile, core game 4v4)

| | games/s | game ticks/s |
|---|---|---|
| scripted vs scripted, E 4,096, 16 games per slot | 810 | 1.13M |
| scripted vs scripted, E 8,192, 16 games per slot | 770 | 1.07M |
| scripted vs scripted, E 4,096, 4 games per slot | 539 | 0.75M |
| network (untrained) vs scripted, E 4,096, 16 games per slot, compile included | 107 | 87k |
| the same, steady (segments 20 to 60) | (122k / 808 ticks a game) | 122k env ticks/s |

The drain is unbiased: the yardstick against itself with fight_speed 1.3, 4v4, scored
0.4656 +- 0.0019 over a whole queue of 65,536 games and 0.4654 +- 0.0034 over the 22,043
games started before a 20 s cutoff and drained (z = 0.07; mean lengths 1,396 and 1,402
ticks). A changed clock (300: longest game 300 ticks) and arena radius (1,200 m: mean
length 1,115 +- 24 compiled, 1,074 +- 25 eager, 1,024 games each) act through the compiled
tick.

A run's drain lasts up to its longest game (the clock), with slots falling idle, so a short
queue wastes most of the GPU: hence at least 16 games per slot. Compile: the scripted tick's
first call 66 s cold (with 512 games of play), 23 s from a warm cache; the network tick
about 110 s cold. Calling the network through torch.func.functional_call (so its weights
are inputs) costs 9% against policy.step directly (122k against 133k env ticks/s).
