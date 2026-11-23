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
seats). New scripted vectors, other networks of the same architecture, team sizes per game,
E and the queue's length are data: no new graph. Every GameConfig field (arena radius, spawn
rule, clock, fog, field count, lineup) is a constant of the graph: a new value compiles anew
(Dynamo's guard), as does another network architecture.
