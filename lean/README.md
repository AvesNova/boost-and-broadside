# Lean core: first cut

Written on 2026-11-22 on a CPU-only machine, by four builders working in parallel against
`CONTRACT.md`, before the framework gate (G1) closed. **Nothing here has run on a GPU,
and the packages have not been run together beyond what each `NOTES.md` says.** Treat it
as a draft to finish, not a working system.

| Package | What it is | Its notes |
|---|---|---|
| `contract.py` | The shared types: `State`, `StepOut`, `Obs`, the action helpers | |
| `game/` | The game step: elimination with no shrink on one fixed map, any team sizes | `game/NOTES.md` |
| `scripted/` | The simplified scripted opponent, its personas, and a weaker chaser | `scripted/NOTES.md` |
| `model/` | The policy: masked attention with rotary position, a Griffin recurrent block per token, categorical heads | `model/NOTES.md` |
| `learn/`, `eval/`, `train.py` | Rollout, PPO with a self-switching BC term, the crossover evaluation, a smoke entry point | `learn/NOTES.md` |

It is in PyTorch because v2's flight lifts into it nearly verbatim and the partial laptop
results did not favour either framework clearly. If G1 chooses JAX, the game step and the
model translate; `docs/rewrite/sections/framework-spike/` has a parity-tested twin of the
trunk.

## When the laptop is back, in this order

Run from the repository root with `PYTHONPATH=.:src`.

1. **Read each `NOTES.md`.** They say what was run, what was not, and what each builder
   had to assume about the others.
2. **Make the packages work together on the CPU.**
   `python -m pytest lean -q`, then `python lean/train.py --smoke`. Fix interface
   mismatches first; they are the most likely failure, since the packages were written
   at the same time.
3. **Finish the framework benchmark** (gates G1 and G2):
   `docs/rewrite/sections/framework-spike/run_laptop.sh`. The first repeat of the
   compile and team-size suites is in `results/laptop/`; steady speed, the whole-loop
   rollout, latency and repeats 2 to 5 are still to run. Set JAX's matrix-multiply
   precision to highest so the two frameworks compute the same thing.
4. **Check the game against v2:** the trajectory comparison in `game/tests/`, following
   `docs/rewrite/sections/game-audit.md` section 5. Then fly it.
5. **Measure the scoreboard's first numbers:** operations per tick for the game step, per
   forward for the model and per call for the scripted opponent, against the tripwires
   in `docs/rewrite/core-brief.md` section 3; steps per second of the smoke run;
   uncompiled against compiled.
6. **Play the scripted opponent against v2's Imugi** to get the exchange rate between the
   old yardstick and the new one (`scripted/NOTES.md`).
7. **Only then a training run:** 4v4, v1's recipe (`docs/rewrite/sections/yardstick.md`),
   to the first learning requirement.

## Decisions still with the author

- What stage 1 reproduces: v1's conditions (mixed spawns, a short clock, a simple
  opponent, 4v4 only) on v2's game, then v2's conditions. See the yardstick record.
- Whether bullets become tokens in the observation, as in v1.
- The operation tripwire for the game: v2's flight alone is over it. The game audit
  proposes a split.
- The time box for the skeleton.
