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

## State at the end of the session

What each builder reported. None of it has been re-checked by anyone else.

| Package | Ran on the CPU | Known to be wrong or unfinished |
|---|---|---|
| `game/` | Invariants pass at 3v5 and 20v20 (no non-finite values; non-existent slots never fire, take damage or get attended) | **Never compared with v2:** the trajectory harness did not complete a run. The team-swap test had one assertion changed and was not re-run. No operation count, no fields, the "sides" spawn untested. A dead ship keeps its speed (v2 zeroes it). The 20v20 test takes 100 s |
| `scripted/` | Six tests pass. Beats the chaser 43 to 1 over 48 games at 5v5; 166 operations per call without memory | Not played against v2's Imugi. Gun range, corner speed and launch speed use one ship type's values for all. Persona spreads are guesses and unplayed. The caller must pass the arena radius |
| `model/` | Ten tests pass: rollout and learner forms agree; padding and permutation leave real ships unchanged; runs at 5v5, 20v20 and 50v50; 616,677 parameters; 289 operations per forward | No backward pass, compile or bf16 run. Value support is a placeholder of -2 to 2. Next-state scales are guesses. The temporal block is the recurrent unit only, not Griffin's full block |
| `learn/`, `eval/`, `train.py` | Ten tests pass. `python -m lean.train --smoke` completes against the real packages: two updates, then a crossover | **The next-state loss is silently zero** (the learner and the model disagree on the target). **About 3 to 5 steps per second on the CPU,** not profiled. Its value helpers are a local copy. The device is hard-wired to the CPU. The model's random position offset is never passed |

Set `OMP_NUM_THREADS=1` for small CPU runs; with default threads on a loaded machine
they are much slower.

Departures from v1's recipe that need the author's eye before a real run are listed in
`learn/NOTES.md`: how kill and damage rewards are shared, a uniform snapshot pool in place
of rating-based sampling, one team per environment training, the BC target and its gate,
and no KL stop or weight averaging.

## When the laptop is back, in this order

Run from the repository root with `PYTHONPATH=.:src`.

1. **Read each `NOTES.md`.** They say what was run, what was not, and what each builder
   had to assume about the others.
2. **Fix the two known faults, then re-run everything on the CPU.** Record the model's
   own next-state target in the rollout so that loss is live; find where the smoke run's
   time goes. Then `python -m pytest lean -q` and `python -m lean.train --smoke`.
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
