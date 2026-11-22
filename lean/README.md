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

Run on the CPU, one thread: `OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m pytest lean -q`
gives **36 passed in 39 s**, and `python -m lean.train --smoke` completes (two updates
against the real packages, then a crossover; all losses finite). **Nothing has run on a
GPU.** The rest is what each builder reported.

| Package | Checked on the CPU | Not checked, or known to be unfinished |
|---|---|---|
| `game/` | **Matches v2 to float32 rounding:** one step from each v2 state over 32 environments, 5v5, 150 ticks (13,553 shots, 169 deaths): position within 6.1e-5 m, speed 1.5e-5 m/s, angle of attack 8.2e-8 rad, no mismatch in deaths, shots, killers, bullet lifetimes or cooldowns; open-loop positions agree to 1.8e-4 m after 150 ticks. Invariants pass at 3v5 and 20v20. **1,312 operations per step** (flight 929, the rest 383) | No game ended in the comparison, so the result coding and the clock rule are uncompared. No run with fields. The "sides" spawn untested. A ship that dies keeps its speed and nose rate (v2 resets them); the comparison excludes ships dying that tick. `reset_done` costs a third of a step on every tick |
| `scripted/` | Six tests. Beats the chaser 43 to 1 over 48 games at 5v5; 166 operations per call without memory | Not played against v2's Imugi. Gun range, corner speed and launch speed use one ship type's values for all. Persona spreads are guesses and unplayed. The caller must pass the arena radius |
| `model/` | Fourteen tests: rollout and learner forms agree; padding and permutation leave real ships unchanged; runs at 5v5, 20v20 and 50v50; every parameter gets a gradient and masked rows contribute none; fits a fixed batch (cross-entropy 3.74 to 0.08 in 60 steps); `torch.compile` works and matches eager, with one recompile at the first new team size by default and none with `dynamic=True`; bf16 runs with position phases and the recurrence pinned to float32. 616,677 parameters; 289 operations per forward | The time loop is unrolled, so a new segment length will recompile. No compiled backward pass. The temporal block is the recurrent unit only, not Griffin's full block (`model/NOTES.md` says what adopting it would change). Next-state scales are rough |
| `learn/`, `eval/`, `train.py` | Twelve tests. The next-state loss is live. Value targets use the model's helpers with an explicit support per level. The random position offset is drawn per match and stored with the rollout. `--device` is wired. Collect plus update run at 150 to 210 steps per second on one CPU thread at smoke size | `--device cuda` untested. The damage level's value support is unmeasured and the kill level's needs widening for larger teams. `eval` passes no position offset. Unequal teams not run with the real game |

Set `OMP_NUM_THREADS=1` for small CPU runs. With default threads on a loaded machine they
are many times slower, which is what made the first smoke run look like 3 to 5 steps per
second.

Departures from v1's recipe that need the author's eye before a real run are listed in
`learn/NOTES.md`: how kill and damage rewards are shared, a uniform snapshot pool in place
of rating-based sampling, one team per environment training, the BC target and its gate,
and no KL stop or weight averaging.

## When the laptop is back, in this order

Run from the repository root with `PYTHONPATH=.:src`.

1. **Read each `NOTES.md`.** They say what was run, what was not, and what each builder
   had to assume about the others.
2. **Re-run everything on the laptop's CPU, then on the GPU:**
   `OMP_NUM_THREADS=1 python -m pytest lean -q`, `python -m lean.train --smoke`, then the
   same smoke run with `--device cuda`.
3. **Finish the framework benchmark** (gates G1 and G2):
   `docs/rewrite/sections/framework-spike/run_laptop.sh`. The first repeat of the
   compile and team-size suites is in `results/laptop/`; steady speed, the whole-loop
   rollout, latency and repeats 2 to 5 are still to run. Set JAX's matrix-multiply
   precision to highest so the two frameworks compute the same thing.
4. **Finish checking the game against v2:** the one-step and open-loop comparisons pass
   (`game/tests/test_against_v2.py`). Still to compare: games that end, the clock rule,
   fields, and the distribution checks in `docs/rewrite/sections/game-audit.md` section
   5. Then fly it.
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
