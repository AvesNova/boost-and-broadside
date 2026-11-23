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

## State on 2026-11-23 (read this first)

What `lean` is for and every deliberate change to v2: `docs/rewrite/core-decisions.md`.
Planned work and status: `docs/rewrite/core-backlog.md`. What was done, when and how it was
checked: `docs/rewrite/core-worklog.md`. Questions for the author: `docs/questions.md`.

- **Game** (`lean/game`): `GameConfig()` is the core game (signed off 2026-11-23): the core
  flight model (weight on the wing, the engine driving speed only, a two-loop store, firing
  cutting the engine), the cheaper head-on damage term, ghost bullets with per-bullet
  outcomes, ten fields drawn per environment, fog, the P-51D only. v2's flight and damage
  models are gone (they were checked against v2 first: `docs/rewrite/game-parity.md`).
  Measured: `docs/rewrite/core-flight-measurements.md`, charts `docs/rewrite/*-core.png`.
- **Sandbox** (`python -m lean.sandbox`): fly the game by keyboard.
- **Model** (`lean/model`): v2's model maths with the decided changes (Griffin block,
  per-kind encoder, fields as attention memory, gated attention, rotary values, 4x MLP, 64
  linear HL-Gauss bins, swappable position code); 2.0M parameters, 509 operations per step.
- **Learner** (`lean/learn`, `lean/learn/NOTES.md`): a BC stage cloning the scripted
  teacher's full distribution (every head and the critics), then RL from its EMA with a
  teacher term that shrinks every update until it switches off; an EMA and a uniform average
  beside the live weights; fused Adam with a 2,000-step warm-up; half self-play, half league
  (the league's routing and schedule are step L2).
- **Scoreboard**: `docs/rewrite/scoreboard.md`.

## Commands

Run from the repository root (`uv run --no-sync` in front of each with the project's venv).

    pytest lean -m "not slow" -q                    # the fast tests (CPU); -m slow: the compiled ones
    python -m lean.train --smoke --device cpu       # BC then RL, a few tiny eager updates
    python -m lean.train                            # BC (96 updates), then RL (--updates)
    python -m lean.train --stage bc --out DIR       # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR   # RL from the BC hand-off
    python -m lean.eval DIR/rl.pt --weights ema --sizes 4         # the crossover against the scripted agent
    python -m lean.sandbox                          # fly the game by keyboard

The sections below describe the first cut (2026-11-22) and are kept for the record.

## State at the end of the session

Run on the CPU, one thread: `OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m pytest lean -q`
gives **36 passed in 39 s**, and `python -m lean.train --smoke` completes (two updates
against the real packages, then a crossover; all losses finite). **Nothing has run on a
GPU.** The rest is what each builder reported.

| Package | Checked on the CPU | Not checked, or known to be unfinished |
|---|---|---|
| `game/` | **Matches v2 to float32 rounding:** one step from each v2 state over 32 environments, 5v5, 150 ticks (13,553 shots, 169 deaths): position within 6.1e-5 m, speed 1.5e-5 m/s, angle of attack 8.2e-8 rad, no mismatch in deaths, shots, killers, bullet lifetimes or cooldowns; open-loop positions agree to 1.8e-4 m after 150 ticks. Invariants pass at 3v5 and 20v20. **1,312 operations per step** (flight 929, the rest 383) | No game ended in the comparison, so the result coding and the clock rule are uncompared. No run with fields. The "sides" spawn untested. A ship that dies keeps its speed and nose rate (v2 resets them); the comparison excludes ships dying that tick. `reset_done` costs a third of a step on every tick |
| `scripted/` | See `scripted/NOTES.md` (step S, 2026-11-23: power rule for the core physics, memory in every caller, head-on side, bounded personas, every number from the game) | Not played against v2's Imugi (decision 5.6: not needed). Persona widths set for competence, not tuned |
| `model/` | Fourteen tests: rollout and learner forms agree; padding and permutation leave real ships unchanged; runs at 5v5, 20v20 and 50v50; every parameter gets a gradient and masked rows contribute none; fits a fixed batch (cross-entropy 3.74 to 0.08 in 60 steps); `torch.compile` works and matches eager, with one recompile at the first new team size by default and none with `dynamic=True`; bf16 runs with position phases and the recurrence pinned to float32. 616,677 parameters; 289 operations per forward | The time loop is unrolled, so a new segment length will recompile. No compiled backward pass. The temporal block is the recurrent unit only, not Griffin's full block (`model/NOTES.md` says what adopting it would change). Next-state scales are rough |
| `learn/`, `eval/`, `train.py` | Twelve tests. The next-state loss is live. Value targets use the model's helpers with an explicit support per level. The random position offset is drawn per match and stored with the rollout. `--device` is wired. Collect plus update run at 150 to 210 steps per second on one CPU thread at smoke size | `--device cuda` untested. The damage level's value support is unmeasured and the kill level's needs widening for larger teams. `eval` passes no position offset. Unequal teams not run with the real game |

**First laptop run (2026-11-22, RTX 4070 Laptop, power profile "performance", eager, no
compile).** `pytest lean -q` on the CPU: 36 passed in 17.5 s. `--smoke --device cpu`: 6.2 s
wall, 234 and 383 steps per second for the two updates, 1.05 GB peak resident memory.
`--smoke --device cuda`: completes with no change to the code, 85 and 209 steps per second
(4 environments, so launch overhead only), peak CUDA memory 25.6 MiB allocated. At the
default size (`--device cuda --updates 3`: 256 environments, 4v4, 128 ticks, collect plus
update) 2,956, 2,923 and 2,882 steps per second, peak 3,576 MiB allocated (4,200 MiB
reserved). That is below v2's floor of about 5,000 before any compilation.

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
