# Frontline shield overhaul — September 18, 2026

Branch: `feat/frontline-shields-5v5`, based on `feat/frontline-overhaul` at `85f2bf3`.
The branch is prepared for review; it has not been merged or pushed.

## Supported rules and decisions

- Play, RL and BC use 5v5. Zones are opaque, with the existing same-core sight
  exemption and successful-shot reveal. Optical fields still bend movement and
  projectiles, and their cores still block sight.
- Frontline's shared `health` channel now means shield level (100 capacity).
  Keeping this resource name avoids duplicating the legacy elimination resource
  pipeline. There is no second hidden hull pool. Zero shield is a vulnerable living
  ship, not a dead ship.
- Projectile impacts are aggregated each physics tick. An enemy hit can kill only
  if shields were already zero before that tick's combat resolution. Breaking a
  shield cannot also kill, regardless of simultaneous hit count. Friendly fire
  depletes shields and delays recovery, but cannot deliver the finishing hit.
- Damage restarts a five-second delay; undamaged ships then recharge at 15/s anywhere.
  Fractional delay expiry and the capacity cap are respected. Same-tick damage blocks
  recharge. Spawn protection masks damage before attribution and does not reset delay.
- Respawns are instant, with 15 shield, 20 power, 30 proper speed, no firing cooldown,
  and the full recharge delay. Initial placement uses the same state: a single
  supported lifecycle rule, immediate steering/shooting, and no exceptional opening
  resource advantage. Statistical playtests showed captures and repeat engagements
  with these values; they have not been exhaustively optimized.
- Ships are invulnerable within their own current spawn. Collision-time membership
  follows movement; capture rotation changes the protection for subsequent collisions.
  Respawns use the newly current spawn.
- Defense-zone damage was removed along with the requested spawn and field hazards.
  Keeping attrition on the objective would prevent natural recovery while defending
  and undermine the intended positioning game. Only the soft outer boundary remains
  hazardous; it also respects shield-break protection and own-spawn invulnerability.

## Reward and learning contract

Actual shield loss, capped before attribution, drives damage rewards. The separate
raw impact matrix still attributes finishing shots at zero shield. Recharge pays the
recovering ship and charges the opposing team by the same total, including unequal
team sizes. Boundary penalties and friendly-fire blame have corresponding opposing
payouts, so environmental damage/recovery cycles cannot mint team reward. Kill and
assist attribution survives the lethal tick and is cleared across respawn lives.

The offensive premium stays at 2:1 through 50M environment decisions, declines
linearly to 1:1 at 300M, then stays there for the final 200M of the 500M default run.
Unpaired shaping also reaches zero at 300M. Kill, damage and capture ratios update
in every primary/auxiliary rollout environment and on checkpoint resume. Shortened
custom runs need correspondingly shortened schedule keypoints.

`shield_delay` is visible/predicted only under the normal observation masks. Hidden
zero-shield enemies remain alive in belief and imagined rollouts. Respawn prediction
labels remain discontinuous while recurrent match memory persists. Fourier attitude
features now encode the actual `atan2` angle; unit-circle prediction targets are retained.
The observation/feature schema is `frontline_shields_v9`; old policy checkpoints are
incompatible and need retraining.

The 26-token default rollout geometry preserves 3,840 environment streams across
four shards of 960, each with a 128-decision horizon: 491,520 environment decisions
per logical update. VRAM launch sizing and checkpoint feature fingerprints were
updated accordingly.

## Removed machinery

Removed field-damage levels/material generation, field-damage observations/features,
interface damage integration and alpha caches, mutable per-projectile damage and its
observation channel, spawn healing, enemy-spawn/defense passive damage, their reward
components/statistics/configuration, and obsolete fixtures/tests. Bullet impact damage
now comes from the configured scalar plus the retained incidence calculation. Optical
index and gradient state remains because refraction still uses it. Renderer field
patterns no longer imply damage; spawn labels say SAFE.

## Validation

Focused tests cover simultaneous overkill, later finishing shots, friendly fire,
protected-spawn hits, recharge delay/cap, boundary interactions, initial/respawn
resources, unequal-team recharge accounting, final reward balance, schedule propagation,
attitude phase, and zero-shield belief/imagination. Existing tests exercise masked
observations, opaque-zone sight/rendering, capture rotation, attribution, recurrent
lifecycle, checkpoint contracts, and launch sizing.

The actual PPO and BC training integration selection passed (13 tests). Both revisions
also completed four eager PPO updates with finite losses in the timing experiment below.
The latest `.venv/bin/pytest -q -n 8` run passed: **1,491 passed, 12 skipped** in
163.73 seconds. An earlier full run exposed a stale CLI shard-count expectation;
it was corrected, the affected 136 CLI/VRAM/belief tests passed, and the full suite
was rerun successfully. Ruff check/format and whitespace validation passed.
CUDA was unavailable; no GPU or full-budget learning run was attempted.

## Performance

Same Python environment (PyTorch 2.13.0+cu130), CPU, one Torch thread, 32 environments.
Simulation rows include movement, fields, bullets, collision, objectives and lifecycle.
Full-path rows additionally compute scripted actions, observation assembly and **bullet
as well as ship visibility**; the default training policy does not request bullet sight.
Each median uses three 50-tick blocks following one warmup block. Revisions were run
sequentially. Shared-host timing varied; inspect the raw ranges before drawing conclusions.

| Configuration | Simulation ms/batch tick | Full scripted/visibility ms | Tensor state bytes/batch |
|---|---:|---:|---:|
| Before, 4v4 | 32.83 | 75.48 | 264,256 |
| Before, 5v5 | 31.92 | 59.89* | 332,224 |
| After, 4v4 | 24.54 | 93.44 | 136,960 |
| After, 5v5 | 31.82 | 126.64 | 173,504 |

*The baseline 5v5 full-path samples ranged from 26.90 to 102.09 ms and are too noisy
for a reliable speedup ratio. The matching 5v5 tensor footprint shrank **47.8%**, and
current 5v5 state is **34.3% smaller** than old 4v4. The full perception path pays for
five additional opaque cores and more ship/projectile sight lines; simplification does
not make the whole observed game faster. Isolated GPU profiling remains future work.

The small eager CPU training benchmark uses eight environments, sixteen rollout steps,
four updates, a 32-wide one-block model, seed 77, and compilation disabled for both:

| Revision | Fleet | 512 environment decisions | Decisions/s |
|---|---|---:|---:|
| Before | 4v4 | 14.70 s | 34.84 |
| After | 5v5 | 14.54 s | 35.22 |

Treat these as comparable smoke timings, not a claim of a 1% speedup or production
throughput. They include rollout/teacher work, auxiliary prediction, optimization,
small evaluations and checkpoint I/O; model/trainer construction is outside the timer.

Raw artifacts: [before](shield-overhaul-before.json), [after](shield-overhaul-after.json),
[training](shield-overhaul-training.json). Reproduce against a clean baseline archive:

```bash
mkdir -p /tmp/frontline-baseline
git archive 85f2bf3 | tar -x -C /tmp/frontline-baseline
PYTHONPATH=/tmp/frontline-baseline/src .venv/bin/python benchmarks/frontline_overhaul_throughput.py --output /tmp/before.json
PYTHONPATH=src .venv/bin/python benchmarks/frontline_overhaul_throughput.py --output /tmp/after.json
PYTHONPATH=/tmp/frontline-baseline/src .venv/bin/python benchmarks/frontline_training_smoke.py 8
PYTHONPATH=src .venv/bin/python benchmarks/frontline_training_smoke.py 10
```

## Scripted teacher and playtest

Depleted living ships retain tactical value through a 0.2 strength floor. Low shields
trigger retreat only under visible local pressure. Retreat points away from the enemy
with a small homeward bias; safe ships keep advancing while recharging. Recovery can
override close-range turn/power behavior while retaining the dogfighter's shooting.
This avoids unnecessary trips to a spawn that no longer heals specially.

Four side-balanced 120-second games per threshold, seed 9751, against the prior
controller (`85f2bf3`) running in the **new** environment:

| Recovery threshold | W/L/D | Captures, candidate:prior | Combat deaths, candidate:prior |
|---|---|---|---|
| 0.3 | 3/0/1 | 7:3 | 87:90 |
| **0.5 retained** | **4/0/0** | **10:2** | **63:72** |
| 0.7 | 4/0/0 | 7:3 | 86:76 |

The middle value converted more captures with fewer deaths than either alternative
in this screen. It is a provisional choice, not statistically established superiority.
[Raw tuning results](shield-overhaul-agent-tuning.json) and the pinned-source
`benchmarks/frontline_shield_agent_tuning.py` make the experiment reproducible.

Eight independent 5v5 self-play matches, seed 20260908, maximum 300 seconds:
all reached first and second capture; five ended at the front threshold and three at
timeout; Team 0/1 won 3/5, with no draws. Means: 8.25 captures, 71.625 combat respawns,
zero boundary deaths, first capture at 36.77 seconds, duration 202.56 seconds. See
[raw match results](shield-overhaul-playtest.json). [Offscreen renderer snapshots](shield-overhaul-playtest.png) at
0/30/60 seconds were inspected for depleted starts, movement and readable zone roles.
No interactive human playtest was performed.

### Defensive-response and pacing follow-up

Visual review found that the teacher rarely entered its own defense. This was a
controller limitation rather than a tuned choice: the shield-era sweep varied only
the recovery threshold, while the inherited aggression formula made baseline offense
7.4 times as attractive as defense and ignored capture progress. Defense now keeps a
one-ship-equivalent demand floor; nonzero enemy capture progress adds immediate urgency
even when zone opacity hides the attacker. The scripted suite reports quiet and
threatened defensive occupancy so future changes measure physical presence rather than
only an internal demand score.

The same review found every pacing constant too quick. Play and training now use a
five-second recharge delay, 15 shield/second recharge, and ten-second one-ship capture,
each a 25% move from the prior 4 s, 20/s and 8 s values. These remain provisional and
need another human playtest; the automated match screen checks that slower pacing does
not stop captures or produce universal timeouts.

In four seeded 5v5 self-play matches, a friendly ship occupied its defense for 18.2%
of quiet defense-team ticks and 50.8% of ticks with enemy capture progress. Every match
reached at least two captures and the mean was 4.5, but all four reached the five-minute
limit (two decided by front position, two level draws). Mean first capture was 129.6 s.
This is evidence that defensive response exists, but it is a small, symmetric sample and
the combined defense/pacing change is substantially more conservative than the prior
teacher. [Raw follow-up results](shield-overhaul-defense-retune.json) retain per-game
occupancy counters.

## Remaining work and uncertainty

GPU throughput/VRAM verification, long training convergence, broader seed/opponent
sweeps and human play remain unmeasured. The five-second delay, 15/s recharge,
ten-second capture and 15/20/30 spawn resources are coherent provisional values, not a
completed balance search. The small teacher sweep does not establish an Elo rating. Older learned-policy
results elsewhere in the repository describe earlier mechanics and are historical.

## Commits

- `318a4e9` — rebuild Frontline combat around shields and 5v5; remove old machinery.
- `edcf11f` — adapt the teacher and record tuning/statistical match evaluations.
- `04315a1` — finish belief, curriculum, renderer and regression-test integration.
- The final documentation/validation commit records this report, benchmark scripts,
  raw performance artifacts and the corrected CLI shard-count expectation.
