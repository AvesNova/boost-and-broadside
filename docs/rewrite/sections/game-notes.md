# Section 1 (game): notes

Branch `section/game`. The v2 spec (`game-spec.md`) is implemented in `bnb.game`. All 93
tests pass (`uv run --no-sync pytest`). The acceptance tests and goldens are unchanged, and
only the adapter (`tests/reference/race_arm.py`, `race_chaser.py`) was rewritten.

## Design

### Layout

| Module | Job |
|---|---|
| `game/rules.py` | `Rules` (sizes and every shared constant, Python numbers) and `FlightConstants`; `rules_of(GameConfig)` |
| `game/ship_types.py` | `ShipTypes`, the ship card: per-type tensors; `ship_types()` defaults; `per_ship(table, ship_type)` |
| `game/state.py` | `Ships`, `Bullets`, `Fields`, `Match`, `State`, `Sweep` (pytree dataclasses); `blank`, `where`, `apply`; canonical race names |
| `game/match.py` | public API: `rules_of`, `ship_types`, `reset`, `move`, `resolve`, `step`, `perceive`, `spread`, `uniforms`, `start_clocks`, `sweep_like` |
| `physics/numerics.py` | normative small-angle forms, soft-min, smoothstep, `lookup` (constant tables as selects) |
| `physics/medium.py` | fields: ln n and its gradient, as a reduction over the field axis |
| `physics/flight.py` | flight setting, kick, one flight tick; `Airframe` (per-ship card values) |
| `physics/bullets.py` | firing, bullet motion with drag |
| `physics/combat.py` | swept hit test (friendly fire), boundary damage, shields and deaths |
| `physics/spawning.py` | point-symmetric maps, formation ranks, spawns; `TickUniforms` slices |
| `physics/step.py` | `reset`, `move` (phases 1-3), `resolve` (phases 4-8), `step` |
| `perception/sight.py` | visibility (sight lines, per-observer range) |
| `perception/observe.py` | the `Obs` / `Privileged` contract, `last_seen` |
| `modes/deathmatch.py` | the skeleton reward stub |

### State types

- `State(ships, bullets, fields, match)`, with the fields of contracts.md section 2:
  - `Ships` (B, N);
  - `Bullets` (B, N, K): `x y dx dy w logn gx gy ttl`;
  - `Fields` (B, M): `x y radius width level`;
  - `Match`: `t`, `axis_x`, `axis_y`, `last_seen` (B, T, N).
- Each is a frozen dataclass registered with `bnb.contracts.tree.contract`, so the runtime
  maps `copy_` and `clone` over it.
- `seat_of_team` is not added. Team deathmatch does not need it (skeleton note, contracts.md).
- `Sweep` is what `move` hands `resolve`: tick-start positions and this tick's
  displacements of ships and bullets. The hit test sweeps exactly these.
- Updates use `dataclasses.replace`; dynamo traces it with `fullgraph=True`.

### Constants

- **`Rules` is static.** Sizes, tick, soft-min exponent, bands and every shared combat
  value are Python numbers, so compiled graphs specialize on them.
- **The ship card is data.** `ShipTypes` holds the 15 parameters of spec 1.1 as tensors,
  shape (num_types,), or (B, num_types) to vary them per environment.
  - `per_ship` reads a ship's value as a select over the types. The type count is the
    table's static shape.
  - A test checks that new card values do not recompile.
- **The `Preset.flight` trap is gone.** Shared flight constants live only in
  `FlightConstants`; per-type ones live only in the card.
- **Public functions take the card last and optionally:** `move(rules, state, action,
  spread, types=None)`. `None` means the default card, built from host values.
  - Captured or swept code must pass the card. The rollout keeps it in its carry, so a
    sweep can `copy_` new values in.
  - Why last: `tests/contract/test_export.py` calls `match.reset(p, uniforms)`, and I may
    not edit it.
- **Randomness is an explicit input**, the spread normals and the tick uniforms of spec 2.
  - `match.spread` and `match.uniforms` draw them (optionally from a generator).
  - The runtime draws them inside its act stage, as before.

### Compile and stage assumptions

- Every function is pure and works eagerly or under `torch.compile(fullgraph=True,
  dynamic=False)`, at any B, on CPU or GPU.
- Nothing in the game copies from the host during a tick: constant tables are select
  chains (`lookup`, `per_ship`). So eager CUDA-graph capture of the tick works (tested).
- **Stage cuts:** the runtime cuts between `move` and `resolve`; `Sweep` is the carry
  between them. `perceive` is separate.
- **No process-global settings** in `bnb.game`: no TF32, threads, seeds, recompile limits
  or guard settings.

## Decisions and deviations

1. **Medium as a field-axis reduction** (spec 4.2, not normative).
   - Why: the unrolled loop over fields made Inductor split one huge kernel into many,
     each storing dozens of intermediates. At M = 54 the bullet move took 3.1 ms.
   - The new form is sums, one product and a leave-one-out sum.
   - **Deviation:** `grad A = clear * sum_i grad alpha_i / (1 - alpha_i)`. The spec says
     "never divide by 1 - alpha_i".
     - I divide only where grad alpha_i != 0.
     - There, 1 - alpha_i is computed in factored form, `w^3 (6w^2 - 15w + 10)` with
       w = 1 - z, so it is strictly positive even where alpha rounds to 1.
     - The spec's concern (0/0) cannot occur. The goldens pass at their 2e-6 tolerance.
2. **Damage matrix and killer by scatter**, O(bullets) instead of O(bullets x ships).
   - The earliest hit is an exact `amin` over an int64 key (tau's float bits, then the
     bullet index), so it is deterministic.
   - `damage` uses `scatter_add`. Its summation order can matter only when three or more
     of one shooter's bullets hit one target in the same tick. That is a negligible
     repeatability cost, within the brief's trade.
3. **Formation phase:** `angle = 2*pi*frac(k/S + psi)`. `frac` keeps float32 rounding to
   about 1e-6 rad; the result equals the spec's angle mod 2*pi.
4. **Stub reward under friendly fire** (`modes/deathmatch.py`).
   - Ally damage and ally kills now count against the shooter.
   - Without this, the learner was rewarded for shooting teammates, and deaths rose
     during training.
5. **Sight test:** the race's loop over cores with bitmask membership is kept.
   - A field-axis reduction like the medium's was measured 1.3-2.5x slower: it recomputes
     both endpoints' membership per pair.
   - The loop stays small for M <= 64 (no kernel splitting at 54 fields).
   - **Vision range is per observer** (`vision_range(rules, state)`, a (B, N) tensor).
     Today it is filled with the shared V.
     - When the ship card adds a per-type range, that function becomes a `per_ship`
       lookup.
     - No fast path assumes a single shared V (coordinator's request).
6. **The skeleton's names are kept where an uneditable test uses them.**
   - `match.preset` is an alias of `rules_of`, and `Rules.num_uniforms` is the tick
     uniform count (`tests/contract/test_export.py`).
   - `State.__getitem__` reads by the race's flat names, read-only
     (`tests/loop/test_lineup.py` reads `carry["state"]["w"]`). Game code never uses it.
     The adapter uses `state.flat` and `unflat`.
7. **The game's `presets.json` is deleted.** Defaults are Python dataclass defaults. A
   section test checks them against `tests/reference/golden/presets.json`.

## Benchmark

`uv run --no-sync python -m bench.run` (seed 20261031); the baseline is commit 4cc7c13c.
The final run is at commit cc49b950 plus the notes (no code changes after it).

| cell | baseline rollout env-ticks/s | after | baseline train | after | VRAM before / after |
|---|---:|---:|---:|---:|---:|
| 114 ships, 22 fields, 360 envs | 70,904 | 111,834 (+58%) | 17,438 | 19,199 | 4.44 / 3.61 GB |
| 40 ships, 8 fields, 1024 envs | 406,851 | 438,961 (+8%) | 56,458 | 57,215 | 4.43 / 3.60 GB |
| 26 ships, 54 fields, 1576 envs | 322,968 | 433,183 (+34%) | 75,825 | 80,715 | 4.77 / 3.94 GB |
| realtime, CPU, 10 ships | p50 0.152 ms, p99 0.168 | p50 0.16, p99 0.18 | | | |

- The after column includes the v2 costs: about twice the hit pairs (friendly fire) and
  drag.
- **Where the rollout time goes now** (114-ship cell, per tick, measured separately):
  - controllers' act stage about 2.2 ms (MLP and scripted, not this section);
  - game about 1.2 ms: move 0.34, resolve 0.54 (hit test 0.34), perceive 0.35-0.42.
- **Training throughput** is dominated by the PPO update (about 2 s per update), which is
  outside this section.
- Realtime is about 5% slower at p50: the B = 1 tick now runs friendly fire, drag and the
  formation ranks. I did not tune it further.
- **What gave the speed:**
  - the field-axis medium (6.6 → 1.4 ms of game time per tick at 54 fields);
  - the scattered damage and killer;
  - card values as selects instead of gathers;
  - the hit test reading displacements.

## Runtime (mechanical changes only)

- **`rollout.py`:**
  - `match.rules_of`;
  - the card in the carry (`carry["types"]`);
  - `uniforms` (B, U) replaces the precomputed `fresh` map;
  - `sweep` (from `match.sweep_like`) replaces `origin`;
  - `Action.unpack` of the packed actions for `move`;
  - `match.start_clocks` for the staggered clocks;
  - `last_seen` copied back through `state.match`.
- **`realtime.py`:** the same API changes; the card is held on the object and passed to
  the compiled tick.
- **`train.py`:** `rollout.p.h` became `rollout.p.tick`.

## Requests and open questions

1. **`tests/loop/test_learning.py` is marginal under v2 rules.**
   - It passes at seed 0. Over seeds 0, 1, 2 the improvement was +0.0156, +0.0156 and
     +0.0044 (the threshold is 0.01).
   - Why: v2 combat (300 J shots, about 186 m range with drag) makes the chaser much less
     lethal. The early reward is about -0.02/s instead of -0.033/s, so there is less to
     learn.
   - Request: the learning section re-tunes this test for v2 (more updates, or a different
     margin).
   - The outcome also changes with any change to compiled numerics, so unrelated game
     edits can flip it.
2. **Eager CUDA-graph capture of the whole rollout fails in `controllers/scripted.py`.**
   - Cause: `torch.tensor([...], device=cuda)`, a host copy.
   - The game side is capture-safe and tested. The controllers section should apply the
     same `lookup` pattern.
3. **`tests/contract/test_export.py` and `tests/loop/test_lineup.py` use the skeleton API**
   (`match.preset`, `match.reset(p, u)`, `carry["state"]["w"]`).
   - Request: move them to `rules_of` and `state.ships.w`. Then the `preset` alias and
     `State.__getitem__` can go.
4. **Contract table, `Match.seat_of_team`:** not implemented; no mode needs it yet.
5. **Contract text, `ShipTypes`:** contracts.md lists it under `GameConfig`'s module. It
   lives in `bnb.game.ship_types`, since `contracts/` was not mine to change.
   - Request: point the contract at it, and note the (B, num_types) per-environment form.
6. **Per-type vision range** (coordinator): ready as described in decision 5.
7. **Possible further speed, not done:**
   - Spawn-point medium: every tick, every ship's would-be spawn point is evaluated
     against the fields (0.13 ms at 54 fields). The spawning set is known at tick start,
     so it could share the flight's medium evaluation, at the cost of coupling phases.
   - Bullet motion and the hit test still run over all N·K slots, though about 5% are
     active mid-game.
   - The observation write (about 70-90 µs) could be tuned.
8. **`bench/results.jsonl`:** the benchmark runs append to it, and I committed those
   appended lines as the record. Nothing else under `bench/` changed.

## Size

Lines per module (`src/bnb/game`, 1269 total; the race arm it replaces was about 1030 plus
`presets.json`):

| module | lines |
|---|---:|
| state.py | 190 |
| physics/flight.py | 181 |
| physics/step.py | 111 |
| rules.py | 105 |
| physics/combat.py | 104 |
| perception/sight.py | 94 |
| physics/spawning.py | 83 |
| ship_types.py | 82 |
| perception/observe.py | 81 |
| physics/bullets.py | 74 |
| physics/medium.py | 53 |
| physics/numerics.py | 45 |
| match.py | 41 |
| modes/deathmatch.py | 25 |

New tests: `tests/sections/test_sight.py` (2 tests) and `test_game_section.py` (8 tests).
`test_perception.py` was ported, and its spawn-leak xfail is removed and now passes.

## Fix round (after the blind review)

The spec amendment 3f583fb0 is pulled: the guarded division in the medium gradient is now
allowed, so decision 1's deviation is gone. 114 tests pass (`uv run --no-sync pytest`,
GPU and slow tests included).

1. **Killer tie-break at tau = 0.**
   - The bug: an exact -0.0 survived the eager clamp, and its int32 bit pattern sorted
     before +0.0. Compiled code flushed it, so eager and compiled disagreed.
   - The fix: `.abs()` after the clamp. It is identity for tau >= 0 and, unlike `+ 0.0`,
     a compiler cannot fold it away.
   - Test: `test_killer_ties_at_tau_zero_go_to_the_lower_bullet_eager_and_compiled`, both
     orders.
2. **Medium gradient at a field's exact centre.**
   - The distance is a double-where sqrt, so autograd is finite everywhere.
   - Tests: the medium against a float64 leave-one-out reference (inner band edge down to
     1e-6 m, 64 coincident fields, nested, 64 staggered, exact centre, battle-scale
     positions), and finite autograd through points and fields for every case.
3. **The ship card is required:** `reset(rules, types, uniforms, fogged=None)`,
   `move(rules, types, ...)`, `resolve(rules, types, ...)`, `step(rules, types, ...)`.
   Every caller passes it.
4. **No legacy left in src:**
   - the `match.preset` alias, `State.__getitem__`, and the canonical, flat and unflat
     helpers are deleted;
   - the race-key translation (state, constants, observation) lives in
     `tests/reference/race_arm.py`;
   - `Visibility` is gone: `visibility()` returns the (B, 2, N) tensor;
   - `tests/contract/test_export.py` and `tests/loop/test_lineup.py` use the new API, and
     `test_contracts.py` the new column count;
   - the card test builds per-environment cards with `ship_types(**overrides)`. A
     (B, types) override gives every parameter the batch axis.
5. **Two teams is a constant,** `rules.TEAMS`. `Rules.num_teams` and the team arguments
   are gone. `view_order`, `team_relative` and the uniform layout stay written as they
   are.
6. **Full-word names everywhere in bnb.game**, including the state fields, the constants
   and the card. Examples: `heading_cos`, `proper_speed`, `angle_of_attack_sin`,
   `log_index_gradient_x`, `moves_left`, `next_slot`, `max_lift_coefficient`,
   `ticks_between_shots`, `tick_length`.
   - Spec symbols are kept in comments, e.g. `# [alpha_s]`.
   - `bnb.game.match` is now `bnb.game.api` (no clash with `Match`).
   - Kept short: shape letters (B, N, K, M, S, P), x and y, cos and sin, the spec's matrix
     entries (m11 ...) and theta.
   - `Rollout.p` keeps its name because `bench/cell.py` reads `rollout.p.episode_ticks`.
7. **Observation columns** are as listed by the coordinator (`contracts/obs.py`:
   SHIP_HIDDEN_ZERO = 16, 24 ship columns, 5 field columns, 4 globals; times in
   seconds).
   - The MLP's scales and the chaser follow.
   - The adapter maps back to the race's 17 columns and 4 field columns.
8. **Fog of war per environment.**
   - `Match.fogged` (B,) bool is set at reset (default: all fogged) and kept through
     episode resets.
   - Without fog, every living enemy is visible, `time_since_seen` is 0 for every ship,
     and `globals.fogged` is 0.
   - `Pool.fog_share` (default 1.0) must give a whole number of environments.
     `pool_sizes` fails otherwise.
   - **The split is fixed and contiguous:** the first `fog_share` of each pool's
     environments are fogged for the whole run. The seat is redrawn each episode, so
     neither side gains from it.
   - Tests cover the unfogged views, the flag in the observation, fog surviving episode
     resets, the rollout's split, and whole-number validation.
9. **New test sizes:** compiled vs eager at odd M (5 fields), M = 64 and battle size
   (100 ships, 40 fields, R 8200 m).
   - Floats agree to rtol and atol 2e-5; flags, counters and indices are compared exactly.
   - The batch-1 `Realtime` path is tested eagerly and compiled, fogged and not, across an
     episode reset.
10. **Sight scaling** at fixed size (40 ships, 1024 environments, RTX 4070 laptop, bench
    lock):

    | M | compile | perceive per tick |
    |---:|---:|---:|
    | 8 | 5.1 s | 115 µs |
    | 16 | 7.1 s | 132 µs |
    | 32 | 9.5 s | 161 µs |
    | 48 | 17.8 s | 203 µs |
    | 64 | 19.0 s | 228 µs |

    Both grow about linearly with M, with no kernel splitting, so nothing is changed. The
    field-axis reduction alternative was measured slower earlier (decision 5).
11. **Costs to know about.**
    - **Eager hit-test memory.** Eagerly, the hit test materializes its (B, N·K, N) pair
      tensors: about 10 float or bool tensors of B·N·K·N elements. At battle size with
      B = 1024 that is 61 M elements each, roughly 3 GB at peak. Compiled, the pair tensors
      fuse into one reduction and are never stored. Eager battle-size runs should use a
      smaller B.
    - **Per-tick spawn work.** The spawn formation is computed for every ship every tick
      and selected by the spawn mask: the formation ranks (B, 2, S, S), the new map for
      every environment, and the medium at every would-be spawn point (B, N, M). Most of
      it is discarded, so its cost is constant per tick rather than per spawn: about
      0.13 ms at 54 fields and 1576 environments. Making it conditional would need a
      data-dependent shape, or sharing the flight's medium evaluation (open item 7).

**Benchmark after the fix round** (same seed):

| cell | rollout env-ticks/s | VRAM |
|---|---:|---:|
| 114 ships, 22 fields | 110,344 (first round 111,834) | 3.86 GB |
| 40 ships, 8 fields | 439,236 (first round 438,961) | 3.85 GB |
| 26 ships, 54 fields | 425,337 (first round 433,183) | 4.27 GB |
| realtime | p50 0.16 ms, p99 0.17 ms | |

- Rollout throughput is unchanged within run-to-run noise.
- VRAM is about 0.25-0.3 GB higher than the first round, from the wider observation
  (19 → 24 ship columns) in the stored trajectory and the larger MLP input.

**Still open:**
- **The learning test passes** at its fixed seed with the new observation, but stays
  marginal across seeds (request 1).
- **The scripted Random controller** still makes a host copy that blocks eager capture of
  the whole rollout (request 2, controllers section).

**Size after the fix round:** `src/bnb/game` is 1322 lines. Full-word names made the
lines longer; the module list is unchanged except `match.py` → `api.py`.
