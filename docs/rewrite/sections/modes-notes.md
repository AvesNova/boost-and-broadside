# Sections 3 and 3b (modes and rewards): notes

Branch `section/modes`. The modes spec (`modes-spec.md`) is implemented in `bnb.game.modes`,
the rewards spec (`rewards-spec.md`) in `bnb.learn.rewards`. All tests pass
(`uv run --no-sync pytest`: 264 passed, none skipped, the 13 rewards equivalence tests
included), and `ruff check` is clean. The fixed tests and goldens are unchanged.

## Design

### Layout

| Module | Job |
|---|---|
| `game/modes/rules.py` | `ModeRules` (static, carried by `Rules.mode`), each mode's clock, the training presets |
| `game/modes/phase.py` | the mode phase; the result at the clock; spawn protection; a new match's mode state and map |
| `game/modes/frontline.py` | v1's Frontline rules (roles, terminal distances, harmonic, membership, the meters and their ledgers, the meter-movement events, the timeout); the v2 ring and mirrored fields |
| `game/modes/spawns.py` | mode-owned placement: team deathmatch (uniform and grouped), Frontline, sandbox (game spec 8's formation) |
| `game/modes/observation.py` | the zone rows and the mode's global columns, per view |
| `learn/rewards.py` | the five levels, `RewardConfig`, the ledger, v1's component functions, `reward_step` |

The skeleton reward stub `game/modes/deathmatch.py` is deleted.

### Configuration and rules

- `GameConfig` gains `mode` (default `"team_deathmatch"`), `spawn_groups` and
  `kills_per_ship` (5). `episode_s` and `respawn_s` default to `None`: the mode's clock
  (team deathmatch 300 s and 5 s, Frontline 300 s and 8 s, sandbox 60 s and 8 s).
- `rules_of` builds `Rules.mode`, a frozen `ModeRules` of Python numbers: K = 8, the
  group and anchor distances, the ring, zone radius, capture time, front threshold and
  the spawn disc. Compiled graphs specialize on it like the rest of `Rules`, and every
  mode hook dispatches on `rules.mode.name` in Python, so a graph holds one mode.
- `modes.rules.PRESETS` holds `deathmatch`, `deathmatch_wide` (40 fields, 8200 m, groups
  (2, 2, 1)) and `frontline`, all 5v5. `WIDE_SHARE = 0.25` records the proposed mix.

### State

`State` gains `mode: ModeState` (pytree, opaque like the rest):
- `score (B, 2)` int32;
- `front (B,)` int32;
- the zones' `zone_x, zone_y, zone_radius (B, Z)`;
- `role (B, Z)` int32, in team 0's frame;
- `progress (B, Z)`;
- `direction (B, Z)` int32, the last push;
- `attack_ledger, defense_ledger (B, Z, N)`;
- `handedness (B,)` (±1) and `ring_rotation (B,)`.

Outside Frontline Z = 0, so these are empty tensors. The reward ledger is not game state:
the runtime carries it (rewards spec 2).

### The tick

`resolve` runs:
1. hits;
2. spawn protection (a hook that is `None` outside Frontline);
3. damage, death and shields;
4. the **mode phase**;
5. the clocks, death and respawn;
6. the episode end.

`done = clock ∨ won`. The result is the win rule's when it ended the match, otherwise
the clock rule's on the state after the phase. It is 0 where not done.

A finished environment gets:
- a new map (Frontline: rotation, handedness, and fields mirrored about the neutral axis);
- a zero mode state, with the ring in Frontline;
- spawns from the mode's placement.

Respawns read the mode state after this tick's captures, so a Frontline ship spawns at its
team's current spawn zone.

`Events` gains `shield_lost` (S⁰ − S after damage, for ships alive at tick start),
`shield_restored` (the recharge applied), `objective_progress`, `objective` and `result`.
`apply_damage` returns them with the ships.

### Spawn draws (the uniform layout)

The tick uniforms are the game's `4M + 1 + 2 + N`, then the mode's (exact functions of
N, K and the groups):

| mode | per environment | per ship |
|---|---|---|
| sandbox | — | — |
| team deathmatch | — | K radii, K angles, heading (2K + 1) |
| grouped | K centre candidates per group (2 each) | 2K + 1, then the anchor (2K + 2) |
| Frontline | ring rotation, handedness | the spawn disc (2) |

- A ship either resets or respawns in a tick, so both rules read the same per-ship
  candidates: the arena disc, the group's 300 m disc, or the anchor's ring.
- **Placement** is computed for every ship every tick and selected by the spawn mask, as
  in the game.
- **Team deathmatch** is one pass for both rules:
  - the candidates are compared with the other team's slots only (team-major: a flip);
  - the target is each enemy's candidate 0 in a resetting environment, else its position
    after the damage phase;
  - an enemy that does not count is moved to infinity;
  - the score is the squared distance to the nearest enemy, `argmax` takes the lowest of
    equal candidates, and with no enemy counting, candidate 0 is taken.
- **Grouped:**
  - the centres are drawn sequentially (G is static);
  - membership is by formation rank;
  - at a reset the rivals are the enemies of the same group;
  - a respawn's anchor is the rank-`⌊u · count⌋` living ship;
  - with no living ship, the uniform rule applies.
- **Frontline:** a uniform point in the disc of 0.35 zone radii around the team's current
  spawn zone, nose toward the zone the team attacks.
- **Sandbox:** game spec 8's formation, unchanged (`physics.spawning.formation`).

### Frontline

`capture_step` is v1's rule exactly (the fixtures): strict-majority meters at
`H(|net|) h / 10 s`, the attempt ledgers (credit present attackers, debit absent
defenders, scale on a fall, clear on empty), and half-presence, half-ledger shares on
completion. A completion zeroes every meter and push, and the roles rotate with the
front.

The mode phase:
- runs membership on the post-move positions of ships alive after the damage phase;
- emits `objective_progress` = held + absent (v1's meter-movement attribution) and
  `objective` = gain − loss;
- wins at |front| ≥ 3.

### Observation

- **Zones,** for each view:
  - x, y, radius;
  - role relative to the viewer (team 1 reads `4 − role`);
  - progress;
  - direction: the push in the viewer's favour (team 0's sign, negated for team 1);
  - the two terminal distances (team 1 reads team 0's swapped).
- **Global columns:**
  - the mode one-hot;
  - the scores by `team_relative`, and the score target (team deathmatch);
  - the front for the viewer, and the threshold (Frontline);
  - zeros where the mode lacks a quantity.

### Yemong

- **Zones are a memory kind.** `Encoder` gains an optional zone projection; the
  constructor's fifth argument and the forward's third stay as they were, so the fixed
  reference arm loads its weights strictly. The memory is the fields then the zones, and
  the rotary tables use both kinds' centres (`Tokens.memory_position`).
- **Zone token (91 inputs):**
  - the position code of the shifted centre;
  - `radius / 1000 m`;
  - the role one-hot (5);
  - progress and direction;
  - `symlog` of both distances.
- **Global token (91 inputs):**
  - the centre's code;
  - episode progress, arena radius / 2600 m, fog;
  - the mode one-hot;
  - both scores divided by the target (0 without one) and `symlog(target)`;
  - the front and its threshold, raw (integers up to 3).
- `obs.zones = None` is read as no zones (a team deathmatch view).
- `YemongConfig.levels` defaults to `REWARD_LEVELS`, with `outcome` as its outcome level.

### Rewards and the ledger

`reward_step(config, ledger (B, N + 1, N), events, team) → (rewards (B, N, 5), ledger')`,
pure, in the rollout's resolve stage. It follows rewards spec 3 step by step:

1. **Applied damage:** what the shield lost, split over raw hits and boundary damage.
2. **Accumulation:** the ship rows by `accumulate_ledger`, the boundary row added.
3. **Kill/death:** `kill_death_events`, on the ship rows.
4. **Recharge attribution:** `attribute_recharge` over all rows, the deficit being the
   column total including the boundary row. Ship rows give `recharge_attributed`; the
   rest is v1's `unattributed`.
5. **Damage:** `damage_events`.
6. **Objective levels:** the mode's events, as they come.
7. **Outcome:** `outcome_events`.
8. **Clearing:** the dead ships' columns are zeroed and done environments' ledgers
   cleared. Rows are kept.

The weights multiply each level's event sum. The component functions are the fixture
interface, so `rewards_arm.py` only renames them. The step passes them the column totals
it already has (keyword-only), because it is memory-bound on its (B, N, N) matrices.

### Runtime

- The rollout carries the ledger, calls `reward_step` in `_resolve`, stores
  `reward (L, R, S, 5)` and adds `zones` to the observation parts.
- PPO sums the levels for its single critic.
- `Trainer.pool_rewards` sums the levels.

## Decisions and deviations

1. **Mode clocks.** `episode_s`/`respawn_s` default to the mode's (spec: 300 s
   default; respawn per mode). Sandbox keeps 60 s and 8 s, the physics game's, so the
   race tests run through an adapter that only selects `mode="sandbox"` (and passes
   sandbox's empty mode state to `State`, which now requires it).
2. **The level names live in `bnb.contracts`** (`REWARD_LEVELS`, next to
   `Trajectory`), and `bnb.learn.rewards` exports them. The spec puts the constant in
   `bnb.learn.rewards`, but Yemong (models, layer 2) defaults to it and may not import
   learn (layer 3, `tests/contract/test_imports.py`).
3. **The harmonic number** is v1's `digamma(n + 1) + γ`.
   - The fixture pins its float32 values (H₀ = −2.4e-7, H₁ = 0.99999988).
   - The meters compute it in their own dtype, which the capture fixture needs in
     float64.
   - H₀ is never used, since meters move only on a strict majority.
4. **Frontline `direction`** is v1's: the majority's sign in team 0's frame, clipped meters
   included, zeroed on a completion (the fixture). The observation shows it in the
   viewer's favour, which reads +1 for a defender whose meter falls: my reading of "+1
   when the meter is moving toward the viewer's capture".
5. **Terminal distances.**
   - The formula is reconstructed from the fixture: with `remaining` captures to the
     end, a zone falls at capture `j ≡ order (mod 5)`.
   - j is taken in the window of five captures starting at `clip(remaining − 4, −4, 1)`.
   - The distance is `remaining − j`.
6. **Roles around the ring** follow the spec and the fixture: by zone at front 0, neutral,
   team-0 spawn, team-0 defense, team-1 defense, team-1 spawn. `modes.md` lists another
   order (request 1).
7. **The ring and the mirror.**
   - The rotation and the handedness come from the mode's uniforms; the spawn-axis
     uniform is unused, as the spec says.
   - An odd last field's signed distance uses its own first uniform, which the
     point-symmetric centre field leaves unused.
8. **Spawn protection** uses the tick-start roles and the post-move position. A
   protected hit is removed from `events.damage` and from the killer; the bullet is spent.
9. **Grouped reset, "restricted to the group"**, is read as the rivals being the same
   group's enemies. A brute-force test pins it, since in practice the groups are too far
   apart for it to matter.
10. **No unheld kill shares.**
    - The ledger half falls back to the raw hits when nothing is outstanding (the
      fixture), so the shares always sum to one.
    - A first version paid `1 − Σ shares` to the opposing team. The mutation check showed
      the term carried only rounding, and it was removed (request 3).
11. **v1's "if none is absent, charge every defender"** and **"all of them, if none is
    outside"** are kept. With equal team sizes they cannot trigger: a strict majority
    leaves the minority side short inside. The fixture interface allows any teams.
12. **Global token scales** beyond "the scores divided by the target" are my choice:
    `symlog(target)`, the front and its threshold raw.
13. **`Events.SCHEMA_VERSION` is 2.** `Obs` stays at 1 because
    `tests/contract/test_contracts.py` asserts it (request 2).
14. **Proposals implemented as written:**
    - team deathmatch respawn 5 s;
    - `deathmatch_wide` at 25% (recorded as `WIDE_SHARE`). One game is one mode, so the
      learner section wires the mix.
15. **Speed: team deathmatch placement.**
    - One pass serves resets and respawns, against the enemy slots only (team-major
      flip), instead of two passes over all N.
    - The targets are stacked into one tensor, so the compiled stage stores them once
      instead of recomputing a provisional point's sqrt, cos and sin for every
      candidate pair.
    - Together, 0.56 → 0.25 ms per tick at 114 ships × 360 environments.
16. **Speed: the reward step.**
    - The ship rows and the boundary row stay apart until the single write.
    - The components take the column totals the step has: the accumulated rows' total is
      the old total plus the applied damage, since a hit target's shares sum to one.
    - Together, 0.70 → 0.63 ms at the same size.
    - Writing the two parts of the ledger as separate tensors would give 0.51 ms, but the
      spec fixes one (B, N + 1, N) tensor (request 5).

## Measurements

RTX 4070 laptop, bench lock held, seed 20261031 (`bench.sizes`), mid-game states.

**Per tick, compiled** (CUDA events; `resolve` is phases 4–8 including the mode;
"mode pieces" is the phase, placement and protection compiled alone):

| cell | mode | resolve ms | vs sandbox | mode pieces ms | reward step ms |
|---|---|---:|---:|---:|---:|
| 114 ships, 22 fields, 360 envs | sandbox | 0.580 | — | 0.043 | 0.626 |
| | team deathmatch | 0.845 | +0.265 | 0.284 | 0.629 |
| | grouped (19, 19, 19) | 0.738 | +0.158 | 0.210 | 0.627 |
| | Frontline | 0.675 | +0.095 | 0.126 | 0.624 |
| 40 ships, 8 fields, 1024 envs | sandbox | 0.261 | — | 0.035 | 0.138 |
| | team deathmatch | 0.427 | +0.166 | 0.132 | 0.139 |
| | grouped (8, 6, 6) | 0.364 | +0.104 | 0.095 | 0.142 |
| | Frontline | 0.363 | +0.102 | 0.129 | 0.166 |
| 26 ships, 54 fields, 1576 envs | sandbox | 0.372 | — | 0.036 | 0.085 |
| | team deathmatch | 0.498 | +0.126 | 0.083 | 0.083 |
| | grouped (5, 4, 4) | 0.480 | +0.109 | 0.097 | 0.084 |
| | Frontline | 0.450 | +0.078 | 0.133 | 0.087 |

- **The mode phase** costs 0.08–0.27 ms per tick:
  - team deathmatch's placement is the largest part, B·N·K·S pairs every tick;
  - Frontline's is about 0.1 ms at every size: its work is (B, N, 5).
- **The reward step** is memory-bound at about 9 passes over a (B, N, N) float32 matrix
  (each about 75 µs at 114 ships × 360). It scales with B·N²: at 5v5 training sizes it
  is tens of microseconds.

**Bench cells, train rollout** (env-ticks/s), and realtime:

`bench.cell` with the mode in the size, one run each. The base is the branch point
572733fd (the game section's game with the skeleton reward stub), run in the same session.

| cell | base | sandbox | team deathmatch | grouped | Frontline |
|---|---:|---:|---:|---:|---:|
| 114 ships, 22 fields, 360 envs | 111,234 | 86,530 | 80,225 | 81,872 | 91,767 |
| 40 ships, 8 fields, 1024 envs | 440,713 | 392,525 | 365,598 | 375,563 | 389,671 |
| 26 ships, 54 fields, 1576 envs | 427,240 | 414,633 | 389,667 | 398,643 | 407,565 |
| train env-ticks/s, 114 / 40 / 26 ships | 18,966 / 56,584 / 78,978 | 18,026 / 55,558 / 78,457 | 17,663 / 54,731 / 77,376 | 17,705 / 55,051 / 77,600 | 18,142 / 55,395 / 78,009 |
| realtime, CPU, 10 ships, p50 / p99 ms | 0.157 / 0.167 | 0.18 / 0.20 | 0.19 / 0.23 | 0.21 / 0.23 | 0.21 / 0.26 |

- **Where the rollout loss goes.**
  - In sandbox it is almost all the reward step: the game's `resolve` is unchanged
    (0.580 ms vs the base's 0.582 ms at 114 ships), against the base's two-pass stub.
  - In team deathmatch the placement adds its 0.13–0.27 ms per tick.
  - The large-N cell feels the O(N²) step most (−22% sandbox, −28% team deathmatch); at
    26 ships it is −3% and −9%.
- **Training throughput** is dominated by the PPO update (about 2 s), so it moves 1–7%.
- **Peak VRAM** is unchanged within 0.1 GB. The ledger is (B, N + 1, N) float32: 19 MB at
  114 ships × 360.
- **Realtime** at B = 1 is 15–35% slower at p50. The reward step does not run there; the
  cost is the mode's code and the wider observation, many small ops at B = 1.
- **Compile times** vary by cache state: the first mode of each size pays the cold
  cache.

## Changes outside `bnb.game`

- **`bnb.contracts`:**
  - `config.py`: `mode`, `spawn_groups`, `kills_per_ship`, the `MODES` names, and
    `episode_s` and `respawn_s` defaulting to the mode's;
  - `events.py`: the five fields, schema 2;
  - `obs.py`: `ZONE_COLUMNS`, `Obs.zones` (default `None`), the seven global columns;
  - `trajectory.py`: `REWARD_LEVELS` and the level axis of `reward`;
  - `__init__.py`.
- **`bnb.models.yemong`:**
  - `inputs.py`: the zone token, the global columns, `Tokens.zone` and `memory_position`;
  - `network.py`: the encoder's optional zone kind;
  - `model.py`: the levels default, the zone kind's initialisation, the memory's rotary
    tables.
- **`bnb.models.mlp`:** `GLOBAL_SCALE` covers the new global columns. This is forced: the
  MLP reads every global column.
- **`bnb.controllers.yemong`:** `reads` includes `zones`.
- **`bnb.runtime`:**
  - `rollout.py`: the ledger in the carry, the reward step, the zones part, the level
    axis;
  - `train.py`: the pool rewards sum the levels.
- **`bnb.learn.ppo`:** GAE on the summed levels.
- **Tests:**
  - `tests/reference/race_arm.py`: sandbox mode, and sandbox's mode state in `unflat`;
  - `tests/reference/rewards/rewards_arm.py`;
  - `tests/contract/test_yemong_contract.py`: the values' level count is now
    `len(REWARD_LEVELS)`. This file is outside my list. The spec's level default forces
    the change; nothing else in it changed.
  - in `tests/sections`: `test_game_section.py` reads the spec defaults in sandbox;
    `test_yemong_model.py` takes the level count; `yemong_cases.py` passes zones and adds
    a short Frontline config;
  - new: `test_modes.py`, `test_rewards.py`, `test_modes_yemong.py`, `modes_cases.py`.

## Tests

- **Modes spec acceptance:**
  - item 1: the fixtures, through the adapter;
  - item 2, team deathmatch: win, draw and clock; every cause of death scores; the
    respawn takes the farthest candidate (brute force); the reset pass is symmetric and
    farthest from the provisional points; the candidates are uniform; the seat is hidden
    in every mode;
  - item 2, grouped: the centres are exact and spread, the groups fill by rank, the reset
    keeps to its group's rivals (brute force), and respawns land 250–600 m from a living
    ship;
  - item 3, Frontline: the mirror (fields, zones, roles); both handednesses; protection;
    an end-to-end capture to the win in 3 × 164 ticks; the timeout and the draw; spawns
    at the current zone facing the attack;
  - items 4 and 5: zero-sum objective events every tick; `objective` sums to ±3 captures
    per team; the shield events match the shield changes through deaths, respawns and
    resets in every mode;
  - item 6: compiled matches eager in Frontline, grouped and sandbox; eager CUDA-graph
    capture works in Frontline and grouped; the race tests pass in sandbox.
- **Rewards spec acceptance:**
  - item 1: the fixtures;
  - items 2 and 3, on lively random play in every mode (weak shields, a short recharge
    delay, small arenas): every level is zero-sum on every tick to 1e-5 of the tick's
    largest reward; each living ship's column equals its deficit within 1e-3 and each dead
    ship's column is 0. Friendly fire, enemy kills, boundary deaths, recharge, respawns
    and resets all occur.
  - item 4, scenarios: the healed duel; the kill shares with ally blame; a boundary
    death; a dead attacker's claims (it dies, or the target heals); a respawned target;
    the outcome;
  - also: the rollout's stored rewards are zero-sum per environment, and the compiled
    step matches eager.
- **Mutation check** (`/tmp/bnb-scratch-modes/mutants.py`, not committed): 28 mutants
  over rewards, Frontline, the phase, spawns, observation and combat.
  - 25 failed the tests at once.
  - Two survivors cannot change any value: the unheld term (decision 10, now removed) and
    "charge only those outside" (decision 11).
  - The third, a grouped reset against every group's enemies, led to the brute-force test,
    which now fails it.

## Requests and open questions

1. **Ring order.**
   - `modes.md` says "in order around the ring: team 0's spawn, team 0's defense,
     neutral, team 1's defense, team 1's spawn".
   - The spec and the fixture give, by zone, neutral, team-0 spawn, team-0 defense,
     team-1 defense, team-1 spawn: the two defense zones adjacent, and neutral between
     the two spawns.
   - I followed the spec. Please confirm, and correct `modes.md`.
2. **`Obs.SCHEMA_VERSION`** should become 2 (zones, seven new global columns), but
   `tests/contract/test_contracts.py` asserts 1.
3. **Rewards spec 3.3's "unheld shares pay the opposing team"** has nothing to apply to:
   the shares always sum to one. Please drop the wording, or say what an unheld share is
   meant to be.
4. **A grouped respawn can land outside the arena:** an anchor near the edge plus up to
   600 m. The spec does not clip it; I did not. Should the candidates be clipped or
   rejected?
5. **The ledger as two tensors** (ship rows (B, N, N) and the boundary row (B, N)) would
   make the reward step about 20% faster (0.63 → 0.51 ms at 114 ships × 360). It needs a
   spec change.
6. **`contracts.md`** (not mine to edit):
   - section 5 still calls the ledger mode state, cleared at respawn;
   - the current-schema section lacks the new `Events`, `Obs.zones`, the global columns,
     `GameConfig.mode` and the reward level axis.
7. **Per-level γ and λ** (rewards.md's starting values) are not in `RewardConfig`. The
   skeleton PPO sums the levels, so section 6 adds them with its critic.
8. **`bnb.cli`** has no mode option; it was outside my boundaries. It trains team
   deathmatch by default now.
9. **`tests/loop/test_learning.py`** passes with the new rewards: 5 levels, 300 s team
   deathmatch, kill/death 0.283 per death. It stays as marginal as the game notes
   describe.
10. **The modes.md open questions** were settled by the spec table: `deathmatch_wide`
    groups (2, 2, 1), and the 25% share (a proposal).
11. **Costs to know about:**
    - placement, the medium at spawn points and the anchors run every tick for every
      ship, whether or not it spawns;
    - making them conditional would need a data-dependent shape (the game notes say the
      same of spawning).

## Size

| module | lines |
|---|---:|
| `game/modes/frontline.py` | 218 |
| `learn/rewards.py` | 198 |
| `game/modes/spawns.py` | 192 |
| `game/modes/phase.py` | 109 |
| `game/modes/rules.py` | 72 |
| `game/modes/observation.py` | 48 |
| changes to the existing `bnb.game` modules (step, spawning, combat, state, rules, observe) | +112 net |

`src/bnb` grew by 1,025 lines net (the stub's 25 deleted). The section tests grew by about
1,065 lines, nearly all in four new files.
