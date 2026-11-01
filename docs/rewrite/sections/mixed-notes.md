# Section 3c (mixed batches): notes

Branch `section/mixed`. The spec (`mixed-spec.md`) is implemented in `bnb.game` (the new
`layout.py`, and the modes and physics taking per-environment rules), `bnb.runtime`, the
`Pool` contract, and Yemong's absent zones. `uv run --no-sync pytest` passes: 294 tests,
GPU and slow ones included, none skipped. `ruff check` is clean. The fixed tests, the
goldens and the four adapters are unchanged. The fix round (end of this file) changed the
design in places; the text above it is updated to match.

## Design

### Configuration

- **`Pool` gains** `mode`, `arena_radius`, `spawn_groups`, `episode_s` and `respawn_s`, all
  `None` by default. `pool_game(config, pool)` is the config with the pool's values
  substituted in (`dataclasses.replace`). The config then validates them: spawn groups only
  in team deathmatch, and so on. The clocks fall back to the mode's defaults, as before.
- **`Rules.pools`**, a tuple of `PoolRules(environments, mode, arena_radius, episode_ticks,
  respawn_ticks)`, replaces `Rules.arena_radius`, `mode`, `episode_ticks` and
  `respawn_ticks`. `rules_of(config, pools=(), num_envs=None)`:
  - **without pools:** one game covering the whole batch, at any batch size
    (`environments=None`). This is what the realtime path, the reference arms and the
    contract tests use;
  - **with pools:** one entry per pool, in pool order. Adjacent pools of the same game
    merge. If only one game is left, it covers the whole batch, so a single-game training
    run compiles exactly as before.
- `Rules.modes` lists the games' modes; `Rules.num_zones` is 5 for every mode (fix round,
  coordinator's decision): a fixed shape of the game, like the field count, so a model
  sees the same memory tokens in any batch. `tick_uniform_count` is the game's count plus
  the widest mode's.
- **Presets:** `deathmatch_wide` is 5v5, 10 fields, 8200 m, groups (2, 2, 1).
  `bnb.runtime.train.TRAINING_MIX` is `(("frontline", 0.5), ("deathmatch", 0.375),
  ("deathmatch_wide", 0.125))`.

### The game (`game/layout.py`, 93 lines)

- **`Environment`:** the rules that vary by environment, each `(B,)`: `arena_radius`,
  `map_scale`, `episode_ticks`, `respawn_ticks`. `environment(rules, B)` builds them from the
  static pools (`torch.full` per pool, then `cat`), so in a compiled graph they are
  constants.
- **`slices(rules, B)`** cuts the batch where the mode changes: adjacent pools of one mode
  share a slice.
- **`per_mode(rules, B, function, *trees)`** slices every tensor of the pytrees, calls
  `function(mode, *slices)` for each slice, and concatenates the results along the batch.
  With one slice it calls the function on the whole batch, without cutting.
- **Shared phases** run once over the batch and read `Environment`: flight, bullets, hits,
  damage, the boundary, the map's fields, the formation, the episode clock, the respawn
  wait, sight, and the observation's `episode_progress` and `arena_radius` columns.
- **Mode-owned steps** run on their mode's slice only:
  - in `resolve`: spawn protection (only when some pool plays Frontline), the mode phase with
    its clock result, the new match (Frontline's ring and mirrored fields), and the spawn
    placement;
  - in `perceive`: the zone rows and the mode's global columns.
  - **Signatures:** each hook takes its `ModeRules` (`mode_rules`) as an argument; nothing reads
    `rules.mode` any more. Each slice reads its own mode's prefix of the tick uniforms.
- **Zones:** every environment of every mode holds five slots. Outside Frontline they stay
  zero and are passed through, never copied: `new_match` zeroes only the score, the team
  deathmatch phase changes only the score, and `state.where` keeps a field that is the same
  tensor on both sides. They show as absent rows.

### Observation

`ZONE_COLUMNS` gains `present`, appended last: 1 in Frontline, and 0 in an absent row, whose
other columns are all 0.

### Yemong

- **`Encoder.absent_zone`:** one `D_m`-wide parameter, initialised `N(0, 1)`, which is the
  scale of a normalised memory token. The encoder creates it empty and Yemong draws it after
  every other layer, so no other initialisation depends on it. After the map-memory norm, `torch.where` puts it in
  place of every zone that is not present. The projection's output for those rows is
  discarded. There is no mask.
- **Rotation:** `compose` places an absent zone at the map centre in the shifted frame
  (`replay.offset`), which is the global token's position. Every memory token has heading
  0, so the absent zone's rotary tables are the global token's (tested). `Tokens` gains
  `zone_present`.
- **The replay is unchanged.** What the step reads about absent zones comes from the
  observation's `zones` part, `present` column included, and the runtime stores that part.

### Runtime and CLI

- **`Rollout`:**
  - builds `rules_of(cfg, pools, num_envs)`;
  - staggers each environment's start clock within its own episode length;
  - reports progress per pool (`Trainer.pool_rewards`, as before).
- **`runtime.train`:**
  - `LINEUP` is the skeleton lineup, now named and no longer only the `TrainConfig` default;
  - `mixed(lineup, mix)` crosses every game of the mix with every lineup pool. The order is
    game by game, so each game is one slice. Pools are named `"frontline/chaser"` and so on,
    and each pool's share is the game's share times the lineup's.
- **`bnb --preset mixed`** trains the default mix. The single-mode presets are unchanged.
  A batch size the pools do not split stops with a message naming the multiple
  (`contracts.pools.environment_multiple`: 8 for the lineup, 64 for the mix).
- **`_assign` skips self-copies** (speed, not in the spec): a carry leaf that a stage returns
  unchanged, which is the buffer itself, is no longer copied onto itself. The advance stage
  did this for the fields, the match and the mode state: 22 of its 32 kernels in Frontline.

## Decisions and deviations

1. **`GameConfig` keeps `mode`, `arena_radius`, `spawn_groups`, `episode_s` and
   `respawn_s`.** They are the pools' defaults, the "GameConfig's value" in the spec.
   - Every one-game caller builds a `GameConfig` alone: realtime, the race arm, the
     contract tests, `bench/cell.py`.
   - "Whether zones exist" became moot in the fix round: every mode has five slots.
2. **Two levels of merging.**
   - `PoolRules`: adjacent pools that play the same game merge.
   - `slices`: adjacent games that share a mode share a slice.
   - The mode code reads the arena radius and the map scale from the per-environment
     tensors, as the spec says. So two pools of one mode with different arenas or clocks
     run in one slice.
3. **Removed from `Rules`:** `arena_radius`, `mode`, `episode_ticks`, `respawn_ticks` and
   `map_scale`. A one-game caller reads `rules.pools[0]`, and the game reads `Environment`.
   No shim was kept. The section tests that read them now use `rules.pools[0]`.
4. **Bit-exact equivalence needs pools sized in whole vector widths on the CPU** (the
   author accepted this; other sizes are held to 1e-6 per tick).
   - Eager torch kernels compute the tail of a tensor (the elements past the last full
     vector) with scalar code. There, `exp` and `sin` can differ from the vectorised
     result in the last bit.
   - With 4 environments per game, a 1-ulp difference in `angle_of_attack` appeared at
     tick 65 and play amplified it.
   - With 32 per game every element takes the same path, and 300 lively ticks match bit
     for bit. The test uses 32.
5. **`Obs.SCHEMA_VERSION` is 3** (fix round; the contract test now asserts it).
6. **`present` is the last zone column,** so existing column indices are unchanged.
7. **The mixed preset's shared config is the first game's preset** (Frontline 5v5,
   10 fields). Every crossed pool sets its own mode, arena and groups.
8. **Proposal implemented as written:** the training mix, Frontline 0.5, team deathmatch
   0.375, the battle map 0.125. At 12 pools the shares are multiples of 1/64, so
   `num_envs` must be a multiple of 64.
9. **Tried and rejected for speed:**
   - **One per-mode region for the whole end of the tick** (phase, episode end, new match,
     placement): 13 fewer kernels, but 1.5% slower in a mixed batch (A/B twice).
   - **Inductor's ConcatKernel in place of pointwise cats** (per-stage compile options):
     slower in a mixed batch. It was 9% faster for `deathmatch_wide` alone, which is a lead
     for the game section (request 3).

## Measurements

RTX 4070 Laptop, bench lock held, a fresh process per cell, float32, TF32 off. Each game is
5v5 with 10 fields. "Weighted" is the share-weighted single-game value (0.5 Frontline,
0.375 squad, 0.125 battle map). Commit c64f8015 (before the fix round; the fix round's
numbers are at the end); the base is the branch point b90f87d5.

**Game tick.** Every seat random, 4096 environments, rollout of 128 ticks, min of 5 after 11
warm-up rollouts. Two separate runs per cell:

| game | ms per tick | base (b90f87d5) |
|---|---:|---:|
| `deathmatch` | 0.662 / 0.677 | 0.704 / 0.698 |
| `deathmatch_wide` | 0.757 / 0.766 | — (40 fields then) |
| `frontline` | 0.732 / 0.740 | 0.754 / 0.755 |
| `mixed` | 0.809 / 0.813 | — |
| weighted | 0.714 | |
| the three games as separate batches at their shares (2048 + 1536 + 512 envs) | 0.761 | |

- **The mixed batch is 13.6% over the weighted value, and 6.6% over running the three games
  as separate batches.** That is not within noise, so the spec's item 5 is not met; the
  author accepted the overhead in the fix round. Run to run, cells vary by under 1.5%.
- **Where the time goes** (per stage, each captured alone):
  - `resolve`, the mode-owned pieces summed per slice:
    - the phase costs 43 µs mixed, against 33 µs for the three games at their shares;
    - the new matches, 23 µs against 21;
    - the placement, 27 µs against 28.

    So the cut itself costs little. The mode-owned kernels at 5v5 sizes are small and do
    not shrink with the slice: Frontline's phase is 40 µs at 4096 environments and 27 µs
    at 2048, and grouped placement is 45 µs at 4096 and 17 µs at 512.
  - **The shared phases cost about 50 µs more** after the cut (inductor fuses them
    differently around the pointwise cats of the slices' results), and this is most of the
    remaining gap.
  - **Every environment carries Frontline's zone state.** This adds about 10 µs to
    `advance` and `perceive` together.
- **No regression for a single game.** The self-copy fix makes a single-game tick 2.5-4.5%
  faster than the branch point: 0.70 → 0.67 ms in team deathmatch, 0.75 → 0.74 ms in
  Frontline.

**Training cells, skeleton MLP learner.** Lineup self-play 0.5, league 0.25, chaser 0.125,
random 0.125; 4096 environments; `bench.cell`'s method (min of 3 after 10 warm-up
rollouts):

| game | rollout ms | learn ms | rollout env-ticks/s | train env-ticks/s | VRAM GB |
|---|---:|---:|---:|---:|---:|
| `deathmatch` | 345.2 | 2190.8 | 1,518,672 | 206,735 | 4.14 |
| `deathmatch_wide` | 354.8 | 2187.9 | 1,477,565 | 206,191 | 4.14 |
| `frontline` | 353.3 | 2187.5 | 1,484,049 | 206,348 | 4.14 |
| `mixed` | 361.8 | 2185.6 | 1,449,281 | 205,816 | 4.15 |
| weighted | 350.5 | | | 206,498 | |
| base `deathmatch` / `frontline` | 347.0 / 352.6 | 2188.0 / 2185.4 | | 206,823 / 206,572 | 4.14 |

- **The mixed rollout is 3.2% over the weighted value; training throughput is 0.3%
  lower.** The skeleton's PPO update (2.19 s) dominates.

**Training cells, Yemong.** Yemong is the learner and the league slot (the same lineup);
1024 environments, so 1536 learner views; float32; min of 3 rollouts after 11. Below it,
unroll forward and backward over one minibatch, 32 learner views × 128 steps, compiled
(min of 5):

| game | rollout ms | rollout env-ticks/s | unroll fwd+bwd ms | peak VRAM GB |
|---|---:|---:|---:|---:|
| `deathmatch` | 3191 | 41,072 | 150.1 | 4.67 |
| `deathmatch_wide` | 3200 | 40,958 | 150.8 | 4.67 |
| `frontline` | 3312 | 39,576 | 157.0 | 4.89 |
| `mixed` | 3317 | 39,518 | 156.1 | 4.89 |
| weighted | 3253 | | 153.6 | |

- **The mixed Yemong rollout is 2.0% over the weighted value, the same as Frontline alone.**
  - The game's cost is negligible next to the 26 ms act stage.
  - A team deathmatch view in a mixed batch attends over 15 memory tokens (10 fields and 5
    absent zones) instead of 10, so it costs what a Frontline view costs.

## Changes outside `bnb.game` and `bnb.runtime`

- **`bnb.contracts`:**
  - `pools.py`: the five `Pool` fields, `pool_game`, `GAME_FIELDS`,
    `environment_multiple`, and `pool_sizes` renamed `environments_per_pool`;
  - `obs.py`: the `present` zone column, the `Obs` docstring, schema 3;
  - `config.py`: docstring only (the fields are the pools' defaults).
- **`bnb.models.yemong`:**
  - `network.py`: `absent_zone` and the `zone_present` argument;
  - `inputs.py`: `Tokens.zone_present` and the absent zones' position at the map centre;
  - `model.py`: passes `zone_present` to the encoder, and draws `absent_zone` last.
- **`bnb.runtime.lineup`:** the renamed `environments_per_pool`.
- **`bnb.cli`:** `--preset mixed`, and the batch-size check.
- **`bench/`** (fix round, permitted): `cell.py` reads the pools' episode lengths and takes a
  `"preset"` in a train cell; `sizes.envs_for` takes a preset and rounds to its multiple.
  `run.py`'s cell list and `results.jsonl` are unchanged.
- **`tests/sections`:**
  - `test_modes.py` reads `rules.pools[0]`, wraps `deathmatch_placement` in a one-game
    helper, and takes the zone width from `ZONE`;
  - `test_game_section.py` reads the clocks from the pool;
  - `test_modes_yemong.py`: a team deathmatch view's five absent slots are the learned
    token (it used to check that no zones and an empty part agree);
  - `test_perception.py`: the renamed `environments_per_pool`;
  - new: `test_mixed.py` (13 tests) and `test_mixed_yemong.py` (5 tests).
- **`tests/contract/test_contracts.py`:** schema 3, the renamed `environments_per_pool`.
- **Unchanged:** `bnb.learn`, `bnb.controllers`, the adapters, the fixed tests, the docs.

## Tests (acceptance items)

1. **Equivalence:**
   - `test_a_mixed_batch_equals_each_game_run_alone`: four games of 32 environments each
     (Frontline, team deathmatch, grouped, sandbox), each with its own arena and clocks,
     half of them fogged. The test runs 300 lively ticks and compares, bit for bit, every
     leaf of the state, the events, the observation, the truth, the rewards and the ledger
     against each game run alone with its prefix of the same uniforms; the absent rows must
     be zero. Each pool has its own respawn wait (fix round).
   - `test_an_unaligned_mixed_batch_equals_each_game_alone_to_float32_rounding` (fix
     round): pools of 5 environments, every tick from the same state, to 1e-6, through
     resets.
   - `test_one_slice_of_several_arenas_and_clocks_equals_each_game_alone` (fix round): two
     team deathmatch pools (600 m and 1500 m) and two Frontline pools (300 m and 700 m),
     each with its own episode length and respawn wait, in two slices, bit for bit.
   - `test_a_compiled_mixed_tick_equals_each_game_compiled_alone`: compiled, from four
     mid-game states, to 1e-6.
2. **Absent zones are inert:**
   - **game:** item 1, plus `test_absent_zones_stay_absent_and_change_nothing`: garbage in
     the absent slots changes no ship, event or score;
   - **Yemong:** `test_absent_zone_columns_change_no_output` (step and unroll, bit for bit),
     and `test_an_absent_zone_is_the_learned_token_at_the_global_tokens_rotation`.
   - **Mutation check:** two mutants each fail two of these tests:
     - the encoder output used instead of the learned token;
     - absent zones placed at the frame's origin.
3. **No seat leak:** `test_a_mixed_batch_does_not_reveal_the_seat`. Every mode's seat checks
   run on 1024 environments per mode in one batch:
   - the own-ship layout;
   - both views' globals and zone rows, `present` included;
   - Frontline's handedness half and half;
   - absent rows zero in both views.
4. **No offset leak:** `test_the_offset_stays_hidden_with_absent_zones`. With the
   position-code columns zeroed, a history shifted by a random per-view offset gives the same
   decision to 1e-8 (float64), with absent and present zones in the batch.
5. **Speed:** the tables above.
6. **Also tested:**
   - pool defaults and validation;
   - merging, and a single game being the one-game rules;
   - the mixed preset's layout;
   - a CPU mixed rollout and PPO update with per-pool rewards;
   - unroll reproducing a stored mixed rollout;
   - a compiled, CUDA-graph-captured mixed rollout agreeing with unroll to 1e-5 (GPU);
   - (fix round) spawn protection behind a team deathmatch pool, each environment's start
     clock within its own episode, the absent slots passed through uncopied, the batch
     multiple.

## Requests and open questions

1. **`contracts.md`** (not mine to edit) needs:
   - the `Pool` fields, `pool_game` and `environment_multiple`;
   - `ZONE_COLUMNS`' `present`, `Obs` schema 3, and five zone slots in every mode;
   - `Rules.pools`, `Environment` and the removal of `Rules.mode`, `arena_radius` and the
     clock fields;
   - section 8's `Pool` block.
2. **Settled in the fix round** (no longer open): the speed overhead is accepted; team
   deathmatch keeps its absent tokens; bit for bit where pools are whole vector widths,
   1e-6 otherwise; the schema is 3.

## Size

| part | lines |
|---|---:|
| `game/layout.py` (new) | 93 |
| `src/bnb` in total | +526 / −198 (net +328) |
| of which the game's modes, physics, perception and state | +206 / −134 |
| `game/rules.py` | +77 / −24 |
| Yemong (network, inputs, model) | +42 / −14 |
| `bench/` (cell, sizes) | +30 / −9 |
| tests | +640 / −46, nearly all in the two new files |

## Fix round (after the review)

Commits a1f898cd and 81f54b32. 294 tests pass (GPU and slow included), ruff is clean.

### Required fixes

1. **bench.** `bench/cell.py` read `rules.episode_ticks`; it now warms up for the longest
   pool's episode. A train cell may name a `"preset"` (a game preset or `"mixed"`), and
   `sizes.envs_for(size, budget, preset)` rounds to the preset's multiple (64 for mixed).
   Checked under the lock: `mixed` at 4096 envs, `deathmatch` at 4096, and a drawn 40-ship
   size at 1024 all run.
2. **Test gaps.** Each of the reviewer's six mutants now fails `tests/sections/test_mixed.py`
   (checked one by one, `/tmp/bnb-scratch-mixed/mutants_fix.py`):
   - the respawn wait read as one value: the pools in the main equivalence test wait 0.8,
     0.5, 0.3 and 1.0 s;
   - placement, Frontline placement or `new_match` reading the slice's first radius or map
     scale: the two-arenas-per-slice equivalence test;
   - the start clock taken modulo the longest pool: each environment's clock is below its
     own episode length, and the long pool's spread past the short one's;
   - protection gated on the first pool: Frontline behind a team deathmatch pool, where a
     hit on a ship in its spawn zone deals nothing and a team deathmatch ship at the same
     spot (the absent zones' centre) takes the hit.
3. **Five zone slots in every mode.** `Rules.num_zones` is `ZONE_SLOTS` = 5 whatever the
   mode. No fixed reference test or golden needed a change (the race arm builds its blank
   state from the rules; no golden reads zones).
   - Single-mode team deathmatch tick (4096 envs, two runs each): 0.673 / 0.673 ms before,
     0.701 / 0.692 ms with the five slots copied through the tick, and **0.668 / 0.676 ms**
     after the pass-through below. Battle map 0.786 → 0.774; Frontline 0.731 → 0.737
     (noise).
   - Yemong pays for the slots in every team deathmatch view (15 memory tokens, as in the
     mix): its rollout went 3191 → 3298 ms (+3.4%), unroll forward and backward 150 → 156 ms.
4. **`Obs.SCHEMA_VERSION` is 3**; `tests/contract/test_contracts.py` asserts it.
5. **`absent_zone` is drawn last:** the encoder creates it with `torch.empty` (no draw), and
   `Yemong.__init__` draws it after every other layer and `_initialise`.

### Quality

- `contracts.pools.pool_sizes` is `environments_per_pool(pools, num_envs)`;
  `layout.pool_sizes` is `sizes_in_batch(rules, batch)`.
- `rule` is `mode_rules` everywhere (the hooks, placement, the observation); team
  deathmatch placement's `mode` parameter too.
- `layout.environment` builds every column, `map_scale` included, through one local helper.
- `api.py`'s docstring: any batch size for one game, the pools' batch size for several.
- `bnb --preset mixed --num-envs 1000` stops with "does not split into the pools' shares:
  use a multiple of 64".
- `TRAINING_MIX` lives in `bnb.runtime.train`.

### Optional (time-boxed)

- **Kept: zone-free slices pass their zone leaves through.** `new_match` outside Frontline
  zeroes only the score and returns the other leaves as they are; `state.where` keeps a
  field that is the same tensor on both sides; the runtime already skips self-copies. So a
  team deathmatch batch copies no absent slot (tested). Team deathmatch alone 0.697 →
  0.672 ms, battle map 0.786 → 0.773, Frontline unchanged, mixed 0.814 → 0.810.
- **Rejected: assembling slices in place** (a fresh batch tensor, each slice's result
  written into its rows instead of `torch.cat`): mixed 0.810 → 0.862 ms.

### Measurements after the fix round (81f54b32)

| cell | `deathmatch` | `frontline` | `deathmatch_wide` | `mixed` | weighted |
|---|---:|---:|---:|---:|---:|
| game tick, ms (two runs) | 0.668 / 0.676 | 0.737 / 0.737 | 0.774 / 0.771 | 0.807 / 0.812 | 0.716 |
| MLP rollout, ms | 341.1 | — | — | 361.9 | |
| MLP train env-ticks/s | 207,759 | — | — | 205,955 | |
| Yemong rollout, ms | 3298 | 3313 | — | 3317 | 3306 |
| Yemong unroll fwd+bwd, ms | 156.3 | 156.9 | — | 157.0 | |

- The mixed tick is 13% over the weighted value (accepted).
- With five slots everywhere, a mixed Yemong rollout costs what the single modes cost
  (+0.3% over the weighted value).
