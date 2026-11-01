# Mixed batches spec (section 3c)

Status: spec, 2026-11-01. Decisions: `learner.md` (author, 2026-11-01).

One batch holds several games in the same tensors and one compiled graph: team
deathmatch on the squad and battle maps, Frontline, fogged and clear. Everything that is
fastest wins; the author's goal is speed and simplicity.

## 1. Configuration

- **`GameConfig`** keeps the shared sizes: ships per team, fields `M`, and whether zones
  exist (`Z` = 5 if any pool plays Frontline, else 0).
- **`Pool` gains** `mode`, `arena_radius`, `spawn_groups`, `episode_s` and `respawn_s`,
  where None means the `GameConfig`'s value or the mode's default.
- **Every environment of a pool shares them.** Pools are contiguous slices of the batch,
  as now.
- **The battle map is 10 fields (decided):** `deathmatch_wide` becomes 5v5, 10 fields,
  arena radius 8200 m, groups (2, 2, 1). Field radii scale with the arena as they already
  do. Training stays at 10 fields; evaluation at 50v50 uses 40. The zero-shot claim covers
  fields too.
- **The default training mix** (proposal): Frontline 0.5, team deathmatch squad 0.375,
  team deathmatch battle map 0.125 (25% of team deathmatch). Fog is per pool, as now.

## 2. The game

- **Per-pool modes as static slices.** Every mode-specific step (map generation, the mode
  phase, spawn placement, spawn protection, the mode's observation) runs on its own pool's
  slices only. The bounds are static, so it is one graph with no work wasted on another
  mode's rules. Results are written back into the batch tensors.
- **Per environment, as tensors:** the arena radius and map scale, the episode length and
  the respawn wait (one value per pool, broadcast). Everything that read `rules` for them
  reads the tensor.
- **Zones:** every environment carries `Z` zone slots. In a mode without zones they are
  **absent**: `present` false, every other column 0, no effect on the game.
- **Fields:** all `M` are present. If a later pool ever needs fewer, an absent field is
  level 0 and radius 0: it changes no index and blocks no sight. That rule is not built
  now.
- **Tick uniforms:** one width for the batch, the largest any pool needs. Each
  environment uses its mode's prefix.

## 3. Observation

`ZONE_COLUMNS` gains `present` (1 or 0). An absent zone's other columns are 0. The global
columns already carry the mode one-hot.

## 4. Yemong

- **Absent zone tokens are a learned vector:** one `D_m`-wide parameter, used as the zone's
  map-memory token wherever `present` is false (selected with `where`; no mask). The
  encoder's output for those rows is discarded.
- **Rotation (decided):** an absent token takes the **global token's rotation** (the map
  centre in the model's shifted frame) with heading angle 0. No rotation (the frame's
  origin) would leak the per-team position offset, since a query could compare the origin
  with the centre.
- **Absent zones in the replay:** whatever the step reads about them is in the replay,
  so `unroll` sees the same.

## 5. Runtime and CLI

- **The rollout** builds one batch from pools with different modes. The trainer's
  progress metrics are per pool.
- **The CLI preset `mixed`** gives the default training mix. The single-mode presets
  stay.

## 6. Acceptance

1. **Equivalence:** a mixed batch equals each game run alone, environment by environment,
   given the same per-environment uniforms. That covers state, events, observation and
   rewards, bit for bit on the CPU in float32 (or within 1e-6 if a fused op reorders),
   eager and compiled.
2. **Absent zones are inert:**
   - in the game, a team deathmatch environment in a mixed batch behaves exactly as
     alone;
   - in Yemong, changing an absent zone's columns changes no output;
   - the absent rotation equals the global token's.
3. **No seat leak:** the seat tests hold in a mixed batch.
4. **No offset leak:** with the encoder's position-code columns zeroed, outputs are
   invariant to the per-team offset, with absent zones present.
5. **Speed:** a mixed batch's tick time is within noise of the share-weighted single-mode
   tick times (no mode pays for another). Report the 5v5 training cells: `deathmatch`,
   `deathmatch_wide` and `frontline` alone, then `mixed`, with the skeleton learner and
   with Yemong.
