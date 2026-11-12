# Bullet damage: progress

Branch feat/bullet-damage (from tune/doctrines 4ba28b54, the doctrine loop's handoff). The
author's decisions: realistic per-type bullet drag (the .50 cal the floor), pure kinetic
damage on the bullet's own speed with a damage handle stated at a reference distance, an
explosive share per type, damage by the closest approach to the target's centre
f = (1 - (b/R)^2)^2, bullets still consumed on a hit, static shapes.

## 2026-11-12 14:00-14:40 KST: the mechanics (game v6), commit bb2cdb61

- **Damage** (`physics.combat.hits`): `D (x + (1 - x)(w / w_ref)^2) (1 - (b/R)^2)^2 A(theta)`.
  `D` the card's `hit_damage` (a centre hit's at the reference distance), `x` the card's
  `explosive_share`, `w` the bullet's proper speed after its move, `w_ref` the card's
  `reference_bullet_speed`, `b^2 = (D0 x Delta)^2 / |Delta|^2` (at most |D0|^2) for the hit
  target only (an O(bullets) gather, not another (B, NK, N) tensor), `R` the target's hit
  radius. `kinetic`, `closeness`, `impact_squared`, `MEAN_CLOSENESS = 8/15` there.
- **Handles**: `bullet_drag` (1/m) after `gun_range`, `explosive_share` after `hit_damage`.
  The card keeps `bullet_drag_scale` (handle / rules.bullet_drag) and gains
  `explosive_share` and `reference_bullet_speed` = w0 exp(-k d), d =
  `Rules.damage_reference_distance` (300 m in the real-scale and design-seed rules.toml;
  300 / 5.82 = 51.5 m standard). Bullet life = ceil ticks of expm1(kR)/(k w0) under the
  type's drag; `range_drag` (the bisection trim) removed; the card's handles read back the
  flown range.
- **Ship files**: every set re-solved (combat fields only; flight untouched): hit damage x
  15/8, explosive share 0, drag by the main gun (`real_scale.GUN_DRAGS`): .50 cal 5e-4
  (floor), Hispano 5.5e-4, MG 151/20 9e-4, MK 108 6.5e-4, light 20 mm (Type 99, Ho-5,
  ShVAK, and the placeholder heavy) 7e-4, UBS and Vickers S 40 mm at the floor. Lives
  0.67-0.77 s (the set's longest 23 ticks, was 18). The standard set keeps the rules' drag.
- `real_scale.py` emits the new handles; the gun ranges keep their values under
  `RANGE_DRAG` (the set's drag before v6). `ship_design` knobs: damage per second is the
  mean hit's.
- Panel: the two handles appear automatically; the share on a linear 0-1 slider; tooltips
  for damage ("a centre hit's at the reference distance"), drag, share, range (flown).
  A rules-drag change now also moves each type's drag handle (`_rerange`).
- Benches: gun_lethality and doctrine_arena card TTKs use the expected hit (D x 8/15);
  gun_lethality's front damage share is now the two head-on aspect classes' damage.
- Tests (quick tier, touched files): all pass but
  `test_ship_sets::test_the_new_match_maps_come_from_the_presets` (fails before this
  branch too: the clover preset).

## 2026-11-12 14:35-14:50 KST: gun-lethality mirrors, before (v5, 4ba28b54) and after (v6)

`bench.gun_lethality play --environments 256 --pairs P-51D:P-51D vizia:vizia Yongma:Yongma Jujak:Jujak` (1v1, 120 s, seed 11; records `duels-v5.jsonl`, `duels-v6.jsonl` here). Passes per kill: the measured TTK median (first hit taken to death) over the head-on pass time (gun range / 2 cruise); aspect classes mutual / offset head-on / flank / tail.

| Mirror | Game | Passes per kill (measured TTK / pass) | Card head-on TTK / pass | First-merge kills | Damage m / o / f / t | Kills m / o / f / t | P(win \| first hit) | Timeouts | Measured TTK (s) |
|---|---|---|---|---|---|---|---|---|---|
| P-51D | v5 | 0.39 | 0.48 | 0.96 ± 0.01 | 0.96 / 0.02 / 0.02 / 0.00 | 0.96 / 0.03 / 0.01 / 0.00 | 0.93 ± 0.02 | 5/256 | 0.70 |
| P-51D | v6 | 0.30 | 0.46 | 0.96 ± 0.01 | 0.97 / 0.02 / 0.02 / 0.00 | 0.95 / 0.04 / 0.01 / 0.00 | 0.70 ± 0.04 | 5/256 | 0.57 |
| vizia | v5 | 11.53 | 2.29 | 0.10 ± 0.02 | 0.87 / 0.10 / 0.03 / 0.00 | 0.81 / 0.17 / 0.03 / 0.00 | 0.84 ± 0.03 | 2/256 | 20.80 |
| vizia | v6 | 1.26 | 2.21 | 0.45 ± 0.03 | 0.88 / 0.09 / 0.03 / 0.00 | 0.83 / 0.14 / 0.03 / 0.00 | 0.83 ± 0.03 | 2/256 | 2.37 |
| Yongma | v5 | 13.06 | 2.35 | 0.06 ± 0.02 | 0.80 / 0.13 / 0.07 / 0.00 | 0.63 / 0.29 / 0.08 / 0.00 | 0.86 ± 0.02 | 5/256 | 23.00 |
| Yongma | v6 | 5.83 | 2.26 | 0.36 ± 0.03 | 0.82 / 0.11 / 0.07 / 0.00 | 0.72 / 0.20 / 0.08 / 0.00 | 0.82 ± 0.03 | 4/256 | 10.67 |
| Jujak | v5 | 9.85 | 2.24 | 0.04 ± 0.01 | 0.91 / 0.07 / 0.02 / 0.00 | 0.86 / 0.11 / 0.03 / 0.00 | 0.81 ± 0.03 | 1/256 | 18.20 |
| Jujak | v6 | 1.20 | 2.15 | 0.55 ± 0.03 | 0.93 / 0.05 / 0.02 / 0.00 | 0.90 / 0.08 / 0.02 / 0.00 | 0.79 ± 0.03 | 2/256 | 2.30 |

Damage per landed hit, v6 over v5: vizia 1.56, Yongma 1.52, Jujak 1.56, P-51D 1.15 (hit
fractions 0.73-0.77, 0.34 for the P-51D). **The 15/8 rescale assumes a uniform impact
parameter; the scripted agent's aimed fire lands near the centre** (vizia's hit radius
10 m against a spread of a few metres at merge ranges), and its merges fire at close range,
where the kinetic term is above 1 (1.35 at the muzzle for a .50). So the card's closed-form
TTK is unchanged but the measured one fell: vizia's mirror 20.8 s -> 2.4 s (11.5 -> 1.3
passes), first-merge kills 0.10 -> 0.45. The aspect mix barely moved (still 80-93%
mutual head-on). The P-51D's P(win | first hit) fell 0.93 -> 0.70: a first graze no
longer decides the duel. For the tuning loop: about 0.65 x the cards' hit damage would
bring vizia's measured damage per hit back to v5's (1 / 1.55), if the old lethality is the
target; or keep the centre-hit meaning and let the closeness reward aim.

## Recompile notes for the tuning loop

- **Card tensors, no recompile:** `hit_damage`, `explosive_share`,
  `reference_bullet_speed`, `bullet_drag_scale` (the per-type drag), `bullet_life`,
  `muzzle_speed`, `fire_rate`, `hit_radius`, `max_shield`, and the scales of the rules
  floats: `head_on_fraction_scale`, `head_on_angle_scale`, `shield_recharge_rate_scale`,
  `shield_recharge_delay_scale`, `bullet_spread_scale`. Sweep those, not the rules.
- **Rules floats, a rebuild in training (static `Rules`):** `head_on_fraction`,
  `head_on_angle`, `shield_recharge_rate`, `shield_recharge_delay`, `bullet_spread`,
  `bullet_drag` (live in the play shell's compiled tick, `tuning.LIVE_RULES`, but static
  elsewhere). Tune them through their card scales above.
- `Rules.damage_reference_distance` is read only by the card solve (the tick reads the
  card's `reference_bullet_speed`): changing it needs the cards re-solved, not a rebuild.
- **Shapes:** `Rules.longest_bullet_moves` (from the card's longest life) and the bullet
  slots (`bullet_slots_for`, life over ticks between shots) are static. The real-scale set's
  longest life is now 23 ticks (was 18, the MG 151/20's 9e-4 drag), so the first build on
  v6 compiles new shapes; afterwards a drag, range or muzzle change that raises the longest
  life (or a slot count) rebuilds, one that stays under it does not.
- Gun ranges: the card now flies its range rounded up to a whole tick (the P-51D's 587 m
  handle flies 610 m); the scripted agent reads the flown range (`characteristics`).
