# Item 12: nose armour, the aspect curve on the sine, the circle deflector, the sight egg

Branch `rules-12` (from core 748680bc). Author decisions 2026-11-28 ~16:30 (docs/rewrite/queue.md
item 12). Scripts and data: docs/rewrite/sections/rules-12/ (behaviour.py, behaviour.sh),
/home/vizia/avesnova/boost-and-broadside/data/rules-12/ (per-condition JSON), replays in
/home/vizia/avesnova/boost-and-broadside/data/replays-2026-11-28-rules12/.

## 16:40 start

- Tried: core's rules measured first in a detached worktree of core (v2-core12, 748680bc) with
  the same behaviour script, so the old-rules numbers come from the old code, not a switch.
- Worked: the first core pass (2,048 / 1,024 / 256 games) took 8 min; the counts were doubled
  for the real pass (4,096 / 2,048 / 512 games per condition).

## 17:00 the rules built

- Card: `nose_inner_angle` / `nose_outer_angle` (30 / 45 deg, the bullet armour), new
  `blame_inner_angle` / `blame_outer_angle` (0 / 45 deg), `vision_ahead` / `vision_side` /
  `vision_behind` (1,500 / 850 / 850 m) replacing `vision_range_scale` and
  `vision_eccentricity`; `card.VISION_RANGE` removed. Hit model: `deflector_radius` (24 m) and
  `deflector_turn` replace the egg's three semi-axes; `card.NOSE_CONE` 1.
- `core.aspect_factor(hit_cos, hit_sin, inner_sin, inverse_width)`: t on the sine, t = 1 behind
  the beam; `_cone_rows` builds both pairs' constants (outer angle at most 90 deg).
- `core.circle_entry` (one quadratic, no nose) and `core.circle_turn` replace `egg_entry` /
  `egg_turn`; the hit width is one number per ship (`hit_width`, 4.75 m), the 17-point
  aspect table and `core.hit_width`'s interpolation are gone; the scripted gun, trigger and
  Triton kernel read the one number and the sine.
- `core.sight_pairs`: the egg (multiplies and a sign check); tracking circle radius
  `tracking_range` = the longest reach.
- Tools, sandbox, replay script (`--vision-ahead/side/behind`), hit bench and tests follow.

## Next

Behaviour tables (old vs new rules), replays, tick timing, the end-of-branch checks.

## 17:22 behaviour measured; trimmed (author ~17:20)

- The author trimmed the matrix (~17:20): new rules with the yardstick at 1v1 / 4v4, core's
  rules with the round-3 centre at 1v1 as the degenerate baseline. Cells already finished
  are kept in docs/rewrite/sections/rules-12/tables.md (the new-rules 16v16 cells were not
  run; the run was stopped after the yardstick's 4v4 at 17:21).
- Worked: the degenerate baseline shows in the tail-chase share, not in the collision
  numbers: centre 1v1 under core 4.9% of bullet hits from the tail (the yardstick under the
  new rules 75.7%); collision deaths under core are 4.0% of deaths, mutual kills 6.6% of fatal
  collisions (rare: the replays' mutual collisions are not a large share of deaths).
- Didn't: the centre (tuned without armour) under the new rules collides more: collision
  deaths 21.0% at 1v1, 41.5% of fatal collisions kill both; its shots fall 39 -> 15 per
  ship-minute. Not re-tuned (a persona re-run is a separate proposal).
- Next (ETA 18:15, worst 18:45): yardstick replays (1v1, 4v4; running, CPU), tick timing egg
  vs circle on a quiet machine, compile check, tests, docs.
