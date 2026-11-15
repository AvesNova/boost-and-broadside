# Final freeze check: progress

Branch `check/freeze-final` (from `feat/imugi-boundary` a8919b27). Notes in order.

## 1. vizia out of the real-scale set (2026-11-15)

- **Tried:** removed `vizia` from `order.toml` and deleted `vizia.toml`; updated the tests,
  two bench defaults and the docs that listed it.
- **Worked:** Yongma's and Jujak's files hold absolute handles; their notes keep the values
  taken from vizia ("vizia's 162.7 x 1.0292"), so nothing reads `vizia.toml` to build or
  solve them. The derivation's table stays in `archetypes.md` section 4; the file itself is
  in git before this commit. The game-version reference still matches (vizia was type 2;
  the default lineup only flies types 0 and 1), so **no version bump** for this step.
- **Found and fixed:** Imugi chose its stored doctrines by the card's type count alone. With
  vizia gone the real-scale set has two types, like the standard set, so a standard card
  would have flown Yongma's and Jujak's doctrines (`test_imugi_memory` caught it). The
  lookup now matches the type count and the scale (`real_scale_card` of the card's gravity):
  `imugi_doctrine.for_card`, `rows_by_kind`, `ImugiStep.table` (a `torch.where` on the
  gravity, as the step is exported). Real-scale Imugi flies the same doctrines as before.
- **Note:** the real-scale card now has 2 types instead of 3, so anything sized by the type
  count (Yemong's card input) differs from a model saved before this change.
- Two quick-tier failures exist on `feat/imugi-boundary` before this change and are not
  touched here: `test_ship_sets.py::test_training_takes_its_games_set_card` (a set without
  drawn lineup types) and `test_play_render.py::test_an_edited_colour_draws_on_the_next_frame`.

## 2. Shield check: a frontal hit with no damage (2026-11-15)

- **Found a bug.** `combat.apply_damage` restarted the recharge wait on `total > 0` only.
  Yongma's and Jujak's head-on fraction is 0 with a sharpness of 3, so a bullet within about
  0.2 degrees of the nose deals exactly 0.0 in float32: such a hit was a hit (the bullet is
  spent, it has a first shooter) but the wait ran on and the shield recharged that tick.
  Rare in play (a tiny cone; just outside it the damage is tiny but positive, which did
  restart the wait), but against the rule that any hit restarts it.
- **Fix:** `apply_damage` takes `struck` (a bullet hit the ship this tick, from the tick's
  `first_shooter`, after Frontline's protection); a hit or damage restarts the wait and
  blocks that tick's recharge. Damage alone still decides breaks and kills.
- **Tests:** `test_shield_recharge.py`: an end-to-end real-scale tick with a bullet exactly
  through the target's nose (no damage, the wait at the delay, nothing restored; the quiet
  shooter recharges). It fails on `feat/imugi-boundary` (wait 0.0 against 3.0 s) and passes
  here; plus a unit case of `struck` without damage.
- **Game version 22** (one bump for the check; the docstring notes the shield and vizia).
  The committed reference was regenerated; its 48 random ticks had no such hit, so its numbers
  did not change.

## 3. Respawn check (report only, nothing changed)

`tune/respawn-waves` was deleted in a cleanup; its last commit 972fd4dc is still readable
(`git show 972fd4dc:docs/rewrite/sections/respawn-waves/progress.md`). Its recommendation
(2026-11-13): Frontline a minimum 8 s then waves every 20 s, no capture effect; deathmatch a
fixed 8 s, no waves. What it found about waves: at 5v5 they leave P(same team next) at
0.47-0.53 (no momentum), but they do shorten Frontline games (median 506 against 570 s,
first capture earlier); in deathmatch they cost 7-14% of the kill rate.

The game now (v22, `api.rules_of` of each preset):

| preset | respawn wait | waves |
|---|---|---|
| real_scale_deathmatch | 10 s fixed (preset override) | none (waves are Frontline only) |
| real_scale_frontline | 8 s minimum | every 20 s per team (`[mode] wave_seconds`) |
| real_scale_elimination | none (no respawn) | none |

So Frontline follows the recommendation; deathmatch does not: it stays at 10 s, which
`modes/rules.py` records as the author's choice of 2026-11-13 ("4200 m, with a 10 s respawn
wait (the clock's would be 8 s)"). The branch measured 8 s at +3% kills a minute against
10 s, and life and trades unchanged.
