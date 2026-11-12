# Tank armour and dive strength: progress

Branch `tune/tank-armour` (from `origin/tune/balance-sizes` 7cf7f846). The task: end the
mirror joust and balance the duels through the ship card, not the agent (duel-deny's
finding: turning off a head-on line loses to a ship that holds it, so the fix must make the
joust a bad trade). Hard stop 01:30 KST. Power profile: **performance** (read 23:03). Every
GPU job a `systemd-run --user` unit with `MemoryMax=8G`, one at a time (two orphaned
multiprocessing workers of 128 MiB each, more than a day old, sat on the GPU throughout;
left alone on the coordinator's word).

## Code

- **The armour cone's sharpness** (`head_on_sharpness`, a card field, the target's): the
  aspect is now `1 - (1 - f0) exp(-ln2 (theta / theta_h)^p)` (`physics/combat.py`), 2 by
  default (the old Gaussian cone, every ship file at 2.0). A high `p` is a tank's nose:
  flat near `f0` inside the cone and full damage soon past `theta_h`. It is a card tensor
  (`per_ship`), so a sweep never recompiles; it is a handle too (`Handles.head_on_sharpness`,
  carried through `weapons.combat_card` / `combat_handles` unchanged), so the tuning panel
  and the solver keep working. Test: `test_head_on_sharpness_is_the_exponent_of_the_targets_cone`.
- **Fight statistics** in `bench/game_loop.py` (`fight_choices` in every job summary),
  from the post-merge window (`CIRCLE_WINDOW`, 3 s after a head-on merge ends):
  - `two_circles`: each two-circle fight's turn circles, from both ships' mean turn
    centre (position + speed / turn rate along the left normal of the flight path) and
    radius over the window's ticks where both turn: **overlapping** when the centres are
    nearer than the radii's sum, **side by side** otherwise; with the share that ends in a
    kill in the pair before its next merge ends (or either ship dies or respawns).
  - `post_merge_choices`, by the chooser's type: **same sense** (turning in the bandit's
    rotation sense, the two-circle choice), **opposite sense** (one-circle), **extend**
    (turning less than half the window), **turn v extender** (the bandit extended); each
    with its outcome until the pair's next merge ends: the chooser kills, is killed,
    neither. A choice still open when the game ends is not counted.

## Tools

`sizes.py`, `trajectories.py` (duel-deny's, unchanged), `table.py` (all the goals in one
row; `--fights` for the new statistics), `make_map.py` (balance-sizes'), `launch.sh`.
Candidates set the armour on both types with `{"set": value}`: `f0` as
`head_on_fraction_scale` (of 0.3), `theta_h` as `head_on_angle_scale` (of 42.27 degrees;
today's cards fly 0.5, 21.1 degrees), `p` as `head_on_sharpness`. The dive exchange as
`dive_exchange_scale` (of `flight.dive_exchange` 9.81 m/s^2: today Yongma 0.626 = 6.1,
Jujak 0.314 = 3.1).
