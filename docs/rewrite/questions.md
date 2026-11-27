# Questions for the author (branch scripted-lookahead, overnight 2026-11-27)

Raised by the round-3 agent; each with the decision-safe choice made meanwhile.

1. **The cold compile is still over the 260 s budget.** The steady-flight table removed every
   flight-function evaluation from the agent's graph (`act` 7,864 -> 4,494 traced operations;
   cold 333 -> 280.2 s on a quiet laptop at the final pass; 311.8 s at step 1 with another
   agent's CPU work running). What is left is the unrolled segment loop: per segment the medium (5 x about 90 operations), the nose spring in closed
   form (5 x 60), the lookup (8 x 40), plus the path arithmetic. Options: (a) the paths as one
   Triton kernel, like the pair stage (the speed order's last step; the largest cut, a day's
   work with its twin and test); (b) fewer path segments (4 instead of 5); (c) a larger budget
   for the agent's graph. Meanwhile: nothing changed beyond the table; the baseline is not
   updated.
2. **How much should the store be worth (`store_weight`)?** 0.25 is set (store mean 0.44-0.51,
   dive 23-25%, climb 22-28% of ship-ticks; beats round 2 68% in 1v1, 57% in 4v4). In 1v1 duels
   more keeps winning (0.4 beats 0.25 58%, 0.6 beats 0.4 62%) while 4v4 is even and the dive
   goes out of use (9% at 0.4, 1-5% at 0.6) with fewer kills per minute. Which matters more for
   the yardstick: 1v1 strength, or balanced use of the powers and a livelier 4v4? Progress.md
   round 3 step 3 has the tables.
3. **Weakening the preferred-speed term** (your suggestion) was measured: `speed_weight` 0.15
   with store weight 0.25 makes ships slow (0.61-0.68 corner) and loses to round 2 (44% 1v1,
   47% 4v4). Kept at 0.3.

# Questions for the author (branch `bullet-deflector`, overnight 2026-11-27)

Each has the decision-safe choice I made so the work could go on; all of it is a proposal.

1. **The bend's angle.** "The matching small angle" is not pinned down. I turn the bullet by
   atan(D_eff / deflector_length) with deflector_length 48 m (4 hit radii) in every preset: the
   frontal glance is 14 deg, the 3 m side push 3.6 deg. A runtime input (workbench, sandbox
   slider), so it can be set by eye. Is a fixed length right, or should the angle follow from
   something else (the speed, the field's size)?
2. **Reach of the field.** It pushes near misses too, out to hit radius + D (pushed to at most
   hit radius + D), so a bullet just missing a ship is bent a little. Keep, or bend hits only?
3. **Dead-centre bullets under "frontal".** The side of the push is the smooth sign b /
   sqrt(b^2 + eps^2) (eps 0.1 m), so a bullet within about 0.1 m of the centre is barely pushed
   or turned: it flies through the ship (no damage: contact and damage use |b| + D). Visible in
   bend_still.png. Rare in play (none counted in the god-mode runs as damage); acceptable?
4. **The side D (3 m) costs hits everywhere.** Symmetric and frontal both shrink the effective
   hit radius off the nose from 8.5 to 5.5 m (-35%); in god-mode play damaging hits per minute
   fall about 25% and kills per minute in 1v1 about 20% (progress.md). Is that wanted (longer
   fights), or should the side D be smaller (1 m: 7.5 m effective)?
5. **Default preset.** I set `GameConfig.deflector = "frontal"` (features on by default); the
   measured play numbers of symmetric and frontal are within noise of each other; frontal
   replaces the nose cone's special damage rule with the deflector and makes the frontal
   bullets visibly glance off (and fly on: they can hit a ship further along).
6. **Collisions' blame** still uses the nose cone's angles (`aspect_factor`), not the deflector's.
   Under "frontal" the nose cone's damage rule is off but its angles still set the blame.

# Questions for the author (branch `deflector-lookahead`, the egg, 2026-11-27 morning)

Each with the decision-safe choice made meanwhile; all of it is a proposal.

1. **The turned bullet's velocity after the tick.** The turn is in the relative frame (bullet
   minus ship), so the bullet's new map velocity is the ship's average velocity over the tick
   plus the turned relative velocity: its speed relative to the ship is kept, its map speed
   changes by the ship's part (up to about 130 m/s at a 45 deg turn of a ship at cruise). The
   other choice (rotate the map direction, map speed kept) bends the next ticks' relative path
   by (R - I) x the ship's velocity: it narrowed the egg's hit width astern at 100-200 m from
   4.7 to 3.7-3.9 m. Chosen: the relative frame (the sketch's numbers at every range).
2. **One egg per bullet per tick.** A bullet is turned by the first egg it enters in a tick;
   its contacts with other ships in that tick stay on the straight path (a second egg in the
   same 60 m move is not seen). Rare in 1v1; in tight 4v4 formations it can matter. Keep?
3. **The egg's frame** uses the ship's heading after the tick's move (as the hit angle does).
4. **The agent's lead error** (`lead_error` 0.25 of the target's sustained turn acceleration,
   spread over the flight time with the bullet spread into the trigger's hit chance) is a
   reasoned choice, not tuned; the egg experiments report how often it fires.
5. **Under GameConfig(deflector="off")** the tick skips the egg's code, but the hit-width
   column follows the rows (the "off" preset's turn 0: 24 m), so the agent stays consistent;
   a slider that sets a turn under "off" would make the agent's table disagree with the tick.
