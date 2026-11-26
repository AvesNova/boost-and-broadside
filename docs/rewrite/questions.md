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
