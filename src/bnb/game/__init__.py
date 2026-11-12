"""The game: its rules, state, physics, modes and perception, used through `bnb.game.api`.

VERSION rises whenever a change alters what the game computes (a rule, a constant, the
order of a sum that changes its rounding), so policy files and ratings say which game they
were made in. A refactor that computes the same numbers keeps it. FINGERPRINT is a hash
of a short seeded match in each game preset at this version
(`tests/sections/test_game_version.py`), so a change to the numbers fails that test until
the version is raised.

Version 1 named the game from 2026-11-03 to 2026-11-10, though 36 changes to what it
computes went in under it (the observation's accelerations, whole-Hz fire rates and
instant bullet life, the time scale, the real-scale card): ratings keyed on version 1 mix
those games. Version 2 is the game at 2026-11-10. Version 3 (2026-11-11): the shield's
recharge pauses while the pilot gives an input other than a quiet one
(`physics.combat.quiet`), and the arena preset's radius is 1950 m (was 1300 m). On
2026-11-12 each type's bullet life became a card field derived from its gun range, equal to
the old shared life on every ship set, so the game's numbers stayed version 3; the
fingerprint changed only because a play-shell match allocates 60 bullet slots (was 45),
and at 45 it is the old one. Also on 2026-11-12 the elimination mode came in (its presets
real_scale_elimination and clover, fixed map layouts): a new game, rated under its own mode
and map, while every existing game computes the same numbers (the old presets' fingerprint
without the observation's new mode_elimination column, always 0 in them, is the old one),
so the version stayed 3; the fingerprint now covers the two new presets as well.
"""

VERSION = 3
FINGERPRINT = "bd69c68587244307"
