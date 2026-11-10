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
those games. Version 2 is the game at 2026-11-10.
"""

VERSION = 2
FINGERPRINT = "f7a190912e80a08a"
