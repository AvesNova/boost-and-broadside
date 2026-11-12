"""Write r9 with the confirmed maneuvers as a player (`Player.named` form) to
`champion.json`: the duel one-circle sense (mirror only, alone) and the crowd extension,
on both types. Not stored as the agent's default (`scripted_doctrines.json`): that is
the author's call.

    python docs/rewrite/sections/maneuvers/champion.py
"""

import dataclasses
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "game-loop"))
sys.path.insert(0, str(HERE))

import league  # noqa: E402
from make import DUEL, STRAIGHT  # noqa: E402
from sizes import overridden  # noqa: E402

FIELDS = dict(DUEL, **STRAIGHT)


def main() -> None:
    r9 = next(p for p in league.load() if p.name == "r9")
    player = dataclasses.replace(
        r9,
        name="r9m",
        energy=overridden(r9.energy, FIELDS),
        turn=overridden(r9.turn, FIELDS),
    )
    (HERE / "champion.json").write_text(json.dumps(player.named(), indent=1) + "\n")
    print(json.dumps(FIELDS))


if __name__ == "__main__":
    main()
