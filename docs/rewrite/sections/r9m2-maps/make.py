"""The play maps as maneuvers-2 `sizes.py` candidates, each flown by r9 and by r9m2.

    python docs/rewrite/sections/r9m2-maps/make.py [r9m2-maps] [r9 r9m2 r9m]

Each map's Yongma and Jujak card fields become absolute card values ({"set": value}) on
`bench.game_loop`'s base card (the real-scale Yongma and Jujak rows: the map's unlisted
fields are the ship files', as in the play shell). r9 is the league player; r9m2 is r9 with
every doctrine field and scripted parameter of maneuvers-2's `champion2.json`; r9m is
maneuvers-2's `champion.json` (the player `sizes.py` adds by that name). Writes
`../maneuvers-2/<tag>/candidates.json` (sizes.py reads candidates beside itself).
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[3]
MAPS = (
    "arch-standard-hit-and-run",
    "arch-rate-speed",
    "arch-standard-brawler",
    "arch-gunner-close",
    "armour-tank",
    "balance-d2",
)


def card(ships: dict, key: str) -> dict:
    return {name: {"set": float(value)} for name, value in ships.get(key, {}).items()}


def main() -> None:
    champion = json.loads((HERE.parent / "maneuvers-2" / "champion2.json").read_text())
    r9m2 = dict(
        doctrine=dict(energy=champion["yongma"], turn=champion["jujak"]),
        parameters={str(i): v for i, v in enumerate(champion["parameters"])},
    )
    tag = sys.argv[1] if len(sys.argv) > 1 else "r9m2-maps"
    doctrines = sys.argv[2:] or ["r9", "r9m2"]
    extra = {"r9": {}, "r9m2": r9m2, "r9m": dict(player="r9m")}
    out = []
    for name in MAPS:
        data = json.loads((ROOT / "src/bnb/game/maps" / f"{name}.json").read_text())
        energy, turn = card(data["ships"], "yongma"), card(data["ships"], "jujak")
        for doctrine in doctrines:
            out.append(
                dict(name=f"{name} | {doctrine}", energy=energy, turn=turn, **extra[doctrine])
            )
    path = HERE.parent / "maneuvers-2" / tag / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
