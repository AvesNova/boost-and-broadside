"""A candidate card as a saved project map the play shell opens by name: the real-scale
deathmatch game on the real-scale set, with the candidate's Yongma and Jujak card fields
as the map's card tuning (`bnb.play.maps`; the ship files and the play default untouched).

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/tank-armour/make_map.py round1/candidates.json 1 armour-x
    bnb play --preset armour-x
"""

import dataclasses
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

from bench.game_loop import PRESET, base_card, variant_types  # noqa: E402
from bnb.game import api  # noqa: E402
from bnb.play.maps import PROJECT, from_json  # noqa: E402


def main() -> None:
    path, index, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    candidate = json.loads((HERE / path).read_text())[index]
    if candidate.get("doctrine") or candidate.get("parameters"):
        raise SystemExit("a map carries a card, not doctrines or scripted parameters")
    base = base_card("cpu")
    types = variant_types(base, [candidate["energy"]], [candidate["turn"]])
    ships = {}
    for row, key in enumerate(("yongma", "jujak")):
        changed = {
            field.name: float(getattr(types, field.name)[row])
            for field in dataclasses.fields(types)
            if float(getattr(types, field.name)[row]) != float(getattr(base, field.name)[row])
        }
        if changed:
            ships[key] = changed
    game = dataclasses.asdict(api.PRESETS[PRESET])
    data = {"game": game, "lineup": None, "ships": ships}
    from_json(name, json.loads(json.dumps(data)), "project")  # fails on anything unknown
    PROJECT.mkdir(parents=True, exist_ok=True)
    (PROJECT / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"{PROJECT / f'{name}.json'}: {candidate['name']}: {ships}")


if __name__ == "__main__":
    main()
