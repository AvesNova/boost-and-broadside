"""Put the third ship into the game on this branch: its ship file as the real-scale set's
third type (Yongma 0, Jujak 1, Bonghwang 2), its doctrine into `imugi_doctrines.json`, the
balance agent's v23 Jujak card (tune/balance-final c875bb84: sustained turn x0.93) into
jujak.toml, and play maps for its matchups.

    PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/third-ship/apply.py \
        candidates/FILE.json CARD_NAME [DOCTRINE_FILE.json DOCTRINE_NAME]

Run it again with another candidate to replace the ship (its file, doctrine and maps).
"""

import argparse
import json
import pathlib
import tomllib

import numpy

from bnb.controllers import imugi_doctrine
from bnb.game import ships

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
KEY, NAME = "bonghwang", "Bonghwang"
SOURCE = "the third-ship search (docs/rewrite/sections/third-ship/results.md, 2026-11-16)"
JUJAK_V23 = HERE / "candidates" / "jujak-v23-card.json"
REASONS = {
    "stall_speed": "the Bf 109K-4's 94.4 (real 75-105.6): a heavy wing; the power-and-roll fighter",
    "g_limit": "the Fw 190A-8's and Bf 109K-4's 7 (real 6-8)",
    "cruise_speed": "330 (real 261-336; the Bf 109K-4's 330.6)",
    "climb_time": "the Tempest V's 290 s, the real best (real 290-555)",
    "low_speed_acceleration": "the Bf 109K-4's 18.97, the real best (real 8.87-18.97)",
    "sustained_turn_rate": "0.63 rad/s, between Yongma's and Jujak's (real 0.503-0.785)",
    "nose_reversal_time": "the Fw 190A-8's 0.249 s, the real fastest roll (real 0.249-0.623)",
    "dive_acceleration": "the A6M5's 43.6, the real floor (real 43.6-44.3): no dive escape",
    "zero_lift_drag": "7.0e-5 (real 4.97e-5 to 1.06e-4; the Bf 109K-4's 7.03e-5): the draggiest of the three",
    "max_shield": "between Jujak's 80 and Yongma's 142",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cards", type=pathlib.Path)
    parser.add_argument("name")
    parser.add_argument("doctrines", type=pathlib.Path, nargs="?")
    parser.add_argument("doctrine_name", nargs="?")
    parser.add_argument(
        "--option",
        help="a balance option's folder (candidates/options/NAME): its Yongma and Jujak files "
        "replace the set's instead of the v23 Jujak",
    )
    args = parser.parse_args()
    entry = next(e for e in json.loads((HERE / args.cards).read_text()) if e["name"] == args.name)
    directory = ships.directory_of(SET)
    jujak_path = directory / "jujak.toml"
    if args.option:  # the balance option's pair, as its files stand
        for key in ("yongma", "jujak"):
            source = HERE / "candidates" / "options" / args.option / f"{key}.toml"
            (directory / f"{key}.toml").write_text(source.read_text())
    else:
        # Jujak's v23 card (the balance agent's adoption)
        data = tomllib.loads(jujak_path.read_text())
        for field, change in json.loads(JUJAK_V23.read_text()).items():
            data["card"][field] = change["set"]
        data["handles"]["sustained_turn_rate"] = 0.7646540176662846 * 0.93
        tag = "; the energy-turn balance (tune/balance-final, game v23): x0.93"
        note = data["notes"]["sustained_turn_rate"].split(tag)[0]
        data["notes"]["sustained_turn_rate"] = note + tag
        ships.write_ship(
            ships.Ship("jujak", data["name"], data["handles"], data["card"], data["notes"]), SET
        )
    # the third ship
    handles = dict(entry["solved_handles"])
    jujak = tomllib.loads(jujak_path.read_text())["handles"]
    notes = {}
    for name, value in handles.items():
        if name in REASONS:
            notes[name] = REASONS[name]
        elif value == jujak.get(name):
            notes[name] = "Jujak's"
        else:
            notes[name] = SOURCE
    notes["hit_damage"] = (
        f"x{handles['hit_damage'] / 19.1877:.3f} Jujak's 19.19 (Yongma's 22.98): the set's own "
        "damage scale, tuned for the balance"
    )
    # the card as the game stores it: float32 values (a combat-only variant's are float64)
    card = {k: float(numpy.float32(v)) for k, v in entry["card"].items()}
    ship = ships.Ship(KEY, NAME, handles, card, notes)
    order = tomllib.loads((directory / "order.toml").read_text())["order"]
    if KEY in order:
        ships.write_ship(ship, SET)
    else:
        ships.append(ship, SET)
    # the doctrine
    stored = json.loads(imugi_doctrine.DOCTRINES.read_text())
    base = stored[SET]["jujak"]["doctrine"]
    doctrine = dict(base)
    if args.doctrines:
        candidate = next(
            e
            for e in json.loads((HERE / args.doctrines).read_text())
            if e["name"] == args.doctrine_name
        )
        doctrine |= candidate.get("doctrine", {})
    stored[SET][KEY] = {
        "source": f"{SOURCE}: Jujak's stored doctrine with the search's fields",
        "doctrine": doctrine,
    }
    imugi_doctrine.DOCTRINES.write_text(json.dumps(stored, indent=1) + "\n")
    # play maps (a map's lineup is both teams'): the mirror, each pair mixed, all three
    maps = pathlib.Path(ships.__file__).parent / "maps"
    template = json.loads((maps / "yongma-jujak.json").read_text())
    for file, lineup in {
        "bonghwang-mirror.json": ["Bonghwang"] * 5,
        "bonghwang-jujak.json": ["Bonghwang", "Jujak", "Bonghwang", "Jujak", "Bonghwang"],
        "bonghwang-yongma.json": ["Bonghwang", "Yongma", "Bonghwang", "Yongma", "Bonghwang"],
        "three-ships.json": ["Bonghwang", "Yongma", "Jujak", "Bonghwang", "Jujak"],
    }.items():
        (maps / file).write_text(json.dumps(template | {"lineup": lineup}, indent=1) + "\n")
    print(f"wrote {directory / (KEY + '.toml')}, jujak.toml, the doctrine and 4 maps")


if __name__ == "__main__":
    main()
