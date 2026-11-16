"""Write a balance candidate (`candidates/final.json`, or another file in candidates/ with
`--candidates`) into the game: its handle changes into the real-scale ship files (the card
fields from its handle solve, `handle_cards.py`), its Yongma and Jujak doctrine fields into
`imugi_doctrines.json`, and copies of the files into after/ (before/ holds them as they
were).

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python \
        docs/rewrite/sections/balance-final/apply.py jt0.93

`--restore` copies before/'s files back into src/ instead (the author's before check).
"""

import argparse
import json
import pathlib
import shutil
import tomllib

from bnb.controllers import imugi_doctrine
from bnb.game import ships

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
KEYS = {"energy": "yongma", "turn": "jujak"}
HANDLE_SIDES = {"energy_handles": "yongma", "turn_handles": "jujak"}
NOTE = "the energy-turn balance (docs/rewrite/sections/balance-final/results.md, 2026-11-16)"
SOURCE = (
    "Baekho (2026-11-13) as stored for game v22, with the energy-turn balance's doctrine "
    "search (docs/rewrite/sections/balance-final/results.md, 2026-11-16) on top"
)


def files() -> dict[str, pathlib.Path]:
    directory = ships.directory_of(SET)
    return {
        "yongma.toml": directory / "yongma.toml",
        "jujak.toml": directory / "jujak.toml",
        "imugi_doctrines.json": imugi_doctrine.DOCTRINES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--candidates", default="final.json", help="a file in candidates/")
    args = parser.parse_args()
    if args.restore:
        for name, path in files().items():
            shutil.copyfile(HERE / "before" / name, path)
            print(f"restored {path}")
        return
    entries = json.loads((HERE / "candidates" / args.candidates).read_text())
    entry = next(e for e in entries if e["name"] == args.candidate)
    card = entry.get("card", {})
    for side, key in KEYS.items():
        path = ships.directory_of(SET) / f"{key}.toml"
        data = tomllib.loads((HERE / "before" / f"{key}.toml").read_text())
        handles, notes = dict(data["handles"]), dict(data.get("notes", {}))
        factors = entry.get("handles", {}).get(f"{side}_handles", {})
        for name, factor in factors.items():
            old = handles[name]
            handles[name] = old * factor
            notes[name] = f"{notes.get(name, '')} {NOTE}: {old:.6g} -> {handles[name]:.6g}"
            print(f"{key} {name}: {old:.6g} -> {handles[name]:.6g}")
        fields = dict(data["card"])
        for name, value in card.get(side, {}).items():
            print(f"  {key} card {name}: {fields[name]:.6g} -> {value['set']:.6g}")
            fields[name] = value["set"]
        ships.write_ship(ships.Ship(key, data["name"], handles, fields, notes), SET)
    stored = json.loads((HERE / "before" / "imugi_doctrines.json").read_text())
    for side, key in KEYS.items():
        if entry.get(side):
            record = stored["real-scale"][key]
            record["doctrine"] = record["doctrine"] | entry[side]
            record["source"] = SOURCE
            print(f"{key} doctrine: {sorted(entry[side])}")
    imugi_doctrine.DOCTRINES.write_text(json.dumps(stored, indent=1) + "\n")
    # after/ holds the v23 after (A); any other candidate goes to options/<name>/
    after = HERE / "after" if args.candidate == "jt0.93" else HERE / "options" / args.candidate
    after.mkdir(parents=True, exist_ok=True)
    for name, path in files().items():
        shutil.copyfile(path, after / name)
    print(f"wrote the candidate into src/ and after/ ({args.candidate})")


if __name__ == "__main__":
    main()
