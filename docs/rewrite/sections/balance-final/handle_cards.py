"""Card candidates in handle space: each ship file's handles times factors, re-solved into the
card (`balance-sizes/handles.py`'s `card_variant`), written as sweep.py candidates (the card
fields that change, as {"set": value}).

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/balance-final/handle_cards.py \
        SPEC.json OUT.json

SPEC: a list of {"name", "energy_handles": {handle: factor}, "turn_handles": {...}, plus
doctrine fields as in sweep.py ("energy", "turn")}.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "balance-sizes"))

from handles import card_variant  # noqa: E402


def main() -> None:
    spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out = []
    for entry in spec:
        entry = dict(entry)
        energy = card_variant("yongma", entry.pop("energy_handles", {}))
        turn = card_variant("jujak", entry.pop("turn_handles", {}))
        if energy or turn:
            entry["card"] = {"energy": energy, "turn": turn}
        out.append(entry | {"handles": spec[len(out)]})
        print(entry["name"], {k: round(v["set"], 5) for k, v in (energy | turn).items()}, flush=True)
    pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
