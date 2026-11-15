"""Card candidates in handle space: a ship file's handles times factors, re-solved into the
card (`api.solve_handles`, as the tuning panel does), written as sweep.py candidates: the
card fields that change from the file's own card, as {"set": value}.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/balance-final/handle_cards.py \
        SPEC.json OUT.json [--processes 8]

SPEC: a list of {"name", "energy_handles": {handle: factor}, "turn_handles": {...}, plus
doctrine fields as in sweep.py ("energy", "turn")}. The solves run in parallel processes;
a handle set that does not solve (Infeasible) is reported and left out.
"""

import argparse
import concurrent.futures
import json
import pathlib
import tomllib

KEYS = {"energy_handles": "yongma", "turn_handles": "jujak"}
SET = "real-scale"


def solve_one(key: str, factors: dict) -> dict | str:
    """The card fields that change when the file's handles are multiplied (or a failure)."""
    import torch

    torch.set_num_threads(2)
    from bnb.game import api, ships
    from bnb.game.handling import Handles
    from bnb.game.handling.errors import Infeasible

    text = (ships.directory_of(SET) / f"{key}.toml").read_text()
    data = tomllib.loads(text)
    handles = dict(data["handles"])
    for name, factor in factors.items():
        handles[name] = handles[name] * factor
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))
    try:
        solution = api.solve_handles(
            Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
            rules,
        )
    except Infeasible as error:
        return f"infeasible: {error}"
    card = data["card"]
    out = {}
    for field in api.ship_types(ship_set=SET).__dataclass_fields__:
        value = float(getattr(solution.types, field)[0])
        if abs(value - card[field]) > 1e-5 * max(abs(card[field]), 1e-9):
            out[field] = {"set": value}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        futures = {
            (number, side): pool.submit(solve_one, key, entry[side])
            for number, entry in enumerate(spec)
            for side, key in KEYS.items()
            if entry.get(side)
        }
        results = {k: f.result() for k, f in futures.items()}
    out = []
    for number, entry in enumerate(spec):
        cards = {side: results.get((number, side), {}) for side in KEYS}
        failed = [v for v in cards.values() if isinstance(v, str)]
        if failed:
            print(entry["name"], failed[0][:200], flush=True)
            continue
        candidate = {k: v for k, v in entry.items() if k not in KEYS}
        candidate["handles"] = {k: entry[k] for k in KEYS if k in entry}
        if any(cards.values()):
            candidate["card"] = {"energy": cards["energy_handles"], "turn": cards["turn_handles"]}
        changed = {k: round(v["set"], 5) for c in cards.values() for k, v in c.items()}
        print(entry["name"], changed, flush=True)
        out.append(candidate)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
