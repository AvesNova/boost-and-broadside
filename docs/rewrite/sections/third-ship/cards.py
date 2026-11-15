"""Third-ship cards: handle sets re-solved into cards (`api.solve_handles`, as the tuning
panel does), each a base ship file's handles with some handles multiplied or set.

    PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/third-ship/cards.py \
        SPEC.json OUT.json [--processes 8]

SPEC: a list of {"name", "base": "jujak"|"yongma", "handles": {handle: factor or
{"set": value}}, plus anything else (kept, e.g. a "doctrine")}. OUT: the same entries with
"card": every card field of the solve (absolute values) and "solved_handles": the handles
the solve was given. A handle set that does not solve is reported and left out.
"""

import argparse
import concurrent.futures
import json
import pathlib
import tomllib

SET = "real-scale"


def handles_of(entry: dict) -> dict:
    """The entry's absolute handles: its base file's with the entry's changes."""
    from bnb.game import ships

    text = (ships.directory_of(SET) / f"{entry.get('base', 'jujak')}.toml").read_text()
    handles = dict(tomllib.loads(text)["handles"])
    for name, change in entry.get("handles", {}).items():
        if isinstance(change, dict):
            handles[name] = float(change["set"])
        else:
            handles[name] = handles[name] * float(change)
    return handles


def solve_one(entry: dict) -> dict | str:
    """The entry's full card (or the failure's text)."""
    import torch

    torch.set_num_threads(2)
    from bnb.game import api
    from bnb.game.handling import Handles
    from bnb.game.handling.errors import Infeasible

    handles = handles_of(entry)
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"], types=api.ship_types(ship_set=SET))
    try:
        solution = api.solve_handles(
            Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
            rules,
        )
    except Infeasible as error:
        return f"infeasible: {error}"
    fields = api.ship_types(ship_set=SET).__dataclass_fields__
    card = {f: float(getattr(solution.types, f)[0]) for f in fields}
    return {"card": card, "solved_handles": handles}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    done = {}
    if args.out.exists():
        done = {e["name"]: e for e in json.loads(args.out.read_text())}
    todo = [e for e in spec if e["name"] not in done or done[e["name"]].get("spec") != e]
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        results = dict(zip([e["name"] for e in todo], pool.map(solve_one, todo), strict=True))
    out = []
    for entry in spec:
        if entry["name"] in results:
            result = results[entry["name"]]
            if isinstance(result, str):
                print(entry["name"], result[:300], flush=True)
                continue
            out.append(entry | result | {"spec": entry})
            print(entry["name"], "solved", flush=True)
        elif entry["name"] in done:
            out.append(done[entry["name"]])
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
