"""The balance study: scrambled-Sobol cards over the balance map's handle space
(`bench.balance_map.SPACE`), every card on all 8 conditions against one rotated reference
card, about 6% of the games card against card; the data for per-condition ability models.

    # the worker: every condition's compiled loop in one process, playing the queues
    PYTHONPATH=src:. python -m bench.balance_study worker --out OUT --lock $GPU_LOCK
    # the driver: Sobol cards solved, their tables' characteristics measured, jobs queued
    PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= python -m bench.balance_study drive --out OUT \\
        --until 00:00

Records under OUT: references.json (the reference cards), cards.jsonl (every Sobol point:
index, coordinates, handles, the solved card or the solver's failure), evaluations.jsonl
(every entry and condition: the matchup, games, kill share, kills and deaths a minute, the
score), batches.jsonl (a line a batch: counts, timings, throughput). Jobs and outputs under
OUT/queue/<condition>/ and OUT/results/<condition>/. A restarted driver continues the Sobol
sequence and submits unfinished batches again; a restarted worker resumes its queue.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import random
import time

from bench import balance_map
from bench.balance_sweep import (
    BALANCE,
    CONDITIONS,
    as_pair,
    balance_loop,
    card_table,
    chunks,
    mode_parameters,
    sweep,
    warm,
)

CONDITION_ORDER = balance_map.EVERY
REFERENCES = ("jujak", "yongma", "g-yongma")
ENVIRONMENTS = 16384
VARIANTS = 512
GAMES = 32
CROSS = 32  # card-against-card entries a batch
CARDS = VARIANTS - CROSS  # card-against-reference entries a batch
DEVICE_KEY = "cuda:0"


def kst(after: float = 0.0, seconds: bool = True) -> str:
    return balance_map.clock(after, seconds=seconds)


# --- the worker ------------------------------------------------------------------------------


def worker(args) -> None:
    """Every condition's loop compiled in turn (one cold compile at a time), then the queues
    played oldest job first under the GPU lock; each entry's line is appended as its chunk
    ends, a finished job moves to done/."""
    from bench.balance_worker import held

    loops = {}
    for name in args.conditions:
        begun = time.time()
        condition = CONDITIONS[name]
        loop = balance_loop(condition, args.environments, args.variants, BALANCE)
        warm(loop, mode_parameters(condition), args.variants)
        loops[name] = loop
        print(f"{kst()} {name}: compiled and warm in {time.time() - begun:.0f} s", flush=True)
        _rss()
    for name in loops:
        (args.out / "queue" / name / "done").mkdir(parents=True, exist_ok=True)
    print(f"{kst()} worker ready: {', '.join(loops)}", flush=True)
    while True:
        jobs = sorted(
            (path.name, name, path)
            for name in loops
            for path in (args.out / "queue" / name).glob("*.json")
        )
        if not jobs:
            time.sleep(2.0)
            continue
        _, name, path = jobs[0]
        job = json.loads(path.read_text())
        out = pathlib.Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {json.loads(line)["name"] for line in out.read_text().splitlines()}
        entries = [e for e in job["pairs"] if e["name"] not in done]
        begun = time.time()
        if entries:
            with held(args.lock):
                sweep(
                    loops[name],
                    entries,
                    out,
                    int(job["games"]),
                    0,
                    int(job["seed"]),
                    args.variants,
                    name,
                    mode_parameters(CONDITIONS[name]),
                    log=lambda line: None,
                )
        path.rename(path.parent / "done" / path.name)
        print(
            f"{kst()} {name} {path.name}: {len(entries)} entries in {time.time() - begun:.0f} s",
            flush=True,
        )


def _rss() -> None:
    try:
        text = pathlib.Path("/proc/self/status").read_text()
        line = next(line for line in text.splitlines() if line.startswith("VmRSS"))
        print(f"  {line}", flush=True)
    except OSError, StopIteration:
        pass


# --- the driver ------------------------------------------------------------------------------


def reference_cards(out: pathlib.Path) -> dict[str, dict]:
    """The reference cards ({name: {field: value}}): v23 Jujak's and Yongma's own cards, and
    balance-final's option G Yongma (v23 shields, the pilot's seed card: its handles through
    the search space, so the unsearched handles are Jujak's) solved; written once."""
    path = out / "references.json"
    if path.exists():
        return json.loads(path.read_text())
    from bnb.game import api, ships

    full = api.ship_types(ship_set=balance_map.SHIP_SET)
    keys = [ship.key for ship in ships.load(balance_map.SHIP_SET)]
    cards = {
        key: {
            f.name: float(getattr(full, f.name)[keys.index(key)]) for f in dataclasses.fields(full)
        }
        for key in ("jujak", "yongma")
    }
    # as the pilot's seed: through the search space (its low-speed acceleration 8.794 -> 8.8,
    # the space's floor; everything else exact)
    base = dict(balance_map.reference_ships()[balance_map.REFERENCE].handles)
    g = balance_map.decode(balance_map.encode(dict(balance_map.seeds())["G-yongma"]), base)
    solved = balance_map.solve_cards([g], 1)[0]
    if "card" not in solved:
        raise RuntimeError(f"G Yongma does not solve: {solved}")
    cards["g-yongma"] = solved["card"]
    path.write_text(json.dumps(cards, indent=1))
    return cards


def read_lines(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prewarm(entries: list[dict], processes: int) -> float:
    """The batch's card table characteristics measured once (row slices in parallel
    processes) and written to the disk cache under every loop's key (the key reads the
    flight rules, which no mode changes; the device is the worker's). Seconds taken."""
    import concurrent.futures
    import multiprocessing

    begun = time.time()
    pairs = [as_pair(entry) for entry in entries]
    for _, _, a_cards, b_cards in chunks(pairs, VARIANTS):
        table = card_table(a_cards, b_cards)
        rows = table.max_energy.shape[0]
        size = math.ceil(rows / processes)
        slices = [(start, min(start + size, rows)) for start in range(0, rows, size)]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(len(slices), mp_context=context) as pool:
            parts = list(pool.map(_measure_rows, [(table, s) for s in slices]))
        _store(table, parts)
    return time.time() - begun


def _rules():
    import torch

    from bench.game_loop import MODES, base_card, variant_types
    from bnb.game import api

    types = variant_types(base_card(torch.device("cpu")), [{}] * VARIANTS, [{}] * VARIANTS)
    return api.rules_of(api.PRESETS[MODES["deathmatch"]], types=types)


def _measure_rows(argument):
    import torch

    torch.set_num_threads(1)
    from bnb.game.handling import characteristics as module
    from bnb.game.ship_types import ShipTypes

    table, (start, stop) = argument
    part = ShipTypes(
        **{f.name: getattr(table, f.name)[start:stop] for f in dataclasses.fields(table)}
    )
    measured = module.measure_characteristics(part, _rules())
    return {f.name: getattr(measured, f.name) for f in dataclasses.fields(measured)}


def _store(table, parts: list[dict]) -> None:
    import os

    import torch

    from bnb.game.handling import characteristics as module

    key = list(module._key(table, _rules()))
    key[1] = DEVICE_KEY
    path = module._disk_path(tuple(key))
    if path.exists():
        return
    fields = {}
    for name in parts[0]:
        values = [p[name] for p in parts]
        fields[name] = torch.cat(values) if isinstance(values[0], torch.Tensor) else values[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial")
    torch.save(fields, partial)
    partial.replace(path)


def entries_of(batch: int, cards: list[dict], references: dict, rng: random.Random):
    """A batch's entries: each card against its rotated reference (the card's Sobol index
    mod 3), then CROSS card-against-card pairs drawn from the batch's cards."""
    entries = []
    for card in cards:
        reference = REFERENCES[card["index"] % len(REFERENCES)]
        entries.append(
            balance_map.pair_entry(f"{card['id']}|{reference}", card["card"], references[reference])
        )
    for _ in range(CROSS):
        a, b = rng.sample(cards, 2)
        entries.append(balance_map.pair_entry(f"{a['id']}~{b['id']}", a["card"], b["card"]))
    return entries


def submit(
    out: pathlib.Path,
    label: str,
    entries: list[dict],
    games: int,
    seed: int,
    conditions=CONDITION_ORDER,
) -> dict[str, pathlib.Path]:
    outputs = {}
    for condition in conditions:
        queue = out / "queue" / condition
        queue.mkdir(parents=True, exist_ok=True)
        result = out / "results" / condition / f"{label}.jsonl"
        result.parent.mkdir(parents=True, exist_ok=True)
        job = dict(pairs=entries, out=str(result), games=games, seed=seed)
        partial = queue / f"{label}.json.partial"
        partial.write_text(json.dumps(job))
        partial.rename(queue / f"{label}.json")
        outputs[condition] = result
    return outputs


def finished(outputs: dict[str, pathlib.Path], count: int) -> bool:
    return all(p.exists() and sum(1 for _ in p.open()) >= count for p in outputs.values())


def evaluations_of(label: str, outputs: dict[str, pathlib.Path]) -> list[dict]:
    rows = []
    for condition, path in outputs.items():
        for record in read_lines(path):
            job = record["jobs"]["energy v turn"]
            name = record["name"]
            a, b = name.split("~") if "~" in name else name.split("|")
            rows.append(
                dict(
                    name=name,
                    a=a,
                    b=b,
                    kind="cross" if "~" in name else "reference",
                    condition=condition,
                    batch=label,
                    seed=record["seed"],
                    games=job["games"],
                    kill_share=job["kill_share_a"],
                    point_share=job["point_share_a"],
                    score=job["score"],
                    kills_per_minute=job["kills_per_minute"],
                    deaths_per_minute=job["deaths_per_minute"],
                    match_seconds=job["match_seconds"],
                    kills_per_game=job.get("kills_per_game"),
                )
            )
    return rows


def until_seconds(clock: str) -> float:
    """Seconds from now to the next HH:MM KST."""
    hours, minutes = map(int, clock.split(":"))
    now = time.gmtime(time.time() + 9 * 3600)
    target = hours * 3600 + minutes * 60
    current = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    return (target - current) % 86400


def drive(args) -> None:
    """Sobol cards, solved in the persistent pool, CARDS valid ones a batch, their table's
    characteristics measured, queued on all 8 conditions; up to --ahead batches in flight;
    each finished batch's evaluations appended. Stops queuing at --until (KST)."""
    from scipy.stats import qmc

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    references = reference_cards(out)
    base = dict(balance_map.reference_ships()[balance_map.REFERENCE].handles)
    cards = read_lines(out / "cards.jsonl")
    evaluated = {row["batch"] for row in read_lines(out / "evaluations.jsonl")}
    sobol = qmc.Sobol(len(balance_map.SPACE), scramble=True, seed=args.seed)
    next_index = max((c["index"] for c in cards), default=-1) + 1
    if next_index:
        sobol.fast_forward(next_index)
    batched = {c["id"] for c in cards if c.get("batch") is not None}
    unused = [c for c in cards if "card" in c and c.get("batch") is None and c["id"] not in batched]
    batches = sorted({c["batch"] for c in cards if c.get("batch") is not None})
    in_flight = {}
    rng = random.Random(args.seed)
    for label in batches:  # an unfinished batch: queued again
        if label not in evaluated:
            mine = [c for c in cards if c.get("batch") == label]
            entries = entries_of(label, mine, references, random.Random(label))
            in_flight[label] = dict(
                entries=entries,
                cards=len(mine),
                begun=time.time(),
                outputs=submit(out, label, entries, args.games, 0),
            )
            print(f"{kst()} {label}: queued again", flush=True)
    number = len(batches)
    deadline = time.time() + until_seconds(args.until)
    started, played_cards = time.time(), 0
    while True:
        for label in [k for k, v in in_flight.items() if finished(v["outputs"], len(v["entries"]))]:
            flight = in_flight.pop(label)
            rows = evaluations_of(label, flight["outputs"])
            append(out / "evaluations.jsonl", rows)
            played_cards += flight["cards"]
            took = time.time() - flight["begun"]
            rate = played_cards / max(time.time() - started, 1.0) * 3600.0
            line = dict(
                batch=label,
                time=kst(),
                cards=flight["cards"],
                entries=len(flight["entries"]),
                seconds=round(took),
                cards_per_hour=round(rate),
                solved=sum(1 for c in read_lines(out / "cards.jsonl") if "card" in c),
                asked=next_index,
            )
            append(out / "batches.jsonl", [line])
            left = max(deadline - time.time(), 0.0)
            print(
                f"{kst()} {label} done: {flight['cards']} cards x 8 conditions in "
                f"{took / 60:.1f} min; {rate:.0f} cards an hour so far; queuing stops at "
                f"{args.until} KST (about {rate * left / 3600:.0f} more cards)",
                flush=True,
            )
        if time.time() >= deadline:
            if not in_flight:
                break
            time.sleep(5.0)
            continue
        if len(in_flight) >= args.ahead:
            time.sleep(5.0)
            continue
        begun = time.time()
        while len(unused) < CARDS:  # Sobol points until a batch's worth solve
            points = sobol.random(128)
            handle_sets = [balance_map.decode(x, base) for x in points]
            results = balance_map.solve_cards(handle_sets, args.processes)
            rows = []
            for offset, (x, handles, result) in enumerate(
                zip(points, handle_sets, results, strict=True)
            ):
                handles = dict(handles, low_speed_acceleration=result["low_speed_acceleration"])
                row = dict(
                    id=f"s{next_index + offset:06d}",
                    index=next_index + offset,
                    coordinates=[round(float(v), 6) for v in x],
                    handles={k: handles[k] for k in balance_map.STORED},
                    low_speed_limit=result["low_speed_limit"],
                )
                row.update(
                    {"card": result["card"]} if "card" in result else {"failure": result["failure"]}
                )
                rows.append(row)
            next_index += len(points)
            append(out / "cards.jsonl", rows)  # unbatched until used
            unused += [r for r in rows if "card" in r]
        solved_seconds = time.time() - begun
        label = f"b{number:04d}"
        mine, unused = unused[:CARDS], unused[CARDS:]
        append(out / "cards.jsonl", [dict(c, batch=label) for c in mine])  # the batch's copy
        entries = entries_of(label, mine, references, rng)
        characteristic_seconds = prewarm(entries, args.processes)
        in_flight[label] = dict(
            entries=entries,
            cards=len(mine),
            begun=time.time(),
            outputs=submit(out, label, entries, args.games, 0),
        )
        print(
            f"{kst()} {label}: queued {len(mine)} cards + {CROSS} card-against-card "
            f"(solve {solved_seconds:.0f} s, characteristics {characteristic_seconds:.0f} s; "
            f"Sobol index {next_index})",
            flush=True,
        )
        number += 1
    print(f"{kst()} drive done", flush=True)


def confirm(args) -> None:
    """Pairs ({"a", "b"} card ids, `study_analysis` pairs.json) played against each other
    on all 8 conditions at --games (seed 2, games no batch saw); appends confirm.jsonl and
    prints each pair's win-rate equivalents and its worst condition."""
    out = args.out
    cards = {c["id"]: c for c in read_lines(out / "cards.jsonl") if "card" in c}
    pairs = json.loads(args.pairs.read_text())
    entries = [
        balance_map.pair_entry(f"{p['a']}~{p['b']}", cards[p["a"]]["card"], cards[p["b"]]["card"])
        for p in pairs
    ]
    outputs = submit(out, args.label, entries, args.games, 2)
    while not finished(outputs, len(entries)):
        time.sleep(10.0)
    rows = evaluations_of(args.label, outputs)
    append(out / "confirm.jsonl", rows)
    print("| pair | " + " | ".join(CONDITION_ORDER) + " | worst |")
    for entry in entries:
        found = {r["condition"]: r for r in rows if r["name"] == entry["name"]}
        wins = {
            c: balance_map.win_equivalent(found[c]["kill_share"][0], c) for c in CONDITION_ORDER
        }
        worst = max(abs(v - 50.0) for v in wins.values())
        print(
            f"| {entry['name']} | "
            + " | ".join(f"{wins[c]:.1f}" for c in CONDITION_ORDER)
            + f" | {worst:.1f} |",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("worker")
    one.add_argument("--out", type=pathlib.Path, required=True)
    one.add_argument("--conditions", nargs="*", default=list(CONDITION_ORDER))
    one.add_argument("--environments", type=int, default=ENVIRONMENTS)
    one.add_argument("--variants", type=int, default=VARIANTS)
    one.add_argument("--lock", type=pathlib.Path, default=None)
    one = sub.add_parser("drive")
    one.add_argument("--out", type=pathlib.Path, required=True)
    one.add_argument("--until", default="00:00", help="KST: queue no batch after this")
    one.add_argument("--games", type=int, default=GAMES)
    one.add_argument("--ahead", type=int, default=2, help="batches queued at once")
    one.add_argument("--processes", type=int, default=8)
    one.add_argument("--seed", type=int, default=0)
    one = sub.add_parser("confirm")
    one.add_argument("--out", type=pathlib.Path, required=True)
    one.add_argument("--pairs", type=pathlib.Path, required=True)
    one.add_argument("--games", type=int, default=2048)
    one.add_argument("--label", default="confirm")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    elif args.command == "confirm":
        confirm(args)
    else:
        drive(args)


if __name__ == "__main__":
    main()
