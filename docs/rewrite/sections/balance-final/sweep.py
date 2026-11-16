"""Doctrine candidates for the energy-against-turn balance (game v22, real scale), on the
game-loop harness (`bench.game_loop.Loop`: every candidate's games in one compiled graph).

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/balance-final/sweep.py \
        --mode deathmatch --candidates candidates/round1.json --out data/round1-dm.jsonl

A candidate is the stored doctrines (`imugi_doctrines.json` as it was before this work:
before/imugi_doctrines.json) with some fields replaced, {"name", "energy": {...},
"turn": {...}} (`imugi_doctrine.named`'s form; a logistic is [centre, scale]); an optional "card":
{"energy": {...}, "turn": {...}} multiplies the card's fields (`bench.game_loop.variant_types`:
a field name -> factor, or {"set": value}) for all three of its jobs. Per
candidate, as side A, the jobs:

- "energy v turn" against itself: a Yongma team against a Jujak team, both flying the
  candidate's doctrines (A's score is the Yongma team's), `--games` games;
- "mixed" and "mixed reversed" against the before doctrines (`--games` / 2 each): the
  candidate's strength against the current Imugi on mixed lineups.

The game: the mode's preset with its fields, its own map, 5v5 (`--ships`), Imugi's mode
parameter set on both sides (`imugi_defaults.MODE_DEFAULTS`), fog on; deathmatch for
`--seconds` (300 s by default) with no early win (the result is the sign of the points),
elimination and Frontline for the preset's clock with their own results. Seats alternate.
Each candidate's jobs are one JSON line of `--out` (a candidate already there is skipped,
so a stopped sweep resumes).
"""

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
import pathlib
import time
import tomllib

import torch

from bench.game_loop import Job, Loop, Player, batches, clock, job_summary, run, variant_types
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_doctrine import from_named, named, stored
from bnb.controllers.imugi_parameters import pack
from bnb.game import api
from bnb.game.handling import characteristics as characteristics_module

HERE = pathlib.Path(__file__).parent
BEFORE = HERE / "before" / "imugi_doctrines.json"
KEYS = ("yongma", "jujak")  # the base card's rows (`bench.game_loop.base_card`)
KEPT = (  # what a job's line keeps of `bench.game_loop.summary`
    "games",
    "score",
    "kill_score",
    "kills_per_minute_mean",
    "kills_per_game",
    "kill_share",
    "kill_share_by_pair",
    "passes_per_kill",
    "no_merge_kills",
    "trade_fraction",
    "wins",
    "draws",
)


def before_card(base):
    """The base card rows (Yongma, Jujak) with every field from before/'s ship files, so a
    sweep plays the version-22 card whatever src/ holds (a candidate's card is relative to it)."""
    values = {}
    cards = [tomllib.loads((HERE / "before" / f"{key}.toml").read_text())["card"] for key in KEYS]
    for field in dataclasses.fields(base):
        column = getattr(base, field.name)
        if field.name in cards[0]:
            column = torch.tensor([c[field.name] for c in cards], dtype=column.dtype)
        values[field.name] = column.to(getattr(base, field.name).device)
    return dataclasses.replace(base, **values)


def before_doctrines() -> dict:
    """The before doctrines by type key, in `named` form."""
    doctrines = stored(BEFORE)["real-scale"]
    return {key: named(doctrine, figures=6) for key, doctrine in doctrines.items()}


def player_of(entry: dict, parameters: tuple[float, ...]) -> Player:
    """A candidate's player: the before doctrines with the entry's fields replaced."""
    base = before_doctrines()
    return Player(
        entry["name"],
        from_named(base["yongma"] | entry.get("energy", {})),
        from_named(base["jujak"] | entry.get("turn", {})),
        parameters=parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deathmatch", "elimination", "frontline"))
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--ships", type=int, default=5)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--mixed-games", type=int, default=None, help="default: --games")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=1, help="card variants a batch holds")
    parser.add_argument(
        "--base", choices=("before", "src"), default="before",
        help="the base card: before/'s ship files (version 22) or src/'s",
    )  # fmt: skip
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true", help="no torch.compile (a CPU check)")
    args = parser.parse_args()
    preset = f"real_scale_{args.mode}"
    seconds = args.seconds or (300.0 if args.mode == "deathmatch" else None)
    seconds = seconds or api.PRESETS[preset].episode_seconds
    parameters = tuple(pack(MODE_DEFAULTS.sets[preset]))
    entries = json.loads(args.candidates.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(line)["name"] for line in args.out.read_text().splitlines()}
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("every candidate is done", flush=True)
        return
    mixed = (args.mixed_games or args.games) // 2
    loop = Loop(
        args.environments,
        args.variants,
        seconds,
        ships_per_team=args.ships,
        mode=args.mode,
        fields=True,
        fixed_map=True,
        device=args.device,
        compiled=not args.eager,
    )
    if args.base == "before":
        loop.base = before_card(loop.base)
    sweep(
        loop, entries, args.out, args.games, mixed, args.seed, args.variants, args.mode,
        args.ships, seconds, parameters,
    )  # fmt: skip


def chunk_cards(loop: Loop, entries: list[dict], games: int, mixed: int, variants: int):
    """Each chunk's card variants as `sweep` plays them: (first index, candidates in the chunk,
    energy dicts, turn dicts)."""
    chunk = max(1, loop.B // (games + 2 * mixed))
    if any("card" in entry for entry in entries):
        chunk = min(chunk, variants)
    for first in range(0, len(entries), chunk):
        cards = [entry.get("card", {}) for entry in entries[first : first + chunk]][:variants]
        cards += [{}] * (variants - len(cards))
        size = len(entries[first : first + chunk])
        yield first, size, [c.get("energy", {}) for c in cards], [c.get("turn", {}) for c in cards]


def _prewarm_one(base, energy, turn, rules, device: str) -> None:
    """One chunk's card characteristics into the disk cache under the loop's device key (the
    entry `Loop.set_types` then loads), measured on this process's CPU."""
    torch.set_num_threads(1)
    key = list(characteristics_module._key(variant_types(base, energy, turn), rules))
    key[1] = device
    path = characteristics_module._disk_path(tuple(key))
    if path.exists():
        return
    measured = characteristics_module.measure_characteristics(
        variant_types(base, energy, turn), rules
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial")
    torch.save({f.name: getattr(measured, f.name) for f in dataclasses.fields(measured)}, partial)
    partial.replace(path)


def prewarm(
    loop: Loop, entries: list[dict], games: int, mixed: int, variants: int, processes: int = 12
) -> None:
    """Measure every chunk's card characteristics before the GPU is taken: `set_types` runs
    them on the CPU (`top_speed` flies a boost until the store is spent, minutes for a large
    store on a slow CPU), so a job would otherwise hold the GPU lock with the GPU idle."""
    base = type(loop.base)(
        **{f.name: getattr(loop.base, f.name).cpu() for f in dataclasses.fields(loop.base)}
    )
    device = str(loop.types.max_energy.device)
    tasks = [
        (base, e, t, loop.rules, device)
        for _, _, e, t in chunk_cards(loop, entries, games, mixed, variants)
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        min(processes, len(tasks)), mp_context=context
    ) as pool:
        for future in [pool.submit(_prewarm_one, *task) for task in tasks]:
            future.result()


def warm(loop: Loop, parameters: tuple[float, ...], variants: int, ticks: int = 3) -> None:
    """Compile the tick a job plays, outside any lock: a few ticks of a job-like batch (a card
    on every variant, the three lineups with a candidate and the before doctrines), then the
    base card back. The filler game alone left Frontline 10v10 to recompile on its first job."""
    card = {"max_shield": 1.0}  # a card dict on every variant, as a job sets (values as base)
    loop.set_types([card] * variants, [card] * variants)
    candidate = player_of({"name": "warm"}, parameters)
    before = player_of({"name": "before"}, parameters)
    share = max(2, loop.B // (3 * variants)) // 2 * 2
    jobs = []
    for variant in range(variants):
        jobs.append(Job(variant, "energy v turn", candidate, candidate, share))
        jobs.append(Job(variant, "mixed", candidate, before, share))
        jobs.append(Job(variant, "mixed reversed", candidate, before, share))
    groups = [group for _, group in batches(jobs, loop.B)[0]]
    arguments = loop.start(groups)
    for _ in range(ticks):
        arguments = loop._tick(*arguments)
    torch.cuda.synchronize(loop.device) if loop.device.type == "cuda" else None
    loop.set_types([{}] * variants, [{}] * variants)


def sweep(
    loop: Loop,
    entries: list[dict],
    out: pathlib.Path,
    games: int,
    mixed: int,
    seed: int,
    variants: int,
    mode: str,
    ships: int,
    seconds: float,
    parameters: tuple[float, ...],
) -> None:
    """Every entry's three jobs on a built loop, a chunk of candidates per batch; each
    candidate's line is appended to `out` as its chunk finishes (`worker.py` calls this on its
    one compiled loop)."""
    before = player_of({"name": "before"}, parameters)
    started = time.time()
    # a chunk of candidates fills about one batch, so each line is written as it finishes
    for first, size, energy, turn in chunk_cards(loop, entries, games, mixed, variants):
        part = entries[first : first + size]
        players = [player_of(entry, parameters) for entry in part]
        loop.set_types(energy, turn)
        jobs = []
        for variant, candidate in enumerate(players):
            variant = variant if variants > 1 else 0
            jobs.append(Job(variant, "energy v turn", candidate, candidate, games))
            jobs.append(Job(variant, "mixed", candidate, before, mixed))
            jobs.append(Job(variant, "mixed reversed", candidate, before, mixed))
        torch.manual_seed(seed + first)
        outs = run(loop, jobs, log=lambda line: None)
        rows = [job_summary(j, o, seconds) for j, o in zip(jobs, outs, strict=True)]
        with out.open("a") as handle:
            for number, entry in enumerate(part):
                three = rows[3 * number : 3 * number + 3]
                line = dict(
                    name=entry["name"],
                    mode=mode,
                    ships=ships,
                    seconds=seconds,
                    candidate=entry,
                    jobs={r["lineup"]: {k: r[k] for k in KEPT} for r in three},
                )
                handle.write(json.dumps(line) + "\n")
                energy = three[0]
                mixed_score = (three[1]["score"][0] + three[2]["score"][0]) / 2
                mixed_kill = (three[1]["kill_score"][0] + three[2]["kill_score"][0]) / 2
                print(
                    f"{entry['name']}: Yongma team {energy['score'][0]:.3f} ± "
                    f"{energy['score'][1]:.3f} (kills {energy['kill_score'][0]:.3f}); "
                    f"mixed v before {mixed_score:.3f} (kills {mixed_kill:.3f})",
                    flush=True,
                )
        elapsed = time.time() - started
        finished = first + len(part)
        left = elapsed / finished * (len(entries) - finished)
        print(f"{finished}/{len(entries)} candidates; done about {clock(left)} KST", flush=True)


if __name__ == "__main__":
    main()
