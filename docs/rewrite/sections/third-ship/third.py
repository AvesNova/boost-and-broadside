"""Third-ship screens on the game-loop harness (`bench.game_loop.Loop`), with no change to
the harness: a candidate ship C takes a slot of the two-type variant table.

    PYTHONPATH=$PWD/src:$PWD .venv/bin/python docs/rewrite/sections/third-ship/third.py \
        --candidates candidates/screen1.json --out data/screen1-dm.jsonl

The variant table holds 1 + 2 * chunk variants (`--chunk` candidates per batch): variant 0 is
the release pair (Yongma, Jujak); a candidate's variants are (C, Jujak) and (Yongma, C), C's
card set field by field ({"set": value}). Each candidate's jobs, side A always C's team:

- "C v J": (C, Jujak), lineup "energy v turn" (A's team all C, B's all Jujak);
- "C v Y": (Yongma, C), lineup "turn v energy" (A's team all C, B's all Yongma);
- "C mirror": (C, Jujak), lineup "energy mirror" (`--mirror-games`; its score is 1/2 by
  symmetry, kept for the pace, the passes and the aspects);
- "C+J v C+J" ("mixed", A's doctrine for C against B's, `--mixed-games`; only when the
  candidate has a "versus" doctrine to compare with).

A player flies the stored doctrines (`imugi_doctrines.json`) for Yongma and Jujak and the
candidate's doctrine for C: its "doctrine" fields over the stored doctrine of its
"doctrine_base" (jujak by default; "off": the default Doctrine, every element off).
`--yongma-card` / `--jujak-card` (JSON: field -> factor or {"set": value}) change the
release cards (an entry's own "yongma_card" / "jujak_card" override them for its jobs, so one
run can hold several balance options; a reference for an option is an entry whose card is
that option's Yongma, flying Yongma's doctrine: its "C v J" is the option's Y v J) (the balance agent's adopted changes). Each run also plays the release pair
once ("Y v J", variant 0) as the reference. Each candidate's line goes to `--out` as it
finishes (a candidate already there is skipped, so a stopped run resumes).
"""

import argparse
import json
import pathlib
import time

import torch

from bench.game_loop import Job, Loop, Player, clock, job_summary, run
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_doctrine import Doctrine, from_named, named, stored
from bnb.controllers.imugi_parameters import pack
from bnb.game import api

KEPT = (
    "games",
    "score",
    "kill_score",
    "kills_per_minute_mean",
    "kills_per_minute_median",
    "kills_per_game",
    "kill_share",
    "damage_share",
    "kill_share_by_pair",
    "passes_per_kill",
    "no_merge_kills",
    "first_merge_kills",
    "first_hit_wins",
    "ttk_median",
    "circles",
    "trade_fraction",
    "wins",
    "draws",
)


def stored_named() -> dict:
    """The stored doctrines of the real-scale set by key, in `named` form."""
    return {k: named(d, figures=6) for k, d in stored()["real-scale"].items()}


def doctrine_of(entry: dict, key: str = "doctrine") -> Doctrine:
    """A candidate's C doctrine: its fields over its base."""
    base_key = entry.get(f"{key}_base", "jujak")
    base = named(Doctrine(), figures=6) if base_key == "off" else stored_named()[base_key]
    return from_named(base | entry.get(key, {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="deathmatch", choices=("deathmatch", "elimination", "frontline"))
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="candidate names to run")
    parser.add_argument("--ships", type=int, default=5)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--mirror-games", type=int, default=64)
    parser.add_argument("--mixed-games", type=int, default=0)
    parser.add_argument("--reference-games", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yongma-card", type=json.loads, default={})
    parser.add_argument("--jujak-card", type=json.loads, default={})
    parser.add_argument("--yongma-card-file", type=pathlib.Path, help="--yongma-card from a file")
    parser.add_argument("--jujak-card-file", type=pathlib.Path, help="--jujak-card from a file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.yongma_card_file:
        args.yongma_card = json.loads(args.yongma_card_file.read_text())
    if args.jujak_card_file:
        args.jujak_card = json.loads(args.jujak_card_file.read_text())
    preset = f"real_scale_{args.mode}"
    seconds = args.seconds or (300.0 if args.mode == "deathmatch" else None)
    seconds = seconds or api.PRESETS[preset].episode_seconds
    parameters = tuple(pack(MODE_DEFAULTS.sets[preset]))
    entries = json.loads(args.candidates.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.only:
        entries = [e for e in entries if e["name"] in args.only]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["name"] for line in args.out.read_text().splitlines()}
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("every candidate is done", flush=True)
        return
    base = stored()["real-scale"]
    K = 1 + 2 * args.chunk
    loop = Loop(
        args.environments,
        K,
        seconds,
        ships_per_team=args.ships,
        mode=args.mode,
        fields=True,
        fixed_map=True,
        device=args.device,
        compiled=not args.eager,
    )
    reference = Player("release", base["yongma"], base["jujak"], parameters=parameters)
    started = time.time()
    first_chunk = True
    for first in range(0, len(entries), args.chunk):
        part = entries[first : first + args.chunk]
        energy = [dict(args.yongma_card)]
        turn = [dict(args.jujak_card)]
        for entry in part:
            c = {name: {"set": value} for name, value in entry["card"].items()}
            # an entry's own opponents ("yongma_card", "jujak_card": a balance option's cards)
            energy += [c, dict(entry.get("yongma_card", args.yongma_card))]
            turn += [dict(entry.get("jujak_card", args.jujak_card)), c]
        while len(energy) < K:
            energy.append({})
            turn.append({})
        loop.set_types(energy, turn)
        jobs, keys = [], []
        if first_chunk and args.reference_games:
            jobs.append(Job(0, "energy v turn", reference, reference, args.reference_games))
            keys.append(("release", "Y v J"))
            if args.mirror_games:
                jobs.append(Job(0, "energy mirror", reference, reference, args.mirror_games))
                keys.append(("release", "Y mirror"))
                jobs.append(Job(0, "turn mirror", reference, reference, args.mirror_games))
                keys.append(("release", "J mirror"))
        for number, entry in enumerate(part):
            c_doctrine = doctrine_of(entry)
            with_j = Player(entry["name"], c_doctrine, base["jujak"], parameters=parameters)
            with_y = Player(entry["name"], base["yongma"], c_doctrine, parameters=parameters)
            va, vb = 1 + 2 * number, 2 + 2 * number
            jobs.append(Job(va, "energy v turn", with_j, with_j, args.games))
            keys.append((entry["name"], "C v J"))
            jobs.append(Job(vb, "turn v energy", with_y, with_y, args.games))
            keys.append((entry["name"], "C v Y"))
            if args.mirror_games:
                jobs.append(Job(va, "energy mirror", with_j, with_j, args.mirror_games))
                keys.append((entry["name"], "C mirror"))
            if args.mixed_games and "versus" in entry:
                other = Player("versus", doctrine_of(entry, "versus"), base["jujak"], parameters=parameters)
                jobs.append(Job(va, "mixed", with_j, other, args.mixed_games // 2))
                keys.append((entry["name"], "mixed"))
                jobs.append(Job(va, "mixed reversed", with_j, other, args.mixed_games // 2))
                keys.append((entry["name"], "mixed reversed"))
        torch.manual_seed(args.seed + first)
        outs = run(loop, jobs, log=lambda line: print(line, flush=True))
        rows = [job_summary(j, o, seconds) for j, o in zip(jobs, outs, strict=True)]
        by_name: dict[str, dict] = {}
        for (name, label), row in zip(keys, rows, strict=True):
            by_name.setdefault(name, {})[label] = {k: row[k] for k in KEPT if k in row}
        with args.out.open("a") as handle:
            for name, jobs_of in by_name.items():
                entry = next((e for e in part if e["name"] == name), {"name": name})
                line = dict(
                    name=name,
                    mode=args.mode,
                    ships=args.ships,
                    seconds=seconds,
                    yongma_card=args.yongma_card,
                    jujak_card=args.jujak_card,
                    candidate={k: v for k, v in entry.items() if k not in ("card", "solved_handles", "spec")},
                    jobs=jobs_of,
                )
                handle.write(json.dumps(line) + "\n")
                text = "; ".join(
                    f"{label} {r['score'][0]:.3f}±{r['score'][1]:.3f} (k {r['kill_score'][0]:.3f})"
                    for label, r in jobs_of.items()
                    if "mirror" not in label
                )
                for label in ("C mirror", "Y mirror", "J mirror"):
                    if label in jobs_of:
                        m = jobs_of[label]
                        text += f"; {label} kpm {m['kills_per_minute_mean']:.2f} ppk {m['passes_per_kill']:.2f}"
                print(f"{name}: {text}", flush=True)
        first_chunk = False
        elapsed = time.time() - started
        finished = first + len(part)
        left = elapsed / finished * (len(entries) - finished)
        print(f"{finished}/{len(entries)} candidates; done about {clock(left)} KST", flush=True)


if __name__ == "__main__":
    main()
