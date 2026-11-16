"""Does a worker's start-up compile the graph its jobs play? Builds the worker's loop (small
`--environments`), runs its start-up (the filler tick and `sweep.warm`), then a few ticks of a
real job's first batch; run with TORCH_LOGS=recompiles to see any guard that fails after the
start-up.

    TORCH_LOGS=recompiles PYTHONPATH=src:. python \\
        docs/rewrite/sections/balance-final/recompile_check.py \\
        --mode frontline --ships 10 --environments 64 --candidates candidates/probe3.json
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from sweep import MODE_DEFAULTS, Loop, api, before_card, chunk_cards, pack, player_of, warm  # noqa: E402

from bench.game_loop import FILLER, Group, Job, batches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--ships", type=int, default=5)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--no-warm", action="store_true")
    args = parser.parse_args()
    preset = f"real_scale_{args.mode}"
    seconds = 300.0 if args.mode == "deathmatch" else api.PRESETS[preset].episode_seconds
    parameters = tuple(pack(MODE_DEFAULTS.sets[preset]))
    loop = Loop(
        args.environments, args.variants, seconds, ships_per_team=args.ships, mode=args.mode,
        fields=True, fixed_map=True, device="cuda", compiled=True,
    )  # fmt: skip
    loop.base = before_card(loop.base)
    loop._tick(*loop.start([Group(0, "mixed", FILLER, FILLER, loop.B)]))
    print("filler tick done", flush=True)
    if not args.no_warm:
        warm(loop, parameters, args.variants)
        print("warm done", flush=True)
    entries = json.loads(args.candidates.read_text())
    games, mixed = 4, 2
    _, size, energy, turn = next(iter(chunk_cards(loop, entries, games, mixed, args.variants)))
    loop.set_types(energy, turn)
    before = player_of({"name": "before"}, parameters)
    jobs = []
    for variant, entry in enumerate(entries[:size]):
        player = player_of(entry, parameters)
        jobs += [Job(variant, "energy v turn", player, player, games),
                 Job(variant, "mixed", player, before, mixed),
                 Job(variant, "mixed reversed", player, before, mixed)]  # fmt: skip
    arguments = loop.start([group for _, group in batches(jobs, loop.B)[0]])
    for _ in range(5):
        arguments = loop._tick(*arguments)
    torch.cuda.synchronize()
    print("job ticks done", flush=True)


if __name__ == "__main__":
    main()
