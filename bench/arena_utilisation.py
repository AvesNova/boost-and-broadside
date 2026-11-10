"""How much of a ratings batch is wasted, and what counting more games per batch would buy.

A batch (`Arena.play`) runs until every environment's first game has ended; an environment
whose game ended early keeps simulating a second game that is never counted. This plays
`batches` steady batches, reads which rows have finished after every segment (one host
read each, as `play` already makes), and reports the share of row-ticks spent on counted
games, per block the first games' lengths, and then, by simulation from those measured
lengths, the "every game started before T, each played to its end" rule: games counted
per batch tick and the share of row-ticks counted, for several T. That rule is a stopping
rule (the next game starts only after the last has ended), so wins / games stays a
consistent estimate (Wald's identity); counting the games that ended before T would not.

Usage: python bench/arena_utilisation.py players=a,b,c [preset=frontline] [batches=3]\n       [save=ticks.npy]
Prints one JSON line.
"""

import json
import sys

import numpy

from bnb.game.api import PRESETS
from bnb.league.matches import SEAT, Arena, ArenaConfig, player


def _simulate(lengths: list[numpy.ndarray], horizon: float, rng, repeats: int = 200) -> dict:
    """The started-before-`horizon` rule (horizon 0: each environment's first game only),
    each block's environments drawing i.i.d. game lengths from that block's measured
    ones: (games counted per tick of the batch, the share of row-ticks in counted games,
    the batch's length in ticks), means over `repeats` batches."""
    rates, shares, ends = [], [], []
    for _ in range(repeats):
        busy, counted, end = 0.0, 0, 0.0
        for measured in lengths:
            for _ in range(len(measured)):
                clock = 0.0
                while True:
                    length = float(rng.choice(measured))
                    clock += length
                    counted += 1
                    if clock >= horizon:
                        break
                busy += clock
                end = max(end, clock)
        rows = sum(len(measured) for measured in lengths)
        rates.append(counted / end)
        shares.append(busy / (rows * end))
        ends.append(end)
    return dict(
        games_per_tick=float(numpy.mean(rates)),
        counted_share=float(numpy.mean(shares)),
        batch_ticks=float(numpy.mean(ends)),
    )


def main() -> None:
    """Play the batches, measure, simulate; print one JSON line."""
    import torch

    options = dict(a.split("=", 1) for a in sys.argv[1:])
    names = options["players"].split(",")
    batches = int(options.get("batches", 3))
    game = PRESETS[options.get("preset", "frontline")]
    device = torch.device("cuda")
    config = ArenaConfig()
    arena = Arena([player(name) for name in names], game, device, config)
    count = len(names)
    pairs = [(a, b) for a in range(count) for b in range(count) if a != b]
    matchups = [pairs[i % len(pairs)] for i in range(config.blocks)]
    arena.play(matchups)  # compiles and captures

    seat, history = arena.seat, []
    original = seat.finished

    def recorded(carry) -> bool:
        finished = carry.finished.cpu()
        history[-1].append(finished)
        return bool(finished.all())

    seat.finished = recorded
    end_ticks = []  # per batch (rows,): the segment end by which each row's first game ended
    for _ in range(batches):
        history.append([])
        arena.play(matchups)
        states = torch.stack(history[-1])  # (checks, rows); check 0 before any segment
        states = states[:, seat.first_seat.cpu()]  # a row per game: its block's first seat's
        ended = states.int().argmax(0)  # the first check at which the row had finished
        end_ticks.append((ended * config.length).numpy())
    seat.finished = original
    if options.get("save"):  # the raw end ticks (batches, games), before any analysis
        numpy.save(options["save"], numpy.stack(end_ticks))

    ticks = numpy.stack(end_ticks)  # (batches, rows)
    rows = ticks.shape[1]
    batch_ticks = ticks.max(1)
    counted_share = float((ticks.sum(1) / (rows * batch_ticks)).mean())
    per_block = ticks.reshape(batches, config.blocks, config.block_environments)
    lengths = [per_block[:, b].reshape(-1).astype(float) for b in range(config.blocks)]
    median = float(numpy.median(ticks))
    rng = numpy.random.default_rng(0)
    rules = {
        f"T={factor}x_median": _simulate(lengths, factor * median, rng)
        for factor in (0.0, 1.0, 2.0, 4.0)
    }
    first = rules["T=0.0x_median"]["games_per_tick"]
    for rule in rules.values():
        rule["games_per_tick_vs_first_only"] = rule["games_per_tick"] / first
    print(
        json.dumps(
            dict(
                preset=options.get("preset", "frontline"),
                players=names,
                rows=rows,
                segment_ticks=config.length,
                measured=dict(
                    batch_ticks=batch_ticks.tolist(),
                    counted_share=counted_share,
                    median_game_ticks=median,
                    block_mean_ticks=[float(x.mean()) for x in lengths],
                    block_max_ticks=[float(x.max()) for x in lengths],
                ),
                simulated=rules,
            )
        )
    )


if __name__ == "__main__":
    main()
