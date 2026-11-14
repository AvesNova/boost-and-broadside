"""Which tournament layout rates fastest: games per second and planned information per second
of the match engine (`bnb.league.matches.Arena`) for several layouts (blocks, games per block,
ticks per segment), so the rating defaults can be chosen.

For each layout an engine is built on the field (compiled, as `bnb ratings` builds it), one
warm-up batch is played (the compile and the graph capture), then `batches` timed batches.
Each batch's blocks are planned as a tournament plans them (`bnb.league.design.plan_batch`,
on a posterior of the field with no games, so every pair is open), within the engine's lanes
and `players_per_batch`. Reported per layout:

- games per second: the batch's games over its wall time (the mean over the timed batches);
- information per second: one batch's planned decrease of the weighted A-optimal criterion
  (in units of the targets' variances) on a reference field (`reference_players` players
  `reference_spacing` Elo apart, each neighbour pair with `reference_games` games, every
  target `reference_stderr`), over the batch's measured seconds. A layout of more blocks
  spreads a batch over more pairs, so its information per game is higher than one of fewer,
  larger blocks at the same games;
- the planning seconds of one batch on the reference field.

Every timed batch is appended as a JSON line to `out` when it ends (with the layout's
summary after its last batch), and each line printed with the clock time the probe expects to
end. Bound: `batches` timed batches per layout, after one warm-up.

Usage: `uv run --no-sync python bench/rating_layout_probe.py players=scripted,PATH_A,PATH_B
[preset=real_scale_frontline] [layouts=2x96x32,4x96x32,8x128x32] [batches=3] [device=cuda]
[compiled=1] [lineup=Yongma,Yongma,Jujak,Jujak,Jujak] [players_per_batch=] [out=probe.jsonl]
[episode_seconds=]` (episode_seconds shortens the game, for a smoke test on the CPU).
"""

import dataclasses
import datetime
import json
import pathlib
import sys
import time

import torch

from bnb.game.api import PRESETS
from bnb.league import design
from bnb.league.matches import Arena, ArenaConfig, Player, default_lanes, network_groups, player
from bnb.league.tournament import outcomes


def parse_layouts(text: str) -> list[tuple[int, int, int]]:
    """ "BxEXL,..." as (blocks, games per block, ticks per segment) triples (L defaults to 32)."""
    layouts = []
    for item in text.split(","):
        parts = [int(part) for part in item.split("x")]
        if len(parts) == 2:
            parts.append(32)
        if len(parts) != 3:
            raise ValueError(f"layout {item!r}: blocks x environments [x length]")
        layouts.append(tuple(parts))
    return layouts


def reference_information(
    blocks: int,
    block_games: int,
    players_per_batch: int | None,
    players: int = 16,
    spacing: float = 120.0,
    games: float = 64.0,
    stderr: float = 45.0,
) -> tuple[float, float]:
    """(one batch's planned criterion decrease on the reference field, the planning seconds)."""
    truth = torch.arange(players, dtype=torch.float64) * spacing
    wins = torch.zeros((players, players), dtype=torch.float64)
    for index in range(players - 1):
        probability = float(torch.sigmoid((truth[index] - truth[index + 1]) / design.SCALE))
        wins[index, index + 1] = games * probability
        wins[index + 1, index] = games * (1.0 - probability)
    posterior = design.fit_posterior(wins, 0)
    targets = design.Targets(torch.full((players,), stderr, dtype=torch.float64))
    allowed = ~torch.eye(players, dtype=torch.bool)
    started = time.perf_counter()
    plan = design.plan_batch(
        posterior, targets, allowed, design.Layout(blocks, block_games, players_per_batch)
    )
    return plan.criterion - plan.after, time.perf_counter() - started


def _matchups(field: list[Player], layout: design.Layout) -> list[tuple[int, int]]:
    """A batch's blocks as a tournament would plan them over the field with no games."""
    count = len(field)
    posterior = design.fit_posterior(torch.zeros((count, count), dtype=torch.float64), 0)
    targets = design.Targets(torch.full((count,), 45.0, dtype=torch.float64))
    allowed = ~torch.eye(count, dtype=torch.bool)
    return design.plan_batch(posterior, targets, allowed, layout).blocks


def measure(
    field: list[Player],
    game,
    layouts: list[tuple[int, int, int]],
    device: torch.device,
    batches: int = 3,
    compiled: bool = True,
    players_per_batch: int | None = None,
    out: pathlib.Path | None = None,
    log=print,
) -> list[dict]:
    """Each layout's summary (module docstring); every batch line written to `out` as it
    ends."""
    groups = network_groups(field)
    summaries = []
    started_all = time.perf_counter()
    total = len(layouts) * (batches + 1)
    done = 0
    for blocks, environments, length in layouts:
        config = ArenaConfig(blocks, environments, length, compiled)
        lanes = default_lanes(groups, 2 * blocks)
        plan_layout = design.Layout(blocks, environments, players_per_batch, tuple(groups), lanes)
        matchups = _matchups(field, plan_layout)
        built = time.perf_counter()
        arena = Arena(field, game, device, config, lanes)
        outcomes(arena, matchups)  # the warm-up: the compile and the capture
        warm = time.perf_counter() - built
        done += 1
        seconds = []
        for batch in range(batches):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            results, _ = outcomes(arena, matchups)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - started)
            done += 1
            line = dict(
                layout=[blocks, environments, length],
                batch=batch,
                seconds=seconds[-1],
                games=int(results.sum()),
                warm_up_seconds=warm,
                wall_time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            )
            _write(out, line)
            elapsed = time.perf_counter() - started_all
            end = datetime.datetime.now().astimezone() + datetime.timedelta(
                seconds=elapsed / done * (total - done)
            )
            log(f"{json.dumps(line)}  (ends about {end.strftime('%H:%M')})")
        mean = sum(seconds) / len(seconds)
        information, planning = reference_information(blocks, environments, players_per_batch)
        summary = dict(
            layout=[blocks, environments, length],
            games_per_batch=blocks * environments,
            batch_seconds=mean,
            games_per_second=blocks * environments / mean,
            information_per_batch=information,
            information_per_second=information / mean,
            planning_seconds=planning,
            warm_up_seconds=warm,
            device=str(device),
            players=[one.name for one in field],
        )
        summaries.append(summary)
        _write(out, dict(summary=summary))
        log(json.dumps(dict(summary=summary)))
        del arena
    return summaries


def _write(out: pathlib.Path | None, line: dict) -> None:
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as file:
        file.write(json.dumps(line) + "\n")
        file.flush()


def main() -> None:
    """Parse the key=value arguments and measure."""
    options = dict(argument.split("=", 1) for argument in sys.argv[1:])
    game = PRESETS[options.get("preset", "real_scale_frontline")]
    if options.get("episode_seconds"):
        game = dataclasses.replace(game, episode_seconds=float(options["episode_seconds"]))
    field = [player(name) for name in options["players"].split(",")]
    if options.get("lineup"):
        from bnb.runtime.train import lineup_indices

        indices = lineup_indices(tuple(options["lineup"].split(",")), game)
        field = [dataclasses.replace(one, ship_type=indices) for one in field]
    measure(
        field,
        game,
        parse_layouts(options.get("layouts", "2x96x32,4x96x32,8x128x32")),
        torch.device(options.get("device", "cuda")),
        int(options.get("batches", 3)),
        options.get("compiled", "1") == "1",
        int(options["players_per_batch"]) if options.get("players_per_batch") else None,
        pathlib.Path(options["out"]) if options.get("out") else None,
    )


if __name__ == "__main__":
    main()
