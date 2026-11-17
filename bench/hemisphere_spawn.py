"""The hemisphere spawn's check (the release run, 2026-11-17): Imugi against Imugi in
`real_scale_elimination_hemisphere`, Jujak in every slot, 5v5 (or a learner-side lineup),
from the match's start for a bounded time. Per game: the first shot, the first enemy damage,
the boundary deaths (a death on a tick the ship took boundary damage and no enemy hit), and
the spawn positions; a still of a few games' spawns.

    PYTHONPATH=src python -m bench.hemisphere_spawn --games 256 --seconds 90 --out review/spawn

Writes OUT/games.jsonl (one line per game, as each batch ends), OUT/summary.json and
OUT/spawns.png. Bounded by --seconds per batch; progress with the batch's ETA on stdout.
"""

import argparse
import datetime
import json
import math
import pathlib
import statistics
import time

import torch

from bnb.contracts import Pool
from bnb.controllers.imugi import Imugi
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import lineup_indices

PRESET = "real_scale_elimination_hemisphere"  # --preset: another elimination game to compare
LENGTH = 30  # ticks per segment (1 s)


def batch(
    games: int, seconds: float, seed: int, device: torch.device, compiled: bool, preset: str
) -> dict:
    """One batch of `games` games from their start: per game the event times (s, None if
    not within `seconds`), the boundary deaths' times and every ship's spawn."""
    game = api.PRESETS[preset]
    types = api.ship_types(device, ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    S, N = game.ships_per_team, game.num_ships
    team = torch.tensor(lineup_indices(("Jujak",) * S, game), dtype=torch.int32)
    ship_type = team.repeat(2)[None].expand(games, N).contiguous()
    features = characteristics(types, rules)
    seen: dict = {}
    original = api.resolve

    def resolve(*args, **kwargs):
        state, events = original(*args, **kwargs)
        seen["events"] = events
        return state, events

    api.resolve = resolve
    try:
        rollout = Rollout(
            game,
            types,
            (Pool("imugi", 1.0, ("first", "second")),),
            {"first": Imugi(features, S), "second": Imugi(features, S)},
            "first",
            games,
            LENGTH,
            device,
            compiled=compiled,
            staggered=False,
            ship_type=ship_type,
            seed=seed,
        )
        state = rollout.carry["state"]
        spawn = dict(
            # copies: the rollout writes its state in place, and .cpu() of a CPU tensor is
            # the tensor itself
            x=state.ships.x.detach().cpu().clone(),
            y=state.ships.y.detach().cpu().clone(),
            axis=torch.stack([state.match.spawn_axis_x, state.match.spawn_axis_y], -1).cpu(),
        )
        act, advance, resolve_stage, perceive = rollout.stages
        tick_length = rules.tick_length
        first_shot = torch.full((games,), math.inf, device=device)
        first_damage = torch.full((games,), math.inf, device=device)
        ended = torch.zeros(games, dtype=torch.bool, device=device)
        border_deaths: list[tuple[int, int, float]] = []
        deaths = torch.zeros(games, device=device)
        clock = {"tick": 0}

        def watched(carry):
            resolve_stage(carry)
            events = seen["events"]
            now = (clock["tick"] + 1) * tick_length
            playing = ~ended
            fired = events.fired.any(-1) & playing
            hit = (events.damage.sum((-1, -2)) > 0) & playing
            first_shot.copy_(torch.where(fired & first_shot.isinf(), now, first_shot))
            first_damage.copy_(torch.where(hit & first_damage.isinf(), now, first_damage))
            enemy_hit = events.damage.sum(-2) > 0  # (B, N) damage taken from a ship
            border = events.died & (events.boundary_damage > 0) & ~enemy_hit & playing[:, None]
            for environment, ship in border.nonzero().tolist():
                border_deaths.append((environment, ship, now))
            deaths.add_((events.died & playing[:, None]).sum(-1))
            ended.copy_(ended | events.done)
            clock["tick"] += 1

        rollout.stages = (act, advance, watched, perceive)
        for _ in range(math.ceil(seconds / (LENGTH * tick_length))):
            rollout.run()
    finally:
        api.resolve = original

    def times(values: torch.Tensor) -> list[float | None]:
        return [None if math.isinf(v) else round(v, 3) for v in values.tolist()]

    return dict(
        first_shot=times(first_shot),
        first_damage=times(first_damage),
        border_deaths=border_deaths,
        deaths=deaths.tolist(),
        spawn=spawn,
    )


def median(values: list) -> float | None:
    """The median of the observed values, the unobserved (None) counted as past every one
    (so it is the median of all games while at most half are unobserved)."""
    known = sorted(v for v in values if v is not None)
    if len(known) * 2 <= len(values):
        return None
    return statistics.median(known + [math.inf] * (len(values) - len(known)))


def still(spawns: list[dict], path: pathlib.Path, game_radius: float, rules) -> None:
    """Six games' spawns: team 0 blue, team 1 orange, the dividing line, the bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = rules.modes[0]
    half = 0.5 * mode.hemisphere_middle_seconds * rules.spawn_speed
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    S = rules.ships_per_team
    for axis, spawn in zip(axes.flat, spawns, strict=False):
        x, y, (ax, ay) = spawn["x"], spawn["y"], spawn["axis"]
        angle = torch.linspace(0, 2 * math.pi, 361)
        axis.plot(game_radius * torch.cos(angle), game_radius * torch.sin(angle), "k-", lw=1)
        edge = spawn["edge"]
        axis.plot(edge * torch.cos(angle), edge * torch.sin(angle), "k:", lw=0.8)
        for offset in (-half, 0.0, half):  # the dividing line and the middle band
            along = math.sqrt(max(game_radius**2 - offset**2, 0.0))
            px, py = -ay * along, ax * along  # along the line
            cx, cy = ax * offset, ay * offset
            axis.plot([cx - px, cx + px], [cy - py, cy + py], "k-" if offset == 0 else "k:", lw=0.8)
        axis.scatter(x[:S], y[:S], c="tab:blue", s=18, label="team 0")
        axis.scatter(x[S:], y[S:], c="tab:orange", s=18, label="team 1")
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    axes.flat[0].legend(loc="lower left", fontsize=8)
    figure.suptitle(
        f"{PRESET}: spawns of six games (dotted: the edge band and the middle band)", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main() -> None:
    """The command: the batches, their records, the summary and the still."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--out", default="review/spawn")
    parser.add_argument("--preset", default=PRESET)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    game = api.PRESETS[args.preset]
    rules = api.rules_of(game, types=api.ship_types(ship_set=game.ship_set))
    records = out / "games.jsonl"
    records.write_text("")
    rows, spawns, started = [], [], time.time()
    batches = math.ceil(args.games / args.batch)
    for number in range(batches):
        size = min(args.batch, args.games - number * args.batch)
        result = batch(size, args.seconds, args.seed + number, device, args.compiled, args.preset)
        border = {}
        for environment, ship, at in result["border_deaths"]:
            x = float(result["spawn"]["x"][environment, ship])
            y = float(result["spawn"]["y"][environment, ship])
            spawned = math.hypot(x, y) / game.arena_radius
            border.setdefault(environment, []).append(
                dict(ship=ship, at=at, spawn_radius_fraction=round(spawned, 4))
            )
        with records.open("a") as handle:
            for index in range(size):
                row = dict(
                    batch=number,
                    game=index,
                    first_shot=result["first_shot"][index],
                    first_damage=result["first_damage"][index],
                    border_deaths=border.get(index, []),
                    deaths=result["deaths"][index],
                    spawn_x=[round(v, 1) for v in result["spawn"]["x"][index].tolist()],
                    spawn_y=[round(v, 1) for v in result["spawn"]["y"][index].tolist()],
                )
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
        if number == 0:
            for index in range(min(6, size)):
                spawns.append(
                    dict(
                        x=result["spawn"]["x"][index],
                        y=result["spawn"]["y"][index],
                        axis=result["spawn"]["axis"][index].tolist(),
                    )
                )
        elapsed = time.time() - started
        eta = started + elapsed / (number + 1) * batches
        print(
            f"batch {number + 1}/{batches} done in {elapsed:.0f} s, ETA "
            f"{datetime.datetime.fromtimestamp(eta):%H:%M:%S}",
            flush=True,
        )
    shots = [row["first_shot"] for row in rows]
    damage = [row["first_damage"] for row in rows]
    border_times = [death["at"] for row in rows for death in row["border_deaths"]]
    border_spawns = [
        death["spawn_radius_fraction"] for row in rows for death in row["border_deaths"]
    ]
    every_spawn = [
        math.hypot(x, y) / game.arena_radius
        for row in rows
        for x, y in zip(row["spawn_x"], row["spawn_y"], strict=True)
    ]
    summary = dict(
        preset=args.preset,
        games=len(rows),
        seconds=args.seconds,
        median_first_shot=median(shots),
        median_first_damage=median(damage),
        first_shot_quartiles=statistics.quantiles([s for s in shots if s is not None], n=4)
        if sum(s is not None for s in shots) > 1
        else None,
        first_damage_quartiles=statistics.quantiles([d for d in damage if d is not None], n=4)
        if sum(d is not None for d in damage) > 1
        else None,
        games_without_a_shot=sum(s is None for s in shots),
        games_without_damage=sum(d is None for d in damage),
        border_deaths=len(border_times),
        border_deaths_first_10s=sum(at <= 10.0 for at in border_times),
        border_death_times=sorted(border_times),
        border_deaths_spawn_radius_fraction_median=statistics.median(border_spawns)
        if border_spawns
        else None,
        spawn_radius_fraction_median=statistics.median(every_spawn),
        spawns_beyond_0_9_radius=sum(r > 0.9 for r in every_spawn) / len(every_spawn),
        deaths_per_game=statistics.mean(row["deaths"] for row in rows),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    # the edge band as the game computes it (the boundary's first 5 s of travel)
    mode = rules.modes[0]
    ramp = rules.boundary_ramp_width * game.arena_radius / 2600.0
    closing = (game.arena_radius + ramp) / (game.episode_seconds - mode.shrink_end_lead_seconds)
    for spawn in spawns:
        spawn["edge"] = game.arena_radius - mode.hemisphere_edge_seconds * closing
    if game.spawn == "hemisphere":
        still(spawns, out / "spawns.png", game.arena_radius, rules)


if __name__ == "__main__":
    main()
