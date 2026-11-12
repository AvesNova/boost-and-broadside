"""Trajectories of scripted fights: a candidate (`sizes.py`'s form) at one fleet size and
lineup, a few games recorded tick by tick (positions, headings, hits, deaths), the fight
shape measured from the tracks, and PNG plots (each ship's path coloured by time, light to
dark; merges, hits and the kill marked; the arena edge dashed).

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/balance-sizes/trajectories.py --candidates baseline/candidates.json \
        --size 1 --lineup "energy v turn" --games 32 --plots 4 --out images/frozen-1v1-evt

Per fight (a 1v1 from both ships' spawn to the first death): merges before the kill (both
noses within 30 degrees of the line, inside the longer gun range, a new merge only after 1
s out of one), the circles each ship turned (its heading's total turn over 2 pi), the most
separation (in gun ranges), the time spent outside 0.9 of the arena radius, and the kill's
time. Writes <out>.json with every fight's numbers.
"""

import argparse
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "game-loop"))

import league  # noqa: E402
from sizes import player_of  # noqa: E402

from bench.game_loop import Group, Loop  # noqa: E402
from bnb.contracts import Action, Obs  # noqa: E402
from bnb.game import api  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # Yongma, Jujak (the palette's slots 1 and 2)
TYPE_NAMES = ("Yongma", "Jujak")
MERGE_CONE = math.radians(30.0)


@torch.no_grad()
def record(loop: Loop, groups: list[Group]) -> dict:
    """Every tick's x, y, heading, alive, hits and deaths of every environment, on the host."""
    state, observation, carry, _ = loop.start(groups)
    B, N = loop.B, loop.N
    xs, ys, cs, ss, alive, hits, died, killer = [], [], [], [], [], [], [], []
    for t in range(loop.ticks - 1):
        view = Obs(
            **{
                name: getattr(observation, name).flatten(0, 1)
                for name in ("ships", "fields", "globals", "zones")
            }
        )
        reset = torch.full((2 * B,), t == 0, dtype=torch.bool, device=loop.device)
        action, carry, _ = loop.agent.step(carry, view, reset)
        action = Action.unpack(action.packed().reshape(B, N, 3).to(torch.int32))
        moved, sweep = api.move(
            loop.rules, loop.types, state, action, api.spread(loop.rules, B, loop.device)
        )
        state, events = api.resolve(
            loop.rules, loop.types, moved, sweep, action, api.uniforms(loop.rules, B, loop.device)
        )
        shown = api.display(state)
        xs.append(shown["ships.x"].clone())
        ys.append(shown["ships.y"].clone())
        cs.append(shown["ships.heading_cos"].clone())
        ss.append(shown["ships.heading_sin"].clone())
        alive.append(shown["ships.alive"].bool().clone())
        hits.append(events.damage > 0.0)  # (B, shooter, target)
        died.append(events.died.clone())
        killer.append(events.killer.clone())
        state, observation, _ = api.perceive(loop.rules, state)
    stack = lambda values: torch.stack(values, 1).cpu()  # noqa: E731
    return dict(
        x=stack(xs),
        y=stack(ys),
        cos=stack(cs),
        sin=stack(ss),
        alive=stack(alive),
        hits=stack(hits),
        died=stack(died),
        killer=stack(killer),
        ship_type=loop.ship_type.cpu(),
        dt=loop.rules.tick_length,
    )


def merges_of(rec: dict, b: int, i: int, j: int, reach: float, start: int, end: int) -> list[int]:
    """The ticks where merges of ships i and j begin, in [start, end)."""
    x, y, c, s = rec["x"][b], rec["y"][b], rec["cos"][b], rec["sin"][b]
    dx, dy = x[start:end, j] - x[start:end, i], y[start:end, j] - y[start:end, i]
    distance = torch.sqrt(dx * dx + dy * dy)
    off_i = torch.atan2(
        (c[start:end, i] * dy - s[start:end, i] * dx).abs(),
        c[start:end, i] * dx + s[start:end, i] * dy,
    )
    off_j = torch.atan2(
        (c[start:end, j] * dy - s[start:end, j] * dx).abs(),
        -(c[start:end, j] * dx + s[start:end, j] * dy),
    )
    on = ((distance < reach) & (off_i < MERGE_CONE) & (off_j < MERGE_CONE)).tolist()
    out, quiet, gap = [], 10**9, int(round(1.0 / rec["dt"]))
    for t, value in enumerate(on):
        if value:
            if quiet >= gap:
                out.append(start + t)
            quiet = 0
        else:
            quiet += 1
    return out


def turned(rec: dict, b: int, i: int, start: int, end: int) -> float:
    """Circles ship i's heading turned in [start, end) (the total of the absolute turns)."""
    angle = torch.atan2(rec["sin"][b, start:end, i], rec["cos"][b, start:end, i])
    step = torch.remainder(angle[1:] - angle[:-1] + math.pi, 2 * math.pi) - math.pi
    return float(step.abs().sum()) / (2 * math.pi)


def reversals(rec: dict, b: int, i: int, start: int, end: int) -> int:
    """Turn-sense reversals of ship i (a held turn > 0.1 rad/s for 1 s, then the other way)."""
    angle = torch.atan2(rec["sin"][b, start:end, i], rec["cos"][b, start:end, i])
    rate = (torch.remainder(angle[1:] - angle[:-1] + math.pi, 2 * math.pi) - math.pi) / rec["dt"]
    hold = int(round(1.0 / rec["dt"]))
    if len(rate) < hold:
        return 0
    smooth = torch.nn.functional.avg_pool1d(rate[None, None], hold, 1)[0, 0]
    sense = torch.where(smooth > 0.1, 1, torch.where(smooth < -0.1, -1, 0)).tolist()
    count, last = 0, 0
    for value in sense:
        if value != 0 and last != 0 and value != last:
            count += 1
        if value != 0:
            last = value
    return count


def fights_1v1(rec: dict, loop: Loop, arena: float) -> list[dict]:
    """Each 1v1 fight: from both ships alive to the first death (or the game's end)."""
    out = []
    reach = float(loop.card.gun_range.max())
    T = rec["x"].shape[1]
    for b in range(rec["x"].shape[0]):
        alive = rec["alive"][b]
        both = (alive[:, 0] & alive[:, 1]).tolist()
        t = 0
        while t < T:
            if not both[t]:
                t += 1
                continue
            start = t
            while t < T and both[t]:
                t += 1
            end = t
            victim = None
            if end < T:
                died = rec["died"][b, end]
                victim = 0 if bool(died[0]) else (1 if bool(died[1]) else None)
            if end - start < 30:
                continue
            dx = rec["x"][b, start:end, 1] - rec["x"][b, start:end, 0]
            dy = rec["y"][b, start:end, 1] - rec["y"][b, start:end, 0]
            separation = torch.sqrt(dx * dx + dy * dy)
            radius = torch.sqrt(rec["x"][b, start:end] ** 2 + rec["y"][b, start:end] ** 2)
            types = [TYPE_NAMES[int(rec["ship_type"][b, i] >= loop.K)] for i in range(2)]
            merges = merges_of(rec, b, 0, 1, reach, start, end)
            out.append(
                dict(
                    environment=b,
                    start=start,
                    end=end,
                    seconds=(end - start) * rec["dt"],
                    types=types,
                    victim=None if victim is None else types[victim],
                    victim_index=victim,
                    merges=len(merges),
                    merge_ticks=merges,
                    circles=[turned(rec, b, i, start, end) for i in range(2)],
                    reversals=[reversals(rec, b, i, start, end) for i in range(2)],
                    max_separation=float(separation.max()) / reach,
                    extensions=extensions(separation / reach),
                    edge_seconds=float((radius > 0.9 * arena).any(1).float().sum()) * rec["dt"],
                )
            )
    return out


def extensions(separation: torch.Tensor) -> int:
    """Times the pair opened past 2 gun ranges after being inside 1 and came back inside 1."""
    count, inside, opened = 0, False, False
    for value in separation.tolist():
        if value < 1.0:
            if opened:
                count += 1
            inside, opened = True, False
        elif value > 2.0 and inside:
            opened = True
    return count


def ramp(hex_colour: str) -> LinearSegmentedColormap:
    """A one-hue ramp, light (early) to the full hue (late)."""
    return LinearSegmentedColormap.from_list("ramp", ["#e6e6e3", hex_colour])


def plot(
    rec: dict,
    loop: Loop,
    b: int,
    start: int,
    end: int,
    title: str,
    path: pathlib.Path,
    arena: float,
    kill_tick: int | None,
) -> None:
    """One window of one environment: every ship's track, time light to dark."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    x, y = rec["x"][b, start:end], rec["y"][b, start:end]
    alive = rec["alive"][b, start:end]
    reach = float(loop.card.gun_range.max())
    N = x.shape[1]
    S = N // 2
    for i in range(N):
        kind = int(rec["ship_type"][b, i] >= loop.K)
        colour = BLUE if kind == 0 else ORANGE
        points = torch.stack([x[:, i], y[:, i]], -1)
        keep = alive[:, i]
        segments = torch.stack([points[:-1], points[1:]], 1)[keep[1:] & keep[:-1]]
        if len(segments) == 0:
            continue
        times = torch.arange(len(points) - 1)[keep[1:] & keep[:-1]].float()
        lines = LineCollection(
            segments.numpy(),
            cmap=ramp(colour),
            linewidths=1.6,
            linestyles="solid" if i < S else (0, (4, 1.5)),
        )
        lines.set_array(times.numpy())
        lines.set_clim(0, len(points))
        ax.add_collection(lines)
        first = int(keep.float().argmax())
        ax.plot(x[first, i], y[first, i], "o", color=colour, ms=7, mec="white", mew=1.5)
    # merges (pair midpoints) and hits (on the target)
    for i in range(S):
        for j in range(S, N):
            for t in merges_of(rec, b, i, j, reach, start, end):
                mx = 0.5 * (rec["x"][b, t, i] + rec["x"][b, t, j])
                my = 0.5 * (rec["y"][b, t, i] + rec["y"][b, t, j])
                ax.plot(mx, my, "D", color="#555555", ms=6, mec="white", mew=1.0)
    hits = rec["hits"][b, start:end]  # (T, shooter, target)
    for shooter in range(N):
        targets = hits[:, shooter, :]
        ticks, victims = torch.nonzero(targets, as_tuple=True)
        enemy = (victims < S) != (shooter < S)
        ticks, victims = ticks[enemy], victims[enemy]
        if len(ticks):
            colour = BLUE if int(rec["ship_type"][b, shooter] >= loop.K) == 0 else ORANGE
            ax.plot(
                x[ticks, victims], y[ticks, victims], "x", color=colour, ms=5, mew=1.2, alpha=0.8
            )
    if kill_tick is not None:
        died = torch.nonzero(rec["died"][b, kill_tick]).flatten().tolist()
        for v in died:
            ax.plot(
                rec["x"][b, kill_tick - 1, v],
                rec["y"][b, kill_tick - 1, v],
                "X",
                color="#1a1a19",
                ms=16,
                mec="white",
                mew=1.5,
            )
    theta = torch.linspace(0, 2 * math.pi, 400)
    ax.plot(arena * torch.cos(theta), arena * torch.sin(theta), "--", color="#999999", lw=1)
    live = alive.any(1)
    xs, ys = x[alive], y[alive]
    if len(xs):
        cx, cy = float(xs.mean()), float(ys.mean())
        half = 0.55 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 4 * reach)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    del live
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(color="#eeeeee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=BLUE, lw=2, label="Yongma (energy)"),
        Line2D([], [], color=ORANGE, lw=2, label="Jujak (turn)"),
        Line2D([], [], color="#777777", lw=1.6, label="team 0 solid, team 1 dashed"),
        Line2D([], [], marker="o", ls="", color="#777777", label="start"),
        Line2D([], [], marker="D", ls="", color="#555555", label="merge"),
        Line2D([], [], marker="x", ls="", color="#777777", label="hit (on the target)"),
        Line2D([], [], marker="X", ls="", color="#1a1a19", ms=10, label="kill"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.85)
    ax.text(
        0.01,
        0.01,
        "time: light (start) to dark (end); arena edge dashed",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--lineup", default="energy v turn")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--plots", type=int, default=4)
    parser.add_argument("--window", type=float, default=90.0, help="s before the kill (2v2+)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    candidates = json.loads((HERE / args.candidates).read_text())
    candidate = candidates[args.candidate]
    by = {p.name: p for p in league.load()}
    player = player_of(candidate, by, 0)
    loop = Loop(
        args.games, 1, args.seconds, device=args.device, compiled=False, ships_per_team=args.size
    )
    loop.set_types([candidate.get("energy", {})], [candidate.get("turn", {})])
    rec = record(loop, [Group(0, args.lineup, player, player, args.games)])
    arena = float(loop.game.arena_radius)
    out = pathlib.Path(HERE / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = f"{candidate['name']}, {args.size}v{args.size} {args.lineup}"
    result = dict(candidate=candidate, size=args.size, lineup=args.lineup, games=args.games)
    if args.size == 1:
        fights = fights_1v1(rec, loop, arena)
        result["fights"] = fights
        shown = [f for f in fights if f["victim"] is not None][: args.plots]
        for n, f in enumerate(shown):
            kill = f["end"]
            title = (
                f"{label}\ngame {f['environment']}, {f['seconds']:.0f} s, {f['merges']} merges, "
                f"circles {f['circles'][0]:.1f} / {f['circles'][1]:.1f}, reversals "
                f"{f['reversals'][0]} / {f['reversals'][1]}; {f['victim']} killed"
            )
            plot(
                rec,
                loop,
                f["environment"],
                max(f["start"], kill - int(args.window * 4 / rec["dt"])),
                kill + 1,
                title,
                out.with_name(f"{out.name}-{n}.png"),
                arena,
                kill,
            )
        if not shown and fights:
            f = fights[0]
            plot(
                rec,
                loop,
                f["environment"],
                f["start"],
                min(f["end"], f["start"] + 3600),
                f"{label}\ngame {f['environment']}, no kill in {f['seconds']:.0f} s",
                out.with_name(f"{out.name}-0.png"),
                arena,
                None,
            )
    else:
        made = 0
        window = int(args.window / rec["dt"])
        for b in range(args.games):
            deaths = torch.nonzero(rec["died"][b].any(1)).flatten().tolist()
            deaths = [t for t in deaths if t > window // 3]
            if not deaths:
                continue
            kill = deaths[0]
            plot(
                rec,
                loop,
                b,
                max(0, kill - window),
                kill + 1,
                f"{label}\ngame {b}, the {kill * rec['dt']:.0f} s to the first kill",
                out.with_name(f"{out.name}-{made}.png"),
                arena,
                kill,
            )
            made += 1
            if made >= args.plots:
                break
    out.with_name(out.name + ".json").write_text(json.dumps(result, indent=1))
    if "fights" in result:
        fights = result["fights"]
        killed = [f for f in fights if f["victim"]]
        mean = lambda key: sum(f[key] for f in killed) / max(len(killed), 1)  # noqa: E731
        print(
            f"{label}: {len(fights)} fights, {len(killed)} kills; merges a kill mean "
            f"{mean('merges'):.2f}, seconds {mean('seconds'):.0f}, extensions "
            f"{mean('extensions'):.2f}, edge s {mean('edge_seconds'):.1f}"
        )


if __name__ == "__main__":
    main()
