"""Vision range in scripted 4v4: how far each ship is from its target and from the nearest
enemy, by tactic, and how often a turning fight loses sight of its target. Measures; changes
nothing in the game or the agent.

The game is the training game (GameConfig(training_spawn=True): drawn fields, fog, ghost
bullets, the 1,800-tick clock), 4v4, the default scripted agent (ScriptedConfig(), persona
None: the yardstick) on both teams in one call, E environments for T ticks, finished
environments re-dealt. The tactic label is replay.py's: the arg-max tactic weight, a turning
fight split by the sign of the agent's last-pass plan (2c two-circle, 1c one-circle, "?" no
plan yet). "Visible" is the fog as the agent reads it: the enemy team saw the ship this tick
(ticks_since_seen == 0: in some enemy's sight, or revealed by firing or being hit).

    # collect (GPU), one chunk file per batch of ticks:
    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/vision/vision_measure.py \
        collect --envs 1024 --ticks 3600 --out DIR
    # analyse (CPU): tables to stdout, charts beside this file
    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/vision/vision_measure.py \
        analyse --out DIR
"""

import argparse
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.game import core
from lean.scripted.opponent import TACTICS, explain, sample

LABELS = ("SEEK", "MERGE", "OFF", "TURN 1c", "TURN 2c", "TURN ?", "BREAK")
TURNING = (3, 4, 5)
CANDIDATES = (1000.0, 1250.0, 1500.0, 1750.0, 2000.0)
HERE = Path(__file__).parent


# ---------------------------------------------------------------------- collection


def first(value) -> float:
    """A ship constant's value (every ship is the same type): a float or a tensor's first."""
    return float(value.flatten()[0]) if isinstance(value, torch.Tensor) else float(value)


def label(weights: torch.Tensor, last_pass: torch.Tensor) -> torch.Tensor:
    """(E, N) int8 index into LABELS, as replay.py derives it."""
    top = weights.argmax(-1)  # TACTICS order: disengaged, merge, offensive, turning, defensive
    plan = torch.sign(last_pass)
    turning = torch.where(plan > 0, 4, torch.where(plan < 0, 3, 5))
    mapped = torch.tensor([0, 1, 2, -1, 6], device=top.device)[top]
    return torch.where(top == 3, turning, mapped).to(torch.int8)


@torch.no_grad()
def collect(arguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs, ticks, batch = arguments.envs, arguments.ticks, arguments.batch
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    config = core.GameConfig(training_spawn=True)
    sizes = torch.tensor([[4, 4]] * envs, device=device)
    state = game.deal(config, envs, 8, sizes, generator, device)
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, slots, generator, device)
    episode = torch.zeros(envs, dtype=torch.int32, device=device)
    c = state.constants
    meta = {
        "envs": envs,
        "ticks": ticks,
        "slots": slots,
        "tick_length": config.tick_length,
        "clock_ticks": config.clock_ticks,
        "arena_radius": config.arena_radius,
        "vision_range": first(c["vision_range"]),
        "gun_range": first(c["gun_range"]),
        "sustained_radius": first(c["sustained_radius"]),
        "sustained_speed": first(c["sustained_speed"]),
        "corner_speed": first(c["corner_speed"]),
        "corner_rate": first(c["corner_rate"]),
        "device": str(device),
        "labels": LABELS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    began = time.time()
    buffer: dict[str, list] = {}
    for step in range(ticks):
        reasons = explain(agent, state, None, None, memory)
        weights = torch.stack([reasons.weights[name] for name in TACTICS], -1)
        parts = reasons.marginals
        controlled = parts.controlled
        target = torch.where(controlled, parts.target, -1)
        alive = state.alive & state.exists
        enemy = state.team[:, :, None] != state.team[:, None, :]
        dx = state.x[:, :, None] - state.x[:, None, :]
        dy = state.y[:, :, None] - state.y[:, None, :]
        distance = torch.sqrt(dx * dx + dy * dy)  # (E, own, other)
        live_enemy = enemy & alive[:, None, :]
        seen = state.ticks_since_seen == 0  # (E, N): the enemy team sees this ship now
        sighted = core.sighted_by_enemy(state) & state.exists
        far = torch.full_like(distance, math.inf)
        nearest = torch.where(live_enemy, distance, far).amin(-1)
        nearest_visible = torch.where(live_enemy & seen[:, None, :], distance, far).amin(-1)
        safe = target.clamp_min(0)
        to_target = torch.where(target >= 0, distance.gather(-1, safe[..., None])[..., 0], math.nan)
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        action = sample(parts, uniforms)
        memory = {
            "target": target,
            "side": memory["side"],
            "sense": torch.where(controlled, reasons.sense, 0.0),
            "last_pass": torch.where(controlled, reasons.last_pass, 0.0),
        }
        heading = torch.atan2(state.heading_sin, state.heading_cos)
        record = {
            "label": label(weights, reasons.last_pass),
            "alive": alive,
            "target": target.to(torch.int8),
            "to_target": to_target.to(torch.float16),
            "nearest": nearest.clamp_max(6e4).to(torch.float16),
            "nearest_visible": nearest_visible.clamp_max(6e4).to(torch.float16),
            "seen": seen,
            "sighted": sighted,
            "speed": state.speed.to(torch.float16),
            "heading": heading,
            "episode": episode.clone(),
            "game_tick": state.tick.to(torch.int16).clone(),
        }
        state, result = game.tick(config, state, action, generator)
        done = result.done
        record["done"] = done
        for name, value in record.items():
            buffer.setdefault(name, []).append(value.cpu())
        fresh = game.deal(config, envs, slots, sizes, generator, device)
        state = game.take_deal(state, done, fresh)
        new_memory = scripted.fresh_memory(envs, slots, generator, device)
        memory = {
            name: torch.where(
                done.view(-1, *[1] * (memory[name].dim() - 1)), new_memory[name], memory[name]
            )
            for name in memory
        }
        episode = episode + done.to(torch.int32)
        if (step + 1) % batch == 0 or step + 1 == ticks:
            chunk = {name: torch.stack(values).numpy() for name, values in buffer.items()}
            index = step // batch
            np.savez(out / f"chunk_{index:03d}.npz", **chunk)
            buffer = {}
            elapsed = time.time() - began
            rate = elapsed / (step + 1)
            finish = datetime.datetime.now() + datetime.timedelta(seconds=rate * (ticks - step - 1))
            print(
                f"{datetime.datetime.now():%H:%M:%S} tick {step + 1} of {ticks}, "
                f"{elapsed:.0f} s, {1000 * rate:.1f} ms/tick, games ended "
                f"{int(episode.sum())}, chunk {index} saved, ETA {finish:%H:%M:%S}",
                flush=True,
            )
    print("done", flush=True)


# ---------------------------------------------------------------------- analysis


def load(out: Path) -> tuple[dict, dict]:
    meta = json.loads((out / "meta.json").read_text())
    files = sorted(out.glob("chunk_*.npz"))
    parts: dict[str, list] = {}
    for file in files:
        with np.load(file) as chunk:
            for name in chunk.files:
                parts.setdefault(name, []).append(chunk[name])
    data = {name: np.concatenate(values) for name, values in parts.items()}
    return meta, data


def percentiles(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [math.nan] * 4
    return [float(v) for v in np.percentile(values, (50, 90, 95, 99))]


def reverse_next(flag: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """(T, ...) the first tick >= t where flag is True, within the same stretch (a stretch
    ends at the ticks where boundary is True, inclusive); T + 1 when none."""
    ticks = flag.shape[0]
    result = np.empty(flag.shape, dtype=np.int32)
    running = np.full(flag.shape[1:], ticks + 1, dtype=np.int32)
    for t in range(ticks - 1, -1, -1):
        running = np.where(boundary[t], ticks + 1, running)
        running = np.where(flag[t], t, running)
        result[t] = running
    return result


def analyse(arguments) -> None:
    meta, d = load(arguments.out)
    tick_length = meta["tick_length"]
    minute = 60.0 / tick_length
    ticks, envs, slots = d["label"].shape
    label_, alive, target = d["label"], d["alive"], d["target"].astype(np.int64)
    to_target = d["to_target"].astype(np.float32)
    seen, done = d["seen"], d["done"]
    has = alive & (target >= 0)
    safe = np.clip(target, 0, slots - 1)
    target_seen = np.take_along_axis(seen, safe, -1) & has
    target_alive = np.take_along_axis(alive, safe, -1) & has
    target_sighted = np.take_along_axis(d["sighted"], safe, -1) & has
    turning = np.isin(label_, TURNING)
    episodes = int(done.sum())
    lines = [
        f"{envs} environments x {ticks} ticks, 4v4, {episodes} games ended, "
        f"{int(alive.sum())} living ship-ticks ({alive.sum() * tick_length / 60:.0f} ship-minutes)",
        f"vision {meta['vision_range']:.0f} m, gun range {meta['gun_range']:.0f} m, "
        f"arena radius {meta['arena_radius']:.0f} m",
    ]

    # Consecutive pairs (t-1, t) in the same game, the ship alive at both, the same target.
    same = (~done[:-1])[:, :, None] & alive[:-1] & alive[1:] & (target[:-1] == target[1:])
    same &= has[:-1] & target_alive[1:]
    lost = same & target_seen[:-1] & ~target_seen[1:]  # visible -> not visible
    # Regain: the lost enemy is seen again by the team before the game ends, either ship dies.
    boundary = np.broadcast_to(done[:, :, None], alive.shape)
    next_seen = reverse_next(seen, boundary)  # per enemy slot
    next_gone = reverse_next(~alive, boundary)  # per slot
    next_end = reverse_next(np.broadcast_to(done[:, :, None], alive.shape), np.zeros_like(boundary))
    event_t, event_e, event_i = np.nonzero(lost)
    t1 = event_t + 1
    j = target[t1, event_e, event_i]
    again = next_seen[t1, event_e, j]
    limit = np.minimum.reduce(
        [next_gone[t1, event_e, event_i], next_gone[t1, event_e, j], next_end[t1, event_e, j]]
    )
    censored = limit > ticks  # the run ended first: unknown
    regained = again < limit
    gap = (again - t1) * tick_length
    event_label = label_[event_t, event_e, event_i]

    table = [
        "| tactic | ship-min | share | target p50 | p90 | p95 | p99 | target > 1,000 m | "
        "target seen | nearest p50 | p95 | losses / min | of them > 1 s | regained | "
        "regain p50 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    stats = {}
    groups = [(name, label_ == k) for k, name in enumerate(LABELS)]
    groups.append(("TURN all", turning))
    for name, mask in groups:
        m = mask & has
        m_all = mask & alive
        dist = to_target[m]
        near = d["nearest"][m_all].astype(np.float32)
        near = near[near < 6e4]
        minutes = m_all.sum() / minute
        events = np.isin(event_label, [LABELS.index(name)] if name in LABELS else TURNING)
        known = events & ~censored
        p = percentiles(dist)
        back = known & regained
        middle_gap = float(np.median(gap[back])) if back.any() else math.nan
        stats[name] = {
            "minutes": float(minutes),
            "p": p,
            "beyond": float((dist > 1000).mean()) if dist.size else math.nan,
            "seen": float(target_seen[m].mean()) if m.any() else math.nan,
            "losses": float(events.sum() / max(minutes, 1e-9)),
            "events": int(events.sum()),
            "long": float((events & ~(regained & (gap <= 1.0))).sum() / max(minutes, 1e-9)),
            "regained": float(regained[known].mean()) if known.any() else math.nan,
        }
        table.append(
            f"| {name} | {minutes:.0f} | {m_all.sum() / alive.sum():.1%} | "
            + " | ".join(f"{v:.0f}" for v in p)
            + f" | {stats[name]['beyond']:.1%} | {stats[name]['seen']:.1%} | "
            + " | ".join(f"{v:.0f}" for v in percentiles(near)[0::2])
            + f" | {stats[name]['losses']:.2f} | {stats[name]['long']:.2f}"
            + f" | {stats[name]['regained']:.0%} of "
            f"{int(known.sum())} | {middle_gap:.1f} |"
        )
    lines += ["", "Per tactic (distances in m, true positions)", *table]

    # Fog versus range at 1,000 m inside turning fights.
    tm = turning & has
    in_range = to_target <= meta["vision_range"]
    lines += [
        "",
        f"Turning fights, target visible (fog): {target_seen[tm].mean():.1%}; in the sight test "
        f"{target_sighted[tm].mean():.1%}; within {meta['vision_range']:.0f} m: "
        f"{in_range[tm].mean():.1%}; visible though beyond range (shared sight or a shot's "
        f"reveal): {(target_seen & ~in_range)[tm].mean():.1%}; within range but hidden (a field "
        f"core blocks): {(~target_seen & in_range)[tm].mean():.1%}",
    ]
    turning_events = np.isin(event_label, TURNING)
    lost_beyond = to_target[t1, event_e, event_i][turning_events] > meta["vision_range"]
    lines.append(
        f"Turning-fight contact losses with the target beyond {meta['vision_range']:.0f} m at "
        f"the loss: {lost_beyond.mean():.1%} of {turning_events.sum()}"
    )

    # Candidate vision ranges, offline (distance only).
    fight_minutes = (turning & alive).sum() / minute
    candidate_rows = [
        "| vision m | turning ticks in range | losses / fight-min | of them > 1 s | "
        "TURN 2c in range | TURN 1c in range |",
        "|---|---|---|---|---|---|",
    ]
    candidates = {}
    for vision in CANDIDATES:
        inside = (to_target <= vision) & has
        drop = same & turning[:-1] & inside[:-1] & ~inside[1:]
        share = inside[tm].mean()
        two = inside[(label_ == 4) & has].mean()
        one = inside[(label_ == 3) & has].mean()
        rate = drop.sum() / fight_minutes
        back = reverse_next(inside, boundary)  # the ship's next tick with its target in range
        drop_t, drop_e, drop_i = np.nonzero(drop)
        out_for = back[drop_t + 1, drop_e, drop_i] - (drop_t + 1)
        long_rate = (out_for * tick_length > 1.0).sum() / fight_minutes
        candidates[vision] = (float(share), float(rate), float(two), float(one), long_rate)
        candidate_rows.append(
            f"| {vision:.0f} | {share:.1%} | {rate:.2f} | {long_rate:.2f} | {two:.1%} | {one:.1%} |"
        )
    lines += ["", "Candidate ranges (distance to target only)", *candidate_rows]

    # Measured turn radius in turning fights: speed / heading rate.
    heading = d["heading"]
    turn = np.angle(np.exp(1j * (heading[1:] - heading[:-1]))) / tick_length
    steady = same & turning[:-1] & (np.abs(turn) > 0.05)
    radius = d["speed"][:-1].astype(np.float32)[steady] / np.abs(turn[steady])
    speed = d["speed"][:-1].astype(np.float32)[turning[:-1] & alive[:-1]]
    lines += [
        "",
        f"Turning fights: speed p10/50/90 {np.percentile(speed, 10):.0f}/"
        f"{np.median(speed):.0f}/{np.percentile(speed, 90):.0f} m/s; flown radius (speed / "
        f"heading rate) p10/50/90 {np.percentile(radius, 10):.0f}/{np.median(radius):.0f}/"
        f"{np.percentile(radius, 90):.0f} m",
    ]

    # Geometry: the card's turn readouts (game units) and a two-circle fight's largest
    # separation, about two turn diameters plus the merge offset. The merge offset's proxy:
    # the distance to the target at the closest tick of each MERGE stretch that ends in a
    # turning fight against the same target.
    from lean.game import tuning

    readout = tuning.performance(tuning.cached_real_row("p51d"))
    corner_radius = readout["corner_speed"] / readout["corner_rate"]
    merge = (label_ == 4) & has
    entering = same & (label_[:-1] == 1) & merge[1:]
    offsets = []
    t_in, e_in, i_in = np.nonzero(entering)
    for t, e, i in zip(t_in[:20000], e_in[:20000], i_in[:20000]):
        start = t
        while start > 0 and label_[start - 1, e, i] == 1 and same[start - 1, e, i]:
            start -= 1
        offsets.append(float(to_target[start : t + 1, e, i].min()))
    offset = float(np.median(offsets)) if offsets else math.nan
    lines += [
        "",
        f"Card readouts (tuning.performance): best sustained turn {readout['sustained_rate']:.3f} "
        f"rad/s at {readout['sustained_speed']:.0f} m/s, radius {readout['sustained_radius']:.0f}"
        f" m; tightest sustained radius {readout['min_radius']:.0f} m at "
        f"{readout['min_radius_speed']:.0f} m/s; corner {readout['corner_speed']:.0f} m/s at "
        f"{readout['corner_rate']:.3f} rad/s, radius {corner_radius:.0f} m",
        f"Merge offset proxy (closest approach of MERGE stretches entering TURN 2c, "
        f"{len(offsets)} stretches): p50 {offset:.0f} m, p90 "
        f"{np.percentile(offsets, 90) if offsets else math.nan:.0f} m",
        f"Two-circle largest separation 4R + offset: at the sustained radius "
        f"{4 * readout['sustained_radius'] + offset:.0f} m, at corner "
        f"{4 * corner_radius + offset:.0f} m, at the flown median radius "
        f"{4 * np.median(radius) + offset:.0f} m",
    ]

    # Seekers late in the game.
    game_tick = d["game_tick"][:, :, None]
    seek = (label_ == 0) & alive
    for name, late in (
        ("whole game", np.ones_like(seek)),
        ("game tick >= 1,200 (last 20 s)", np.broadcast_to(game_tick >= 1200, seek.shape)),
    ):
        mask = seek & late
        if not mask.any():
            continue
        near = d["nearest"][mask].astype(np.float32)
        near_visible = d["nearest_visible"][mask].astype(np.float32)
        lines.append(
            f"SEEK, {name}: {mask.sum() / minute:.1f} ship-min "
            f"({mask.sum() / (alive & late).sum():.1%} of living ship-ticks); "
            f"nearest enemy p50/p90 {np.median(near):.0f}/"
            f"{np.percentile(near, 90):.0f} m, > 1,500 m {(near > 1500).mean():.1%}; no enemy "
            f"visible to the team {(near_visible >= 6e4).mean():.1%}, nearest visible enemy "
            f"> 1,000 m or none {(near_visible > 1000).mean():.1%}"
        )

    text = "\n".join(lines)
    print(text)
    (arguments.out / "results.md").write_text(text + "\n")
    charts(meta, to_target, label_, has, turning, candidates, arguments.charts)


def charts(meta, to_target, label_, has, turning, candidates, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e3e2de"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )
    directory.mkdir(parents=True, exist_ok=True)

    # 1: share of turning ticks in range vs vision range (continuous), candidates marked.
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    grid_v = np.linspace(500, 2500, 81)
    for mask, colour, name in (
        ((label_ == 4) & has, blue, "TURN 2c"),
        ((label_ == 3) & has, orange, "TURN 1c"),
    ):
        values = np.sort(to_target[mask])
        share = np.searchsorted(values, grid_v, side="right") / values.size
        axis.plot(grid_v, 100 * share, color=colour, linewidth=2.2)
        axis.annotate(
            name,
            (grid_v[-1], 100 * share[-1]),
            xytext=(-4, -14 if name == "TURN 1c" else 6),
            textcoords="offset points",
            ha="right",
            color=colour,
            fontsize=12,
        )
    for vision in candidates:
        axis.axvline(vision, color=grid, linewidth=1, zorder=0)
    axis.axvline(meta["vision_range"], color=muted, linewidth=1.2, linestyle="--")
    axis.text(meta["vision_range"] + 20, 3, "now", color=muted, fontsize=11)
    axis.set_xlabel("vision range (m)")
    axis.set_ylabel("ticks with target in range (%)")
    axis.set_ylim(0, 101)
    axis.set_title("Turning-fight ticks within vision range", loc="left", fontsize=14, color=ink)
    figure.tight_layout()
    figure.savefig(directory / "in-range-vs-vision.png")
    plt.close(figure)

    # 2: distance histogram for TURN 2c with the gun and vision ranges marked.
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    values = to_target[(label_ == 4) & has]
    axis.hist(values, bins=np.arange(0, 2600, 50), color=blue, edgecolor="white", linewidth=0.5)
    top = axis.get_ylim()[1]
    for x, name, height in (
        (meta["gun_range"], "gun", 0.95),
        (meta["vision_range"], "vision", 0.82),
    ):
        axis.axvline(x, color=ink, linewidth=1.2, linestyle="--")
        axis.text(x + 25, top * height, f"{name} {x:.0f} m", color=ink, fontsize=11)
    axis.set_xlabel("distance to target (m)")
    axis.set_ylabel("ship-ticks")
    axis.set_title("TURN 2c: distance to target", loc="left", fontsize=14, color=ink)
    figure.tight_layout()
    figure.savefig(directory / "turn2c-distance.png")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("collect", "analyse"))
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--charts", type=Path, default=HERE)
    arguments = parser.parse_args()
    if arguments.mode == "collect":
        collect(arguments)
    else:
        analyse(arguments)


if __name__ == "__main__":
    main()
