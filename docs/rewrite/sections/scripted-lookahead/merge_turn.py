"""Merge turn for the look-ahead agent (the measurement of branch merge-turn,
docs/rewrite/sections/merge-turn/merge_turn.py, without the old agent's explanation rows):
how long a pilot flies on after a pass before its heading has changed by 30 and by 90 degrees.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/merge_turn.py \
        --agent look-ahead --out DIR

A pass event: two aircraft of opposite teams at a local minimum of their distance (the closest
approach at tick k) while each nose is within 60 degrees of the other aircraft one second
earlier; the same pair again within 2 s is not a new event. Per aircraft: the time from k until
its heading has changed by 30 and by 90 degrees, capped at 10 s (a capped aircraft counts as
10 s, a lower bound), censored (left out) when it is removed or its match ends first. Matches
as on merge-turn: `game.reset` with team sizes (S, S), seed 0, the same agent on both teams,
fog on; 256 environments for 1v1 and 64 for 4v4, 1,800 ticks, no re-deal. Setups: 1v1 without
fields, 1v1 with fields, 4v4 with fields. Prints median / 75th / 90th percentiles and writes
them to DIR/merge_turn_<agent>.json.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.scripted.rule_based import opponent as rule_based
from lean.scripted.rule_based.parameters import ScriptedConfig as RuleBasedConfig

SETUPS = {
    "1v1-nofields": (1, False, 256),
    "1v1-fields": (1, True, 256),
    "4v4-fields": (4, True, 64),
}
WINDOW = 300  # ticks: 10 s
FPS = 30


def simulate(agent: str, size: int, fields: bool, envs: int, ticks: int, device):
    config = game.GameConfig() if fields else game.GameConfig(field_count=0)
    generator = torch.Generator(device).manual_seed(0)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    memory = rule_based.fresh_memory(envs, slots, generator, device)
    tracks = {name: [] for name in ("x", "y", "heading", "alive", "over")}
    over = torch.zeros(envs, dtype=torch.bool, device=device)
    team = state.team.cpu().numpy()
    with torch.no_grad():
        for _ in range(ticks):
            for name, value in (
                ("x", state.x),
                ("y", state.y),
                ("heading", torch.atan2(state.heading_sin, state.heading_cos)),
                ("alive", state.alive & state.exists),
                ("over", over),
            ):
                tracks[name].append(value.cpu().numpy())
            if agent == "look-ahead":
                uniforms = scripted.draw_uniforms(envs, slots, generator, device)
                action = scripted.act(scripted.ScriptedConfig(), state, None, None, uniforms)
            else:
                uniforms = rule_based.draw_uniforms(envs, slots, generator, device)
                action, memory = rule_based.act(
                    RuleBasedConfig(), state, None, None, uniforms, memory
                )
            state, out = game.tick(config, state, action, generator)
            over = over | out.done
            if bool(over.all()):
                break
    return {name: np.stack(values) for name, values in tracks.items()}, team


def events_of(tracks, team):
    """Per aircraft in a pass: (ticks to 30 deg or None, ticks to 90 deg or None, end)."""
    x, y, heading, alive, over = (tracks[name] for name in ("x", "y", "heading", "alive", "over"))
    ticks, envs, slots = x.shape
    records = []
    for e in range(envs):
        for a in range(slots):
            for b in range(a + 1, slots):
                if team[e, a] == team[e, b]:
                    continue
                d = np.hypot(x[:, e, b] - x[:, e, a], y[:, e, b] - y[:, e, a])
                valid = alive[:, e, a] & alive[:, e, b] & ~over[:, e]
                last = -10 * FPS
                for k in range(FPS, ticks - 1):
                    if not (valid[k] and valid[k - 1] and valid[k + 1] and valid[k - FPS]):
                        continue
                    if not (d[k] <= d[k - 1] and d[k] < d[k + 1]) or k - last < 2 * FPS:
                        continue
                    facing = True
                    for s, p in ((a, b), (b, a)):
                        ex = x[k - FPS, e, p] - x[k - FPS, e, s]
                        ey = y[k - FPS, e, p] - y[k - FPS, e, s]
                        norm = math.hypot(ex, ey)
                        h = heading[k - FPS, e, s]
                        if norm <= 0 or (math.cos(h) * ex + math.sin(h) * ey) / norm < 0.5:
                            facing = False
                    if not facing:
                        continue
                    last = k
                    for s in (a, b):
                        records.append(turn_times(heading[:, e, s], alive[:, e, s], over[:, e], k))
    return records


def turn_times(heading, alive, over, k):
    t30 = t90 = None
    for rel in range(1, WINDOW + 1):
        t = k + rel
        if t >= len(heading) or over[t] or not alive[t]:
            return t30, t90, "censored" if t30 is None else "partial"
        change = abs(math.degrees(math.remainder(heading[t] - heading[k], 2 * math.pi)))
        if t30 is None and change >= 30.0:
            t30 = rel
        if t90 is None and change >= 90.0:
            t90 = rel
            return t30, t90, "reached"
    return t30, t90, "capped"


def summary(records):
    def times(index):
        values = [
            (r[index] if r[index] is not None else WINDOW) / FPS
            for r in records
            if r[index] is not None or r[2] == "capped"
        ]
        if not values:
            return None
        array = np.array(values)
        return [float(np.percentile(array, q)) for q in (50, 75, 90)]

    slow = sum(1 for r in records if (r[0] is None and r[2] == "capped") or (r[0] or 0) > 45)
    return {
        "aircraft": len(records),
        "censored_before_30": sum(1 for r in records if r[2] == "censored"),
        "t30": times(0),
        "t90": times(1),
        "slow_over_1_5_s": slow,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("look-ahead", "rule-based"), default="look-ahead")
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, (size, fields, envs) in SETUPS.items():
        if arguments.only and name != arguments.only:
            continue
        tracks, team = simulate(arguments.agent, size, fields, envs, arguments.ticks, device)
        results[name] = summary(events_of(tracks, team))
        row = results[name]
        print(
            f"{arguments.agent} {name}: {row['aircraft']} aircraft, t30 {row['t30']}, "
            f"t90 {row['t90']}, slow {row['slow_over_1_5_s']}",
            flush=True,
        )
        arguments.out.mkdir(parents=True, exist_ok=True)
        path = arguments.out / f"merge_turn_{arguments.agent}.json"
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
