"""Head-on passes and the loops after them (problem 3 of scripted-ghosts: diagnosis only).

    loops.py OUT [--sizes 1 1] [--envs 256] [--seed 0]

Scripted yardstick on both teams, CPU, GameConfig(training_spawn=True) without fields (fog on,
the training spawn), `envs` games of one team size each, every game to its end (no re-deal).
Per tick it records every ship's pose, stick and power actions, nose rate and the reasons of
its decision (lean.scripted.opponent.explain: target, tactic weights, plan, stick cap, speed
target), and the gates the loop depends on (engaged: range / engage range; level cap:
speed / middle stick speed).

A pass: a mutual pair (each the other's target) at a local minimum of their distance while
closing, with their headings within 30 degrees of opposite. Per pass: the lateral separation
(the closest approach), each ship's time from the pass to its first full-stick tick, the
loop (from the pass to the first tick both noses are within 30 degrees of the other ship
again, or the next pass, a death, the game's end): its length, stick level and turn rate,
tactic weights, plan, gates, and the range at its end. Per pair, the run of consecutive
passes before one of the two dies. Writes OUT (json) and prints a summary.
"""

import argparse
import dataclasses
import json
import math
import sys
import time

import numpy as np
import torch

import lean
from lean import game, scripted
from lean.scripted.opponent import TACTICS, explain
from lean.scripted.parameters import GATES

FACING = math.radians(30.0)


def gate(config, name: str, quantity: torch.Tensor) -> torch.Tensor:
    row = GATES[name]
    centre = config.values[row.centre]
    temperature = config.values["temperature"] if row.tempered else 1.0
    sharpness = (1.0 if row.rising else -1.0) / (row.scale * temperature)
    logit = (sharpness * torch.log(quantity.clamp_min(1e-6) / centre)).clamp(-30, 30)
    return row.floor + (1 - 2 * row.floor) * torch.sigmoid(logit)


@torch.no_grad()
def simulate(sizes, envs: int, seed: int) -> dict:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    config = dataclasses.replace(game.GameConfig(training_spawn=True), field_count=0)
    team_sizes = torch.tensor([list(sizes)]).expand(envs, 2)
    state = game.reset(config, envs, team_sizes, generator, device)
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, slots, generator, device)
    c = state.constants
    over = torch.zeros(envs, dtype=torch.bool)
    names = (
        "x",
        "y",
        "hc",
        "hs",
        "speed",
        "rate",
        "alive",
        "target",
        "stick",
        "power",
        "plan",
        "cap",
        "speed_target",
        "range_ratio",
        "level_cap",
        "store",
        "seen",
    )
    record = {name: [] for name in names + TACTICS}
    started = time.time()
    for tick in range(config.clock_ticks + 1):
        reasons = explain(agent, state, None, None, memory)
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        action, memory, _ = scripted.decide(agent, state, None, None, uniforms, memory)
        alive = state.alive & state.exists & ~over[:, None]
        target = reasons.marginals.target
        dx = state.x.gather(1, target) - state.x
        dy = state.y.gather(1, target) - state.y
        rows = {
            "x": state.x,
            "y": state.y,
            "hc": state.heading_cos,
            "hs": state.heading_sin,
            "speed": state.speed,
            "rate": state.nose_rate,
            "alive": alive,
            "target": target,
            "stick": action[..., 0],
            "power": action[..., 1],
            "plan": torch.sign(reasons.last_pass),
            "cap": reasons.cap,
            "speed_target": reasons.speed_target,
            "range_ratio": torch.sqrt(dx * dx + dy * dy)
            / c["gun_range"]
            / agent.values["engage_range"],
            "level_cap": gate(agent, "level_cap", state.speed / c["middle_stick_speed"]),
            "store": state.energy * c["inverse_max_energy"],
            "seen": state.ticks_since_seen == 0,
        }
        rows.update({name: reasons.weights[name] for name in TACTICS})
        for name, value in rows.items():
            record[name].append(value.clone())
        if bool(over.all()):
            break
        state, out = game.tick(config, state, action, generator)
        over = over | out.done
        if tick % 300 == 299:
            print(f"{time.strftime('%H:%M:%S')} {sizes} tick {tick + 1}", file=sys.stderr)
    data = {name: torch.stack(values).numpy() for name, values in record.items()}  # (T, E, N)
    data["team"] = state.team.numpy()
    data["constants"] = {
        name: float(c[name])
        for name in (
            "sustained_radius",
            "sustained_rate",
            "sustained_speed",
            "corner_speed",
            "gun_range",
            "middle_stick_speed",
            "vision_range",
        )
    }
    data["constants"]["tick_length"] = config.tick_length
    print(f"simulated {sizes} in {time.time() - started:.0f} s", file=sys.stderr)
    return data


def level_of(stick: np.ndarray) -> np.ndarray:
    """0 straight, 1 gradual, 2 middle, 3 full."""
    return (stick + 1) // 2


def analyse(data: dict, sizes) -> dict:
    constants = data["constants"]
    h = constants["tick_length"]
    T, E, N = data["x"].shape
    passes, runs = [], []
    for e in range(E):
        x, y = data["x"][:, e], data["y"][:, e]
        alive = data["alive"][:, e]
        target = data["target"][:, e]
        heading = np.arctan2(data["hs"][:, e], data["hc"][:, e])
        for i in range(N):
            for j in range(i + 1, N):
                if data["team"][e, i] == data["team"][e, j]:
                    continue
                both = alive[:, i] & alive[:, j]
                if not both.any():
                    continue
                mutual = both & (target[:, i] == j) & (target[:, j] == i)
                distance = np.hypot(x[:, j] - x[:, i], y[:, j] - y[:, i])
                opposite = np.abs(np.angle(np.exp(1j * (heading[:, i] - heading[:, j] - np.pi))))
                bearing_i = np.abs(
                    np.angle(
                        np.exp(
                            1j * (np.arctan2(y[:, j] - y[:, i], x[:, j] - x[:, i]) - heading[:, i])
                        )
                    )
                )
                bearing_j = np.abs(
                    np.angle(
                        np.exp(
                            1j * (np.arctan2(y[:, i] - y[:, j], x[:, i] - x[:, j]) - heading[:, j])
                        )
                    )
                )
                facing = (bearing_i < FACING) & (bearing_j < FACING)
                ticks = [
                    t
                    for t in range(1, T - 1)
                    if mutual[t]
                    and both[t + 1]
                    and distance[t] <= distance[t - 1]
                    and distance[t] < distance[t + 1]
                    and opposite[t] < FACING
                ]
                pair_passes = []
                for number, t in enumerate(ticks):
                    following = ticks[number + 1] if number + 1 < len(ticks) else T - 1
                    dead = np.nonzero(~both[t:])[0]
                    stop = min(following, t + dead[0] if len(dead) else T - 1)
                    # The loop ends when both face each other again (after leaving the pass).
                    end = None
                    for u in range(t + 1, stop):
                        if facing[u] and distance[u] > distance[t] + 100.0:
                            end = u
                            break
                    last = end if end is not None else stop
                    span = slice(t + 1, max(last, t + 2))
                    item = {
                        "env": e,
                        "pair": [i, j],
                        "tick": t,
                        "lateral": float(distance[t]),
                        "closing": float(-(distance[t] - distance[t - 1]) / h),
                        "loop_closed": end is not None,
                        "loop_seconds": float((last - t) * h),
                        "range_at_face": float(distance[end]) if end is not None else None,
                        "ended_by": "face"
                        if end is not None
                        else (
                            "pass"
                            if stop == following and following < T - 1
                            else ("death" if len(dead) else "clock")
                        ),
                    }
                    for side, ship in (("a", i), ("b", j)):
                        level = level_of(data["stick"][:, e, ship])
                        full = np.nonzero(level[t + 1 : stop] >= 3)[0]
                        middle = np.nonzero(level[t + 1 : stop] >= 2)[0]
                        item[side] = {
                            "to_full_s": float((full[0] + 1) * h) if len(full) else None,
                            "to_middle_or_more_s": float((middle[0] + 1) * h)
                            if len(middle)
                            else None,
                            "stick_level_mean": float(level[span].mean()),
                            "full_share": float((level[span] >= 3).mean()),
                            "rate_over_sustained": float(
                                np.abs(data["rate"][span, e, ship]).mean()
                                / constants["sustained_rate"]
                            ),
                            "speed_over_corner": float(
                                data["speed"][span, e, ship].mean() / constants["corner_speed"]
                            ),
                            "speed_target_over_corner": float(
                                data["speed_target"][span, e, ship].mean()
                                / constants["corner_speed"]
                            ),
                            "store": float(data["store"][span, e, ship].mean()),
                            "weights": {
                                name: float(data[name][span, e, ship].mean()) for name in TACTICS
                            },
                            "plan": float(data["plan"][span, e, ship].mean()),
                            "two_circle_share": float((data["plan"][span, e, ship] > 0).mean()),
                            "range_over_engage": float(data["range_ratio"][span, e, ship].mean()),
                            "range_over_engage_max": float(
                                data["range_ratio"][span, e, ship].max()
                            ),
                            "level_cap_gate": float(data["level_cap"][span, e, ship].mean()),
                            "cap": float(data["cap"][span, e, ship].mean()),
                            "seen_share": float(data["seen"][span, e, ship].mean()),
                            # The first 3 s after the pass.
                            "first3": {
                                "stick_level_mean": float(level[t + 1 : t + 91].mean()),
                                "disengaged": float(
                                    data["disengaged"][t + 1 : t + 91, e, ship].mean()
                                ),
                                "turning": float(data["turning"][t + 1 : t + 91, e, ship].mean()),
                                "merge": float(data["merge"][t + 1 : t + 91, e, ship].mean()),
                                "cap": float(data["cap"][t + 1 : t + 91, e, ship].mean()),
                                "rate_over_sustained": float(
                                    np.abs(data["rate"][t + 1 : t + 91, e, ship]).mean()
                                    / constants["sustained_rate"]
                                ),
                            },
                        }
                    pair_passes.append(item)
                passes += pair_passes
                if pair_passes:
                    died = not both[-1] and (
                        alive[-1, i] != alive[-1, j] or not (alive[-1, i] or alive[-1, j])
                    )
                    runs.append(
                        {
                            "env": e,
                            "pair": [i, j],
                            "passes": len(pair_passes),
                            "one_died": bool(died),
                        }
                    )
    return {
        "sizes": list(sizes),
        "envs": E,
        "constants": constants,
        "passes": passes,
        "runs": runs,
        "summary": summary(passes, runs, constants, E, T, h),
    }


def quantiles(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {"n": 0}
    p = np.percentile(values, [10, 50, 90])
    return {
        "n": int(len(values)),
        "p10": float(p[0]),
        "p50": float(p[1]),
        "p90": float(p[2]),
        "mean": float(values.mean()),
    }


def summary(passes, runs, constants, envs, ticks, h) -> dict:
    ships = [side for item in passes for side in (item["a"], item["b"])]
    radius = constants["sustained_radius"]
    out = {
        "passes": len(passes),
        "passes_per_game_minute": len(passes) / (envs * ticks * h / 60.0),
        "lateral_m": quantiles([p["lateral"] for p in passes]),
        "lateral_over_sustained_radius": quantiles([p["lateral"] / radius for p in passes]),
        "share_lateral_below_radius": float(np.mean([p["lateral"] < radius for p in passes]))
        if passes
        else None,
        "closing_speed": quantiles([p["closing"] for p in passes]),
        "loop_closed_share": float(np.mean([p["loop_closed"] for p in passes])) if passes else None,
        "ended_by": {
            k: sum(p["ended_by"] == k for p in passes) for k in ("face", "pass", "death", "clock")
        },
        "loop_seconds_closed": quantiles([p["loop_seconds"] for p in passes if p["loop_closed"]]),
        "range_at_face_m": quantiles([p["range_at_face"] for p in passes]),
        "to_full_s": quantiles([s["to_full_s"] for s in ships]),
        "never_full_share": float(np.mean([s["to_full_s"] is None for s in ships]))
        if ships
        else None,
        "to_middle_or_more_s": quantiles([s["to_middle_or_more_s"] for s in ships]),
    }
    for key in (
        "stick_level_mean",
        "full_share",
        "rate_over_sustained",
        "speed_over_corner",
        "speed_target_over_corner",
        "store",
        "plan",
        "two_circle_share",
        "range_over_engage",
        "range_over_engage_max",
        "level_cap_gate",
        "cap",
        "seen_share",
    ):
        out[key] = quantiles([s[key] for s in ships])
    for name in TACTICS:
        out[f"weight_{name}"] = quantiles([s["weights"][name] for s in ships])
    for key in ("stick_level_mean", "disengaged", "turning", "merge", "cap", "rate_over_sustained"):
        out[f"first3_{key}"] = quantiles([s["first3"][key] for s in ships])
    out["passes_per_pair"] = quantiles([r["passes"] for r in runs])
    out["passes_per_pair_ending_in_death"] = quantiles([r["passes"] for r in runs if r["one_died"]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs=2, default=[1, 1])
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    data = simulate(tuple(arguments.sizes), arguments.envs, arguments.seed)
    result = analyse(data, arguments.sizes)
    result["lean"] = lean.__file__
    with open(arguments.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
