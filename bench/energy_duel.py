"""Energy duel: 1v1 head-on with hand-written controllers in the lean game (lean.game.core).

Question (author, 2026-11-25): with identical ships, can energy management (power 1 dive,
power 2 climb, power 0 coast) break the head-on stalemate that the kinematic model shows?

Setup: no fields, fog off, Blue (slot 0, team 0) and Red (slot 1, team 1) start 3 km apart
nose to nose at the sustained-turn speed with the store at half (the deal's spawn store).
Both steer at full stick toward the scripted agent's lead point (straight once the nose is
within the gun tolerance of it). Red coasts all game. Blue at each merge (Red more than 90 deg
off its nose) commits a turn sense (two-circle: toward Red's side; one-circle: away) and holds
full stick that way until its nose is within ACQUIRE of the lead point; its power follows the
mode's schedule on the circle phase (its heading change since the first merge in the
committed sense, modulo 360 deg). Both fire when the nose is within the gun tolerance of the
lead point and inside gun range.

Usage (from the repository root): PYTHONPATH=. uv run --no-sync python bench/energy_duel.py
OUT_DIR [--envs 64] [--ticks 1800]
Writes OUT_DIR/results.json, OUT_DIR/traces.pt and the charts (bench/energy_duel_charts.py).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from lean.game import card, core
from lean.scripted.parameters import ScriptedConfig

COAST, DIVE, CLIMB = 0, 1, 2
FULL_LEFT, FULL_RIGHT = 6, 5  # stick indices: even turns left (card.STICK_LEVELS)
START_DISTANCE = 3000.0  # m
START_LATERAL = 30.0  # m: Red's lateral offset, so the pass has a side (left to left)
ACQUIRE = math.radians(10.0)
MERGE_BEARING = 0.5 * math.pi
DAMAGING = 1.0  # shield points: a hit "damaging" at or above this (a full hit is 21)
PASS_RANGE = 1500.0  # m: a closest approach under this is a pass
PASS_REARM = 200.0  # m the distance must open after a pass before the next one counts
PASS_LOOK = 30  # ticks after a pass at which Blue's angles off are read (1 s)
TWO_PI = 2.0 * math.pi


@dataclass
class Condition:
    name: str
    mode: int  # 1..5
    plan: str  # "one-circle" or "two-circle"
    switch: float  # degrees: modes 2/3 the boost/climb switch; modes 4/5 the near-Red half-width
    balanced: bool = False


def default_switch(mode: int) -> float:
    return {1: 0.0, 2: 180.0, 3: 180.0, 4: 90.0, 5: 90.0}[mode]


def blue_power(mode: Tensor, switch: Tensor, phase: Tensor, merged: Tensor) -> Tensor:
    """Blue's power index (E,) from the mode, the switch (radians) and the circle phase."""
    first = phase < switch  # modes 2, 3: the first part of the circle
    near = (phase < switch) | (phase > TWO_PI - switch)  # modes 4, 5: the part near Red
    dive = torch.where(
        mode == 2, first, torch.where(mode == 3, ~first, torch.where(mode == 4, near, ~near))
    )
    power = torch.where(dive, DIVE, CLIMB)
    return torch.where((mode == 1) | ~merged, COAST, power)


def exchange_rate(constants, speed: Tensor, stick: Tensor, power: Tensor) -> Tensor:
    """Blue's store rate without the store's edges (E,): speed x the commanded exchange for a
    dive; for a climb the same with the opposite sign, except at full stick, where the turn
    takes the whole lift budget and the climb makes nothing (lean.game.core._setting;
    lean.game.core.steady_rates gives these rates for a steady turn)."""
    climb = constants["commanded_table"][:, CLIMB, 0]  # (E, 3, N): Blue's row (slot 0)
    full = stick >= FULL_RIGHT
    sign = torch.where(power == DIVE, -1.0, torch.where((power == CLIMB) & ~full, 1.0, 0.0))
    return sign * speed * climb


def signed_angle(cos, sin, x, y):
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


def run(conditions: list[Condition], envs: int, ticks: int, seed: int, keep_traces: bool):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    config = core.GameConfig(field_count=0, fog=False, clock_ticks=ticks)
    C = len(conditions)
    E = C * envs
    state = core.reset(config, E, torch.tensor([[1, 1]]).expand(E, 2), generator, "cpu")
    c = state.constants
    scripted = ScriptedConfig().values
    gun_range = c["gun_range"]
    bullet_drag, launch = c["bullet_drag"], c["bullet_launch_speed"]
    hit_radius, h = c["hit_radius"], config.tick_length

    # Head-on placement through the game's spawn, then the start speed.
    half = 0.5 * START_DISTANCE
    x = torch.tensor([[-half, half]]).expand(E, 2).clone()
    y = torch.tensor([[0.0, START_LATERAL]]).expand(E, 2).clone()
    heading = torch.tensor([[0.0, math.pi]]).expand(E, 2)
    chosen = torch.ones(E, 2, dtype=torch.bool)
    state = core.spawn(
        config, state, chosen, x, y, torch.cos(heading), torch.sin(heading), generator
    )
    speed0 = c["sustained_speed"]
    speed = torch.full((E, 2), speed0)
    state = state.replace(
        speed=speed,
        velocity_x=speed * state.heading_cos,
        velocity_y=speed * state.heading_sin,
    )

    mode = torch.tensor([cond.mode for cond in conditions]).repeat_interleave(envs)
    switch = torch.tensor([math.radians(cond.switch) for cond in conditions]).repeat_interleave(
        envs
    )
    two_circle = torch.tensor([cond.plan == "two-circle" for cond in conditions]).repeat_interleave(
        envs
    )
    balanced = torch.tensor([cond.balanced for cond in conditions]).repeat_interleave(envs)

    # Blue's controller memory.
    blue_phase = torch.zeros(E, dtype=torch.long)  # 0 approach / tracking, 1 committed
    sense = torch.zeros(E)  # committed turn sense, +1 left
    first_sense = torch.zeros(E)
    merged = torch.zeros(E, dtype=torch.bool)
    turned = torch.zeros(E)  # heading change since the first merge, in the first sense
    tracking_armed = torch.ones(E, dtype=torch.bool)  # a new merge may commit
    done = torch.zeros(E, dtype=torch.bool)
    result = torch.full((E,), -1, dtype=torch.long)
    end_tick = torch.full((E,), ticks, dtype=torch.long)

    # Records.
    damage = torch.zeros(E, 2)  # dealt by [blue, red]
    first_hit = torch.full((E, 2), math.nan)  # seconds, by attacker
    shots = torch.zeros(E, 2)
    speed_sum = torch.zeros(E, 2)
    rate_sum = torch.zeros(E, 2)
    live_ticks = torch.zeros(E)
    circle_store = [[] for _ in range(E)]
    circle_total = [[] for _ in range(E)]
    circle_time = [[] for _ in range(E)]
    circle_exchange = [[] for _ in range(E)]
    exchanged = torch.zeros(E)
    circles_done = torch.zeros(E, dtype=torch.long)
    passes = [[] for _ in range(E)]  # [time, distance, crossing angle, Blue's angles off 1 s on]
    last_distance = torch.full((E,), math.inf)
    prev_distance = torch.full((E,), math.inf)
    pass_armed = torch.ones(E, dtype=torch.bool)
    pass_min = torch.full((E,), math.inf)
    measure_at = torch.full((E,), -1, dtype=torch.long)
    traces = {k: [] for k in ("x", "y", "speed", "energy", "power")} if keep_traces else None
    hits = []  # (env, tick, attacker, damage, x, y) of damaging hits (traced envs only)
    trace_envs = torch.arange(C) * envs

    started = time.time()
    for t in range(ticks):
        # Geometry from each ship to the other (slot 0 <-> 1).
        other = torch.tensor([1, 0])
        ox = state.x[:, other] - state.x
        oy = state.y[:, other] - state.y
        distance = torch.sqrt(ox * ox + oy * oy).clamp_min(1e-6)
        vx, vy = state.velocity_x, state.velocity_y
        # The scripted agent's lead (lean.scripted.opponent.explain): drag-law time of flight
        # to the target's range, relative velocity, its lead gain and cap.
        lead_distance = (distance / gun_range).clamp_max(scripted["lead_cap"])
        flight = torch.expm1(bullet_drag * gun_range * lead_distance) * (
            scripted["lead_gain"] / (bullet_drag * launch)
        )
        aim_x = ox + (vx[:, other] - vx) * flight
        aim_y = oy + (vy[:, other] - vy) * flight
        error = signed_angle(state.heading_cos, state.heading_sin, aim_x, aim_y)
        bearing = signed_angle(state.heading_cos, state.heading_sin, ox, oy)
        # The scripted agent's gun tolerance: aligned x the target's angular size (clamped).
        size = (2.0 * torch.atan(hit_radius / distance)).clamp(
            scripted["target_size_minimum"], scripted["target_size_maximum"]
        )
        tolerance = scripted["aligned"] * size
        on = error.abs() < tolerance
        fire = on & (distance < gun_range)
        toward = torch.where(error > 0.0, FULL_LEFT, FULL_RIGHT)
        stick = torch.where(on, 0, toward)

        # Blue's merge logic.
        blue_bearing = bearing[:, 0]
        aft = blue_bearing.abs() > MERGE_BEARING
        commit = aft & tracking_armed & (blue_phase == 0)
        towards_red = torch.where(blue_bearing > 0.0, 1.0, -1.0)
        new_sense = torch.where(two_circle, towards_red, -towards_red)
        sense = torch.where(commit, new_sense, sense)
        first_sense = torch.where(commit & ~merged, new_sense, first_sense)
        merged = merged | commit
        blue_phase = torch.where(commit, 1, blue_phase)
        tracking_armed = tracking_armed & ~commit
        acquired = (blue_phase == 1) & (error[:, 0].abs() < ACQUIRE)
        blue_phase = torch.where(acquired, 0, blue_phase)
        # Re-arm once Red is forward of the 3/9 line again.
        tracking_armed = tracking_armed | (blue_bearing.abs() < MERGE_BEARING)
        committed_stick = torch.where(sense > 0.0, FULL_LEFT, FULL_RIGHT)
        blue_stick = torch.where(blue_phase == 1, committed_stick, stick[:, 0])
        phase = torch.remainder(turned, TWO_PI)
        power_blue = blue_power(mode, switch, phase, merged)
        # Balanced variants climb at the middle stick: at full stick the turn takes the whole
        # lift budget and a climb gains no store (lean.game.core._setting).
        middle = balanced & (power_blue == CLIMB) & (blue_stick >= FULL_RIGHT)
        blue_stick = torch.where(middle, blue_stick - 2, blue_stick)
        sticks = torch.stack([blue_stick, stick[:, 1]], 1)
        powers = torch.stack([power_blue, torch.zeros_like(power_blue)], 1)
        action = torch.stack([sticks, powers, fire.long()], -1)

        old_cos, old_sin = state.heading_cos[:, 0].clone(), state.heading_sin[:, 0].clone()
        applied = state.pending[:, 0]  # what this tick applies (the one-tick delay)
        exchanged = exchanged + h * exchange_rate(
            state.constants, state.speed[:, 0], applied[:, 0], applied[:, 1]
        )
        state, out = core.tick(config, state, action, generator)
        live = ~done
        # Records for live environments.
        dealt = torch.stack([out.hit_damage[:, 0, 1], out.hit_damage[:, 1, 0]], 1)
        damage += torch.where(live[:, None], dealt, 0.0)
        new_hit = live[:, None] & (dealt >= DAMAGING) & torch.isnan(first_hit)
        first_hit = torch.where(new_hit, (t + 1) * h, first_hit)
        shots += torch.where(live[:, None], state.fired.float(), 0.0)
        speed_sum += torch.where(live[:, None], state.speed, 0.0)
        rate_sum += torch.where(live[:, None], state.nose_rate.abs(), 0.0)
        live_ticks += live.float()
        delta = signed_angle(old_cos, old_sin, state.heading_cos[:, 0], state.heading_sin[:, 0])
        turned = turned + torch.where(merged & live, delta * first_sense, 0.0)
        # Circle boundaries: the store and total energy at the merge and at each 360 deg.
        circle_now = torch.where(merged, torch.floor(turned / TWO_PI).long() + 1, 0)
        crossed = (circle_now != circles_done) & live
        if crossed.any():
            store = state.energy[:, 0]
            total = store + 0.5 * state.speed[:, 0] ** 2
            for e in torch.nonzero(crossed)[:, 0].tolist():
                circle_store[e].append(float(store[e]))
                circle_total[e].append(float(total[e]))
                circle_time[e].append((t + 1) * h)
                circle_exchange[e].append(float(exchanged[e]))
        circles_done = torch.where(live, circle_now, circles_done)
        # Passes: closest approaches under PASS_RANGE.
        new_distance = torch.sqrt(
            (state.x[:, 1] - state.x[:, 0]) ** 2 + (state.y[:, 1] - state.y[:, 0]) ** 2
        )
        minimum = (
            live
            & pass_armed
            & (last_distance < prev_distance)
            & (last_distance <= new_distance)
            & (last_distance < PASS_RANGE)
        )
        if minimum.any():
            # The heading crossing angle at the closest approach (180 deg: exactly head-on).
            crossing = signed_angle(
                state.heading_cos[:, 0],
                state.heading_sin[:, 0],
                state.heading_cos[:, 1],
                state.heading_sin[:, 1],
            ).abs()
            for e in torch.nonzero(minimum)[:, 0].tolist():
                passes[e].append(
                    [t * h, float(last_distance[e]), math.degrees(float(crossing[e])), None]
                )
            measure_at = torch.where(minimum, t + PASS_LOOK, measure_at)
            pass_armed = pass_armed & ~minimum
            pass_min = torch.where(minimum, last_distance, pass_min)
        pass_min = torch.minimum(pass_min, new_distance)
        pass_armed = pass_armed | (new_distance > pass_min + PASS_REARM)
        measure = (measure_at == t) & live
        if measure.any():
            # Blue's angles off (lean.scripted.situation: aspect + |bearing|; 180 deg neutral,
            # below 180 Blue has gained, 0 Blue on Red's tail nose on), PASS_LOOK ticks after.
            dx = state.x[:, 1] - state.x[:, 0]
            dy = state.y[:, 1] - state.y[:, 0]
            blue_bearing_now = signed_angle(
                state.heading_cos[:, 0], state.heading_sin[:, 0], dx, dy
            )
            red_bearing_now = signed_angle(
                state.heading_cos[:, 1], state.heading_sin[:, 1], -dx, -dy
            )
            angles_off = math.pi - red_bearing_now.abs() + blue_bearing_now.abs()
            for e in torch.nonzero(measure)[:, 0].tolist():
                passes[e][-1][3] = math.degrees(float(angles_off[e]))
        prev_distance, last_distance = last_distance, new_distance

        if keep_traces:
            for name, value in (
                ("x", state.x),
                ("y", state.y),
                ("speed", state.speed),
                ("energy", state.energy),
            ):
                traces[name].append(value[trace_envs].clone())
            traces["power"].append(powers[trace_envs].clone())
            for slot in (0, 1):
                for index, e in enumerate(trace_envs.tolist()):
                    if live[e] and dealt[e, slot] >= DAMAGING:
                        target = 1 - slot
                        hits.append(
                            (
                                index,
                                t + 1,
                                slot,
                                float(dealt[e, slot]),
                                float(state.x[e, target]),
                                float(state.y[e, target]),
                            )
                        )
        finished = out.done & live
        result = torch.where(finished, out.result, result)
        end_tick = torch.where(finished, t + 1, end_tick)
        done = done | out.done
        if t % 300 == 0:
            print(
                f"tick {t}  {time.time() - started:.1f} s  done {int(done.sum())}/{E}", flush=True
            )
        if bool(done.all()) and not keep_traces:
            break

    alive = state.alive
    kills = torch.where(
        ~alive[:, 1] & alive[:, 0], 0, torch.where(~alive[:, 0] & alive[:, 1], 1, 2)
    )
    kills = torch.where(~alive[:, 0] & ~alive[:, 1], 3, kills)  # both dead
    summary = []
    for index, cond in enumerate(conditions):
        rows = slice(index * envs, (index + 1) * envs)
        net_store, net_total, circle_seconds, circles, net_exchange = [], [], [], [], []
        by_circle = {}
        for e in range(index * envs, (index + 1) * envs):
            s, tt, tm, ex = circle_store[e], circle_total[e], circle_time[e], circle_exchange[e]
            circles.append(len(s) - 1 if s else 0)
            for k in range(1, len(s)):
                net_store.append(s[k] - s[k - 1])
                net_exchange.append(ex[k] - ex[k - 1])
                by_circle.setdefault(k, []).append((s[k] - s[k - 1], tt[k] - tt[k - 1], s[k]))
                net_total.append(tt[k] - tt[k - 1])
                circle_seconds.append(tm[k] - tm[k - 1])
        pass_rows = [passes[e] for e in range(index * envs, (index + 1) * envs)]
        most = max((len(p) for p in pass_rows), default=0)
        per_pass = []
        for k in range(min(most, 12)):
            values = [p[k] for p in pass_rows if len(p) > k]
            later = [v[3] for v in values if v[3] is not None]
            per_pass.append(
                {
                    "n": len(values),
                    "time": sum(v[0] for v in values) / len(values),
                    "distance": sum(v[1] for v in values) / len(values),
                    "crossing_deg": sum(v[2] for v in values) / len(values),
                    "blue_angles_off_deg": sum(later) / len(later) if later else None,
                }
            )
        k = kills[rows]
        first = first_hit[rows]
        ticks_live = live_ticks[rows][:, None].clamp_min(1)

        def mean_or_none(values):
            return sum(values) / len(values) if values else None

        def nan_mean(column):
            ok = ~torch.isnan(column)
            return float(column[ok].mean()) if ok.any() else None

        summary.append(
            {
                "name": cond.name,
                "mode": cond.mode,
                "plan": cond.plan,
                "switch_deg": cond.switch,
                "balanced": cond.balanced,
                "envs": envs,
                "blue_kill": float((k == 0).float().mean()),
                "red_kill": float((k == 1).float().mean()),
                "clock": float((k == 2).float().mean()),
                "both_dead": float((k == 3).float().mean()),
                "clock_result_blue": float(((k == 2) & (result[rows] == 0)).float().mean()),
                "clock_result_red": float(((k == 2) & (result[rows] == 1)).float().mean()),
                "damage_by_blue": float(damage[rows, 0].mean()),
                "damage_by_red": float(damage[rows, 1].mean()),
                "damage_by_blue_std": float(damage[rows, 0].std()),
                "damage_by_red_std": float(damage[rows, 1].std()),
                "first_hit_by_blue_s": nan_mean(first[:, 0]),
                "first_hit_by_red_s": nan_mean(first[:, 1]),
                "blue_lands_any": float((~torch.isnan(first[:, 0])).float().mean()),
                "red_lands_any": float((~torch.isnan(first[:, 1])).float().mean()),
                "shots_blue": float(shots[rows, 0].mean()),
                "shots_red": float(shots[rows, 1].mean()),
                "blue_speed": float((speed_sum[rows, 0:1] / ticks_live).mean()),
                "red_speed": float((speed_sum[rows, 1:2] / ticks_live).mean()),
                "blue_rate_deg": math.degrees(float((rate_sum[rows, 0:1] / ticks_live).mean())),
                "red_rate_deg": math.degrees(float((rate_sum[rows, 1:2] / ticks_live).mean())),
                "fight_seconds": float(end_tick[rows].float().mean()) * h,
                "circles_mean": sum(circles) / len(circles),
                "net_store_per_circle": mean_or_none(net_store),
                "net_exchange_per_circle": mean_or_none(net_exchange),
                "net_total_per_circle": mean_or_none(net_total),
                "circle_seconds": mean_or_none(circle_seconds),
                "circle_count": len(net_store),
                "by_circle": [
                    {
                        "circle": k,
                        "n": len(v),
                        "net_store": sum(a for a, _, _ in v) / len(v),
                        "net_total": sum(b for _, b, _ in v) / len(v),
                        "store_after": sum(c for _, _, c in v) / len(v),
                    }
                    for k, v in sorted(by_circle.items())
                ],
                "passes": per_pass,
            }
        )
    out = {"summary": summary}
    if keep_traces:
        out["traces"] = {name: torch.stack(v, 0) for name, v in traces.items()}
        out["hits"] = hits
        out["end_tick"] = end_tick[trace_envs]
    return out


def base_conditions() -> list[Condition]:
    conditions = []
    for mode in (1, 2, 3, 4, 5):
        for plan in ("one-circle", "two-circle"):
            conditions.append(Condition(f"mode {mode}, {plan}", mode, plan, default_switch(mode)))
    return conditions


def solo_net(modes: list[int], switches: list[float], ticks: int, middle_climb: bool) -> list[dict]:
    """Blue alone (Red not alive), turning left from the start on its schedule: full stick
    in the dive, the middle stick in the climb when `middle_climb` (else full stick). Returns
    per (mode, switch) the store at each completed circle and the mean net store and total
    energy per circle over circles 2-4 (the first is the entry from the start speed)."""
    config = core.GameConfig(field_count=0, fog=False, clock_ticks=10 * ticks)
    E = len(modes)
    generator = torch.Generator().manual_seed(0)
    state = core.reset(config, E, torch.tensor([[1, 1]]).expand(E, 2), generator, "cpu")
    c = state.constants
    speed = torch.full((E, 2), c["sustained_speed"])
    state = state.replace(
        speed=speed,
        velocity_x=speed * state.heading_cos,
        velocity_y=speed * state.heading_sin,
        alive=torch.tensor([[True, False]]).expand(E, 2).clone(),
        x=torch.tensor([[0.0, 3000.0]]).expand(E, 2).clone(),
    )
    mode = torch.tensor(modes)
    switch = torch.tensor([math.radians(value) for value in switches])
    merged = torch.ones(E, dtype=torch.bool)
    turned = torch.zeros(E)
    boundaries = [[float(state.energy[e, 0])] for e in range(E)]
    totals = [[float(state.energy[e, 0] + 0.5 * state.speed[e, 0] ** 2)] for e in range(E)]
    exchanged = torch.zeros(E)
    exchange_marks = [[0.0] for _ in range(E)]
    done_circles = torch.zeros(E, dtype=torch.long)
    low_store = torch.full((E,), math.inf)
    high_store = torch.full((E,), -math.inf)
    for _ in range(ticks):
        power = blue_power(mode, switch, torch.remainder(turned, TWO_PI), merged)
        stick = torch.full((E,), FULL_LEFT)
        if middle_climb:
            stick = torch.where(power == CLIMB, FULL_LEFT - 2, stick)
        action = torch.zeros(E, 2, 3, dtype=torch.long)
        action[:, 0, 0], action[:, 0, 1] = stick, power
        old_cos, old_sin = state.heading_cos[:, 0].clone(), state.heading_sin[:, 0].clone()
        # The pending action is what this tick applies (one-tick delay).
        applied = state.pending[:, 0]
        exchanged = exchanged + config.tick_length * exchange_rate(
            state.constants, state.speed[:, 0], applied[:, 0], applied[:, 1]
        )
        state, _ = core.tick(config, state, action, generator)
        turned = turned + signed_angle(
            old_cos, old_sin, state.heading_cos[:, 0], state.heading_sin[:, 0]
        )
        circles = torch.floor(turned / TWO_PI).long()
        measured = (circles >= 1) & (circles < 4)
        low_store = torch.where(measured, torch.minimum(low_store, state.energy[:, 0]), low_store)
        high_store = torch.where(
            measured, torch.maximum(high_store, state.energy[:, 0]), high_store
        )
        for e in torch.nonzero(circles != done_circles)[:, 0].tolist():
            boundaries[e].append(float(state.energy[e, 0]))
            totals[e].append(float(state.energy[e, 0] + 0.5 * state.speed[e, 0] ** 2))
            exchange_marks[e].append(float(exchanged[e]))
        done_circles = circles
    rows = []
    for e in range(E):
        b, tt = boundaries[e], totals[e]
        nets = [b[k] - b[k - 1] for k in range(2, min(len(b), 5))]
        marks = exchange_marks[e]
        exchange_nets = [marks[k] - marks[k - 1] for k in range(2, min(len(marks), 5))]
        net_totals = [tt[k] - tt[k - 1] for k in range(2, min(len(tt), 5))]
        # A store that reached an edge band (10% of full) in circles 2-4: its net is clipped.
        edge = 0.1 * c["max_energy"]
        saturated = (
            "bottom"
            if float(low_store[e]) < edge
            else "top"
            if float(high_store[e]) > c["max_energy"] - edge
            else None
        )
        rows.append(
            {
                "mode": modes[e],
                "switch_deg": switches[e],
                "circles": len(b) - 1,
                "saturated": saturated,
                "net_store": sum(nets) / len(nets) if nets else None,
                "net_exchange": sum(exchange_nets) / len(exchange_nets) if exchange_nets else None,
                "net_total": sum(net_totals) / len(net_totals) if net_totals else None,
            }
        )
    return rows


def solo_balance(ticks: int, middle_climb: bool = True, rounds: int = 3):
    """The switch per mode (2-5) with Blue's solo net exchange per circle (the store's
    change without its edges, `exchange_rate`) nearest zero: a grid
    over the switch, then two finer grids around the sign change (linear interpolation)."""
    best, history = {}, []
    ranges = {mode: (1.0, 359.0 if mode in (2, 3) else 179.0) for mode in (2, 3, 4, 5)}
    for round_index in range(rounds):
        modes, switches = [], []
        for mode, (low, high) in ranges.items():
            for k in range(25):
                modes.append(mode)
                switches.append(low + (high - low) * k / 24)
        rows = solo_net(modes, switches, ticks, middle_climb)
        history.extend({"round": round_index, **row} for row in rows)
        for mode in ranges:
            mine = [r for r in rows if r["mode"] == mode and r["net_exchange"] is not None]

            def signed(row):
                return row["net_exchange"]

            crossing = None
            for a, b in zip(mine, mine[1:]):
                if (signed(a) <= 0.0) != (signed(b) <= 0.0):
                    fraction = signed(a) / (signed(a) - signed(b))
                    crossing = a["switch_deg"] + fraction * (b["switch_deg"] - a["switch_deg"])
                    step = b["switch_deg"] - a["switch_deg"]
                    break
            if crossing is None:
                nearest = min(mine, key=lambda r: abs(signed(r)))
                crossing, step = nearest["switch_deg"], (ranges[mode][1] - ranges[mode][0]) / 24
            best[mode] = crossing
            ranges[mode] = (max(crossing - step, 0.5), crossing + step)
        print({mode: round(value, 2) for mode, value in best.items()}, flush=True)
    check = solo_net(list(best), [best[m] for m in best], ticks, middle_climb)
    return best, history, check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solo-ticks", type=int, default=1500)
    args = parser.parse_args()
    torch.set_num_threads(8)

    config = core.GameConfig(field_count=0, fog=False)
    card_values = core.ship_card(config).uniform
    setup = {
        "start_speed": card_values["sustained_speed"],
        "start_store": card_values["spawn_energy"],
        "max_store": card_values["max_energy"],
        "gun_range": card_values["gun_range"],
        "hit_damage": card_values["hit_damage"],
        "max_shield": card_values["max_shield"],
        "head_on": card_values["head_on"],
        "fire_interval_ticks": card_values["fire_interval"],
        "sustained_rate_deg": math.degrees(card_values["sustained_rate"]),
        "corner_speed": card_values["corner_speed"],
        "cruise_speed": card_values["cruise_speed"],
        "spawn_speed_standard": card.SPAWN_SPEED,
        "scripted_aligned": ScriptedConfig().values["aligned"],
        "scripted_lead_gain": ScriptedConfig().values["lead_gain"],
        "fog": False,
        "fields": 0,
        "start_distance": START_DISTANCE,
        "start_lateral": START_LATERAL,
        "envs": args.envs,
        "ticks": args.ticks,
    }
    print(setup, flush=True)

    base = run(base_conditions(), args.envs, args.ticks, args.seed, keep_traces=True)
    # The equal splits flown solo with the climb at full stick (as in the fight), for the
    # store's balance outside the fight.
    equal_full = solo_net(
        [2, 3, 4, 5], [default_switch(m) for m in (2, 3, 4, 5)], args.solo_ticks, middle_climb=False
    )
    equal_middle = solo_net(
        [2, 3, 4, 5], [default_switch(m) for m in (2, 3, 4, 5)], args.solo_ticks, middle_climb=True
    )
    best, history, check = solo_balance(args.solo_ticks)
    balanced = [
        Condition(f"mode {m} balanced, {p}", m, p, best[m], True)
        for m in (2, 3, 4, 5)
        for p in ("one-circle", "two-circle")
    ]
    final = run(balanced, args.envs, args.ticks, args.seed, keep_traces=True)
    summary = base["summary"] + final["summary"]
    with open(f"{args.out}/results.json", "w") as handle:
        json.dump(
            {
                "setup": setup,
                "summary": summary,
                "solo_equal_full_stick_climb": equal_full,
                "solo_equal_middle_stick_climb": equal_middle,
                "solo_balanced_switch_deg": best,
                "solo_balanced_check": check,
                "solo_balance_history": history,
            },
            handle,
            indent=1,
        )
    torch.save(
        {
            "names": [row["name"] for row in summary],
            "traces": {
                name: torch.cat([base["traces"][name], final["traces"][name]], 1)
                for name in base["traces"]
            },
            "hits": base["hits"]
            + [(i + len(base["summary"]),) + tuple(rest) for i, *rest in final["hits"]],
            "end_tick": torch.cat([base["end_tick"], final["end_tick"]]),
            "tick_length": config.tick_length,
        },
        f"{args.out}/traces.pt",
    )
    for row in summary:
        print(
            f"{row['name']:28s} B {row['blue_kill']:.2f} R {row['red_kill']:.2f} "
            f"clk {row['clock']:.2f} dmgB {row['damage_by_blue']:.1f} "
            f"dmgR {row['damage_by_red']:.1f} netS {row['net_store_per_circle']} "
            f"vB {row['blue_speed']:.0f} vR {row['red_speed']:.0f}"
        )


if __name__ == "__main__":
    main()
