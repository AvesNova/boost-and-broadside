"""Merge turn: how long the scripted pilot flies straight after a pass, and why.

    uv run --no-sync python docs/rewrite/sections/merge-turn/merge_turn.py --out <scratch dir>

A pass event: two aircraft on opposite teams at a local minimum of their distance (the closest
approach, CPA, at tick k) while each nose is within 60 degrees of the other aircraft one second
earlier (tick k - 30). For each aircraft in a pass: the ticks from k until its heading has
changed by 30 and by 90 degrees (capped at 10 s; censored when it is removed or the match ends).
Every 3 ticks in the window the pilot's `Explanation` is logged, with the stick level sampled,
speed over corner speed, whether the partner is visible (fog), the counterpart chosen, and the
boundary steering weight (recomputed with the pilot's own formula, same inputs).

Matches are the first-touch episodes' setup (`game.reset` with team sizes (S, S), seed 0,
scripted pilot on both teams with `ScriptedConfig()`, fog on): 1v1 without fields, 1v1 with
fields, 4v4 with fields. Output: raw per-aircraft summaries and rows as JSON in --out, and a
printed summary of the percentiles and the slow-case shares.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.game.core import enemy_view
from lean.scripted.opponent import engage, explain, map_velocity
from lean.scripted.parameters import value

torch.set_num_threads(4)

TACTICS = ("disengaged", "merge", "offensive", "turning", "defensive")
SETUPS = {
    "1v1-nofields": (1, False),
    "1v1-fields": (1, True),
    "4v4-fields": (4, True),
}
WINDOW_TICKS = 300  # 10 s at 30 Hz
LOG_EVERY = 3
SLOW_SECONDS = 1.5


def pilot_extras(config, state, memory, reasons):
    """The boundary steering weight and whether the pilot has a target, as `explain` computes
    them (same formulas and parameters, recomputed here because `explain` keeps them local)."""
    c = state.constants
    enemies = enemy_view(state)
    fight = engage(
        state,
        None,
        value(config, None, "weakest_weight"),
        memory["target"],
        value(config, None, "target_keep_factor"),
        clearance=value(config, None, "ally_clearance") * c["hit_radius"],
        enemies=enemies,
        ghost_cutoff=value(config, None, "ghost_cutoff"),
    )
    has_target = fight.has_target
    distance_from_centre = torch.sqrt(state.x * state.x + state.y * state.y).clamp_min(1e-6)
    ratio = state.speed / c["corner_speed"]
    turn_radius = c["sustained_radius"] * (ratio * ratio).clamp_min(1.0)
    margin = value(config, None, "boundary_margin") * turn_radius
    depth = (distance_from_centre - (c["arena_radius"] - margin)) / margin
    low = value(config, None, "boundary_ramp_low")
    high = value(config, None, "boundary_ramp_high")
    ramp = ((depth - low) / (high - low)).clamp(0.0, 1.0)
    vx, vy, _ = map_velocity(state)
    outward = state.x * vx + state.y * vy > 0.0
    weight = torch.where(outward & has_target, ramp, torch.zeros_like(ramp))
    weight = torch.where(has_target, weight, torch.ones_like(weight))
    return weight, has_target


def step_pilot(config, state, memory, uniforms):
    """One decision for both teams: `scripted.decide` with the explanation kept (same memory
    update as `decide`, team None)."""
    reasons = explain(config, state, None, None, memory)
    parts = reasons.marginals
    action = scripted.sample(parts, uniforms)
    controlled = parts.controlled
    new = {
        "target": torch.where(controlled, parts.target, -1),
        "side": memory["side"],
        "sense": torch.where(controlled, reasons.sense, 0.0),
        "last_pass": torch.where(controlled, reasons.last_pass, 0.0),
    }
    return action, new, reasons


def snapshot(config, state, memory, action, reasons):
    """Everything the rows need about one tick, as numpy (E, N) arrays (pairs (E, N, N))."""
    weight, has_target = pilot_extras(config, state, memory, reasons)
    present = (state.exists & state.alive).cpu()
    team = state.team.cpu()
    x = state.x.cpu().double()
    y = state.y.cpu().double()
    opposite = team[:, :, None] != team[:, None, :]
    upper = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)
    valid = present[:, :, None] & present[:, None, :] & opposite & upper
    dx = x[:, None, :] - x[:, :, None]
    dy = y[:, None, :] - y[:, :, None]
    if state.ticks_since_seen is None:
        visible = torch.ones_like(present)
    else:
        visible = (state.ticks_since_seen == 0).cpu()
    stick = action[..., 0].cpu()
    return {
        "x": x.numpy(),
        "y": y.numpy(),
        "hc": state.heading_cos.cpu().double().numpy(),
        "hs": state.heading_sin.cpu().double().numpy(),
        "speed": state.speed.cpu().double().numpy(),
        "alive": present.numpy(),
        "team": team.numpy(),
        "dist": torch.sqrt(dx * dx + dy * dy).numpy(),
        "valid": valid.numpy(),
        "weights": torch.stack([reasons.weights[name] for name in TACTICS], -1).cpu().numpy(),
        "sense": reasons.sense.cpu().numpy(),
        "cap": reasons.cap.cpu().numpy(),
        "stick_error": reasons.stick_error.cpu().numpy(),
        "gun_error": reasons.gun_error.cpu().numpy(),
        "speed_target": reasons.speed_target.cpu().numpy(),
        "stick": stick.numpy(),
        "target": reasons.marginals.target.cpu().numpy(),
        "visible": visible.numpy(),
        "boundary": weight.cpu().numpy(),
        "has_target": has_target.cpu().numpy(),
        "corner": float(state.constants["corner_speed"]),
    }


def angle_change(hc0, hs0, hc1, hs1):
    """Unsigned heading change in degrees."""
    cross = hc0 * hs1 - hs0 * hc1
    dot = hc0 * hc1 + hs0 * hs1
    return math.degrees(abs(math.atan2(cross, dot)))


def run_setup(name, size, fields, envs, ticks, seed, fps):
    config = scripted.ScriptedConfig()
    game_config = game.GameConfig() if fields else game.GameConfig(field_count=0)
    generator = torch.Generator().manual_seed(seed)
    state = game.reset(
        game_config, envs, torch.tensor([[size, size]]).expand(envs, 2), generator, "cpu"
    )
    slots = state.exists.shape[1]
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    ring_len = fps + 5
    ring = [None] * ring_len
    over = np.zeros(envs, dtype=bool)
    active = []  # pending events
    events = []  # finished events
    last_pass = {}  # (env, a, b) -> tick of the last CPA logged
    tick_length = game_config.tick_length
    constants_corner = None

    def update(event, t):
        """Log tick t of the event (both aircraft) and close what has ended."""
        entry = ring[t % ring_len]
        rel = t - event["k"]
        for side in event["sides"]:
            if side["done"]:
                continue
            s = side["ship"]
            partner = event["b"] if s == event["a"] else event["a"]
            if rel > 0 and not entry["alive"][event["env"], s]:
                side["end"] = "removed"
                side["done"] = True
                continue
            if over[event["env"]]:
                side["end"] = "match ended"
                side["done"] = True
                continue
            e = event["env"]
            h0 = event["head"][s]
            change = angle_change(
                h0[0], h0[1], entry["hc"][e, s], entry["hs"][e, s]
            )
            if side["t30"] is None and change >= 30.0:
                side["t30"] = rel
            if side["t90"] is None and change >= 90.0:
                side["t90"] = rel
            if rel % LOG_EVERY == 0 or side["t90"] == rel:
                target = int(entry["target"][e, s])
                side["rows"].append(
                    {
                        "rel": rel,
                        "change": change,
                        "weights": {
                            n: float(entry["weights"][e, s, i]) for i, n in enumerate(TACTICS)
                        },
                        "sense": float(entry["sense"][e, s]),
                        "cap": float(entry["cap"][e, s]),
                        "stick_error": float(entry["stick_error"][e, s]),
                        "gun_error": float(entry["gun_error"][e, s]),
                        "speed_target_ratio": float(entry["speed_target"][e, s] / entry["corner"]),
                        "stick": int(entry["stick"][e, s]),
                        "speed_ratio": float(entry["speed"][e, s] / entry["corner"]),
                        "visible_partner": bool(entry["visible"][e, partner]),
                        "target": target,
                        "partner": int(partner),
                        "target_is_partner": target == partner,
                        "target_at_cpa": int(event["target0"][s]),
                        "boundary": float(entry["boundary"][e, s]),
                        "has_target": bool(entry["has_target"][e, s]),
                    }
                )
            if side["t90"] is not None or rel >= WINDOW_TICKS:
                side["done"] = True
                if side["t90"] is None:
                    side["end"] = "capped"
                else:
                    side["end"] = "reached"

    for t in range(ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        action, new_memory, reasons = step_pilot(config, state, memory, uniforms)
        entry = snapshot(config, state, memory, action, reasons)
        entry["over"] = over.copy()
        ring[t % ring_len] = entry
        if constants_corner is None:
            constants_corner = entry["corner"]

        # Candidate closest approaches at k = t - 1: a local minimum of the pair distance.
        if t >= 2:
            k = t - 1
            prev, cur, nxt = ring[(t - 2) % ring_len], ring[k % ring_len], entry
            cand = (
                cur["valid"]
                & ~cur["over"][:, None, None]
                & (cur["dist"] <= prev["dist"])
                & (cur["dist"] < nxt["dist"])
            )
            if k - fps >= 0:
                earlier = ring[(k - fps) % ring_len]
                for e, a, b in np.argwhere(cand):
                    if over[e]:
                        continue
                    if (k - fps) < 0 or not earlier["valid"][e, a, b]:
                        continue
                    ok = True
                    for s, p in ((a, b), (b, a)):
                        ex = earlier["x"][e, p] - earlier["x"][e, s]
                        ey = earlier["y"][e, p] - earlier["y"][e, s]
                        dot = earlier["hc"][e, s] * ex + earlier["hs"][e, s] * ey
                        norm = math.hypot(ex, ey)
                        if norm <= 0.0 or dot / norm < math.cos(math.radians(60.0)):
                            ok = False
                    if not ok:
                        continue
                    key = (int(e), int(a), int(b))
                    if key in last_pass and k - last_pass[key] < 2 * fps:
                        continue
                    last_pass[key] = k
                    event = {
                        "setup": name,
                        "env": int(e),
                        "a": int(a),
                        "b": int(b),
                        "k": k,
                        "cpa_distance": float(cur["dist"][e, a, b]),
                        "speed": {
                            "a": float(cur["speed"][e, a]) / constants_corner,
                            "b": float(cur["speed"][e, b]) / constants_corner,
                        },
                        "head": {
                            int(a): (float(cur["hc"][e, a]), float(cur["hs"][e, a])),
                            int(b): (float(cur["hc"][e, b]), float(cur["hs"][e, b])),
                        },
                        "target0": {
                            int(a): int(cur["target"][e, a]),
                            int(b): int(cur["target"][e, b]),
                        },
                        "sides": [
                            {"ship": int(a), "t30": None, "t90": None, "done": False,
                             "end": None, "rows": []},
                            {"ship": int(b), "t30": None, "t90": None, "done": False,
                             "end": None, "rows": []},
                        ],
                    }
                    # the event's tick k is logged from its own snapshot; then tick t
                    update(event, k)
                    active.append(event)

        for event in active:
            update(event, t)
        still = []
        for event in active:
            if all(side["done"] for side in event["sides"]):
                events.append(event)
            else:
                still.append(event)
        active = still

        if t + 1 >= ticks:
            break
        new_state, out = game.tick(game_config, state, action, generator)
        over = over | out.done.cpu().numpy()
        state = new_state
        memory = new_memory
        if over.all():
            break
    # Events still open at the end.
    for event in active:
        for side in event["sides"]:
            if not side["done"]:
                side["end"] = "match ended"
        events.append(event)
    return events


def jsonable(event):
    out = dict(event)
    out["head"] = {str(k): v for k, v in event["head"].items()}
    out["target0"] = {str(k): v for k, v in event["target0"].items()}
    out["speed"] = dict(event["speed"])
    out["sides"] = [
        {key: value for key, value in side.items()} for side in event["sides"]
    ]
    return out


def percentile_row(values):
    if not values:
        return "n/a"
    arr = np.array(values, dtype=float)
    return f"{np.median(arr):.2f} / {np.percentile(arr, 75):.2f} / {np.percentile(arr, 90):.2f}"


def aircraft_records(events, tick_seconds):
    """One record per aircraft in a pass: times in seconds (None when not reached), the censor
    reason, and its rows."""
    records = []
    for event in events:
        for side in event["sides"]:
            rows = side["rows"]
            records.append(
                {
                    "setup": event["setup"],
                    "t30": None if side["t30"] is None else side["t30"] * tick_seconds,
                    "t90": None if side["t90"] is None else side["t90"] * tick_seconds,
                    "end": side["end"],
                    "cpa_distance": event["cpa_distance"],
                    "speed_ratio": event["speed"]["a" if side["ship"] == event["a"] else "b"],
                    "rows": rows,
                    "target_at_cpa": event["target0"][str(side["ship"])],
                    "partner": event["b"] if side["ship"] == event["a"] else event["a"],
                }
            )
    return records


def row_features(record, tick_seconds):
    """Booleans for each row of the first 1.5 s of the window (rel <= 1.5 s)."""
    limit = SLOW_SECONDS / tick_seconds
    out = []
    for row in record["rows"]:
        if row["rel"] > limit:
            continue
        weights = row["weights"]
        out.append(
            {
                "tactic": max(weights, key=weights.get),
                "ahead_tactic": weights["merge"] + weights["offensive"] > 0.5,
                "disengaged_tactic": weights["disengaged"] > 0.5,
                "aim_within_2_deg": abs(row["stick_error"]) < math.radians(2.0),
                "no_sense": row["sense"] == 0.0,
                "cap_folds_full": row["cap"] > 0.5,
                "straight_stick": row["stick"] == 0,
                "partner_hidden": not row["visible_partner"],
                "target_not_partner": not row["target_is_partner"],
                "target_changed": row["target"] != record["target_at_cpa"],
                "boundary_steer": row["boundary"] > 0.0,
                "speed_above_corner_1_5": row["speed_ratio"] > 1.5,
                "pointing_below_turn_gate": abs(row["stick_error"]) < 0.002868,
                "aim_shifted_from_lead": abs(row["stick_error"] - row["gun_error"]) > 1e-3,
            }
        )
    return out


def summarise(out_dir):
    tick_seconds = 1.0 / 30.0
    report = {}
    for name in SETUPS:
        path = out_dir / f"{name}.json"
        if not path.exists():
            continue
        events = json.loads(path.read_text(encoding="utf-8"))
        records = aircraft_records(events, tick_seconds)
        censored = Counter(r["end"] for r in records if r["t30"] is None and r["end"] != "capped")
        # 10 s cap: a time not reached counts as 10 s for the percentiles (a lower bound).
        t30 = [r["t30"] if r["t30"] is not None else 10.0 for r in records
               if r["t30"] is not None or r["end"] == "capped"]
        t90 = [r["t90"] if r["t90"] is not None else 10.0 for r in records
               if r["t90"] is not None or r["end"] == "capped"]
        slow = [r for r in records if (r["t30"] is None and r["end"] == "capped")
                or (r["t30"] is not None and r["t30"] > SLOW_SECONDS)]
        fast = [r for r in records if r["t30"] is not None and r["t30"] <= SLOW_SECONDS]
        features = {}
        for group_name, group in (("slow", slow), ("fast", fast)):
            per_aircraft = [row_features(r, tick_seconds) for r in group]
            per_aircraft = [p for p in per_aircraft if p]
            tallies = Counter()
            for rows in per_aircraft:
                for key in rows[0]:
                    if key == "tactic":
                        continue
                    share = np.mean([row[key] for row in rows])
                    if share > 0.5:
                        tallies[key] += 1
                tactic = Counter(row["tactic"] for row in rows).most_common(1)[0][0]
                tallies[f"dominant tactic {tactic}"] += 1
            at_cpa = [r["rows"][0] for r in group if r["rows"] and r["rows"][0]["rel"] == 0]
            tallies_cpa = {
                "partner_hidden_at_cpa": sum(not row["visible_partner"] for row in at_cpa),
                "target_is_partner_at_cpa": sum(row["target_is_partner"] for row in at_cpa),
                "disengaged_at_cpa": sum(row["weights"]["disengaged"] > 0.5 for row in at_cpa),
            }
            signature = Counter()
            for rows in per_aircraft:
                signature[
                    (
                        "aim<2deg" if np.mean([r["aim_within_2_deg"] for r in rows]) > 0.5 else "aim off",
                        "partner hidden" if np.mean([r["partner_hidden"] for r in rows]) > 0.5 else "partner seen",
                        "no sense" if np.mean([r["no_sense"] for r in rows]) > 0.5 else "sense held",
                    )
                ] += 1
            features[group_name] = {
                "signature_aim_partner_sense": {" | ".join(k): v for k, v in signature.most_common()},
                "aircraft": len(per_aircraft),
                "majority_counts": dict(tallies),
                "at_cpa_counts_of": len(at_cpa),
                "at_cpa": tallies_cpa,
            }
        report[name] = {
            "pass_events": len(events),
            "aircraft": len(records),
            "censored_before_30": dict(censored),
            "t30_median_p75_p90_s": percentile_row(t30),
            "t90_median_p75_p90_s": percentile_row(t90),
            "t30_n": len(t30),
            "t90_n": len(t90),
            "slow_aircraft": len(slow),
            "fast_aircraft": len(fast),
            "features": features,
            "closest_distance_median_m": float(np.median([e["cpa_distance"] for e in events])),
            "collisions_cpa_under_10m": int(sum(e["cpa_distance"] < 10 for e in events)),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarise", action="store_true", help="only summarise --out")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default=None, help="one setup name")
    args = parser.parse_args()
    if args.summarise:
        print(json.dumps(summarise(args.out), indent=1))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / game.GameConfig().tick_length)
    for name, (size, fields) in SETUPS.items():
        if args.only and name != args.only:
            continue
        envs = args.envs if size == 1 else max(args.envs // 4, 16)
        events = run_setup(name, size, fields, envs, args.ticks, args.seed, fps)
        (args.out / f"{name}.json").write_text(
            json.dumps([jsonable(e) for e in events]), encoding="utf-8"
        )
        print(f"{name}: {len(events)} pass events", flush=True)


if __name__ == "__main__":
    main()
