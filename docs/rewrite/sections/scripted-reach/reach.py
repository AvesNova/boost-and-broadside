"""Reach and effect of the scripted agent's mechanisms (docs/rewrite/scripted-reach.md).

    reach.py DATA collect [--games 320]   # scripted-vs-scripted games, sparse ship-states
    reach.py DATA collect --crossing 2    # the same games with the overshoot's crossing gate
                                          # widened 2x (states_crossing2.pt; for its counts)
    reach.py DATA check                   # probe.explain with nothing off == lean's marginals
    reach.py DATA analyse                 # reach, JS and action changes -> reach.json

DATA: ~/avesnova/boost-and-broadside/data/scripted-screening/reach-2026-11-25. Run with the
repository root on PYTHONPATH.

Collect: the core game at its defaults (GameConfig(): arena 4,200 m, mixed spawn, fields on,
standard spawn), 4v4. Each team is flown by one agent: the yardstick with probability 1/2,
else one of four personas drawn at spread 1. The first snapshot comes 20 to 80 ticks in, then
one every 40 to 80 ticks; every living ship of a snapshot is one ship-state, stored with the
memory it decided from. Over every tick of every game it also counts, from the memory alone
(no instrumentation of lean/), the ship-ticks flown under a turn sense that an overshoot
reversed, and under a sense committed by a plan the switch on gain flipped.

Analyse: on every stored ship-state, the decision of the agent that flew it (its rows), with
the mechanism on and off (probe.py OFF, or a row override), as the joint distribution over
the 42 actions (stick x power x fire).
"""

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch

from lean import game, scripted
from lean.contract import State
from lean.learn.rollout import teacher_joint
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import PARAMETERS, SAMPLED, box_point, rows_of, vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

HERE = Path(__file__).resolve().parent
SLOTS = 8
PERSONAS = 4
SEED = 20261125
FIRST_LOW, FIRST_HIGH = 20, 80
GAP_LOW, GAP_HIGH = 40, 80
EPS = 0.05  # a blend weight this large counts as reached
CHUNK = 4096


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agent_table() -> torch.Tensor:
    """Row 0 the yardstick, rows 1..PERSONAS personas at spread 1 (seeded)."""
    config = ScriptedConfig()
    generator = torch.Generator().manual_seed(SEED)
    unit = torch.rand((PERSONAS, len(SAMPLED)), generator=generator, dtype=torch.float64)
    drawn = box_point(config, unit, 1.0)
    vectors = [vector(config)]
    for index in range(PERSONAS):
        values = dict(config.values)
        values.update({name: float(drawn[name][index]) for name in SAMPLED})
        vectors.append(vector(ScriptedConfig(values)))
    return torch.tensor(vectors, dtype=torch.float32)


def map_state(state: State, function) -> State:
    values = {}
    for field in dataclasses.fields(state):
        item = getattr(state, field.name)
        if item is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(item, table=function(item.table, field.name))
        else:
            values[field.name] = function(item, field.name)
    return State(**values)


def select_rows(state: State, index: torch.Tensor) -> State:
    """Environments `index`; bullets cut to one empty slot (the scripted agent reads none)."""

    def pick(item, name):
        item = item.index_select(0, index)
        if name in ("bullets", "bullet_counts", "bullet_real"):
            item = item[..., :1].clone().zero_()
        return item

    return map_state(state, pick)


def cat_states(states: list[State]) -> State:
    first = states[0]
    values = {}
    for field in dataclasses.fields(first):
        items = [getattr(state, field.name) for state in states]
        if items[0] is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(
                items[0], table=torch.cat([item.table for item in items])
            )
        else:
            values[field.name] = torch.cat(items)
    return State(**values)


# --------------------------------------------------------------------------- collect


@torch.no_grad()
def collect(data: Path, games: int, crossing: float = 1.0) -> None:
    path = data / ("states.pt" if crossing == 1.0 else f"states_crossing{crossing:g}.pt")
    if path.exists():
        log(f"{path} exists; kept")
        return
    here = device()
    table = agent_table()
    table[:, list(PARAMETERS).index("overshoot_crossing")] *= crossing
    table = table.to(here)
    config = game.GameConfig()
    generator = torch.Generator(device=here).manual_seed(SEED)
    torch.manual_seed(SEED)

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device=here)

    sizes = torch.full((games, 2), 4, device=here)
    state = game.deal(config, games, SLOTS, sizes, generator, here)
    memory = scripted.fresh_memory(games, SLOTS, generator, here)

    def team_agent():
        persona = 1 + (uniform(games) * PERSONAS).long().clamp(max=PERSONAS - 1)
        return torch.where(uniform(games) < 0.5, 0, persona)

    agent = torch.where(state.team == 0, team_agent()[:, None], team_agent()[:, None])
    rows = rows_of(table, agent)
    next_snap = (FIRST_LOW + uniform(games) * (FIRST_HIGH - FIRST_LOW)).long()
    over = torch.zeros(games, dtype=torch.bool, device=here)
    reversed_flag = torch.zeros((games, SLOTS), dtype=torch.bool, device=here)
    switched_flag = torch.zeros((games, SLOTS), dtype=torch.bool, device=here)
    counts = {
        name: 0
        for name in (
            "ship_ticks",
            "overshoot_events",
            "under_overshoot",
            "switch_commits",
            "under_switch",
            "commits",
        )
    }
    snapshots, memories, agents, ticks, flags = [], [], [], [], []
    tick, started = 0, time.perf_counter()
    while not bool(over.all()) and tick < config.clock_ticks + 2:
        take = (~over) & (state.tick >= next_snap)
        if bool(take.any()):
            index = take.nonzero().squeeze(-1)
            snapshots.append(map_state(select_rows(state, index), lambda x, _: x.cpu()))
            memories.append(
                {name: value.index_select(0, index).cpu() for name, value in memory.items()}
            )
            agents.append(agent.index_select(0, index).cpu())
            ticks.append(state.tick.index_select(0, index).cpu())
            flags.append(
                torch.stack([reversed_flag, switched_flag], -1).index_select(0, index).cpu()
            )
            gap = (GAP_LOW + uniform(games) * (GAP_HIGH - GAP_LOW)).long()
            next_snap = torch.where(take, state.tick + gap, next_snap)
        live = (state.alive & state.exists) & ~over[:, None]
        uniforms = scripted.draw_uniforms(games, SLOTS, generator, here)
        action, new, _ = scripted.decide(ScriptedConfig(), state, None, rows, uniforms, memory)
        # Memory-only footprints (no lean/ instrumentation): a sense flipped to its opposite
        # with the same target is an overshoot reversal (release and commit cannot happen in
        # one decision; a target change is excluded); a commit with the same target whose
        # last pass flips sign is a plan the switch on gain flipped.
        same = live & (new["target"] == memory["target"])
        flip = same & (memory["sense"] != 0.0) & (new["sense"] == -memory["sense"])
        changed = new["sense"] != memory["sense"]
        commit = live & changed & (memory["sense"] == 0.0) & (new["sense"] != 0.0)
        switch = (
            commit
            & same
            & (memory["last_pass"] != 0.0)
            & (torch.sign(new["last_pass"]) == -torch.sign(memory["last_pass"]))
        )
        reversed_flag = torch.where(flip, True, torch.where(changed, False, reversed_flag))
        switched_flag = torch.where(switch, True, torch.where(changed, False, switched_flag))
        holding = live & (new["sense"] != 0.0)
        counts["ship_ticks"] += int(live.sum())
        counts["overshoot_events"] += int(flip.sum())
        counts["under_overshoot"] += int((reversed_flag & holding).sum())
        counts["switch_commits"] += int(switch.sum())
        counts["commits"] += int(commit.sum())
        counts["under_switch"] += int((switched_flag & holding).sum())
        memory = new
        noise = game.draw_noise(config, games, SLOTS, generator, here)
        state, out = game.step(config, state, action, noise)
        over = over | out.done
        tick += 1
        if tick % 500 == 0:
            ships = sum(int((s.alive & s.exists).sum()) for s in snapshots)
            log(
                f"tick {tick}: {int(over.sum())}/{games} games over, {ships} ship-states, "
                f"{time.perf_counter() - started:.0f} s"
            )
    snapshot = cat_states(snapshots)
    ships = int((snapshot.alive & snapshot.exists).sum())
    torch.save(
        {
            "state": snapshot,
            "memory": {name: torch.cat([item[name] for item in memories]) for name in memory},
            "agent": torch.cat(agents),
            "tick": torch.cat(ticks),
            "flags": torch.cat(flags),  # (S, N, 2): sense from an overshoot reversal, from a switch
            "table": table.cpu(),
            "rows": list(PARAMETERS),
            "games": games,
            "ticks_played": tick,
            "counts": counts,
            "device": str(here),
        },
        path,
    )
    log(
        f"wrote {path}: {ships} ship-states in {snapshot.x.shape[0]} snapshots; counts {counts}; "
        f"{time.perf_counter() - started:.0f} s"
    )


# --------------------------------------------------------------------------- analyse


def load(data: Path, name: str = "states.pt"):
    record = torch.load(data / name, weights_only=False)
    return record


def chunks(record, here):
    state, memory, agent = record["state"], record["memory"], record["agent"]
    table = record["table"].to(here)
    envs = state.x.shape[0]
    for start in range(0, envs, CHUNK):
        index = torch.arange(start, min(start + CHUNK, envs))
        part = map_state(state, lambda x, _: x.index_select(0, index).to(here))
        mem = {name: value.index_select(0, index).to(here) for name, value in memory.items()}
        rows = rows_of(table, agent.index_select(0, index).to(here))
        mem["flags"] = record["flags"].index_select(0, index).to(here)
        yield part, mem, rows, agent.index_select(0, index).to(here)


def joint(parts) -> torch.Tensor:
    return teacher_joint(torch.cat([parts.stick, parts.power, parts.fire[..., None]], -1))


def js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)

    def kl(a, b):
        return torch.where(
            a > 0, a * (torch.log(a.clamp_min(1e-30)) - torch.log(b.clamp_min(1e-30))), 0.0
        ).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def check(data: Path) -> None:
    here = device()
    record = load(data)
    worst = 0.0
    for state, memory, rows, _ in chunks(record, here):
        memory.pop("flags")
        lean_parts = scripted.decide(
            ScriptedConfig(), state, None, rows, torch.zeros(*state.x.shape, 3, device=here), memory
        )[2]
        mine, _ = probe.explain(ScriptedConfig(), state, None, rows, memory)
        mask = lean_parts.controlled
        for name in ("stick", "power", "fire"):
            a, b = getattr(lean_parts, name)[mask], getattr(mine, name)[mask]
            worst = max(worst, float((a - b).abs().max()))
    log(f"check: largest difference between probe.explain and lean's marginals {worst:.2e}")
    if worst > 1e-5:
        raise SystemExit("probe.explain does not reproduce lean's marginals")


def variants():
    """name -> (off set, off row overrides {row: function of the on value}, on row overrides
    (the widened overshoot), off memory transform or None). The "held sense" variants act
    through memory: on the states whose held turn sense came from an overshoot reversal (or
    from a plan the switch flipped), off is the same state with that sense the other way, the
    sense the ship would hold without the mechanism."""
    widen = lambda k: {"overshoot_crossing": lambda v: v * k}  # noqa: E731
    return {
        "overshoot": ({"overshoot"}, {}, {}),
        "overshoot (held sense)": (set(), {}, {}, flip_sense(0)),
        "overshoot_x2": ({"overshoot"}, {}, widen(2.0)),
        "overshoot_x3": ({"overshoot"}, {}, widen(3.0)),
        "spend (all three gates)": ({"spend"}, {}, {}),
        "spend_need": ({"spend_need"}, {}, {}),
        "spend_range": ({"spend_range"}, {}, {}),
        "spend_share": ({"spend_share"}, {}, {}),
        "weakest_weight": (set(), {"weakest_weight": lambda v: v * 0.0}, {}),
        "switch_gain": ({"switch"}, {}, {}),
        "switch_gain (held sense)": (set(), {}, {}, flip_sense(1)),
        "scissors_range": ({"scissors"}, {}, {}),
        "cash_angle": ({"cash"}, {}, {}),
        "relax_angle": ({"relax"}, {}, {}),
        "defend_range": ({"defend"}, {}, {}),
        "boundary_margin (the steer)": ({"boundary"}, {}, {}),
        "speed_floor": ({"speed_floor"}, {}, {}),
        "turning_floor": (set(), {"turning_floor": lambda v: v * 0.0 + 1e-3}, {}),
    }


def flip_sense(column: int):
    def transform(memory):
        flagged = memory["flags"][..., column] & (memory["sense"] != 0.0)
        return {**memory, "sense": torch.where(flagged, -memory["sense"], memory["sense"])}

    return transform


def reach_rule(name: str, on: dict, off: dict, memory: dict) -> torch.Tensor:
    w = on["weights"]
    g = on["gate"]
    spend_weight = w["offensive"] + w["turning"]
    base = name.split(" ")[0]
    if name.endswith("(held sense)"):
        column = 0 if base == "overshoot" else 1
        return memory["flags"][..., column] & (memory["sense"] != 0.0)
    if base.startswith("overshoot"):
        return on["overshoot"]
    if base == "spend":
        return spend_weight * on["near"] > EPS
    if base == "spend_need":
        return spend_weight * g["shot_near"] * g["spend_range"] * (1 - g["spend_need"]) > EPS
    if base == "spend_range":
        return spend_weight * g["shot_near"] * g["spend_need"] * (1 - g["spend_range"]) > EPS
    if base == "spend_share":
        return spend_weight * g["spend_need"] * g["spend_range"] * (1 - g["shot_near"]) > EPS
    if base == "weakest_weight":
        return on["has_target"] & (on["target"] != off["target"])
    if base == "switch_gain":
        return on["plan_switched"] & on["commit"]
    if base == "scissors_range":
        return w["turning"] * (1 - on["two_circle"]) * g["scissors"] > EPS
    if base == "cash_angle":
        return w["turning"] * on["two_circle"] * g["cash"] > EPS
    if base == "relax_angle":
        return w["defensive"] * g["defend"] * (1 - g["relax"]) > EPS
    if base == "defend_range":
        return w["defensive"] * g["relax"] * (1 - g["defend"]) > EPS
    if base == "boundary_margin":
        return on["centre_weight"] > EPS
    if base == "speed_floor":
        return on["log_target_raw"] < on["log_floor"] - 0.01
    if base == "turning_floor":
        return on["has_target"] & (
            ((on["own_sense"] - off["own_sense"]).abs() > EPS)
            | ((on["target_sense"] - off["target_sense"]).abs() > EPS)
        )
    raise KeyError(name)


def overridden(rows: dict, overrides: dict) -> dict:
    return {
        name: (overrides[name](value) if name in overrides else value)
        for name, value in rows.items()
    }


@torch.no_grad()
def analyse(data: Path, states_name: str = "states.pt") -> None:
    here = device()
    record = load(data, states_name)
    names = list(variants())
    sums = {
        name: {
            "states": 0,
            "reached": 0,
            "reached_yardstick": 0,
            "js_reached": 0.0,
            "change_reached": 0,
            "js_all": 0.0,
            "change_all": 0,
            "head_js_reached": [0.0, 0.0, 0.0],
        }
        for name in names
    }
    yardstick_states = 0
    conditions = {}
    started = time.perf_counter()
    for state, memory, rows, agent in chunks(record, here):
        cache = {}
        flags = memory.pop("flags")
        for name, item in variants().items():
            off_set, off_rows, on_rows = item[:3]
            transform = item[3] if len(item) > 3 else None
            on_key = name if on_rows else ""
            if on_key not in cache:
                cache[on_key] = probe.explain(
                    ScriptedConfig(), state, None, overridden(rows, on_rows), memory
                )
            on_parts, on_probe = cache[on_key]
            off_memory = transform({**memory, "flags": flags}) if transform else memory
            off_memory = {key: value for key, value in off_memory.items() if key != "flags"}
            off_parts, off_probe = probe.explain(
                ScriptedConfig(),
                state,
                None,
                overridden(overridden(rows, on_rows), off_rows),
                off_memory,
                frozenset(off_set),
            )
            mask = on_parts.controlled
            reached = reach_rule(name, on_probe, off_probe, {**memory, "flags": flags}) & mask
            p, q = joint(on_parts), joint(off_parts)
            divergence = js(p, q)
            change = p.argmax(-1) != q.argmax(-1)
            heads = [
                js(on_parts.stick, off_parts.stick),
                js(on_parts.power, off_parts.power),
                js(
                    torch.stack([1 - on_parts.fire, on_parts.fire], -1),
                    torch.stack([1 - off_parts.fire, off_parts.fire], -1),
                ),
            ]
            s = sums[name]
            s["states"] += int(mask.sum())
            s["reached"] += int(reached.sum())
            s["reached_yardstick"] += int((reached & (agent == 0)).sum())
            s["js_reached"] += float(divergence[reached].sum())
            s["change_reached"] += int(change[reached].sum())
            s["js_all"] += float(divergence[mask].sum())
            s["change_all"] += int(change[mask].sum())
            for k in range(3):
                s["head_js_reached"][k] += float(heads[k][reached].sum())
            if name.startswith("overshoot") and "held" not in name:
                c = conditions.setdefault(name, {})
                running = on_probe["overshoot_conditions"]["aft_held"] & mask
                c["aft_held"] = c.get("aft_held", 0) + int(running.sum())
                for cond in ("from_my_side", "rate", "range", "crossing", "slower"):
                    c[cond + "_alone"] = c.get(cond + "_alone", 0) + int(
                        (on_probe["overshoot_conditions"][cond] & mask).sum()
                    )
                    running = running & on_probe["overshoot_conditions"][cond]
                    c["then_" + cond] = c.get("then_" + cond, 0) + int(running.sum())
        yardstick_states += int(((agent == 0) & cache[""][0].controlled).sum())
    results = {}
    for name, s in sums.items():
        n, r = s["states"], s["reached"]
        results[name] = {
            "states": n,
            "reached": r,
            "reach": r / n,
            "reached_yardstick_flown": s["reached_yardstick"],
            "js_reached": s["js_reached"] / r if r else None,
            "change_reached": s["change_reached"] / r if r else None,
            "js_all": s["js_all"] / n,
            "change_all": s["change_all"] / n,
            "head_js_reached": [v / r for v in s["head_js_reached"]] if r else None,
        }
    output = {
        "results": results,
        "overshoot_conditions": conditions,
        "yardstick_flown_states": yardstick_states,
        "collect_counts": record["counts"],
        "games": record["games"],
        "snapshots": int(record["state"].x.shape[0]),
        "eps": EPS,
        "ln2": math.log(2.0),
    }
    stem = (
        "reach"
        if states_name == "states.pt"
        else "reach_" + Path(states_name).stem.split("_", 1)[1]
    )
    (HERE / f"{stem}.json").write_text(json.dumps(output, indent=1))
    log(f"analyse: {time.perf_counter() - started:.0f} s")
    print(
        f"{'mechanism':30s} {'reach%':>8s} {'JS|r':>8s} {'chg|r':>7s} {'JS all':>9s} "
        f"{'chg all':>8s}  n"
    )
    for name, row in results.items():
        jr = f"{row['js_reached']:.4f}" if row["js_reached"] is not None else "-"
        cr = f"{row['change_reached']:.3f}" if row["change_reached"] is not None else "-"
        print(
            f"{name:30s} {100 * row['reach']:8.3f} {jr:>8s} {cr:>7s} {row['js_all']:9.5f} "
            f"{row['change_all']:8.4f}  {row['reached']}"
        )
    print(json.dumps(conditions, indent=1))
    print(json.dumps(record["counts"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("step", choices=("collect", "check", "analyse"))
    parser.add_argument("--games", type=int, default=320)
    parser.add_argument("--crossing", type=float, default=1.0)
    parser.add_argument("--states", default="states.pt")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    if args.step == "collect":
        collect(args.data, args.games, args.crossing)
    elif args.step == "check":
        check(args.data)
    else:
        analyse(args.data, args.states)


if __name__ == "__main__":
    main()
