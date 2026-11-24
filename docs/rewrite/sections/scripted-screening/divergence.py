"""Behavioural divergence of scripted configurations (docs/rewrite/scripted-divergence.md).

A configuration's behaviour is its joint action distribution (stick x power x fire, the
product of lean.scripted.decide's marginals, as the BC teacher's lean.learn.rollout
teacher_joint) on a fixed set of stored ship-states, each with the memory of the ship that
flew it. Two configurations differ by the Jensen-Shannon divergence of their distributions
(natural log), per ship-state, averaged over the ship-states.

    divergence.py DATA states                  # the state set (games, sparse snapshots)
    divergence.py DATA check                   # bullets are not read; timing of one call
    divergence.py DATA screen                  # JS from the yardstick of the 4,000 + references
    divergence.py DATA saltelli --base 512     # direct Sobol indices of JS (Saltelli sample)
    divergence.py DATA elementary --base 1024  # the same, plus one-row redraws JS(A, AB_i)
    divergence.py DATA sweep                   # one row at a time through the yardstick
    divergence.py DATA pairwise --count 300    # pairwise JS, MDS, families
    divergence_analyse.py DATA                 # divergence.json: indices, families, tables

DATA is the data folder (~/avesnova/boost-and-broadside/data/scripted-screening): the state
set is states.pt there; every step's result is a .npz there, written once per batch, and a
step skips what is already written. Run with the repository root on PYTHONPATH.
"""

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.contract import State
from lean.learn.rollout import teacher_joint, teacher_marginals
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import PARAMETERS, SAMPLED, UPWARD, box_point, rows_of, vector

HERE = Path(__file__).resolve().parent
RADII = (3000.0, 4200.0, 6000.0)
SPAWNS = ("mixed", "sides")
SIZES = (3, 4, 5)
SLOTS = 10
FIXED_CONFIG = ScriptedConfig()
BFM = ScriptedConfig().replace(yo_yo_time=0.5, lead_turn_scale=1.0, relax_angle=1.5 * math.pi, defend_range=1.0)
DEFAULT_UNIT = np.array([0.0 if PARAMETERS[name].law == UPWARD else 0.5 for name in SAMPLED])
LN2 = math.log(2.0)
# The state set: per (radius, spawn) group, GAMES games of team sizes 3v3, 4v4, 5v5 in turn,
# a snapshot of each game every GAP_LOW .. GAP_HIGH ticks (uniform), the first at FIRST_LOW ..
# FIRST_HIGH; every living ship of a snapshot is a ship-state.
GAMES = 440
FIRST_LOW, FIRST_HIGH = 30, 230
GAP_LOW, GAP_HIGH = 150, 250
MIX_SOBOL = 30  # the Sobol configurations that fly the state-set games (the stream's first)


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def game_configs():
    return [dataclasses.replace(game.GameConfig(), arena_radius=radius, spawn=spawn)
            for radius in RADII for spawn in SPAWNS]


def unit_vector(unit_row, spread: float = 1.0, centre: ScriptedConfig | None = None) -> tuple[float, ...]:
    """The full vector at a point of the unit cube (screen.py agent_at)."""
    centre = centre or ScriptedConfig()
    drawn = box_point(centre, torch.as_tensor(np.asarray(unit_row), dtype=torch.float64), spread)
    values = dict(centre.values)
    values.update({row: float(drawn[row]) for row in SAMPLED})
    return vector(ScriptedConfig(values))


def cma_vector() -> tuple[float, ...]:
    record = json.loads((HERE / "cma_candidate.json").read_text())
    return vector(ScriptedConfig(record["values"]))


def skills() -> dict:
    return json.loads((HERE / "skills.json").read_text())


def screen_units() -> np.ndarray:
    return np.array([entry["unit"] for entry in skills()["screen"]])


# --------------------------------------------------------------------------- the state set


def select_rows(state: State, index: torch.Tensor) -> State:
    """The environments `index` of a state (every per-environment tensor; bullets dropped
    to one empty slot: the scripted agent does not read them, `check`)."""
    values = {}
    for field in dataclasses.fields(state):
        item = getattr(state, field.name)
        if item is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(item, table=item.table.index_select(0, index))
        elif field.name in ("bullets", "bullet_counts", "bullet_real"):
            values[field.name] = item.index_select(0, index)[..., :1].clone().zero_()
        else:
            values[field.name] = item.index_select(0, index)
    return State(**values)


def cat_states(states: list[State]) -> State:
    first = states[0]
    values = {}
    for field in dataclasses.fields(first):
        items = [getattr(state, field.name) for state in states]
        if items[0] is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(items[0], table=torch.cat([item.table for item in items]))
        else:
            values[field.name] = torch.cat(items)
    return State(**values)


def state_to(state: State, target) -> State:
    values = {}
    for field in dataclasses.fields(state):
        item = getattr(state, field.name)
        if item is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(item, table=item.table.to(target))
        else:
            values[field.name] = item.to(target)
    return State(**values)


@torch.no_grad()
def collect_group(game_config, table: torch.Tensor, seed: int, games: int) -> dict:
    """`games` games of the group, played whole by scripted agents of `table` (seat A and B
    each the yardstick (row 0) with probability 1/4, the CMA candidate (row 1) 1/4, else a
    uniform Sobol row), with sparse snapshots."""
    here = table.device
    generator = torch.Generator(device=here).manual_seed(seed)
    torch.manual_seed(seed)

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device=here)

    sizes_each = torch.tensor(SIZES, device=here)[torch.arange(games, device=here) % len(SIZES)]
    sizes = torch.stack([sizes_each, sizes_each], -1)
    state = game.deal(game_config, games, SLOTS, sizes, generator, here)
    memory = scripted.fresh_memory(games, SLOTS, generator, here)

    def seat_agent():
        u = uniform(games)
        sobol = 2 + (uniform(games) * (table.shape[0] - 2)).long().clamp(max=table.shape[0] - 3)
        return torch.where(u < 0.25, 0, torch.where(u < 0.5, 1, sobol))

    agent_a, agent_b = seat_agent(), seat_agent()
    side = (uniform(games) < 0.5).long()
    seat_a = state.team == side[:, None]
    agent = torch.where(seat_a, agent_a[:, None], agent_b[:, None])
    rows = rows_of(table, agent)
    next_snap = (FIRST_LOW + uniform(games) * (FIRST_HIGH - FIRST_LOW)).long()
    over = torch.zeros(games, dtype=torch.bool, device=here)
    snapshots, memories, agents, games_of, ticks = [], [], [], [], []
    tick = 0
    while not bool(over.all()) and tick < game_config.clock_ticks + 2:
        take = (~over) & (state.tick >= next_snap)
        if bool(take.any()):
            index = take.nonzero().squeeze(-1)
            snapshots.append(select_rows(state, index))
            memories.append({name: value.index_select(0, index) for name, value in memory.items()})
            agents.append(agent.index_select(0, index))
            games_of.append(index)
            ticks.append(state.tick.index_select(0, index))
            gap = (GAP_LOW + uniform(games) * (GAP_HIGH - GAP_LOW)).long()
            next_snap = torch.where(take, state.tick + gap, next_snap)
        uniforms = scripted.draw_uniforms(games, SLOTS, generator, here)
        action, memory, _ = scripted.decide(FIXED_CONFIG, state, None, rows, uniforms, memory)
        noise = game.draw_noise(game_config, games, SLOTS, generator, here)
        state, out = game.step(game_config, state, action, noise)
        over = over | out.done
        tick += 1
    log(f"  group radius {game_config.arena_radius:.0f} {game_config.spawn}: {tick} ticks, "
        f"{len(snapshots)} snapshot ticks")
    snapshot = state_to(cat_states(snapshots), "cpu")
    return {
        "state": snapshot,
        "memory": {name: torch.cat([item[name] for item in memories]).cpu() for name in memory},
        "agent": torch.cat(agents).cpu(),
        "game": torch.cat(games_of).cpu(),
        "tick": torch.cat(ticks).cpu(),
        "radius": game_config.arena_radius,
        "spawn": game_config.spawn,
    }


def mix_table() -> tuple[torch.Tensor, list[str]]:
    units = screen_units()[:MIX_SOBOL]
    vectors = [vector(ScriptedConfig()), cma_vector()] + [unit_vector(row) for row in units]
    names = ["yardstick", "cma"] + [f"screen{index}" for index in range(MIX_SOBOL)]
    return torch.tensor(vectors, dtype=torch.float32), names


def make_states(data: Path) -> None:
    path = data / "states.pt"
    if path.exists():
        log(f"{path} exists; kept")
        return
    here = device()
    table, names = mix_table()
    table = table.to(here)
    groups = []
    started = time.perf_counter()
    for number, game_config in enumerate(game_configs()):
        groups.append(collect_group(game_config, table, 20261124 + number, GAMES))
        state = groups[-1]["state"]
        ships = int((state.alive & state.exists).sum())
        log(f"group {number + 1}/6: {state.x.shape[0]} snapshots, {ships} ship-states, "
            f"{time.perf_counter() - started:.0f} s")
    torch.save({"groups": groups, "mix": names, "mix_table": table.cpu(), "rows": list(PARAMETERS),
                "games_per_group": GAMES}, path)
    total = sum(int((group["state"].alive & group["state"].exists).sum()) for group in groups)
    log(f"wrote {path}: {total} ship-states, {path.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------------- distributions


class StateSet:
    """The stored ship-states on the device, per group, ready for batched calls."""

    def __init__(self, data: Path, limit: int | None = None):
        here = device()
        record = torch.load(data / "states.pt", weights_only=False)
        self.groups = []
        for group in record["groups"]:
            state = state_to(group["state"], here)
            memory = {name: value.to(here) for name, value in group["memory"].items()}
            if limit is not None:
                index = torch.arange(min(limit, state.x.shape[0]), device=here)
                state = select_rows(state, index)
                memory = {name: value.index_select(0, index) for name, value in memory.items()}
            mask = state.alive & state.exists
            self.groups.append({"state": state, "memory": memory, "mask": mask,
                                "envs": state.x.shape[0], "agent": group["agent"].to(here)})
        self.count = sum(int(group["mask"].sum()) for group in self.groups)
        # Snapshot index of every ship-state, in the order of `marginals` (for cluster errors).
        clusters, offset = [], 0
        for group in self.groups:
            snap = torch.arange(group["envs"], device=here)[:, None].expand_as(group["mask"])
            clusters.append(snap[group["mask"]] + offset)
            offset += group["envs"]
        self.cluster = torch.cat(clusters)
        self.snapshots = offset
        self.cluster_counts = torch.bincount(self.cluster, minlength=offset).float()

    @torch.no_grad()
    def marginals(self, vectors: torch.Tensor) -> torch.Tensor:
        """(K, S, 11) float32: each of K configurations' marginals (stick 7, power 3, fire 1)
        on every ship-state, in one batched call per group (the group repeated K times,
        every parameter a per-ship tensor row)."""
        here = self.groups[0]["state"].x.device
        vectors = vectors.to(here, torch.float32)
        count = vectors.shape[0]
        out = []
        for group in self.groups:
            envs = group["envs"]
            repeat = torch.arange(envs, device=here).repeat(count)
            state = select_rows_keep(group["state"], repeat)
            memory = {name: value.index_select(0, repeat) for name, value in group["memory"].items()}
            agent = torch.arange(count, device=here).repeat_interleave(envs)[:, None].expand(-1, SLOTS)
            zeros = torch.zeros((envs * count, SLOTS, 3), device=here)
            _, _, parts = scripted.decide(FIXED_CONFIG, state, None, rows_of(vectors, agent), zeros, memory)
            record = teacher_marginals(parts).view(count, envs, SLOTS, -1)
            out.append(record[:, group["mask"]])
        return torch.cat(out, 1)


def select_rows_keep(state: State, index: torch.Tensor) -> State:
    """select_rows without touching the (already empty) bullet slots."""
    values = {}
    for field in dataclasses.fields(state):
        item = getattr(state, field.name)
        if item is None:
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(item, table=item.table.index_select(0, index))
        else:
            values[field.name] = item.index_select(0, index)
    return State(**values)


def _entropy_terms(p: torch.Tensor) -> torch.Tensor:
    return torch.special.xlogy(p, p).sum(-1)


def js_joint(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JS divergence (nats) per row of two joint distributions (..., 42)."""
    m = 0.5 * (p + q)
    return 0.5 * (_entropy_terms(p) + _entropy_terms(q)) - _entropy_terms(m)


def js_parts(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per ship-state JS (nats) of the joint and of each head, from marginals (..., 11)."""
    joint = js_joint(teacher_joint(a), teacher_joint(b)).clamp_min(0.0)
    stick = js_joint(a[..., :7], b[..., :7]).clamp_min(0.0)
    power = js_joint(a[..., 7:10], b[..., 7:10]).clamp_min(0.0)
    fa, fb = a[..., 10:], b[..., 10:]
    fire = js_joint(torch.cat([1 - fa, fa], -1), torch.cat([1 - fb, fb], -1)).clamp_min(0.0)
    return {"joint": joint, "stick": stick, "power": power, "fire": fire}


def style_of(marginals: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean action shares over the ship-states (K,): from the distributions themselves."""
    stick, power, fire = marginals[..., :7], marginals[..., 7:10], marginals[..., 10]
    return {"straight": stick[..., 0].mean(-1), "full_stick": stick[..., 5:7].sum(-1).mean(-1),
            "dive": power[..., 1].mean(-1), "climb": power[..., 2].mean(-1), "fire": fire.mean(-1)}


def cluster_error(states: StateSet, values: torch.Tensor) -> torch.Tensor:
    """Standard error of the mean over ship-states of values (K, S), clustered by snapshot
    (the ships of one snapshot are not independent)."""
    sums = torch.zeros(values.shape[0], states.snapshots, device=values.device)
    sums.index_add_(1, states.cluster, values)
    mean = values.mean(-1, keepdim=True)
    residual = sums - states.cluster_counts[None] * mean
    n = states.snapshots
    return torch.sqrt((residual * residual).sum(-1) * n / (n - 1)) / values.shape[-1]


def divergence_from(states: StateSet, reference: torch.Tensor, vectors: torch.Tensor, batch: int) -> dict:
    """JS of every vector from the reference marginals (S, 11): means per head, the joint's
    cluster error, and the distribution-style shares; batched."""
    out = {key: [] for key in ("joint", "stick", "power", "fire", "joint_error",
                               "straight", "full_stick", "dive", "climb", "fire_share")}
    for start in range(0, vectors.shape[0], batch):
        chunk = states.marginals(vectors[start:start + batch])
        parts = js_parts(chunk, reference[None])
        for key, value in parts.items():
            out[key].append(value.mean(-1).cpu())
        out["joint_error"].append(cluster_error(states, parts["joint"]).cpu())
        style = style_of(chunk)
        for key, value in style.items():
            out["fire_share" if key == "fire" else key].append(value.cpu())
    return {key: torch.cat(value).numpy() for key, value in out.items()}


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def batched_run(data: Path, name: str, vectors: np.ndarray, batch: int, chunk: int = 2000, **extra) -> None:
    """JS from the yardstick for `vectors`, written to DATA/name.npz in chunks (resumes)."""
    states = StateSet(data)
    reference = states.marginals(torch.tensor([vector(ScriptedConfig())]))[0]
    log(f"{name}: {len(vectors)} configurations on {states.count} ship-states ({states.snapshots} snapshots)")
    parts_dir = data / f"{name}.parts"
    parts_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    for start in range(0, len(vectors), chunk):
        part = parts_dir / f"{start:07d}.npz"
        if part.exists():
            continue
        result = divergence_from(states, reference, torch.tensor(vectors[start:start + chunk]), batch)
        save_npz(part, **result)
        done = min(start + chunk, len(vectors))
        seconds = time.perf_counter() - started
        log(f"  {name}: {done}/{len(vectors)} in {seconds:.0f} s")
    merged = {}
    for part in sorted(parts_dir.glob("*.npz")):
        with np.load(part) as item:
            for key in item.files:
                merged.setdefault(key, []).append(item[key])
    merged = {key: np.concatenate(value) for key, value in merged.items()}
    save_npz(data / f"{name}.npz", vectors=vectors, **merged, **extra)
    log(f"wrote {data / f'{name}.npz'}")


# --------------------------------------------------------------------------- commands


def check(data: Path) -> None:
    """Bullets are not read (full state against bullets dropped) and one call's time."""
    here = device()
    game_config = game_configs()[2]
    sizes = torch.tensor([[4, 4]] * 64, device=here)
    generator = torch.Generator(device=here).manual_seed(1)
    state = game.deal(game_config, 64, SLOTS, sizes, generator, here)
    memory = scripted.fresh_memory(64, SLOTS, generator, here)
    for _ in range(200):
        action, memory, _ = scripted.decide(FIXED_CONFIG, state, None, None, scripted.draw_uniforms(64, SLOTS, generator, here), memory)
        state, _ = game.step(game_config, state, action, game.draw_noise(game_config, 64, SLOTS, generator, here))
    zeros = torch.zeros((64, SLOTS, 3), device=here)
    table = torch.tensor([vector(ScriptedConfig())], device=here)
    agent = torch.zeros((64, SLOTS), dtype=torch.long, device=here)
    _, _, full = scripted.decide(FIXED_CONFIG, state, None, rows_of(table, agent), zeros, memory)
    dropped = select_rows(state, torch.arange(64, device=here))
    _, _, light = scripted.decide(FIXED_CONFIG, dropped, None, rows_of(table, agent), zeros, memory)
    _, _, plain = scripted.decide(FIXED_CONFIG, state, None, None, zeros, memory)
    a, b, c = teacher_marginals(full), teacher_marginals(light), teacher_marginals(plain)
    print(f"bullets dropped: max |difference| {float((a - b).abs().max()):.3g}; "
          f"rows vs config numbers: {float((a - c).abs().max()):.3g}; bullet slots {state.bullets.shape[-1]}")
    if (data / "states.pt").exists():
        states = StateSet(data)
        for count in (1, 8, 16):
            vectors = torch.tensor([vector(ScriptedConfig())] * count)
            states.marginals(vectors)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(3):
                states.marginals(vectors)
            torch.cuda.synchronize()
            print(f"{count} configurations a call: {(time.perf_counter() - started) / 3 / count * 1000:.1f} ms each, "
                  f"{states.count} ship-states")


def references() -> tuple[list[str], np.ndarray]:
    panel = json.loads((HERE / "panel.json").read_text())
    names = ["yardstick", "cma", "bfm"] + [member["name"] for member in panel["members"][2:]]
    vectors = [vector(ScriptedConfig()), cma_vector(), vector(BFM)] + [member["values"] for member in panel["members"][2:]]
    return names, np.array(vectors, dtype=np.float32)


def screen(data: Path, batch: int) -> None:
    names, reference_vectors = references()
    units = screen_units()
    vectors = np.array([unit_vector(row) for row in units], dtype=np.float32)
    batched_run(data, "screen", np.concatenate([reference_vectors, vectors]), batch,
                names=np.array(names + [f"screen{index}" for index in range(len(units))]), units=units)


def saltelli(data: Path, base: int, batch: int) -> None:
    from SALib.sample import sobol as sobol_sample
    problem = {"num_vars": len(SAMPLED), "names": list(SAMPLED), "bounds": [[0.0, 1.0]] * len(SAMPLED)}
    units = sobol_sample.sample(problem, base, calc_second_order=False, seed=20261124)
    vectors = np.array([unit_vector(row) for row in units], dtype=np.float32)
    batched_run(data, f"saltelli{base}", vectors, batch, units=units)


def elementary(data: Path, base: int) -> None:
    """Saltelli's sample (base rows, each A, AB_1 .. AB_D, B; AB_i is A with row i taken from
    B), with two measures per configuration group: JS from the yardstick of every
    configuration (for direct Sobol indices), and JS(A, AB_i), the behaviour change of
    redrawing row i alone, and JS(A, B), of redrawing every row. Writes elementary{base}.npz,
    resuming by chunks of base rows."""
    from SALib.sample import sobol as sobol_sample
    problem = {"num_vars": len(SAMPLED), "names": list(SAMPLED), "bounds": [[0.0, 1.0]] * len(SAMPLED)}
    units = sobol_sample.sample(problem, base, calc_second_order=False, seed=20261125)
    width = len(SAMPLED) + 2
    states = StateSet(data)
    reference = states.marginals(torch.tensor([vector(ScriptedConfig())]))[0]
    parts_dir = data / f"elementary{base}.parts"
    parts_dir.mkdir(exist_ok=True)
    rows_per_chunk = 64
    started = time.perf_counter()
    for first in range(0, base, rows_per_chunk):
        part = parts_dir / f"{first:06d}.npz"
        if part.exists():
            continue
        from_yardstick = {key: [] for key in ("joint", "stick", "power", "fire")}
        one_row = {key: [] for key in ("joint", "stick", "power", "fire")}
        every_row = {key: [] for key in ("joint", "stick", "power", "fire")}
        for row in range(first, min(first + rows_per_chunk, base)):
            group = units[row * width:(row + 1) * width]
            vectors = torch.tensor(np.array([unit_vector(item) for item in group], dtype=np.float32))
            stored = torch.cat([states.marginals(vectors[start:start + 12]) for start in range(0, width, 12)])
            for key, value in js_parts(stored, reference[None]).items():
                from_yardstick[key].append(value.mean(-1).cpu())
            for key, value in js_parts(stored[1:-1], stored[:1]).items():
                one_row[key].append(value.mean(-1).cpu())
            for key, value in js_parts(stored[-1:], stored[:1]).items():
                every_row[key].append(value.mean(-1).cpu())
        save_npz(part, **{f"yardstick_{key}": torch.cat(value).numpy() for key, value in from_yardstick.items()},
                 **{f"one_row_{key}": torch.stack(value).numpy() for key, value in one_row.items()},
                 **{f"every_row_{key}": torch.cat(value).numpy() for key, value in every_row.items()})
        log(f"  elementary{base}: base rows {min(first + rows_per_chunk, base)}/{base}, {time.perf_counter() - started:.0f} s")
    merged = {}
    for part in sorted(parts_dir.glob("*.npz")):
        with np.load(part) as item:
            for key in item.files:
                merged.setdefault(key, []).append(item[key])
    save_npz(data / f"elementary{base}.npz", units=units, **{key: np.concatenate(value) for key, value in merged.items()})
    log(f"wrote {data / f'elementary{base}.npz'}")


SWEEP_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sweep(data: Path, batch: int) -> None:
    units, labels = [], []
    for column, name in enumerate(SAMPLED):
        for level in SWEEP_LEVELS:
            unit = DEFAULT_UNIT.copy()
            unit[column] = level
            units.append(unit)
            labels.append([column, level])
    vectors = np.array([unit_vector(row) for row in units], dtype=np.float32)
    batched_run(data, "sweep", vectors, batch, chunk=len(vectors), units=np.array(units), labels=np.array(labels))


def pairwise(data: Path, count: int, batch: int) -> None:
    """Pairwise JS of the references and the first count - 9 screening configurations."""
    path = data / f"pairwise{count}.npz"
    if path.exists():
        log(f"{path} exists; kept")
        return
    names, reference_vectors = references()
    units = screen_units()[: count - len(names)]
    vectors = np.concatenate([reference_vectors, np.array([unit_vector(row) for row in units], dtype=np.float32)])
    states = StateSet(data)
    started = time.perf_counter()
    stored = torch.cat([states.marginals(torch.tensor(vectors[start:start + batch]))
                        for start in range(0, len(vectors), batch)])  # (K, S, 11)
    log(f"pairwise: marginals of {len(vectors)} in {time.perf_counter() - started:.0f} s")
    total = len(vectors)
    matrix = torch.zeros(total, total)
    for i in range(total):
        p = teacher_joint(stored[i])
        for start in range(i + 1, total, 16):
            q = teacher_joint(stored[start:start + 16])
            matrix[i, start:start + q.shape[0]] = js_joint(p[None], q).clamp_min(0.0).mean(-1).cpu()
        if i % 50 == 0:
            log(f"  pairwise row {i}/{total}, {time.perf_counter() - started:.0f} s")
    matrix = matrix + matrix.T
    style = {key: value.cpu().numpy() for key, value in style_of(stored).items()}
    save_npz(path, matrix=matrix.numpy(), vectors=vectors,
             names=np.array(names + [f"screen{index}" for index in range(len(units))]), **style)
    log(f"wrote {path} in {time.perf_counter() - started:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("command", choices=("states", "check", "screen", "saltelli", "elementary", "sweep", "pairwise"))
    parser.add_argument("--base", type=int, default=512)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    arguments = parser.parse_args()
    data = Path(arguments.data).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    if arguments.command == "states":
        make_states(data)
    elif arguments.command == "check":
        check(data)
    elif arguments.command == "screen":
        screen(data, arguments.batch)
    elif arguments.command == "saltelli":
        saltelli(data, arguments.base, arguments.batch)
    elif arguments.command == "elementary":
        elementary(data, arguments.base)
    elif arguments.command == "sweep":
        sweep(data, arguments.batch)
    else:
        pairwise(data, arguments.count, arguments.batch)


if __name__ == "__main__":
    main()
