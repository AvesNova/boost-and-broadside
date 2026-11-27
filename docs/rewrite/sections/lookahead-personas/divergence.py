"""Behavioural divergence of look-ahead personas (docs/rewrite/sections/lookahead-personas/).

A configuration's behaviour is its joint 42-way distribution (`scripted.explain`'s
`Choice.joint`, the BC teacher's label) on a fixed set of stored ship-states. Two
configurations differ by the Jensen-Shannon divergence (natural log) of their joints per
ship-state, averaged over the ship-states. Adapted from docs/rewrite/sections/scripted-screening/
divergence.py (the rule-based agent's study, 2026-11-24): the look-ahead agent has no memory,
so a ship-state is just the state.

    divergence.py DATA states              # the state set: two setups (below), sparse snapshots
    divergence.py DATA probe               # bullets not read; one configuration's time
    divergence.py DATA oat                 # each sampled row alone at u = 0, 0.25, 0.75, 1
    divergence.py DATA saltelli --base 256 # Saltelli sample: JS from the yardstick, one-row redraws
    divergence.py DATA population          # 64 personas at spread 0.25, 0.5, 1: label blur

Setups (the game's defaults otherwise, clock 1,800 ticks): "1v1" fog off, no fields; "4v4"
fields (10) and fog. The state-set games: per environment each team is the yardstick (1/3)
or spread-1 personas drawn per ship (2/3), redrawn at each deal; actions sampled from the
agent. Every command writes its results per chunk under DATA and skips chunks already written
(resumable); `--minutes` bounds one invocation (GPU jobs of about 10 minutes).
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.contract import State
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import SAMPLED, box_point, rows_of, vector

SETUPS = {
    "1v1": dict(size=1, changes=dict(fog=False, field_count=0), envs=4096, snapshots=20000),
    "4v4": dict(size=4, changes=dict(), envs=2048, snapshots=12000),
}
FIRST_LOW, FIRST_HIGH = 30, 230
GAP_LOW, GAP_HIGH = 150, 250
YARDSTICK_SHARE = 1.0 / 3.0
DIMENSIONS = len(SAMPLED)
CENTRE = np.full(DIMENSIONS, 0.5)  # every sampled row's box is symmetric: 0.5 is the default
LEVELS = (0.0, 0.25, 0.75, 1.0)
SPREADS = (0.25, 0.5, 1.0)
POPULATION = 64
PAIR_STATES = 20000  # ship-states of the pairwise subset (per setup, the first)


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def game_config(setup: str):
    return dataclasses.replace(game.GameConfig(), **SETUPS[setup]["changes"])


def select_envs(state: State, index: torch.Tensor, drop_bullets: bool = True) -> State:
    """The environments `index` of a state; bullets cut to one empty slot (the agent reads
    no bullet: `probe`)."""
    values = {}
    for item in dataclasses.fields(state):
        value = getattr(state, item.name)
        if value is None:
            values[item.name] = None
        elif item.name == "constants":
            values[item.name] = dataclasses.replace(
                value, ship_type=value.ship_type.index_select(0, index)
            )
        elif drop_bullets and item.name == "bullets":
            values[item.name] = value.index_select(0, index)[..., :1].clone().zero_()
        elif drop_bullets and item.name == "bullet_state":
            values[item.name] = (
                value.index_select(0, index)[..., :1].clone().fill_(int(value.max()))
            )
        else:
            values[item.name] = value.index_select(0, index)
    return State(**values)


def cat_states(states: list[State]) -> State:
    values = {}
    for item in dataclasses.fields(states[0]):
        items = [getattr(state, item.name) for state in states]
        if items[0] is None:
            values[item.name] = None
        elif item.name == "constants":
            values[item.name] = dataclasses.replace(
                items[0], ship_type=torch.cat([value.ship_type for value in items])
            )
        else:
            values[item.name] = torch.cat(items)
    return State(**values)


def state_to(state: State, target) -> State:
    values = {}
    for item in dataclasses.fields(state):
        value = getattr(state, item.name)
        if value is None:
            values[item.name] = None
        elif item.name == "constants":
            values[item.name] = dataclasses.replace(
                value,
                table=value.table.to(target),
                ship_type=value.ship_type.to(target),
                per_type={name: table.to(target) for name, table in value.per_type.items()},
            )
        else:
            values[item.name] = value.to(target)
    return State(**values)


def unit_rows(unit: np.ndarray, spread: float = 1.0) -> dict[str, float]:
    """Every row of the agent at a point of the unit cube (the sampled rows on their boxes)."""
    drawn = box_point(ScriptedConfig(), torch.as_tensor(unit, dtype=torch.float64), spread)
    values = dict(ScriptedConfig().values)
    values.update({name: float(drawn[name]) for name in SAMPLED})
    return values


def unit_vector(unit: np.ndarray, spread: float = 1.0) -> tuple[float, ...]:
    return vector(ScriptedConfig(unit_rows(unit, spread)))


# --------------------------------------------------------------------------- the state set


@torch.no_grad()
def collect(setup: str, seed: int) -> dict:
    here = device()
    spec = SETUPS[setup]
    config = game_config(setup)
    envs, size = spec["envs"], spec["size"]
    generator = torch.Generator(device=here).manual_seed(seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=here)
    state = game.reset(config, envs, sizes, generator, here)
    slots = state.x.shape[1]
    yard = ScriptedConfig()
    default = {name: torch.full((envs, slots), yard.values[name], device=here) for name in SAMPLED}

    def draw_mix():
        drawn = scripted.personas(yard, envs, slots, generator, here, 1.0)
        team_yard = torch.rand((envs, 2), generator=generator, device=here) < YARDSTICK_SHARE
        is_yard = team_yard.gather(1, state.team)
        return {name: torch.where(is_yard, default[name], drawn[name]) for name in SAMPLED}, is_yard

    persona, is_yard = draw_mix()

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device=here)

    next_snap = (FIRST_LOW + uniform(envs) * (FIRST_HIGH - FIRST_LOW)).long()
    taken, flags, persona_rows, ticks = [], [], [], []
    count, started, tick = 0, time.perf_counter(), 0
    while count < spec["snapshots"]:
        take = state.tick >= next_snap
        if bool(take.any()):
            index = take.nonzero().squeeze(-1)
            taken.append(state_to(select_envs(state, index), "cpu"))
            flags.append(is_yard.index_select(0, index).cpu())
            persona_rows.append(
                torch.stack([persona[name].index_select(0, index) for name in SAMPLED], -1).cpu()
            )
            ticks.append(state.tick.index_select(0, index).cpu())
            count += len(index)
            gap = (GAP_LOW + uniform(envs) * (GAP_HIGH - GAP_LOW)).long()
            next_snap = torch.where(take, state.tick + gap, next_snap)
        uniforms = scripted.draw_uniforms(envs, slots, generator, here)
        action = scripted.act(yard, state, None, persona, uniforms)
        state, out = game.tick(config, state, action, generator)
        tick += 1
        done = out.done
        if bool(done.any()):
            fresh, fresh_yard = draw_mix()
            persona = {
                name: torch.where(done[:, None], fresh[name], persona[name]) for name in SAMPLED
            }
            is_yard = torch.where(done[:, None], fresh_yard, is_yard)
            state = game.take_deal(
                state, done, game.deal(config, envs, slots, sizes, generator, here)
            )
            next_snap = torch.where(
                done, (FIRST_LOW + uniform(envs) * (FIRST_HIGH - FIRST_LOW)).long(), next_snap
            )
        if tick % 300 == 0:
            log(f"  {setup}: tick {tick}, {count} snapshots, {time.perf_counter() - started:.0f} s")
    snapshot = cat_states(taken)
    return {
        "state": snapshot,
        "is_yardstick": torch.cat(flags),
        "persona": torch.cat(persona_rows),
        "tick": torch.cat(ticks),
    }


def make_states(data: Path) -> None:
    for number, setup in enumerate(SETUPS):
        path = data / f"states_{setup}.pt"
        if path.exists():
            log(f"{path} exists; kept")
            continue
        group = collect(setup, 20261127 + number)
        state = group["state"]
        ships = int((state.alive & state.exists).sum())
        torch.save(group, path)
        log(f"wrote {path}: {state.x.shape[0]} snapshots, {ships} ship-states")


class StateSet:
    """Both setups' stored ship-states on the device."""

    def __init__(self, data: Path):
        here = device()
        self.groups = {}
        for setup in SETUPS:
            group = torch.load(data / f"states_{setup}.pt", weights_only=False)
            state = state_to(group["state"], here)
            mask = state.alive & state.exists
            self.groups[setup] = {"state": state, "mask": mask}
        self.counts = {setup: int(group["mask"].sum()) for setup, group in self.groups.items()}

    @torch.no_grad()
    def joint(
        self, setup: str, values: dict[str, float] | tuple, chunk: int = 4096
    ) -> torch.Tensor:
        """(S, 42) float32: the agent `values` (a full vector or its rows) on the setup's
        ship-states, every row a per-ship tensor (as the evaluation runner passes them)."""
        group = self.groups[setup]
        state, mask = group["state"], group["mask"]
        if isinstance(values, dict):
            values = vector(ScriptedConfig(values))
        table = torch.tensor([values], dtype=torch.float32, device=state.x.device)
        out = []
        envs = state.x.shape[0]
        for start in range(0, envs, chunk):
            index = torch.arange(start, min(start + chunk, envs), device=state.x.device)
            part = select_envs(state, index, drop_bullets=False)
            agent = torch.zeros(part.x.shape, dtype=torch.long, device=state.x.device)
            choice = scripted.explain(ScriptedConfig(), part, None, rows_of(table, agent))
            out.append(choice.joint[mask.index_select(0, index)])
        return torch.cat(out)


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -torch.special.xlogy(p, p).sum(-1)


def js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row JS divergence (nats) of two distributions (..., 42)."""
    return (entropy(0.5 * (p + q)) - 0.5 * (entropy(p) + entropy(q))).clamp_min(0.0)


def heads(p: torch.Tensor) -> dict[str, torch.Tensor]:
    """The joint's marginals: stick (7), power (3), fire (2)."""
    parts = p.view(*p.shape[:-1], 7, 3, 2)
    return {"stick": parts.sum((-1, -2)), "power": parts.sum((-1, -3)), "fire": parts.sum((-2, -3))}


def js_all(p: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    out = {"joint": float(js(p, q).mean())}
    hp, hq = heads(p), heads(q)
    for name in hp:
        out[name] = float(js(hp[name], hq[name]).mean())
    return out


def shares(p: torch.Tensor) -> dict[str, float]:
    """Mean action shares from the distribution itself."""
    h = heads(p)
    return {
        "straight": float(h["stick"][:, 0].mean()),
        "full_stick": float(h["stick"][:, 5:7].sum(-1).mean()),
        "dive": float(h["power"][:, 1].mean()),
        "climb": float(h["power"][:, 2].mean()),
        "fire_share": float(h["fire"][:, 1].mean()),
        "entropy": float(entropy(p).mean()),
    }


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def reference(states: StateSet, data: Path) -> dict[str, torch.Tensor]:
    """The yardstick's joints (cached on disk)."""
    path = data / "yardstick_joint.pt"
    if path.exists():
        return {k: v.to(device()) for k, v in torch.load(path).items()}
    out = {setup: states.joint(setup, ScriptedConfig().values) for setup in SETUPS}
    torch.save({k: v.cpu() for k, v in out.items()}, path)
    return out


# --------------------------------------------------------------------------- commands


def probe(data: Path) -> None:
    states = StateSet(data)
    here = device()
    for setup in SETUPS:
        group = states.groups[setup]
        index = torch.arange(min(512, group["state"].x.shape[0]), device=here)
        full = select_envs(group["state"], index, drop_bullets=False)
        light = select_envs(group["state"], index, drop_bullets=True)
        a = scripted.explain(ScriptedConfig(), full, None, None).joint
        b = scripted.explain(ScriptedConfig(), light, None, None).joint
        table = torch.tensor([vector(ScriptedConfig())], device=here)
        agent = torch.zeros(full.x.shape, dtype=torch.long, device=here)
        c = scripted.explain(ScriptedConfig(), full, None, rows_of(table, agent)).joint
        print(
            f"{setup}: {states.counts[setup]} ship-states; bullets dropped max |diff| "
            f"{float((a - b).abs().max()):.3g}; rows as tensors vs numbers "
            f"{float((a - c).abs().max()):.3g}"
        )
        states.joint(setup, ScriptedConfig().values)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            states.joint(setup, ScriptedConfig().values)
        torch.cuda.synchronize()
        print(
            "  one configuration on all ship-states: "
            f"{(time.perf_counter() - started) / 3 * 1000:.0f} ms"
        )


def run_configs(
    data: Path, name: str, units: np.ndarray, spreads: np.ndarray, minutes: float, chunk: int = 64
) -> bool:
    """JS from the yardstick (joint, heads) and action shares for every configuration, both
    setups; DATA/name.parts/ chunk files; returns True when complete."""
    states = StateSet(data)
    yard = reference(states, data)
    parts = data / f"{name}.parts"
    parts.mkdir(exist_ok=True)
    started = time.perf_counter()
    for first in range(0, len(units), chunk):
        part = parts / f"{first:06d}.npz"
        if part.exists():
            continue
        if (time.perf_counter() - started) / 60.0 > minutes:
            log(f"{name}: stopping at the time bound, {first}/{len(units)}")
            return False
        rows = {}
        for k in range(first, min(first + chunk, len(units))):
            values = unit_rows(units[k], float(spreads[k]))
            for setup in SETUPS:
                p = states.joint(setup, values)
                for key, value in {**js_all(p, yard[setup]), **shares(p)}.items():
                    rows.setdefault(f"{setup}_{key}", []).append(value)
        save_npz(part, **{key: np.array(value) for key, value in rows.items()})
        log(
            f"  {name}: {min(first + chunk, len(units))}/{len(units)}, "
            f"{time.perf_counter() - started:.0f} s"
        )
    merged = {}
    for part in sorted(parts.glob("*.npz")):
        with np.load(part) as item:
            for key in item.files:
                merged.setdefault(key, []).append(item[key])
    save_npz(
        data / f"{name}.npz",
        units=units,
        spreads=spreads,
        **{key: np.concatenate(value) for key, value in merged.items()},
    )
    log(f"wrote {data / f'{name}.npz'}")
    return True


def oat(data: Path, minutes: float) -> None:
    units, labels = [], []
    for column in range(DIMENSIONS):
        for level in LEVELS:
            unit = CENTRE.copy()
            unit[column] = level
            units.append(unit)
            labels.append((column, level))
    units = np.array(units)
    if run_configs(data, "oat", units, np.ones(len(units)), minutes):
        with np.load(data / "oat.npz") as item:
            merged = dict(item)
        save_npz(data / "oat.npz", labels=np.array(labels), **merged)


def saltelli(data: Path, base: int, minutes: float) -> None:
    """Saltelli's sample (A, AB_1 .. AB_D, B per base row, spread 1): JS from the yardstick of
    each (Sobol indices of that output), and JS(A, AB_i), the behaviour change of redrawing
    row i alone (and JS(A, B), of every row)."""
    from SALib.sample import sobol as sobol_sample

    problem = {"num_vars": DIMENSIONS, "names": list(SAMPLED), "bounds": [[0.0, 1.0]] * DIMENSIONS}
    units = sobol_sample.sample(problem, base, calc_second_order=False, seed=20261127)
    width = DIMENSIONS + 2
    states = StateSet(data)
    yard = reference(states, data)
    parts = data / f"saltelli{base}.parts"
    parts.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows_per_chunk = 8
    for first in range(0, base, rows_per_chunk):
        part = parts / f"{first:06d}.npz"
        if part.exists():
            continue
        if (time.perf_counter() - started) / 60.0 > minutes:
            log(f"saltelli: stopping at the time bound, base rows {first}/{base}")
            return
        out = {}
        for row in range(first, min(first + rows_per_chunk, base)):
            group = units[row * width : (row + 1) * width]
            for setup in SETUPS:
                joints = [states.joint(setup, unit_rows(item)) for item in group]
                from_yard = [js_all(p, yard[setup]) for p in joints]
                redraw = [js_all(joints[k], joints[0]) for k in range(1, width)]
                for key in ("joint", "stick", "power", "fire"):
                    out.setdefault(f"{setup}_yard_{key}", []).append(
                        [item[key] for item in from_yard]
                    )
                    out.setdefault(f"{setup}_redraw_{key}", []).append(
                        [item[key] for item in redraw]
                    )
        save_npz(part, **{key: np.array(value) for key, value in out.items()})
        log(
            f"  saltelli{base}: base rows {min(first + rows_per_chunk, base)}/{base}, "
            f"{time.perf_counter() - started:.0f} s"
        )
    merged = {}
    for part in sorted(parts.glob("*.npz")):
        with np.load(part) as item:
            for key in item.files:
                merged.setdefault(key, []).append(item[key])
    save_npz(
        data / f"saltelli{base}.npz",
        units=units,
        **{key: np.concatenate(value) for key, value in merged.items()},
    )
    log(f"wrote {data / f'saltelli{base}.npz'}")


def population(data: Path, minutes: float) -> None:
    """Label blur: POPULATION personas (uniform on the unit cube, the same points at every
    spread) on every ship-state. Per spread and setup: the mixture's entropy, the members'
    mean entropy (their difference is the generalized JS divergence: the cross-entropy a
    single student fitting the mixed labels pays beyond each member's own), each member's JS
    from the yardstick, and the pairwise JS on the first PAIR_STATES ship-states."""
    states = StateSet(data)
    yard = reference(states, data)
    rng = np.random.default_rng(20261127)
    units = rng.random((POPULATION, DIMENSIONS))
    started = time.perf_counter()
    for spread in SPREADS:
        for setup in SETUPS:
            path = data / f"population_{setup}_{spread}.npz"
            if path.exists():
                continue
            if (time.perf_counter() - started) / 60.0 > minutes:
                log("population: stopping at the time bound")
                return
            mixture = torch.zeros_like(yard[setup])
            member_entropy, from_yard, subset = [], [], []
            for unit in units:
                p = states.joint(setup, unit_rows(unit, spread))
                mixture += p / POPULATION
                member_entropy.append(float(entropy(p).mean()))
                from_yard.append(float(js(p, yard[setup]).mean()))
                subset.append(p[:PAIR_STATES].half())
            subset = torch.stack(subset).float()
            pairwise = np.zeros((POPULATION, POPULATION))
            for i in range(POPULATION):
                pairwise[i] = js(subset[i][None], subset).mean(-1).cpu().numpy()
            # with the yardstick as a member: the mixture of 1/2 yardstick, 1/2 population
            half = 0.5 * (mixture + yard[setup])
            save_npz(
                path,
                units=units,
                member_entropy=np.array(member_entropy),
                from_yardstick=np.array(from_yard),
                mixture_entropy=float(entropy(mixture).mean()),
                yardstick_entropy=float(entropy(yard[setup]).mean()),
                half_mixture_entropy=float(entropy(half).mean()),
                mixture_from_yardstick=float(js(mixture, yard[setup]).mean()),
                pairwise=pairwise,
            )
            log(f"  population {setup} spread {spread}: {time.perf_counter() - started:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("command", choices=("states", "probe", "oat", "saltelli", "population"))
    parser.add_argument("--base", type=int, default=256)
    parser.add_argument("--minutes", type=float, default=8.0)
    arguments = parser.parse_args()
    data = Path(arguments.data).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    if arguments.command == "states":
        make_states(data)
    elif arguments.command == "probe":
        probe(data)
    elif arguments.command == "oat":
        oat(data, arguments.minutes)
    elif arguments.command == "saltelli":
        saltelli(data, arguments.base, arguments.minutes)
    else:
        population(data, arguments.minutes)


if __name__ == "__main__":
    main()
