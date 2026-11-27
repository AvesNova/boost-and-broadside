"""Round 2, step 5: label blur of a persona box around a proposed centre (CPU or GPU, no
compile).

    blur2.py DATA CENTRE.json states --minutes 10    # the state sets (1v1, 4v4, 16v16)
    blur2.py DATA CENTRE.json blur --minutes 12      # 64 personas at spreads 0.25, 0.5, 1
    blur2.py DATA CENTRE.json report OUT.json

CENTRE.json: the centre's rows over the defaults (a JSON object; its crowd rows are fixed rows,
so every persona keeps them). The state sets as round 1's (divergence.collect): sparse
snapshots (first at 30-230 ticks, then every 150-250) of games with sampled actions, fields
and fog, each team either the centre (1/3) or spread-1 personas around it drawn per ship
(2/3), at each size's study clock (skill2.CLOCKS). The blur per spread and size: the 64
personas' mixture entropy minus their mean entropy (generalized JS), with the members' JS from
the centre, as divergence.population. `--drop` names sampled rows held at the centre (the
proposed box's inert rows). Every result is written per (spread, size) file; resumable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import divergence  # noqa: E402
import skill2  # noqa: E402

from lean import game, scripted  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import SAMPLED, box_point, rows_of, vector  # noqa: E402

SIZES = {1: (4096, 12000), 4: (1024, 6000), 16: (256, 2000)}  # envs, snapshots
SPREADS = (0.25, 0.5, 1.0)
POPULATION = 64


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def centre_of(path) -> ScriptedConfig:
    return ScriptedConfig().replace(**json.loads(Path(path).read_text()))


@torch.no_grad()
def collect(centre: ScriptedConfig, size: int, seed: int):
    here = divergence.device()
    envs, snapshots = SIZES[size]
    config = skill2.config_of(size)
    generator = torch.Generator(device=here).manual_seed(seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=here)
    state = game.reset(config, envs, sizes, generator, here)
    slots = state.x.shape[1]
    base = {name: torch.full((envs, slots), centre.values[name], device=here) for name in SAMPLED}

    def draw():
        drawn = scripted.personas(centre, envs, slots, generator, here, 1.0)
        team_centre = torch.rand((envs, 2), generator=generator, device=here) < 1.0 / 3.0
        is_centre = team_centre.gather(1, state.team)
        return {name: torch.where(is_centre, base[name], drawn[name]) for name in SAMPLED}

    persona = draw()

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device=here)

    low, high = divergence.FIRST_LOW, divergence.FIRST_HIGH
    next_snap = (low + uniform(envs) * (high - low)).long()
    taken, count = [], 0
    while count < snapshots:
        take = state.tick >= next_snap
        if bool(take.any()):
            index = take.nonzero().squeeze(-1)
            taken.append(divergence.state_to(divergence.select_envs(state, index), "cpu"))
            count += len(index)
            gap = divergence.GAP_LOW + uniform(envs) * (divergence.GAP_HIGH - divergence.GAP_LOW)
            next_snap = torch.where(take, state.tick + gap.long(), next_snap)
        uniforms = scripted.draw_uniforms(envs, slots, generator, here)
        action = scripted.act(centre, state, None, persona, uniforms)
        state, out = game.tick(config, state, action, generator)
        if bool(out.done.any()):
            fresh = draw()
            persona = {
                name: torch.where(out.done[:, None], fresh[name], persona[name]) for name in SAMPLED
            }
            state = game.take_deal(
                state, out.done, game.deal(config, envs, slots, sizes, generator, here)
            )
            next_snap = torch.where(
                out.done, (low + uniform(envs) * (high - low)).long(), next_snap
            )
    return divergence.cat_states(taken)


class States:
    def __init__(self, data: Path, size: int):
        here = divergence.device()
        self.state = divergence.state_to(
            torch.load(data / f"states_{size}v{size}.pt", weights_only=False), here
        )
        self.mask = self.state.alive & self.state.exists

    @torch.no_grad()
    def joint(self, values: tuple, chunk: int = 1024) -> torch.Tensor:
        state = self.state
        table = torch.tensor([values], dtype=torch.float32, device=state.x.device)
        envs = state.x.shape[0]
        out = []
        for start in range(0, envs, chunk):
            index = torch.arange(start, min(start + chunk, envs), device=state.x.device)
            part = divergence.select_envs(state, index, drop_bullets=False)
            agent = torch.zeros(part.x.shape, dtype=torch.long, device=state.x.device)
            choice = scripted.explain(ScriptedConfig(), part, None, rows_of(table, agent))
            out.append(choice.joint[self.mask.index_select(0, index)])
        return torch.cat(out)


def persona_vector(centre: ScriptedConfig, unit: np.ndarray, spread: float, drop) -> tuple:
    drawn = box_point(centre, torch.as_tensor(unit, dtype=torch.float64), spread)
    values = dict(centre.values)
    values.update({name: float(drawn[name]) for name in SAMPLED if name not in drop})
    return vector(ScriptedConfig(values))


def blur(data: Path, centre: ScriptedConfig, minutes: float, drop, tag: str) -> None:
    started = time.perf_counter()
    units = np.random.default_rng(20261128).random((POPULATION, len(SAMPLED)))
    for size in SIZES:
        states = None
        for spread in SPREADS:
            path = data / f"blur{tag}_{size}v{size}_{spread}.json"
            if path.exists():
                continue
            if (time.perf_counter() - started) / 60.0 > minutes:
                log("blur: time bound")
                return
            states = states or States(data, size)
            centre_joint = states.joint(vector(centre))
            mixture = torch.zeros_like(centre_joint)
            member_entropy, from_centre = [], []
            for unit in units:
                p = states.joint(persona_vector(centre, unit, spread, drop))
                mixture += p / POPULATION
                member_entropy.append(float(divergence.entropy(p).mean()))
                from_centre.append(float(divergence.js(p, centre_joint).mean()))
            mean_entropy = float(np.mean(member_entropy))
            mixture_entropy = float(divergence.entropy(mixture).mean())
            result = {
                "ship_states": int(centre_joint.shape[0]),
                "centre_entropy": float(divergence.entropy(centre_joint).mean()),
                "member_entropy": mean_entropy,
                "mixture_entropy": mixture_entropy,
                "blur": mixture_entropy - mean_entropy,
                "from_centre_mean": float(np.mean(from_centre)),
                "from_centre_max": float(np.max(from_centre)),
                "mixture_from_centre": float(divergence.js(mixture, centre_joint).mean()),
                "drop": list(drop),
            }
            path.write_text(json.dumps(result, indent=1))
            log(f"blur {size}v{size} spread {spread}: {result['blur']:.4f} nats")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("centre")
    parser.add_argument("command", choices=("states", "blur", "report"))
    parser.add_argument("out", nargs="?")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()
    data = arguments.data
    data.mkdir(parents=True, exist_ok=True)
    centre = centre_of(arguments.centre)
    if arguments.command == "states":
        started = time.perf_counter()
        for number, size in enumerate(SIZES):
            path = data / f"states_{size}v{size}.pt"
            if path.exists():
                continue
            if (time.perf_counter() - started) / 60.0 > arguments.minutes:
                log("states: time bound")
                return
            state = collect(centre, size, 20261128 + number)
            torch.save(state, path)
            log(f"wrote {path}: {int((state.alive & state.exists).sum())} ship-states")
    elif arguments.command == "blur":
        blur(data, centre, arguments.minutes, tuple(arguments.drop), arguments.tag)
    else:
        rows = {}
        for path in sorted(data.glob("blur*_*.json")):
            rows[path.stem] = json.loads(path.read_text())
        Path(arguments.out).write_text(json.dumps(rows, indent=1))
        for key, row in rows.items():
            log(
                f"{key}: blur {row['blur']:.4f} of member H {row['member_entropy']:.3f} "
                f"({row['blur'] / row['member_entropy']:.1%}), JS from centre mean "
                f"{row['from_centre_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
