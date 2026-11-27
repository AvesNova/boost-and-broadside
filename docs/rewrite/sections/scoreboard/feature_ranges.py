"""The ship features' raw ranges in scripted play, for the observation's fixed scales
(lean/game/NOTES.md "Feature scales").

    OMP_NUM_THREADS=4 PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scoreboard/feature_ranges.py

Both teams scripted (the yardstick, spread 0, as lean.learn plays it), the core game (4v4
elimination, drawn fields, fog) with the training spawn on, E environments for T ticks, finished
environments re-dealt. Every living ship's raw quantity (as the observation reads it, before any
scale) is sampled every few ticks; prints the 1st and 99th percentiles, the extremes and the
share of exact zeros.
"""

import sys

import torch

from lean.contract import REAL
from lean.game import core
from lean.scripted import ScriptedConfig, act, draw_uniforms, fresh_memory, personas

ENVS, TICKS, EVERY, PER_TEAM = 256, 3000, 3, 4


def raw(config, state):
    """{name: (E, N) raw values} of the ship scalars observe() scales."""
    inverse_index = torch.exp(-state.log_index)
    distance = torch.sqrt(state.x * state.x + state.y * state.y)
    c = state.constants
    return {
        "world velocity x, y (m/s)": torch.stack([state.velocity_x, state.velocity_y]) * inverse_index,
        "world speed (m/s)": state.speed * inverse_index,
        "angle of attack (rad)": state.angle,
        "angle of attack sin": state.angle_sin,
        "nose rate (rad/s)": state.nose_rate,
        "store fraction": state.energy * c["inverse_max_energy"],
        "shield fraction": state.shield / c["max_shield"],
        "shield (points)": state.shield,
        "shield wait (s)": state.shield_wait,
        "gun cooldown (s)": state.cooldown * config.tick_length,
        "inside the boundary (m)": config.arena_radius - distance,
        "real bullet share": (state.bullet_state == REAL).float().mean(-1),
    }


def main() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = core.GameConfig(training_spawn=True)
    scripted = ScriptedConfig()
    sizes = torch.tensor([[PER_TEAM, PER_TEAM]]).expand(ENVS, 2)
    state = core.reset(config, ENVS, sizes, generator, "cpu")
    slots = state.x.shape[1]
    persona = personas(scripted, ENVS, slots, generator, "cpu", 0.0)
    memory = fresh_memory(ENVS, slots, generator, "cpu")
    samples: dict[str, list] = {}
    episodes = 0
    with torch.no_grad():
        for tick in range(TICKS):
            if tick % EVERY == 0:
                living = state.alive
                for name, value in raw(config, state).items():
                    samples.setdefault(name, []).append(value[..., living].flatten())
            action, memory = act(scripted, state, None, persona, draw_uniforms(ENVS, slots, generator, "cpu"), memory)
            state, out = core.step(config, state, action, core.draw_noise(config, ENVS, slots, generator, "cpu"))
            episodes += int(out.done.sum())
            fresh = fresh_memory(ENVS, slots, generator, "cpu")
            memory = {name: torch.where(out.done[:, None], fresh[name], memory[name]) for name in memory}
            state = core.take_deal(state, out.done, core.deal(config, ENVS, slots, sizes, generator, "cpu"))
            if (tick + 1) % 500 == 0:
                print(f"tick {tick + 1} of {TICKS}, episodes ended {episodes}", file=sys.stderr, flush=True)
    print(f"{ENVS} environments x {TICKS} ticks, {PER_TEAM}v{PER_TEAM}, every {EVERY}th tick, {episodes} episodes ended")
    print("| quantity | 1st percentile | 99th percentile | min | max | exact zeros |")
    print("|---|---|---|---|---|---|")
    for name, parts in samples.items():
        values = torch.cat(parts).double()
        low, high = torch.quantile(values[torch.randperm(values.numel())[:5_000_000]], torch.tensor([0.01, 0.99], dtype=torch.float64))
        zeros = float((values == 0).double().mean())
        print(f"| {name} | {low:.4g} | {high:.4g} | {values.min():.4g} | {values.max():.4g} | {zeros:.3f} |")


if __name__ == "__main__":
    main()
