"""Static against dynamic compiled step from the same states (within rounding), and a
hit-model change under the static graph with no recompile (GPU).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/tick-audit/static_check.py
"""

import json

import torch

from lean.contract import FIRES, POWERS, STICKS, compile_graph, whole_state
from lean.game import core

device = torch.device("cuda")
config = core.GameConfig()
g = torch.Generator(device=device).manual_seed(0)
envs = 2048
sizes = torch.tensor([[4, 4]], device=device).expand(envs, 2)
state = whole_state(core.reset(config, envs, sizes, g, device))
slots = state.x.shape[1]
inputs = []
for t in range(60):
    action = torch.stack(
        [
            torch.randint(0, n, (envs, slots), device=device, generator=g)
            for n in (STICKS, POWERS, FIRES)
        ],
        -1,
    )
    noise = core.draw_noise(config, envs, slots, g, device)
    inputs.append((state, action, noise))
    state, _ = core.step(config, state, action, noise)
static = compile_graph(core.step, "static")
dynamic = compile_graph(core.step, "dynamic")
worst = {}
for state, action, noise in inputs[10:]:
    a, ao = static(config, state, action, noise)
    b, bo = dynamic(config, state, action, noise)
    for name in ("ships", "bullets", "seen"):
        u, v = getattr(a, name), getattr(b, name)
        rel = ((u - v).abs() / v.abs().clamp_min(1.0)).max().item()
        worst[name] = max(worst.get(name, 0.0), rel)
    for name in ("bullet_state", "flags"):
        worst[name] = max(worst.get(name, 0), int((getattr(a, name) != getattr(b, name)).sum()))
    worst["outcome"] = max(worst.get("outcome", 0), int((ao.shot_outcome != bo.shot_outcome).sum()))
print("static vs dynamic (relative to max(|v|,1)):", json.dumps(worst))
state, action, noise = inputs[20]
changed = state.replace(
    constants=core.with_hit_model(state.constants, hit_radius=14.0, bullet_spread=40.0)
)
before, _ = static(config, state, action, noise)
after, _ = static(config, changed, action, noise)  # raises if the change recompiles
print(
    "hit model change, no recompile; bullets differ:",
    bool((before.bullets != after.bullets).any()),
    "damage rows differ:",
    bool((before.ships != after.ships).any()),
)
