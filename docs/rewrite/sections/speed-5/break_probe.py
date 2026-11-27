"""GPU probe (queue item 5): the chunked live-policy step under bf16 compiles fullgraph
(a nested compile region decorated inside traced code is a graph break).

    PYTHONPATH=. python docs/rewrite/sections/speed-5/break_probe.py
"""

import torch

from lean import game, model
from lean.learn.rollout import chunked_step
from lean.model.agent import agent_step

device = torch.device("cuda")
torch.manual_seed(0)
config = game.GameConfig(training_spawn=True)
envs = 12
state = game.reset(
    config, envs, torch.full((envs, 2), 4, device=device),
    torch.Generator(device=device).manual_seed(0), device,
)
view = game.observe(config, state, 0)
policy = model.Policy(model.ModelConfig.for_arena(config.arena_radius)).to(device)
carry = policy.initial_carry(envs, state.exists.shape[1], device)
offset = torch.zeros(envs, 2, device=device)
reset = torch.zeros(envs, dtype=torch.bool, device=device)


def tick(view, carry, offset, reset):
    return agent_step(chunked_step(policy, 3), True, view, carry, offset, None, reset)


try:
    torch.compile(tick, fullgraph=True)(view, carry, offset, reset)
    print("OK fullgraph")
except Exception as error:  # noqa: BLE001
    print("BREAK", str(error)[:3000])
