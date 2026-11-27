"""CPU check (queue item 4): Policy.step over seat chunks equals one call (rows independent)."""
import dataclasses

import torch

from lean import game, model

torch.manual_seed(0)
config = game.GameConfig(training_spawn=True)
envs = 12
sizes = torch.full((envs, 2), 4)
state = game.reset(config, envs, sizes, torch.Generator().manual_seed(0), "cpu")
obs = game.observe(config, state, 0)
policy = model.Policy(model.ModelConfig.for_arena(config.arena_radius))
carry = torch.randn(policy.initial_carry(envs, state.exists.shape[1], "cpu").shape)
offset = torch.randn(envs, 2) * 100
reset = torch.rand(envs) < 0.3
whole = policy.step(obs, carry, offset, reset)
parts = []
for part in (slice(0, 5), slice(5, 12)):
    view = dataclasses.replace(obs, **{f.name: getattr(obs, f.name)[part] for f in dataclasses.fields(obs)})
    parts.append(policy.step(view, carry[:, part], offset[part], reset[part]))
for index, name in enumerate(("logits", "value_logits", "carry")):
    joined = torch.cat([p[index] for p in parts], 1 if name == "carry" else 0)
    print(name, (joined - whole[index]).abs().max().item(), whole[index].abs().max().item())
