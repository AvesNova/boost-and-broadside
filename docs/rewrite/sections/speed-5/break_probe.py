import torch
from lean import game, model
from lean.learn.rollout import chunked_step
from lean.model.agent import agent_step
d = torch.device("cuda"); torch.manual_seed(0)
gc = game.GameConfig(training_spawn=True); envs = 12
state = game.reset(gc, envs, torch.full((envs, 2), 4, device=d), torch.Generator(device=d).manual_seed(0), d)
view = game.observe(gc, state, 0)
policy = model.Policy(model.ModelConfig.for_arena(gc.arena_radius)).to(d)
carry = policy.initial_carry(envs, state.exists.shape[1], d)
offset = torch.zeros(envs, 2, device=d); reset = torch.zeros(envs, dtype=torch.bool, device=d)
def f(view, carry, offset, reset):
    return agent_step(chunked_step(policy, 3), True, view, carry, offset, None, reset)
try:
    torch.compile(f, fullgraph=True)(view, carry, offset, reset)
    print("OK fullgraph")
except Exception as e:
    print("BREAK", str(e)[:3000])
