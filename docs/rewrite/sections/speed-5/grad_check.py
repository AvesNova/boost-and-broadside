"""GPU check (queue item 5): one minibatch's gradient at micro-batches of 64, 16 and 8 views
(eager, bf16 as the trainer, every geometry pair so no draw), the default model at 4v4.
Plain SGD at rate 1 without clipping: the step is minus the summed gradient.

    PYTHONPATH=. python docs/rewrite/sections/speed-5/grad_check.py
"""

import copy
import dataclasses
import json

import torch

from lean import game, model, scripted
from lean.learn import LearnConfig, Opponent, collect, start
from lean.learn.store import RolloutStore
from lean.learn.training import update

device = torch.device("cuda")
torch.manual_seed(0)
generator = torch.Generator(device=device).manual_seed(0)
config = dataclasses.replace(
    LearnConfig(),
    epochs=1,
    minibatches=1,
    learning_rate=1.0,
    warmup_steps=1,
    gradient_clip=1e9,
    pair_samples=10**6,
    field_pair_samples=10**6,
)
game_config = game.GameConfig(training_spawn=True)
policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
opponent = Opponent(scripted.ScriptedConfig(), None)
state, carry = start(config, game_config, policy, opponent, 128, generator)
store = RolloutStore(1)
collect(config, game_config, policy, opponent, state, carry, 64, generator, store=store)
print("views", store.views, flush=True)
steps, losses = {}, {}
for size in (64, 16, 8):
    twin = copy.deepcopy(policy)
    before = [parameter.detach().clone() for parameter in twin.parameters()]
    optimiser = torch.optim.SGD(twin.parameters(), lr=1.0, fused=True)
    metrics = update(
        dataclasses.replace(config, micro_batch_views=size),
        twin,
        optimiser,
        store,
        cloning=True,
        generator=torch.Generator(device=device).manual_seed(2),
    )
    steps[size] = [after.detach() - b for after, b in zip(twin.parameters(), before, strict=True)]
    losses[size] = {name: metrics[name] for name in ("loss", "bc", "value", "next_state", "hit")}
    del twin, optimiser
    torch.cuda.empty_cache()
scale = max(step.abs().max().item() for step in steps[64])
norm = torch.sqrt(sum((step.float() ** 2).sum() for step in steps[64])).item()
record = {"gradient_max": scale, "gradient_norm": norm, "losses": losses}
for size in (16, 8):
    worst = max((a - b).abs().max().item() for a, b in zip(steps[size], steps[64], strict=True))
    difference = torch.sqrt(
        sum(((a - b).float() ** 2).sum() for a, b in zip(steps[size], steps[64], strict=True))
    ).item()
    record[f"b{size}_vs_64"] = {
        "max_abs_difference": worst,
        "relative_to_max": worst / scale,
        "relative_norm": difference / norm,
    }
print(json.dumps(record), flush=True)
