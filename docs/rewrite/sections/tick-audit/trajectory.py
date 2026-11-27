"""Eager CPU game trajectory for bitwise / within-rounding checks between trees: reset, then
TICKS steps with seeded random actions and re-deals (clock 60 ticks, 64 environments, 1-4 a
side); saves every tick's state tensors and outcomes. Compare two files with compare.py.

    PYTHONPATH=<tree> uv run --no-sync python docs/rewrite/sections/tick-audit/trajectory.py \
        OUT.pt TICKS [DEFLECTOR]
"""

import dataclasses
import sys

import torch

from lean.contract import FIRES, POWERS, STICKS
from lean.game import core

out, ticks = sys.argv[1], int(sys.argv[2])
deflector = sys.argv[3] if len(sys.argv) > 3 else None
torch.manual_seed(0)
config = (
    core.GameConfig(clock_ticks=60)
    if deflector is None
    else core.GameConfig(clock_ticks=60, deflector=deflector)
)
g = torch.Generator().manual_seed(0)
envs = 64
sizes = torch.randint(1, 5, (envs, 2), generator=g)
state = core.reset(config, envs, sizes, g, "cpu")
fresh = core.deal(config, envs, state.x.shape[1], sizes, g, "cpu")
slots = state.x.shape[1]
record = []
for t in range(ticks):
    action = torch.stack(
        [torch.randint(0, n, (envs, slots), generator=g) for n in (STICKS, POWERS, FIRES)], -1
    )
    noise = core.draw_noise(config, envs, slots, g, "cpu")
    state, so = core.step(config, state, action, noise)
    done = so.done
    state = core.take_deal(state, done, fresh, (torch.cumsum(done.long(), 0) - 1) % envs)
    keep = {
        f.name: getattr(state, f.name).clone()
        for f in dataclasses.fields(state)
        if isinstance(getattr(state, f.name), torch.Tensor)
    }
    keep["outcome"] = so.shot_outcome.clone()
    keep["damage"] = so.hit_damage.clone()
    keep["done"] = done.clone()
    record.append(keep)
torch.save(record, out)
