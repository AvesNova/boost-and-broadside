"""Both collision rules on the same play (docs/rewrite/sections/first-touch/progress.md).

    PYTHONPATH=<tree> uv run --no-sync python docs/rewrite/sections/first-touch/episodes.py

The yardstick on both teams, GameConfig() with and without fields, 256 games each, seed 0. Per
overlap episode (a run of ticks a pair ends inside the two hit radii R or is scored by either
rule): the contacts of the closest-approach rule (core 17e6cf3b) and of the first-touch rule
(2026-11-26), both computed here on the same straight moves, and whether the episode was cut (a
ship of the pair not alive at the next tick start, or the game over). Prints the counts per
(kind, old contacts, new contacts)."""

from collections import Counter

import torch

from lean import game, scripted

torch.set_num_threads(4)


@torch.no_grad()
def run(size, fields, envs=256, seed=0):
    config = game.GameConfig() if fields else game.GameConfig(field_count=0)
    generator = torch.Generator().manual_seed(seed)
    state = game.reset(config, envs, torch.tensor([[size, size]]).expand(envs, 2), generator, "cpu")
    slots = state.exists.shape[1]
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    agent = scripted.ScriptedConfig()
    reach = 2 * float(state.constants["hit_radius"])
    upper = torch.triu(torch.ones(slots, slots, dtype=torch.bool), 1)
    over = torch.zeros(envs, dtype=torch.bool)
    in_episode = torch.zeros(envs, slots, slots, dtype=torch.bool)
    old_count = torch.zeros(envs, slots, slots, dtype=torch.long)
    new_count = torch.zeros_like(old_count)

    tally = Counter()

    def close(mask, kind):
        for old, new in zip(old_count[mask].tolist(), new_count[mask].tolist()):
            tally[(kind, f"old{min(old, 2)}", f"new{min(new, 2)}")] += 1

    for tick in range(config.clock_ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        action, memory = scripted.act(agent, state, None, None, uniforms, memory)
        alive0 = state.alive & state.exists & ~over[:, None]
        new, out = game.tick(config, state, action, generator)
        gap_x = state.x[:, None, :] - state.x[:, :, None]
        gap_y = state.y[:, None, :] - state.y[:, :, None]
        move_x = (new.x - state.x)[:, None, :] - (new.x - state.x)[:, :, None]
        move_y = (new.y - state.y)[:, None, :] - (new.y - state.y)[:, :, None]
        tau = -(gap_x * move_x + gap_y * move_y) / (move_x * move_x + move_y * move_y).clamp_min(
            1e-12
        )
        clamped = tau.clamp(0, 1)
        miss = torch.hypot(gap_x + clamped * move_x, gap_y + clamped * move_y)
        pair = alive0[:, :, None] & alive0[:, None, :] & upper
        old_contact = (tau >= 0) & (tau < 1) & (miss < reach) & pair
        new_contact = (torch.hypot(gap_x, gap_y) >= reach) & (miss < reach) & pair
        end_gap = torch.hypot(gap_x + move_x, gap_y + move_y)
        touching = pair & ((end_gap < reach) | old_contact | new_contact)
        close(in_episode & ~pair, "cut")
        close(in_episode & ~touching & pair, "closed")
        old_count = torch.where(
            touching, torch.where(in_episode, old_count, 0) + old_contact.long(), 0
        )
        new_count = torch.where(
            touching, torch.where(in_episode, new_count, 0) + new_contact.long(), 0
        )
        in_episode = touching
        over = over | out.done
        state = new
        if bool(over.all()):
            break
    close(in_episode, "cut")
    print(size, fields, sorted(tally.items()), flush=True)


for size in (1, 4):
    for fields in (False, True):
        run(size, fields)
