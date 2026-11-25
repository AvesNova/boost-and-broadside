"""Nose cone and collisions: a sanity measurement (docs/rewrite/sections/nose-cone/progress.md).

    PYTHONPATH=<tree> uv run --no-sync python docs/rewrite/sections/nose-cone/sanity.py OUT.json
        [--sizes 1 4] [--envs 256] [--seed 0] [--threads 4]

The yardstick on both teams (ScriptedConfig(), one `act` call for both), CPU, the game as
evaluated (GameConfig(): fog, the standard spawn, the 1,800-tick clock), without fields and with
them; every game runs to its end (no re-deal). Runs against whichever tree is on PYTHONPATH (core
before the change has no collisions: its contacts are the passes that would be collisions).

Per pair and tick, the swept test (straight moves; a pass is the tick holding its closest
approach). A contact is what the tree's `core.collisions` scores (core 17e6cf3b: a pass inside
the two hit radii; from 2026-11-26: the first touch); a tree without it uses the pass rule. The
once-per-pass check: each run of ticks in which a pair is scored or ends the tick inside the two
radii (an overlap episode) should hold one contact. Runs cut by a death or the game's end are
counted apart, and so are runs that began inside the radii (spawned overlapping: never scored
from 2026-11-26); a fast episode is one closing faster than FAST_PASS in any of its ticks. A
head-on pass is a pass inside HEAD_ON_PASS metres with the noses at least HEAD_ON_NOSES apart.
Per real bullet that hit an enemy (shot_outcome 1): the hit angle, the bullet's direction
against the nose of the enemy nearest its last move (mid-tick positions), and whether that
bullet's pair took bullet damage.
Writes one record per (size, fields) to OUT.
"""

import argparse
import json
import math
import sys
import time

import torch

from lean import game, scripted
from lean.game import core

HEAD_ON_PASS = 100.0  # m
HEAD_ON_NOSES = math.radians(150.0)
NOSE_CONE = math.radians(30.0)
FAST_PASS = 100.0  # m/s


@torch.no_grad()
def measure(size: int, fields: bool, envs: int, seed: int) -> dict:
    base = game.GameConfig()
    config = base if fields else game.GameConfig(field_count=0)
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[size, size]]).expand(envs, 2)
    state = game.reset(config, envs, sizes, generator, "cpu")
    slots = state.exists.shape[1]
    agent = scripted.ScriptedConfig()
    memory = scripted.fresh_memory(envs, slots, generator, "cpu")
    c = state.constants
    reach = 2.0 * float(c["hit_radius"])
    h = config.tick_length
    upper = torch.triu(torch.ones(slots, slots, dtype=torch.bool), 1)
    team = state.team
    enemy_pair = team[:, :, None] != team[:, None, :]
    over = torch.zeros(envs, dtype=torch.bool)
    length = torch.zeros(envs, dtype=torch.long)
    result = torch.full((envs,), -1, dtype=torch.long)
    tally = dict(
        contacts=0,
        enemy_contacts=0,
        head_on_passes=0,
        head_on_contacts=0,
        collision_damage=0.0,
        collision_deaths=0,
        deaths=0,
        hits=0,
        hits_in_cone=0,
        episodes=0,
        episodes_cut=0,
        episodes_unscored=0,
        episodes_scored_twice=0,
        episodes_began_inside=0,
        episodes_began_inside_scored=0,
        fast_episodes=0,
        fast_episodes_unscored=0,
        fast_episodes_scored_twice=0,
        damaging_hits_in_cone=0,
        bullet_damage=0.0,
    )
    rule = getattr(core, "collisions", None)
    # Overlap episodes, for the once-per-pass check: runs of ticks in which a pair is scored or
    # ends the tick inside the two hit radii; each should hold exactly one contact.
    in_episode = torch.zeros(envs, slots, slots, dtype=torch.bool)
    episode_contacts = torch.zeros(envs, slots, slots, dtype=torch.long)
    episode_inside = torch.zeros(envs, slots, slots, dtype=torch.bool)  # began inside the radii
    episode_fast = torch.zeros(envs, slots, slots, dtype=torch.bool)
    started = time.time()
    for tick in range(config.clock_ticks):
        uniforms = scripted.draw_uniforms(envs, slots, generator, "cpu")
        action, memory = scripted.act(agent, state, None, None, uniforms, memory)
        playing = ~over
        alive0 = state.alive & state.exists & playing[:, None]
        new, out = game.tick(config, state, action, generator)
        # Swept passes between living ships.
        gap_x = state.x[:, None, :] - state.x[:, :, None]
        gap_y = state.y[:, None, :] - state.y[:, :, None]
        move_x = (new.x - state.x)[:, None, :] - (new.x - state.x)[:, :, None]
        move_y = (new.y - state.y)[:, None, :] - (new.y - state.y)[:, :, None]
        move_squared = (move_x * move_x + move_y * move_y).clamp_min(1e-12)
        tau = -(gap_x * move_x + gap_y * move_y) / move_squared
        clamped = tau.clamp(0.0, 1.0)
        miss = torch.hypot(gap_x + clamped * move_x, gap_y + clamped * move_y)
        in_tick = (tau >= 0) & (tau < 1) & alive0[:, :, None] & alive0[:, None, :] & upper
        noses = (
            new.heading_cos[:, :, None] * new.heading_cos[:, None, :]
            + new.heading_sin[:, :, None] * new.heading_sin[:, None, :]
        )
        if rule is not None:
            scored = rule(state, alive0, h, new.x, new.y, new.heading_cos, new.heading_sin)
            contact = scored.contact & upper
        else:
            contact = in_tick & (miss < reach)
        head_on = in_tick & (miss < HEAD_ON_PASS) & (noses < math.cos(HEAD_ON_NOSES))
        end_gap = torch.hypot(
            new.x[:, None, :] - new.x[:, :, None], new.y[:, None, :] - new.y[:, :, None]
        )
        alive_pair = alive0[:, :, None] & alive0[:, None, :] & upper
        touching = alive_pair & ((end_gap < reach) | contact)
        # An episode ends with both ships still alive (one cut by a death or the game's end
        # before its closest approach is no pass).
        closed = in_episode & ~touching & alive_pair
        tally["episodes_cut"] += int((in_episode & ~alive_pair).sum())
        inside = closed & episode_inside
        outside = closed & ~episode_inside
        fast = outside & episode_fast
        tally["episodes_began_inside"] += int(inside.sum())
        tally["episodes_began_inside_scored"] += int((inside & (episode_contacts > 0)).sum())
        tally["episodes"] += int(outside.sum())
        tally["episodes_unscored"] += int((outside & (episode_contacts == 0)).sum())
        tally["episodes_scored_twice"] += int((outside & (episode_contacts > 1)).sum())
        tally["fast_episodes"] += int(fast.sum())
        tally["fast_episodes_unscored"] += int((fast & (episode_contacts == 0)).sum())
        tally["fast_episodes_scored_twice"] += int((fast & (episode_contacts > 1)).sum())
        begins = touching & ~in_episode
        start_gap = torch.hypot(gap_x, gap_y)
        episode_inside = torch.where(begins, start_gap < reach, episode_inside & touching)
        closing_speed = move_squared.sqrt() / h
        episode_fast = torch.where(
            touching, (episode_fast & ~begins) | (closing_speed > FAST_PASS), False
        )
        episode_contacts = torch.where(touching, episode_contacts + contact.long(), 0)
        in_episode = touching
        tally["contacts"] += int(contact.sum())
        tally["enemy_contacts"] += int((contact & enemy_pair).sum())
        tally["head_on_passes"] += int(head_on.sum())
        tally["head_on_contacts"] += int((head_on & contact).sum())
        crash = torch.zeros(envs, slots, slots)
        if rule is not None:
            crash = rule(
                state, alive0, h, new.x, new.y, new.heading_cos, new.heading_sin
            ).damage.transpose(1, 2)  # [credited, damaged]
            tally["collision_damage"] += float(crash.sum())
            tally["collision_deaths"] += int((out.died & alive0 & (crash.sum(-2) > 0)).sum())
        tally["deaths"] += int((out.died & alive0).sum())
        bullet = (out.hit_damage - crash) * playing[:, None, None]
        tally["bullet_damage"] += float(bullet.sum())
        # Real bullets that hit an enemy this tick: the hit angle on the nearest enemy.
        hit = (out.shot_outcome == 1) & new.bullet_real & playing[:, None, None]
        if bool(hit.any()):
            e, owner, k = hit.nonzero(as_tuple=True)
            bullets = new.bullets[e, :, owner, k]  # (H, columns)
            end_x, end_y = bullets[:, core.BULLET_X], bullets[:, core.BULLET_Y]
            direction_x, direction_y = bullets[:, core.BULLET_DIR_X], bullets[:, core.BULLET_DIR_Y]
            step = bullets[:, core.BULLET_SPEED] * h
            mid_x = end_x - 0.5 * step * direction_x
            mid_y = end_y - 0.5 * step * direction_y
            ship_x = 0.5 * (state.x + new.x)[e]
            ship_y = 0.5 * (state.y + new.y)[e]
            distance = torch.hypot(ship_x - mid_x[:, None], ship_y - mid_y[:, None])
            enemy = (team[e] != team[e, owner][:, None]) & alive0[e]
            target = torch.where(enemy, distance, math.inf).argmin(-1)
            nose_x = new.heading_cos[e, target]
            nose_y = new.heading_sin[e, target]
            hit_cos = -(direction_x * nose_x + direction_y * nose_y)
            in_cone = hit_cos > math.cos(NOSE_CONE)
            damaged = bullet[e, owner, target] > 0.0
            tally["hits"] += int(hit.sum())
            tally["hits_in_cone"] += int(in_cone.sum())
            tally["damaging_hits_in_cone"] += int((in_cone & damaged).sum())
        finished = out.done & playing
        length = torch.where(finished, tick + 1, length)
        result = torch.where(finished, out.result, result)
        over = over | out.done
        state = new
        if bool(over.all()):
            break
    games = envs
    record = dict(
        size=size,
        fields=fields,
        games=games,
        seconds=round(time.time() - started, 1),
        draws=int((result == 2).sum()),
        team_zero_wins=int((result == 0).sum()),
        mean_length_s=float(length.float().mean()) * h,
        **tally,
    )
    record["contacts_per_game"] = tally["contacts"] / games
    record["head_on_passes_per_game"] = tally["head_on_passes"] / games
    record["contacts_per_head_on_pass"] = tally["head_on_contacts"] / max(
        tally["head_on_passes"], 1
    )
    record["share_of_hits_in_cone"] = tally["hits_in_cone"] / max(tally["hits"], 1)
    print(json.dumps(record), file=sys.stderr, flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    records = []
    for size in arguments.sizes:
        for fields in (False, True):
            records.append(measure(size, fields, arguments.envs, arguments.seed))
            with open(arguments.out, "w") as handle:  # each run persisted as it lands
                json.dump(records, handle, indent=1)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
