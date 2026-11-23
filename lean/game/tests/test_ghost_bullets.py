"""Ghost bullets: the hit head's labels (author 2026-11-22). A ghost changes nothing in the game;
every bullet, real or ghost, reports how its flight ended, once."""

import dataclasses

import torch

from lean.game import core

torch.set_num_threads(1)
DEVICE = torch.device("cpu")


def play(config, ticks, seed=5, envs=32, per_team=3, fire_share=0.3):
    """The same actions and noise for any config; returns per-tick states and outputs."""
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[per_team, per_team]]).expand(envs, 2)
    state = core.reset(config, envs, sizes, generator, DEVICE)
    actions = torch.Generator().manual_seed(seed + 1)
    states, outs = [], []
    for _ in range(ticks):
        n = state.x.shape[1]
        stick = torch.randint(0, 7, (envs, n), generator=actions)
        power = torch.randint(0, 3, (envs, n), generator=actions)
        fire = (torch.rand((envs, n), generator=actions) < fire_share).long()
        noise = core.draw_noise(config, envs, n, actions, DEVICE)
        state, out = core.step(config, state, torch.stack([stick, power, fire], -1), noise)
        states.append(state)
        outs.append(out)
    return states, outs


BASE = core.GameConfig(arena_radius=1500.0, clock_ticks=10**6, spawn="mixed")


def test_ghosts_change_nothing_in_the_game():
    """With the same actions and noise, the game with ghosts equals the game without: ships,
    shields, shots, deaths and who damaged whom, exactly."""
    off_states, off_outs = play(dataclasses.replace(BASE, ghost_bullets=False), 400, fire_share=0.7)
    on_states, on_outs = play(BASE, 400, fire_share=0.7)
    for a, b, oa, ob in zip(off_states, on_states, off_outs, on_outs):
        for name in ("x", "y", "speed", "energy", "shield", "alive", "fired"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        for name in ("damage_taken", "died", "hit_damage"):
            assert torch.equal(getattr(oa, name), getattr(ob, name)), name
    assert sum(int(o.died.sum()) for o in off_outs) > 0  # the comparison includes deaths


def test_every_living_ship_emits_a_bullet_every_tick():
    """With ghosts, a living ship writes a bullet each tick (stamped with that tick); only real
    shots are marked real."""
    states, _ = play(BASE, 15, fire_share=0.0)
    before, last = states[-2], states[-1]
    slot = ((last.next_slot.long() - 1) % last.bullet_x.shape[-1])[..., None]
    stamped = last.bullet_fired_tick.gather(-1, slot)[..., 0]
    assert torch.equal(stamped[before.alive], before.tick[:, None].expand_as(stamped)[before.alive])
    assert not last.bullet_real.any()


def test_each_bullet_reports_its_outcome_once():
    """Every written bullet ends exactly once, within its life, with outcome 0, 1 or 2 and the
    tick it was fired; ghosts' outcomes include hits."""
    states, outs = play(BASE, 60, fire_share=0.0)
    life = states[0].constants["bullet_moves"]
    ended = {}
    for tick, out in enumerate(outs):
        index = torch.nonzero(out.shot_outcome >= 0)
        for e, n, k in index.tolist():
            fired_at = int(out.shot_tick[e, n, k])
            key = (e, n, fired_at)
            assert key not in ended, key  # once
            assert 0 <= tick - fired_at <= life
            ended[key] = int(out.shot_outcome[e, n, k])
    assert set(ended.values()) <= {0, 1, 2}
    assert any(v == 1 for v in ended.values())  # some ghosts would have hit an enemy
    total_damage = sum(float(o.damage_taken.sum()) for o in outs)
    boundary_only = all(not bool(s.fired.any()) for s in states)
    assert boundary_only and total_damage >= 0.0
