"""Ghost-only bullets (author 2026-11-27): every ship, alive or dead, writes one bullet a tick into
its ring of BULLET_SLOTS slots, so a bullet lives exactly BULLET_SLOTS ticks; one uint8 state per
bullet, GHOST (no damage; labels the hit head), REAL or CONTACTED (it touched a ship: no further
contact, damage or label; it flies on). Each bullet reports its outcome once: at its first
contact, or a miss on its last tick."""

import math

import pytest
import torch

from lean.contract import BULLET_SLOTS, CONTACTED, GHOST, POWERS, REAL, STICKS
from lean.game import core
from lean.game.tests.test_deflector import CIRCLE, OFF, SLOT, crossing, run

torch.set_num_threads(1)
DEVICE = torch.device("cpu")
K = BULLET_SLOTS


def play(config, ticks, seed=5, envs=32, per_team=3, fire_share=0.3, dead=()):
    """The same actions and noise for any config, the slots in `dead` killed at the deal;
    returns per tick (state before, state after, out)."""
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[per_team, per_team]]).expand(envs, 2)
    state = core.reset(config, envs, sizes, generator, DEVICE)
    if dead:
        alive = state.alive.clone()
        alive[:, list(dead)] = False
        state = state.replace(alive=alive)
    actions = torch.Generator().manual_seed(seed + 1)
    played = []
    for _ in range(ticks):
        n = state.x.shape[1]
        stick = torch.randint(0, STICKS, (envs, n), generator=actions)
        power = torch.randint(0, POWERS, (envs, n), generator=actions)
        fire = (torch.rand((envs, n), generator=actions) < fire_share).long()
        noise = core.draw_noise(config, envs, n, actions, DEVICE)
        after, out = core.step(config, state, torch.stack([stick, power, fire], -1), noise)
        played.append((state, after, out))
        state = after
    return played


BASE = core.GameConfig(arena_radius=1500.0, clock_ticks=10**6, spawn="mixed")


def test_every_ship_writes_a_bullet_every_tick():
    """Each tick every ship, alive or dead, writes the tick's slot (tick mod K) of its ring: a
    real bullet if it fired, else a ghost; a dead ship's bullets are ghosts. The other slots
    keep their states but for contacts."""
    for before, after, _ in play(BASE, 2 * K + 3, fire_share=0.5, dead=(0, 4)):
        slot = int(before.tick[0]) % K
        assert (before.tick == before.tick[0]).all()
        # (A new bullet may touch a ship on its first tick: then it is CONTACTED.)
        written = after.bullet_state[:, :, slot]
        new = written != CONTACTED
        assert new.float().mean() > 0.9
        assert torch.equal(written[new], torch.where(after.fired, REAL, GHOST)[new].to(torch.uint8))
        assert (~before.alive).any() and (written[~before.alive & new] == GHOST).all()
        others = torch.arange(K) != slot
        kept, now = before.bullet_state[..., others], after.bullet_state[..., others]
        assert (now[now != kept] == CONTACTED).all()
    assert after.fired.any()


def test_each_bullet_reports_its_outcome_once():
    """Every bullet written after the deal reports exactly once: at its first contact (1 or 2),
    or a miss (0) on its last tick (age K - 1), with the tick it was fired; ghosts' outcomes
    include hits."""
    ticks, envs, per_team = 60, 32, 3
    ended = {}
    for before, _, out in play(BASE, ticks, envs=envs, per_team=per_team, fire_share=0.0):
        tick = int(before.tick[0])
        for e, n, k in torch.nonzero(out.shot_outcome >= 0).tolist():
            fired_at = int(out.shot_tick[e, n, k])
            key = (e, n, fired_at)
            assert key not in ended, key  # once
            outcome = int(out.shot_outcome[e, n, k])
            assert 0 <= tick - fired_at <= K - 1
            assert outcome != 0 or tick - fired_at == K - 1  # a miss on its last tick only
            ended[key] = outcome
    # Every bullet whose last tick fell inside the run reported.
    whole = sum(1 for (_, _, fired_at) in ended if fired_at <= ticks - K)
    assert whole == envs * 2 * per_team * (ticks - K + 1)
    assert set(ended.values()) <= {0, 1, 2}
    assert any(v == 1 for v in ended.values())  # some ghosts would have hit an enemy


def test_start_of_episode_slots_produce_no_labels():
    """A fresh deal's slots are CONTACTED: a ship sitting on them (the origin) gets no contact
    from them in their K ticks, no label and no damage. As ghosts the same slots would label
    it (the control)."""
    fresh = crossing(OFF, [0.0], [0.0], target_speed=1.0)
    fresh = fresh.replace(
        bullet_state=torch.full_like(fresh.bullet_state, CONTACTED),
        tick=torch.zeros(1, dtype=torch.long),
        x=torch.tensor([[-3000.0, 0.5]]),  # the target on the shooter's parked bullets
    )
    for _, out in run(OFF, fresh, K):
        assert (out.shot_outcome[out.shot_tick < 0] == -1).all()
        assert (out.hit_damage == 0.0).all()
    ghosts = fresh.replace(bullet_state=torch.full_like(fresh.bullet_state, GHOST))
    ((_, out),) = run(OFF, ghosts)
    assert int((out.shot_outcome[0, 0] == 1).sum()) == K - 1  # every old slot, not the new one


def _hitting_bullet(config, state_value):
    """A slow bullet of the shooter at the nearly still target's centre, from astern (beyond the
    nose armour): it is within the hit radius on every tick of its life. Fired on tick SLOT; the
    state's tick is SLOT + 1."""
    state = crossing(config, [0.0], [math.pi], start_x=-2.0, target_speed=1.0)
    bullets = state.bullets.clone()
    bullets[:, core.BULLET_SPEED, 0, SLOT] = 1.0
    bullet_state = state.bullet_state.clone()
    bullet_state[:, 0, SLOT] = state_value
    return state.replace(bullets=bullets, bullet_state=bullet_state)


@pytest.mark.parametrize("config", [OFF, CIRCLE], ids=["off", "circle"])
def test_a_contacted_bullet_never_damages_or_labels_again_and_flies_on(config):
    """A real bullet inside the target's hit radius on every tick hits once (label 1, damage,
    state CONTACTED), then never damages or labels again and flies on to its K-th tick with its
    motion unchanged (the same block as a bullet contacted from the start); on the tick after
    its last its slot is rewritten."""
    real = run(config, _hitting_bullet(config, REAL), K)
    contacted = run(config, _hitting_bullet(config, CONTACTED), K)
    first, out = real[0]
    assert int(out.shot_outcome[0, 0, SLOT]) == 1 and float(out.hit_damage[0, 0, 1]) > 0.0
    assert int(out.shot_tick[0, 0, SLOT]) == SLOT
    # Run tick i is the bullet's age i + 1: its last (age K - 1) is i = K - 2.
    x = []
    for i, ((state, out), (other, other_out)) in enumerate(zip(real[: K - 1], contacted)):
        assert int(state.bullet_state[0, 0, SLOT]) == CONTACTED
        assert torch.allclose(state.bullets[..., 0, SLOT], other.bullets[..., 0, SLOT])
        x.append(float(state.bullets[0, core.BULLET_X, 0, SLOT]))
        if i > 0:
            assert int(out.shot_outcome[0, 0, SLOT]) == -1
            assert float(out.hit_damage[0, 0, 1]) == 0.0
        assert int(other_out.shot_outcome[0, 0, SLOT]) == -1
        assert float(other_out.hit_damage[0, 0, 1]) == 0.0
    assert all(b > a for a, b in zip(x, x[1:]))  # flying on
    overwritten, _ = real[K - 1]  # tick SLOT + K: the slot's next bullet, a ghost at the shooter
    assert int(overwritten.bullet_state[0, 0, SLOT]) == GHOST
    assert float(overwritten.bullets[0, core.BULLET_X, 0, SLOT]) < -2000.0
