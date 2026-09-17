"""Frontline shield lifecycle at physics-tick granularity."""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ZoneRole
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import (
    apply_frontline_tick,
    friendly_spawn_mask,
    frontline_ship_config,
)
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.env.physics import _apply_combat_damage
from boost_and_broadside.env.rewards import LocalDamageDealtEnemyReward, ShieldRechargeReward
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.train.rl.features import AttitudeFourier, Fourier


def env():
    ship = replace(frontline_ship_config(SHIP_CONFIG), bullet_min_damage_frac=1.0)
    result = TensorEnv(1, ship, replace(PLAY_ENV_CONFIG, num_fields=0), "cpu")
    result.reset(seed=4)
    result.state.ship_team_id[:] = torch.tensor([[0] * 5 + [1] * 5])
    result.state.ship_pos[:] = result.state.map_center[:, None] + torch.arange(10) * 100
    result.state.ship_health[:] = 15
    return result


def hit(e, owners=(5,), target=0):
    s = e.state
    s.bullet_active.zero_()
    for owner in owners:
        s.bullet_pos[0, owner, 0] = s.ship_pos[0, target]
        s.bullet_active[0, owner, 0] = True
    _apply_combat_damage(s, e.ship_config, frontline=e.env_config.frontline)


def tick(e):
    apply_frontline_tick(e.state, e.env_config.frontline, e.ship_config)


def test_simultaneous_overkill_breaks_shield_but_next_tick_kills():
    e = env()
    hit(e, (5, 6, 7))
    assert e.state.ship_health[0, 0] == 0
    assert not e.state.ship_combat_death[0, 0]
    payout = LocalDamageDealtEnemyReward(1).compute(e.state, None, e.state, None)
    assert payout.sum() == 15
    tick(e)
    assert not e.state.ship_respawned.any()
    hit(e)
    tick(e)
    assert e.state.ship_respawned[0, 0]
    assert e.state.ship_health[0, 0] == e.env_config.frontline.respawn_health
    assert e.state.ship_power[0, 0] == e.env_config.frontline.respawn_power
    assert e.state.ship_vel[0, 0].abs() == pytest.approx(e.env_config.frontline.respawn_speed)
    assert e.state.ship_shield_delay[0, 0] == e.env_config.frontline.shield_recharge_delay
    assert e.state.ship_alive.all()


def test_friendly_hit_cannot_finish_depleted_shield():
    e = env()
    e.state.ship_health[0, 0] = 0
    hit(e, (1,))
    tick(e)
    assert not e.state.ship_respawned.any()
    assert e.state.ship_shield_delay[0, 0] == e.env_config.frontline.shield_recharge_delay


def test_own_spawn_blocks_hits_and_recharge_delay_reset():
    e = env()
    s = e.state
    i = (s.zone_roles[0] == int(ZoneRole.TEAM0_SPAWN)).nonzero()[0, 0]
    s.ship_pos[0, 0] = s.zone_pos[0, i]
    s.ship_shield_delay[0, 0] = 0
    assert friendly_spawn_mask(s, e.ship_config)[0, 0]
    hit(e, (5, 6))
    tick(e)
    assert s.damage_matrix.sum() == 0
    assert s.ship_health[0, 0] > 15
    assert s.ship_shield_delay[0, 0] == 0


def test_delay_recharge_clamp_and_zero_sum_reward():
    e = env()
    s = e.state
    s.ship_shield_delay[:] = e.ship_config.dt * 2
    tick(e)
    tick(e)
    assert s.ship_shield_recharge.sum() == 0
    tick(e)
    assert torch.all(s.ship_shield_recharge > 0)
    s.ship_health[:] = 99.9
    s.ship_shield_delay.zero_()
    tick(e)
    assert torch.all(s.ship_health == 100)
    r = ShieldRechargeReward(1).compute(s, None, s, None)
    assert r.sum().abs() < 1e-5
    tick(e)
    assert s.ship_shield_recharge.sum() == 0


def test_recharge_timer_is_observed_and_fourier_encodes_phase():
    e = env()
    obs = observation_from_state(e.state, e.ship_config)
    assert torch.equal(obs[ObsKey.SHIELD_DELAY][:, :10, 0], e.state.ship_shield_delay)
    theta = torch.tensor([[-math.pi, 0.4, math.pi]]).unsqueeze(-1)
    cart = torch.cat([theta.cos(), theta.sin()], -1)
    assert torch.allclose(AttitudeFourier()(cart), Fourier(4, 2 * math.pi)(theta), atol=2e-6)
    assert AttitudeFourier().out_dim(2) == 8


@pytest.mark.parametrize('owners', [(5, 6), (1, 5), (1, 2)])
def test_damage_and_recovery_are_zero_sum_at_final_ratio(owners):
    from boost_and_broadside.config.defaults import REWARDS
    from boost_and_broadside.env.rewards import build_reward_components
    e = env()
    hit(e, owners)
    tick(e)
    cfg = replace(REWARDS, kill_payout_ratio=1, damage_payout_ratio=1, capture_payout_ratio=1)
    components = build_reward_components(cfg, e.ship_config)
    total = sum(c.weight * c.compute(e.state, torch.zeros(1,10,3), e.state, torch.zeros(1,dtype=torch.bool))
                for c in components if c.weight)
    assert total.sum().abs() < 1e-5


def test_boundary_cannot_finish_a_shield_broken_this_tick():
    e = env()
    e.state.ship_pos[0,0] = e.state.map_center[0] + e.env_config.frontline.playable_radius + 20
    hit(e, (5,6))
    tick(e)
    assert e.state.ship_health[0,0] == 0
    assert not e.state.ship_respawned[0,0]
    e.state.ship_combat_damage.zero_()
    e.state.damage_matrix.zero_()
    tick(e)
    assert e.state.ship_respawned[0,0]


def test_initial_depletion_matches_respawn_and_masked_reset():
    e = env()
    e.reset(seed=14)
    cfg = e.env_config.frontline
    assert torch.all(e.state.ship_health == cfg.respawn_health)
    assert torch.all(e.state.ship_power == cfg.respawn_power)
    assert torch.all(e.state.ship_shield_delay == cfg.shield_recharge_delay)
    assert torch.allclose(e.state.ship_vel.abs()*e.state.ship_local_index,
                          torch.full_like(e.state.ship_health,cfg.respawn_speed))


def test_curriculum_has_a_long_final_zero_sum_phase():
    from boost_and_broadside.profiles import PROFILES
    schedule = PROFILES['rl'].schedule_spec.compile()
    assert schedule.offensive_bias(0) == 1
    assert schedule.offensive_bias(50_000_000) == 1
    assert schedule.offensive_bias(175_000_000) == pytest.approx(0.5)
    for step in (300_000_000,400_000_000,500_000_000):
        assert schedule.offensive_bias(step) == 0
        assert schedule.shaping_scale(step) == 0
    assert PROFILES['rl'].total_timesteps - 300_000_000 >= 200_000_000


def test_recovery_is_zero_sum_even_with_unequal_teams():
    e=env()
    e.state.ship_team_id[0,4]=1
    e.state.ship_shield_recharge[0,0]=3
    reward=ShieldRechargeReward(1).compute(e.state,None,e.state,None)
    assert reward[0,0] == 3
    assert reward.sum().abs()<1e-6
