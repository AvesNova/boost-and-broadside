"""The game's rules from the author review (docs/rewrite/core-decisions.md): the clock's
tie-break, the spawn rules, the training spawn and the one-tick action delay. Each test names
the rule it guards."""

import pytest
import torch

from lean.contract import NO_PENDING, join_action
from lean.game import GameConfig, card, core, draw_noise, observe, reset, step

torch.set_num_threads(1)

CPU = torch.device("cpu")
PLAIN = GameConfig(field_count=0, fog=False)


def neutral(state):
    return torch.zeros(state.x.shape + (3,), dtype=torch.long)


def no_noise(state):
    return {"spread": torch.zeros(state.x.shape + (2,))}


def at_the_clock(alive, shield):
    """One environment of 2v2 on its last tick, far apart and inside the arena, with `alive`
    and `shield` per slot (team 0 in slots 0 and 1)."""
    state = reset(PLAIN, 1, torch.tensor([[2, 2]]), torch.Generator().manual_seed(0), CPU)
    return core.with_velocity(state.replace(
        x=torch.tensor([[-2000.0, -2000.0, 2000.0, 2000.0]]),
        y=torch.tensor([[-500.0, 500.0, -500.0, 500.0]]),
        heading_cos=torch.ones(1, 4), heading_sin=torch.zeros(1, 4),
        alive=torch.tensor([alive]), shield=torch.tensor([shield]),
        tick=torch.full((1,), PLAIN.clock_ticks - 1),
    ))


@pytest.mark.parametrize(
    "alive, shield, result",
    [
        ([True, True, True, False], [10.0, 10.0, 100.0, 0.0], 0),  # more living ships wins, whatever the shield
        ([True, False, True, True], [100.0, 0.0, 10.0, 10.0], 1),
        ([True, True, True, True], [60.0, 30.0, 40.0, 40.0], 0),  # equal ships: more total shield wins
        ([True, True, True, True], [20.0, 30.0, 40.0, 40.0], 1),
        ([True, True, True, True], [30.0, 50.0, 50.0, 30.0], 2),  # equal on both: a draw
        ([True, False, True, False], [50.0, 90.0, 50.0, 0.0], 2),  # a dead ship's shield does not count
    ],
)
def test_the_clock_decides_by_living_ships_then_shield_then_a_draw(alive, shield, result):
    """Rule (2.3): at the clock more living ships wins; equal living ships, more total shield
    over the living ships wins; equal on both, a draw."""
    state = at_the_clock(alive, shield)
    _, out = step(PLAIN, state, neutral(state), no_noise(state))
    assert bool(out.done[0]) and int(out.result[0]) == result


def test_a_wipe_decides_before_the_clock():
    """Rule: a wipe ends the game at once, and the side with ships left wins (both wiped: a
    draw), however the clock's tie-break would have gone."""
    state = at_the_clock([True, True, False, False], [1.0, 1.0, 100.0, 100.0])
    state = state.replace(tick=torch.zeros(1, dtype=torch.long))
    _, out = step(PLAIN, state, neutral(state), no_noise(state))
    assert bool(out.done[0]) and int(out.result[0]) == 0
    state = state.replace(alive=torch.zeros(1, 4, dtype=torch.bool))
    _, out = step(PLAIN, state, neutral(state), no_noise(state))
    assert bool(out.done[0]) and int(out.result[0]) == 2


@pytest.mark.parametrize("spawn", ["mixed", "sides"])
def test_no_ship_spawns_near_the_edge_facing_outward(spawn):
    """Rule (4.1): under either spawn rule no ship within the edge band starts with its nose
    pointing out of the arena; every ship starts inside the spawn disc."""
    config = GameConfig(spawn=spawn, field_count=0, fog=False)
    state = reset(config, 512, torch.tensor([[8, 8]]).expand(512, 2), torch.Generator().manual_seed(1), CPU)
    distance = torch.sqrt(state.x**2 + state.y**2)
    assert (distance <= card.SPAWN_AREA_FRACTION * config.arena_radius + 1e-3).all()
    outward = (state.heading_cos * state.x + state.heading_sin * state.y) / distance.clamp_min(1e-6)
    near = distance > config.arena_radius - card.SPAWN_EDGE_BAND
    assert near.sum() > 500  # the rule is exercised
    assert (outward[near] <= 1e-5).all()
    assert torch.allclose(state.heading_cos**2 + state.heading_sin**2, torch.ones(()), atol=1e-5)


def test_sides_spreads_each_team_over_its_half_disc():
    """Rule (4.1): "sides" puts each team over most of its half of the arena (not a small zone),
    with the dividing strip between the teams; noses point toward the other side."""
    config = GameConfig(spawn="sides", field_count=0, fog=False)
    envs, per_team = 256, 16
    state = reset(config, envs, torch.tensor([[per_team, per_team]]).expand(envs, 2), torch.Generator().manual_seed(2), CPU)
    position = torch.stack([state.x, state.y], -1)
    first, second = position[:, :per_team], position[:, per_team:]
    gap = card.SIDE_GAP_FRACTION * config.arena_radius
    assert (torch.cdist(first, second) >= 2.0 * gap - 1e-2).all()  # the strip between the halves
    # Spread: a half disc of radius 0.8 R has a mean distance of about 0.38 R from its centroid;
    # the old zone (radius 0.15 R) had 0.1 R.
    for team in (first, second):
        spread = (team - team.mean(1, keepdim=True)).norm(dim=-1).mean()
        assert spread > 0.3 * config.arena_radius
    toward = second.mean(1) - first.mean(1)  # (E, 2), from team 0's side to team 1's
    heading = torch.stack([state.heading_cos, state.heading_sin], -1)
    assert ((heading[:, :per_team] * toward[:, None]).sum(-1) > 0.0).float().mean() > 0.95
    assert ((heading[:, per_team:] * toward[:, None]).sum(-1) < 0.0).float().mean() > 0.95


def standard_rows(config, state):
    """(E, N) bool: the ship has the standard spawn's speed, store and shield."""
    constants = state.constants
    speed = card.SPAWN_SPEED * torch.exp(state.log_index)
    return (
        torch.isclose(state.speed, speed)
        & torch.isclose(state.energy, torch.as_tensor(constants["spawn_energy"]))
        & torch.isclose(state.shield, torch.as_tensor(constants["max_shield"]))
    )


def test_the_training_spawn_draws_speed_shield_and_store_with_one_eighth_standard():
    """Rule (4.4 and its follow-up): with the training spawn each ship's proper speed is uniform
    from its stall speed to its top level speed and its shield and store uniform from 0 to full;
    one eighth of the environments (drawn per environment) deal the standard spawn. Off, every
    ship has the standard spawn."""
    envs = 4096
    sizes = torch.tensor([[2, 2]]).expand(envs, 2)
    config = GameConfig(field_count=0, fog=False, training_spawn=True)
    state = reset(config, envs, sizes, torch.Generator().manual_seed(3), CPU)
    standard = standard_rows(config, state)
    whole = standard.all(-1)
    assert torch.equal(standard.any(-1), whole)  # standard per environment, not per ship
    share = whole.float().mean()
    assert abs(float(share) - 1.0 / 8.0) < 0.02  # 4 standard errors at 4,096 environments
    c = state.constants
    drawn = ~whole
    for value, low, high in (
        (state.speed, c["stall_speed"], c["cruise_speed"]),
        (state.shield, 0.0, c["max_shield"]),
        (state.energy, 0.0, c["max_energy"]),
    ):
        fraction = (value[drawn] - low) / (high - low)
        assert (fraction >= 0.0).all() and (fraction <= 1.0 + 1e-6).all()
        assert abs(float(fraction.mean()) - 0.5) < 0.02
        assert abs(float((fraction < 0.25).float().mean()) - 0.25) < 0.02
    assert 80.0 < c["stall_speed"] < c["cruise_speed"]
    plain = reset(GameConfig(field_count=0, fog=False), envs, sizes, torch.Generator().manual_seed(3), CPU)
    assert standard_rows(config, plain).all()


def test_an_action_takes_effect_exactly_one_tick_later():
    """Rule (3.3): the action chosen on tick t is applied on tick t + 1; a deal's pending action
    is neutral (straight, coast, no fire)."""
    state = at_the_clock([True, True, True, True], [100.0] * 4).replace(tick=torch.zeros(1, dtype=torch.long))
    assert (state.pending == 0).all()
    chosen = neutral(state)
    chosen[0, 0] = torch.tensor([5, 2, 1])  # full stick, climb, fire
    first, _ = step(PLAIN, state, chosen, no_noise(state))
    calm, _ = step(PLAIN, state, neutral(state), no_noise(state))
    assert torch.equal(first.ships, calm.ships) and not first.fired.any()  # nothing applied yet
    assert torch.equal(first.pending, chosen)
    second, _ = step(PLAIN, first, neutral(first), no_noise(first))
    applied, _ = step(PLAIN, first.replace(pending=chosen), torch.zeros_like(chosen), no_noise(first))
    later, _ = step(PLAIN, calm, neutral(calm), no_noise(calm))
    assert torch.equal(second.ships, applied.ships) and second.fired[0].tolist() == [True, False, False, False]
    assert not torch.allclose(second.nose_rate[0, 0], later.nose_rate[0, 0])  # the turn did happen
    assert (second.pending == 0).all()


@pytest.mark.parametrize("fog", [False, True])
def test_the_observation_shows_own_pending_actions_only(fog):
    """Rule (3.3): each own living ship's row carries its pending action (joint index); an enemy
    row, a dead ship and an absent slot carry none (NO_PENDING)."""
    config = GameConfig(field_count=0, fog=fog)
    state = reset(config, 1, torch.tensor([[2, 2]]), torch.Generator().manual_seed(4), CPU)
    chosen = torch.tensor([[[5, 2, 1], [3, 1, 0], [1, 0, 1], [6, 2, 0]]])
    state, _ = step(config, state, chosen, draw_noise(config, 1, 4, torch.Generator().manual_seed(5), CPU))
    joint = join_action(chosen)[0].tolist()
    assert observe(config, state, 0).pending[0].tolist() == [joint[0], joint[1], NO_PENDING, NO_PENDING]
    assert observe(config, state, 1).pending[0].tolist() == [NO_PENDING, NO_PENDING, joint[2], joint[3]]
    dead = state.replace(alive=torch.tensor([[True, False, True, True]]))
    assert observe(config, dead, 0).pending[0].tolist() == [joint[0], NO_PENDING, NO_PENDING, NO_PENDING]
    # A re-deal (1v1 in four slots): the pending action is neutral again, absent slots have none.
    fresh = core.deal(config, 1, 4, torch.tensor([[1, 1]]), torch.Generator().manual_seed(6), CPU)
    padded = core.take_deal(state, torch.ones(1, dtype=torch.bool), fresh)
    assert observe(config, padded, 0).pending[0].tolist() == [0, NO_PENDING, NO_PENDING, NO_PENDING]
