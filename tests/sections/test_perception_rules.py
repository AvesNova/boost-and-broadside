"""Perception's rules the reference fixtures do not reach (game spec 10): the fog-off
counter, the reset flag, the privileged rows and the respawn wait's unit. Mutation-sweep
items 3 to 6."""

import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game import api
from bnb.game.state import replace

TYPES = api.ship_types()
CONFIG = GameConfig(2, 3, 3000.0, mode="sandbox")


def _state(fogged: bool, seed: int = 1):
    rules = api.rules_of(CONFIG)
    generator = torch.Generator().manual_seed(seed)
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 1, generator=generator), torch.tensor([fogged])
    )
    return rules, state, generator


def _idle(rules, state, generator):
    """One tick with every ship coasting and holding fire."""
    action = Action.unpack(torch.zeros(1, rules.num_ships, 3, dtype=torch.int32))
    state, _ = api.step(
        rules,
        TYPES,
        state,
        action,
        api.spread(rules, 1, generator=generator),
        api.uniforms(rules, 1, generator=generator),
    )
    return state


def _kill(state, ship: int, wait: int = 1000):
    ships = state.ships
    dead = torch.arange(ships.x.shape[1]) == ship
    return replace(
        state,
        ships=replace(
            ships,
            alive=ships.alive & ~dead,
            respawn_wait=torch.where(dead, wait, ships.respawn_wait).to(ships.respawn_wait.dtype),
        ),
    )


def test_without_fog_no_ship_goes_unseen_not_even_a_dead_one():
    rules, state, generator = _state(fogged=False)
    state = _kill(state, ship=3)
    for _ in range(4):
        state, _, _ = api.perceive(rules, state)
        (ticks_since_seen,) = api.perceived(state)
        assert int(ticks_since_seen.abs().max()) == 0
        state = _kill(_idle(rules, state, generator), ship=3)


def test_the_reset_flag_is_one_on_an_episodes_first_observation_only():
    rules, state, generator = _state(fogged=True)
    _, observation, _ = api.perceive(rules, state)
    assert torch.all(observation.globals[..., GLOBAL["reset"]] == 1.0)
    state = _idle(rules, state, generator)
    _, observation, _ = api.perceive(rules, state)
    assert torch.all(observation.globals[..., GLOBAL["reset"]] == 0.0)


def _separated(fogged: bool):
    """Team 0 at the west edge, team 1 at the east edge (beyond any vision range), ship 3
    dead: from team 0's view ship 2 is a hidden living enemy and ship 3 a dead one."""
    rules, state, generator = _state(fogged)
    state = _idle(rules, state, generator)  # clear the spawn tick's always-visible flag
    ships = state.ships
    x = torch.tensor([[-2500.0, -2450.0, 2500.0, 2450.0]])
    y = torch.tensor([[0.0, 30.0, 0.0, 30.0]])
    state = replace(state, ships=replace(ships, x=x, y=y))
    return rules, _kill(state, ship=3)


def test_the_privileged_rows_show_every_ship_as_seen_where_it_truly_is():
    rules, state = _separated(fogged=True)
    for _ in range(3):  # let the hidden ships' counters rise
        state, observation, privileged = api.perceive(rules, state)
    team0 = observation.ships[0, 0]  # rows: own ships 0, 1, then enemies 2, 3
    assert float(team0[2, SHIP["visible"]]) == 0.0 and float(team0[2, SHIP["alive"]]) == 1.0
    assert float(team0[3, SHIP["alive"]]) == 0.0
    assert float(team0[2, SHIP["time_since_seen"]]) > 0.0  # the fogged view counts
    truth = privileged.ships[0]  # (T, N, columns) by row
    order = privileged.view_order[0]  # (T, N) sim index of each row
    assert torch.all(truth[..., SHIP["visible"]] == 1.0)
    assert torch.all(truth[..., SHIP["time_since_seen"]] == 0.0)
    torch.testing.assert_close(truth[..., SHIP["x"]], state.ships.x[0][order], rtol=0, atol=0)
    torch.testing.assert_close(truth[..., SHIP["y"]], state.ships.y[0][order], rtol=0, atol=0)


def test_the_observed_respawn_wait_is_in_seconds():
    rules, state, _ = _state(fogged=False)
    wait = 37
    state = _kill(state, ship=1, wait=wait)
    _, observation, _ = api.perceive(rules, state)
    shown = float(observation.ships[0, 0, 1, SHIP["respawn_wait"]])  # own row 1 = ship 1
    assert abs(shown - wait * rules.tick_length) < 1e-6
