"""The fire rule's gates (game spec 6.1): a dead ship fires nothing, and the energy gate
is inclusive. Mutation-sweep items 1 and 2."""

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.physics.bullets import fire
from bnb.game.ship_types import per_ship
from bnb.game.state import replace

TYPES = api.ship_types()


def _ready(alive: bool, energy_over_cost: float):
    """A fresh 2v2 state where ship 0 has its gun cooled down and energy = cost x factor."""
    rules = api.rules_of(GameConfig(2, 3, 1200.0, mode="sandbox"))
    generator = torch.Generator().manual_seed(5)
    state = api.reset(rules, TYPES, api.uniforms(rules, 1, generator=generator))
    ships = state.ships
    cost = per_ship(TYPES.bullet_energy_cost, ships.ship_type)
    ships = replace(
        ships,
        alive=ships.alive.clone().index_fill_(1, torch.tensor([0]), alive),
        gun_cooldown=torch.zeros_like(ships.gun_cooldown),
        energy=torch.where(torch.arange(4) == 0, cost * energy_over_cost, ships.energy),
    )
    spread = torch.zeros(1, rules.num_ships, 2)
    shoot = torch.ones(1, rules.num_ships, dtype=torch.bool)
    return rules, ships, state.bullets, shoot, spread, cost


def test_a_dead_ship_with_a_cool_gun_and_energy_fires_nothing():
    rules, ships, bullets, shoot, spread, _ = _ready(alive=False, energy_over_cost=10.0)
    fired_ships, fired_bullets = fire(rules, TYPES, ships, bullets, shoot, spread)
    assert not bool(fired_ships.fired[0, 0])
    assert torch.equal(fired_ships.energy[0, 0], ships.energy[0, 0])
    assert torch.equal(fired_bullets.moves_left[0, 0], bullets.moves_left[0, 0])
    assert bool(fired_ships.fired[0, 1:].all())  # the living ships beside it do fire


def test_a_ship_with_exactly_the_shot_cost_in_energy_fires():
    rules, ships, bullets, shoot, spread, cost = _ready(alive=True, energy_over_cost=1.0)
    fired_ships, fired_bullets = fire(rules, TYPES, ships, bullets, shoot, spread)
    assert bool(fired_ships.fired[0, 0])
    assert float(fired_ships.energy[0, 0]) == 0.0
    assert int(fired_bullets.moves_left[0, 0, int(ships.next_slot[0, 0])]) == rules.bullet_moves
