"""The game api's display functions (`bnb.game.api`, for the play shell): each is the
game's own rule, read through one public name."""

import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api, ship_types
from bnb.game import rules as game_rules
from bnb.game.modes import frontline
from bnb.game.modes import rules as mode_rules
from bnb.game.modes.phase import protected
from bnb.game.perception import sight
from bnb.game.physics.medium import medium
from bnb.game.state import replace


def _state(config: GameConfig, ticks: int = 0, seed: int = 0, fogged: bool = True):
    generator = torch.Generator().manual_seed(seed)
    rules, types = api.rules_of(config), api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, 1, generator=generator),
                      torch.tensor([fogged]))
    action = torch.tensor([[1, 2, 1]] * rules.num_ships, dtype=torch.int32)
    for _ in range(ticks):
        state, _ = api.step(rules, types, state, Action.unpack(action[None]),
                            api.spread(rules, 1, generator=generator),
                            api.uniforms(rules, 1, generator=generator))
    state, observation, _ = api.perceive(rules, state)
    return rules, state, observation


def test_log_index_is_the_medium_at_the_points():
    rules, state, _ = _state(api.PRESETS["frontline"])
    generator = torch.Generator().manual_seed(1)
    x, y = (torch.rand((2, 1, 4000), generator=generator) * 2 - 1) * 2600.0
    fields = state.fields
    got = api.log_index(fields.x, fields.y, fields.radius, fields.band_width, fields.level, x, y)
    assert torch.equal(got, medium(fields, x, y)[0])
    assert (got != 0).any()


def test_core_radius_is_where_the_sight_test_blocks():
    """An observer far off looks at an enemy on the line through a field's centre: the
    enemy is hidden just inside the core radius and seen just outside it."""
    rules, state, _ = _state(GameConfig(1, 1, 2600.0))
    fields = replace(state.fields, x=torch.tensor([[0.0]]), y=torch.tensor([[0.0]]),
                     radius=torch.tensor([[300.0]]), band_width=torch.tensor([[40.0]]))
    state = replace(state, fields=fields)
    core = float(api.core_radius(state)[0, 0])
    for offset, seen in ((-0.5, False), (0.5, True)):
        ships = replace(state.ships, x=torch.tensor([[900.0, core + offset]]),
                        y=torch.zeros(1, 2), alive=torch.ones(1, 2, dtype=torch.bool),
                        fired=torch.zeros(1, 2, dtype=torch.bool),
                        spawned=torch.zeros(1, 2, dtype=torch.bool))
        visible = sight.visibility(replace(state, ships=ships))
        assert bool(visible[0, 0, 1]) is seen, offset


def test_a_lone_attacker_raises_only_the_meters_zone_sides_name():
    """For every front, one ship alone in one zone: the meter rises exactly where
    `zone_sides` names its team the attacker, and the defense zone's owner is the other
    team."""
    rules, state, _ = _state(api.PRESETS["frontline"])
    Z = frontline.ZONES
    for front in range(-2, 3):
        role = frontline.roles_from_front(torch.tensor([front]))  # (1, Z)
        owner, attacker = api.zone_sides(replace(state, mode=replace(state.mode, role=role)))
        for team in (0, 1):
            for zone in range(Z):
                inside = torch.zeros(1, 2, Z, dtype=torch.bool)
                inside[0, team, zone] = True
                capture = frontline.capture_step(
                    torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool), inside, role,
                    torch.zeros(1, Z), torch.zeros(1, Z, 2), torch.zeros(1, Z, 2),
                    torch.tensor([front]), rules.tick_length, 10.0)
                rises = bool(capture.progress[0, zone] > 0)
                assert rises == (int(attacker[0, zone]) == team), (front, team, zone)
                if int(attacker[0, zone]) >= 0:
                    assert int(owner[0, zone]) == 1 - int(attacker[0, zone])


def test_a_spawn_zones_owner_is_the_team_it_protects():
    """A ship inside a spawn zone takes no damage there exactly when the zone is its
    team's (`zone_sides`' owner); the neutral zone has no owner."""
    rules, state, _ = _state(api.PRESETS["frontline"])
    owner, attacker = api.zone_sides(state)
    mode = state.mode
    spawn_zones = (attacker[0] < 0) & (owner[0] >= 0)
    assert int(spawn_zones.sum()) == 2 and int((owner[0] < 0).sum()) == 1
    for zone in spawn_zones.nonzero()[:, 0].tolist():
        ships = replace(state.ships, x=mode.zone_x[:, zone, None].expand_as(state.ships.x),
                        y=mode.zone_y[:, zone, None].expand_as(state.ships.y))
        safe = protected(rules, rules.pools[0].mode, mode, ships)[0]
        team = torch.arange(rules.num_ships) // rules.ships_per_team
        assert torch.equal(safe, team == int(owner[0, zone])), zone


def test_visible_is_the_sight_test_in_sim_order():
    rules, state, observation = _state(GameConfig(4, 12, 2600.0), ticks=60, seed=2)
    got = api.visible(rules, observation)
    assert torch.equal(got, sight.visibility(state))
    assert not got.all()  # the scenario hides someone


def test_the_reexports_are_the_games_own():
    assert api.vision_range is sight.vision_range
    assert api.per_ship is ship_types.per_ship
    assert api.PRESETS is mode_rules.PRESETS
    assert (api.TEAMS, api.LN_SQRT2, api.MAP_SCALE_RADIUS) == (
        game_rules.TEAMS, game_rules.LN_SQRT2, game_rules.MAP_SCALE_RADIUS)
