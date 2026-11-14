"""The cheese agents: valid actions in every mode, only visible enemies read, deterministic,
their one-trick behaviours on constructed observations, and a short game of each per mode."""

import math

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship, zone

from bnb.contracts import Action, GameConfig, Obs, Pool
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted import Scripted, gun_range
from bnb.controllers.scripted_cheese import NAMES, Cheese, agent_of, player_name
from bnb.game import api
from bnb.runtime.rollout import Rollout

KEY = torch.tensor([12345])
BOOST, STRAIGHT = 1, 0
ZONES = [  # the viewer's ring: own spawn, own defense, neutral, enemy defense, enemy spawn
    zone(-2000.0, 0.0, 0),
    zone(-1000.0, 0.0, 1),
    zone(0.0, 800.0, 2),
    zone(1000.0, 0.0, 3),
    zone(2000.0, 0.0, 4),
]


def _decide(name: str, view: Obs) -> Action:
    return Cheese(CHARACTERISTICS, 2, name).step(None, view, None, KEY)[0]


def _views() -> dict[str, Obs]:
    """One view per mode: Frontline (zones), team deathmatch, elimination (the boundary
    shrunk to half)."""
    own = [ship(x=-200.0, y=50.0), ship(x=-250.0, y=-60.0, heading=2.0)]
    foes = [ship(x=300.0, y=0.0, team_relative=1, heading=3.0), absent()]
    elimination = observation(own, foes)
    elimination.globals[:, GLOBAL["mode_team_deathmatch"]] = 0.0
    elimination.globals[:, GLOBAL["mode_elimination"]] = 1.0
    elimination.globals[:, GLOBAL["boundary_fraction"]] = 0.5
    return {
        "frontline": observation(own, foes, ZONES),
        "team_deathmatch": observation(own, foes),
        "elimination": elimination,
    }


@pytest.mark.parametrize("name", NAMES)
def test_every_agent_plays_valid_actions_in_every_mode(name):
    for mode, view in _views().items():
        action = _decide(name, view)
        for part, count in (
            (action.power, NUM_POWER),
            (action.stick, NUM_STICK),
            (action.shoot, NUM_SHOOT),
        ):
            assert part.shape == (1, 2), (name, mode)
            assert part.dtype == torch.int32
            assert int(part.min()) >= 0 and int(part.max()) < count


@pytest.mark.parametrize("name", NAMES)
def test_a_dead_ship_takes_the_no_op_and_no_enemy_means_no_shot(name):
    view = observation([ship(alive=False), ship()], [absent(), absent()], ZONES)
    action = _decide(name, view)
    assert action.packed()[0, 0].tolist() == [0, 0, 0]
    assert int(action.shoot[0, 1]) == 0


@pytest.mark.parametrize("name", NAMES)
def test_a_hidden_or_dead_enemy_does_not_change_the_action(name):
    """A hidden enemy row's position may be anything (here a nonzero one, which a fogged
    observation never shows): the action is the same as with it absent. So is a dead enemy's."""
    own = [ship(x=-200.0, y=50.0), ship(x=-250.0, y=-60.0)]
    seen = ship(x=400.0, y=-100.0, team_relative=1)
    baseline = _decide(name, observation(own, [seen, absent()], ZONES)).packed()
    for x, y in ((100.0, 10.0), (-500.0, 900.0), (0.0, 0.0)):
        hidden = ship(x=x, y=y, team_relative=1)
        hidden[SHIP["visible"]] = 0.0
        hidden[SHIP["x"]], hidden[SHIP["y"]] = x, y
        view = observation(own, [seen, hidden], ZONES)
        assert torch.equal(_decide(name, view).packed(), baseline)
    dead = ship(x=-210.0, y=50.0, team_relative=1, alive=False)
    dead[SHIP["visible"]] = 1.0
    dead[SHIP["x"]], dead[SHIP["y"]] = -210.0, 50.0
    alone = _decide(name, observation(own, [absent(), absent()], ZONES)).packed()
    assert torch.equal(_decide(name, observation(own, [dead, absent()], ZONES)).packed(), alone)


@pytest.mark.parametrize("name", NAMES)
def test_the_decision_is_deterministic_and_draws_nothing(name):
    view = _views()["frontline"]
    first = _decide(name, view).packed()
    assert torch.equal(first, _decide(name, view).packed())
    other = Cheese(CHARACTERISTICS, 2, name).step(None, view, None, torch.tensor([7]))[0]
    assert torch.equal(first, other.packed())


def test_a_per_view_kind_tensor_plays_each_views_agent():
    view = _views()["frontline"]
    count = len(NAMES) + 1
    views = Obs(
        ships=view.ships.repeat(count, 1, 1),
        fields=None,
        globals=view.globals.repeat(count, 1),
        zones=view.zones.repeat(count, 1, 1),
    )
    kinds = torch.arange(-1, len(NAMES))
    action = Cheese(CHARACTERISTICS, 2, kinds).step(None, views, None, KEY.repeat(count))[0]
    assert action.packed()[0].abs().sum() == 0  # kind -1: the no-op
    for index, name in enumerate(NAMES):
        assert torch.equal(action.packed()[index + 1], _decide(name, view).packed()[0])


def test_names():
    assert agent_of(player_name("kiter")) == "kiter"
    assert agent_of("scripted") is None and agent_of("scripted:nobody") is None
    with pytest.raises(ValueError):
        Cheese(CHARACTERISTICS, 2, "nobody")


# ---- behaviours


def test_the_zone_rusher_boosts_to_the_enemy_defense_zone_and_never_fires():
    view = observation(
        [ship(x=-1500.0, y=0.0), ship(x=-1500.0, y=100.0, heading=math.pi)],
        [ship(x=-1450.0, y=0.0, team_relative=1), absent()],
        ZONES,
    )
    action = _decide("zone_rusher", view)
    assert action.power[0].tolist() == [BOOST, BOOST]
    assert action.stick[0, 0] == STRAIGHT and action.stick[0, 1] != STRAIGHT  # faces away: turns
    assert int(action.shoot.sum()) == 0
    inside = observation([ship(x=1000.0), ship(x=1000.0)], [absent(), absent()], ZONES)
    assert _decide("zone_rusher", inside).power[0].tolist() == [0, 0]  # orbits inside, coasting


def test_the_turtle_fires_only_at_enemies_near_its_zone():
    own = [
        ship(x=-1000.0, y=-200.0, heading=math.pi / 2),
        ship(x=-2000.0, y=-200.0, heading=math.pi / 2),
    ]
    near = observation(own, [ship(x=-1000.0, y=-50.0, team_relative=1), absent()], ZONES)
    action = _decide("turtle", near)
    assert action.shoot[0].tolist() == [
        1,
        0,
    ]  # slot 0: defense zone; slot 1: spawn
    far = observation(own, [ship(x=1000.0, y=300.0, team_relative=1), absent()], ZONES)
    assert int(_decide("turtle", far).shoot.sum()) == 0


def test_the_turtle_ships_head_for_their_own_zones():
    """Between the defense zone (east) and the spawn zone (west), both ships heading east:
    slot 0 has its zone dead ahead, slot 1 must turn for the spawn behind it."""
    own = [ship(x=-1500.0, y=0.0, heading=0.0), ship(x=-1500.0, y=0.0, heading=0.0)]
    view = observation(own, [absent(), absent()], ZONES)
    action = _decide("turtle", view)
    assert action.stick[0, 0] == STRAIGHT and action.stick[0, 1] != STRAIGHT


def test_the_evader_turns_away_and_never_fires():
    own = [ship(x=0.0, y=0.0, heading=0.0), ship(x=0.0, y=0.0, heading=math.pi)]
    view = observation(own, [ship(x=300.0, y=0.0, team_relative=1), absent()])
    action = _decide("evader", view)
    assert action.stick[0, 0] != STRAIGHT and action.stick[0, 1] == STRAIGHT
    assert int(action.shoot.sum()) == 0


def test_the_evader_drifts_to_the_centre_as_the_boundary_shrinks():
    own = [ship(x=1000.0, y=0.0, heading=0.0), ship(x=1000.0, y=0.0, heading=0.0)]
    foes = [ship(x=800.0, y=0.0, team_relative=1), absent()]
    full, shrunk = observation(own, foes), observation(own, foes)
    shrunk.globals[:, GLOBAL["boundary_fraction"]] = 0.45  # radius 1170 m, the ship at 1000 m
    assert _decide("evader", full).stick[0, 0] == STRAIGHT  # away from the enemy: straight on
    assert _decide("evader", shrunk).stick[0, 0] != STRAIGHT  # toward the centre instead


def test_the_jouster_closes_head_on_and_fires_when_aligned():
    own = [ship(x=0.0, y=0.0), ship(x=0.0, y=0.0, heading=math.pi)]
    view = observation(own, [ship(x=200.0, y=0.0, team_relative=1), absent()])
    action = _decide("jouster", view)
    assert action.power[0].tolist() == [BOOST, BOOST]
    assert action.stick[0, 0] == STRAIGHT and action.shoot[0, 0] == 1
    assert action.stick[0, 1] != STRAIGHT and action.shoot[0, 1] == 0


def test_the_kiter_holds_the_standoff():
    reach = gun_range(CHARACTERISTICS)

    def play(distance: float) -> tuple[int, int, int]:
        view = observation(
            [ship(x=0.0, y=0.0), ship(x=0.0, y=0.0)],
            [ship(x=distance, y=0.0, team_relative=1), absent()],
        )
        action = _decide("kiter", view)
        return int(action.power[0, 0]), int(action.stick[0, 0]), int(action.shoot[0, 0])

    close = play(0.5 * reach)
    assert close[0] == BOOST and close[1] != STRAIGHT  # too close: turns away, boosting
    assert play(0.9 * reach) == (0, STRAIGHT, 1)  # in the band: aim, coast, fire
    assert play(1.5 * reach) == (BOOST, STRAIGHT, 0)  # too far: closes


def test_the_deathball_focuses_the_enemy_nearest_its_centre():
    foes = [
        ship(x=200.0, y=0.0, team_relative=1),
        ship(x=-100.0, y=-800.0, team_relative=1),
    ]
    view = observation([ship(x=0.0, y=0.0), ship(x=0.0, y=10.0)], foes)
    action = _decide("deathball", view)
    assert action.stick[0].tolist() == [0, 0] and action.shoot[0].tolist() == [1, 1]
    reach = gun_range(CHARACTERISTICS)
    apart = observation([ship(x=0.0), ship(x=0.0, y=-4 * reach, heading=math.pi / 2)], foes)
    # the straggler, flying north, has the centre ahead and fires at nothing it is not aimed at
    assert int(_decide("deathball", apart).stick[0, 1]) == STRAIGHT


def test_without_an_enemy_the_agents_patrol_the_map_centre():
    view = observation([ship(x=1500.0, y=0.0), ship(x=1500.0, y=0.0)], [absent(), absent()])
    for name in ("evader", "deathball", "jouster", "kiter"):
        action = _decide(name, view)
        assert action.stick[0, 0] != STRAIGHT, name  # flying out: turns back toward the centre
        assert int(action.shoot.sum()) == 0


# ---- smoke games

MODES = ("frontline", "team_deathmatch", "elimination")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", NAMES)
def test_a_short_game_of_each_agent_in_each_mode(name, mode):
    """The agent against the scripted agent and against itself, a short segment on the CPU."""
    game = GameConfig(2, 2, 1200.0, mode=mode)
    controllers = {
        "cheese": Cheese(CHARACTERISTICS, 2, name),
        "scripted": Scripted(CHARACTERISTICS, 2),
    }
    pools = (
        Pool("versus", 0.5, ("cheese", "scripted")),
        Pool("mirror", 0.5, ("cheese", "cheese")),
    )
    rollout = Rollout(
        game,
        api.ship_types(),
        pools,
        controllers,
        "cheese",
        4,
        12,
        torch.device("cpu"),
        compiled=False,
    )
    trajectory = rollout.run()
    assert trajectory.obs.ships.shape[0] == 12
    assert torch.isfinite(trajectory.obs.ships).all()


# ---- the match engine


def test_cheese_players_are_named_scripted_colon_name_and_play_in_the_match_engine():
    from bnb.league.matches import STATISTICS, Arena, ArenaConfig, player

    assert [player(player_name(name)).cheese for name in NAMES] == list(NAMES)
    with pytest.raises(ValueError, match="cheese agent"):
        player("scripted:nobody")
    game = GameConfig(2, 2, 1500.0, episode_seconds=12.0, respawn_seconds=1.0)
    field = [player("scripted"), player("scripted:evader"), player("scripted:jouster")]
    arena = Arena(
        field,
        game,
        torch.device("cpu"),
        ArenaConfig(blocks=2, block_environments=4, length=16, compiled=False),
    )
    results = arena.play([(0, 1), (2, 0)])
    assert results.sum(1).tolist() == [4, 4]
    shots = arena.statistics()[..., STATISTICS.index("shots")]  # (blocks, 2)
    assert shots[0, 1] == 0  # the evader never fires
    assert shots[0, 0] > 0  # the scripted agent fires at it, so the game was played
