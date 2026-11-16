"""Imugi's memory (its carry): soft hysteresis on the plan choices, dead reckoning
of hidden enemies, the resets, the mixture rungs' agent flags and the per-mode parameters."""

import dataclasses
import math
import pathlib
import sys

import pytest
import torch
from imugi_cases import CHARACTERISTICS, absent, observation, ship, value

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.imugi import Imugi, distribution, mode_index
from bnb.controllers.imugi_combat import BOOST, TARGET_DOUBT_SECONDS, TARGET_LOST_SECONDS
from bnb.controllers.imugi_memory import FORGET_WEIGHT, remember, ways_round
from bnb.controllers.imugi_parameters import (
    MODES,
    REAL_SCALE_MODES,
    STANDARD_MODES,
    ImugiParameters,
    ModeParameters,
    Switch,
    switch_probability,
)
from bnb.game.rules import Rules

pytestmark = pytest.mark.usefixtures("global_generator")
PARAMETERS = ImugiParameters()
VIEWS = 2000
CPU = torch.device("cpu")


def _enemy(x: float, y: float = 0.0, visible: bool = True, since: float = 0.0, velocity=None):
    row = ship(x=x, y=y, team_relative=1, visible=visible, velocity=velocity)
    row[SHIP["time_since_seen"]] = since
    return row


def _view(enemies: list[torch.Tensor], views: int = VIEWS, reset: bool = False) -> Obs:
    """Own ship 0 at the origin (nose +x), the second own slot absent; `views` copies."""
    obs = observation([ship(), absent(team_relative=0)], enemies)
    obs = Obs(
        ships=obs.ships.expand(views, -1, -1).clone(),
        fields=None,
        globals=obs.globals.expand(views, -1).clone(),
        zones=obs.zones.expand(views, -1, -1).clone(),
    )
    obs.globals[:, GLOBAL["reset"]] = float(reset)
    return obs


def _agent(**changes) -> Imugi:
    return Imugi(CHARACTERISTICS, 2, dataclasses.replace(PARAMETERS, **changes))


def _run(agent: Imugi, views: list[Obs], carry=None):
    """The agent's carry after each view in turn (the first a reset)."""
    carry = agent.init_carry(views[0].ships.shape[0], CPU) if carry is None else carry
    out = []
    for index, view in enumerate(views):
        reset = torch.full((view.ships.shape[0],), index == 0 and carry.fresh.all().item())
        keys = torch.arange(view.ships.shape[0]) + index * view.ships.shape[0]
        _, carry = agent.decide(carry, view, reset, keys)
        out.append(carry)
    return out


def test_the_switching_rate_rises_smoothly_with_the_advantage():
    shape = Switch(10.0, 0.3, 0.1)
    advantage = torch.linspace(-0.3, 0.9, 121)  # within the logit limit
    probability = switch_probability(advantage, shape, 1.0, 1 / 30)
    assert bool((probability.diff() > 0).all())  # strictly rising: no step
    assert float(probability[0]) < 1e-3 and float(probability[-1]) > 0.28  # near 1 - exp(-1/3)
    hot = switch_probability(advantage, shape, 4.0, 1 / 30)
    assert float(hot[0]) > float(probability[0]) and float(hot[-1]) < float(probability[-1])


def test_a_near_tie_keeps_the_committed_target_and_a_clearly_nearer_one_takes_over_late():
    gun = value("gun_range", 0)
    agent = _agent()
    # Commit to enemy 1 (alone in view), then enemy 0 appears just nearer: kept.
    first = _view([absent(), _enemy(gun * 2.0)])
    tie = _view([_enemy(gun * 1.97), _enemy(gun * 2.0)])
    carries = _run(agent, [first] + [tie] * 90)
    assert bool((carries[0].target[:, 0] == 1).all())
    assert bool((carries[-1].target[:, 0] == 1).all())
    # The memoryless agent takes the nearer at once.
    off = Imugi(CHARACTERISTICS, 2, PARAMETERS, memory=False)
    assert bool((_run(off, [first, tie])[-1].target[:, 0] == 0).all())
    # A clearly nearer enemy (a third of the distance) takes over only once it has been
    # so for TARGET_DOUBT_SECONDS (60 ticks), all at once (no draw)
    near = _view([_enemy(gun * 0.67), _enemy(gun * 2.0)])
    ticks = round(TARGET_DOUBT_SECONDS / Rules.tick_length)
    carries = _run(agent, [first] + [near] * (ticks + 2))
    assert bool((carries[ticks - 2].target[:, 0] == 1).all())
    assert bool((carries[-1].target[:, 0] == 0).all())


def test_a_lost_target_forces_a_re_choice():
    gun = value("gun_range", 0)
    agent = _agent()
    both = _view([_enemy(gun * 1.5), _enemy(gun * 1.0)])
    dead = _view([_enemy(gun * 1.5), absent()])  # enemy 1 died: not alive
    carries = _run(agent, [both, dead])
    assert bool((carries[0].target[:, 0] == 1).all()) and bool((carries[1].target[:, 0] == 0).all())
    assert not bool(carries[1].known[:, 1].any())


def test_a_hidden_enemy_is_dead_reckoned_fades_and_is_forgotten():
    vision = value("vision_range", 0)
    speed = 150.0
    agent = _agent()
    start = 1.2 * vision
    seen = _view([_enemy(start, velocity=(speed, 0.0)), absent()], views=1)
    hidden = [
        _view([_enemy(0.0, visible=False, since=t), absent()], views=1) for t in (0.5, 1.5, 6.0)
    ]
    carries = _run(agent, [seen] + hidden)
    for carry in carries[:3]:  # remembered and committed while it fades
        assert bool(carry.known[0, 0]) and int(carry.target[0, 0]) == 0
    assert float(carries[1].enemy_x[0, 0]) == pytest.approx(start)  # the last seen state
    # the prediction: the last seen position plus its velocity times the time since seen
    _, recall = remember(
        carries[0], hidden[0], torch.zeros(1, dtype=torch.bool), CHARACTERISTICS, PARAMETERS, 1.0
    )
    assert float(recall.x[0, 0]) == pytest.approx(start + speed * 0.5)
    assert 0.0 < float(recall.weight[0, 0]) < 1.0
    fade = PARAMETERS.memory_fade_in_seconds
    late = 6.0  # far past the fade's centre: below FORGET_WEIGHT
    assert 1.0 / (1.0 + (late / fade.centre) ** (1.0 / fade.scale)) < FORGET_WEIGHT
    assert not bool(carries[3].known[0, 0]) and int(carries[3].target[0, 0]) == -1


def test_a_prediction_inside_the_own_view_is_dropped():
    vision = value("vision_range", 0)
    agent = _agent()
    seen = _view([_enemy(1.1 * vision, velocity=(-400.0, 0.0)), absent()], views=1)
    # 0.5 s later the prediction is at 0.9 vision ranges: it would be seen, so it was wrong
    hidden = _view([_enemy(0.0, visible=False, since=0.5), absent()], views=1)
    carries = _run(agent, [seen, hidden])
    assert not bool(carries[1].known[0, 0]) and int(carries[1].target[0, 0]) == -1


def test_a_reset_and_the_ships_death_clear_the_memory():
    gun = value("gun_range", 0)
    agent = _agent()
    far = value("vision_range", 0)  # beyond it: a prediction there is not contradicted
    obs = _view([_enemy(far * 1.3), _enemy(far * 1.2)], views=4)
    carry = _run(agent, [obs, obs])[-1]
    assert not bool(carry.fresh.any()) and bool(carry.known.all())
    hidden = _view([_enemy(0.0, visible=False, since=0.1), absent()], views=4)
    reset = torch.tensor([True, False, False, False])
    _, after = agent.decide(carry, hidden, reset, torch.arange(reset.shape[0]))
    assert not bool(after.known[0].any()) and bool(after.known[1:, 0].all())
    dead_own = _view([_enemy(gun * 1.5), _enemy(gun * 1.0)], views=1)
    dead_own.ships[0, 0, SHIP["alive"]] = 0.0
    one = type(carry)(**{f.name: getattr(carry, f.name)[:1] for f in dataclasses.fields(carry)})
    # a dead ship is fresh before its choices: it re-chooses outright
    _, after = agent.decide(
        one, dead_own, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.int64)
    )
    assert not bool(after.fresh.any())


def test_the_turn_side_commits_past_its_angle_and_switches_by_the_shorter_way():
    left, right = ways_round(torch.tensor([0.5, -0.5, math.pi - 0.01]))
    torch.testing.assert_close(left + right, torch.full((3,), 2 * math.pi))
    assert float(left[0]) == pytest.approx(0.5) and float(right[1]) == pytest.approx(0.5)
    gun = value("gun_range", 0)
    agent = _agent()
    behind_left = _view([_enemy(-gun, 0.02 * gun), absent()])  # just left of dead astern
    behind_right = _view([_enemy(-gun, -0.02 * gun), absent()])  # just right of it
    carries = _run(agent, [behind_left] + [behind_right] * 10)
    assert bool(carries[0].combat_left[:, 0].all())
    # the other way round is barely shorter: the committed side holds
    assert float(carries[-1].combat_left[:, 0].float().mean()) > 0.9
    _, stick, _ = (p[:, 0] for p in agent.marginals(behind_right, carries[-1]))
    committed = carries[-1].combat_left[:, 0]
    assert float(stick[committed][:, 2::2].sum(-1).mean()) > 0.9  # turning left
    ahead_right = _view([_enemy(gun, -0.5 * gun), absent()])  # now clearly right
    carries = _run(agent, [behind_left] + [ahead_right] * 15)  # half a second
    assert float(carries[-1].combat_left[:, 0].float().mean()) < 0.05


def test_memory_off_is_the_memoryless_agent():
    gun = value("gun_range", 0)
    off = Imugi(CHARACTERISTICS, 2, PARAMETERS, memory=False)
    seen = _view([_enemy(gun * 1.5), _enemy(gun * 1.0)], views=8)
    hidden = _view([_enemy(0.0, visible=False, since=0.2), _enemy(gun * 1.4)], views=8)
    carries = _run(off, [seen, hidden])
    expected = distribution(hidden, CHARACTERISTICS, PARAMETERS)
    torch.testing.assert_close(off.action_distribution(hidden, carries[-1]), expected)
    assert not bool(carries[-1].known[:, 0].any())


def test_a_mixture_rungs_ships_play_the_agent_by_its_share():
    agent = Imugi(CHARACTERISTICS, 2, PARAMETERS, temperature=2.0, share=0.3)
    obs = _view([_enemy(500.0), absent()])
    carry = agent.init_carry(VIEWS, CPU)
    _, carry = agent.decide(carry, obs, torch.ones(VIEWS, dtype=torch.bool), torch.arange(VIEWS))
    share = float(carry.agent.float().mean())
    assert abs(share - 0.3) < 0.03
    _, kept = agent.decide(
        carry, obs, torch.zeros(VIEWS, dtype=torch.bool), torch.arange(VIEWS) + VIEWS
    )
    assert torch.equal(kept.agent, carry.agent)  # drawn per game
    parts = agent.marginals(obs, carry)
    uniform = ~carry.agent[:, 0]
    torch.testing.assert_close(
        parts.stick[uniform, 0], torch.full_like(parts.stick[uniform, 0], 1 / 7)
    )
    with pytest.raises(ValueError):
        Imugi(CHARACTERISTICS, 2, share=1.5)


def test_each_view_plays_its_modes_parameters():
    sets = {mode: PARAMETERS for mode in MODES}
    sets["deathmatch_wide"] = dataclasses.replace(PARAMETERS, lead_cap_in_gun_range=0.3)
    per_mode = ModeParameters(sets)
    narrow = observation([ship(), absent(team_relative=0)], [_enemy(400.0, 300.0), absent()])
    wide = observation(
        [ship(), absent(team_relative=0)], [_enemy(400.0, 300.0), absent()], arena_radius=8200.0
    )
    assert mode_index(narrow, CHARACTERISTICS).tolist() == [MODES.index("deathmatch")]
    assert mode_index(wide, CHARACTERISTICS).tolist() == [MODES.index("deathmatch_wide")]
    both = Obs(
        ships=torch.cat([narrow.ships, wide.ships]),
        fields=None,
        globals=torch.cat([narrow.globals, wide.globals]),
        zones=torch.cat([narrow.zones, wide.zones]),
    )
    mixed = distribution(both, CHARACTERISTICS, per_mode)
    torch.testing.assert_close(mixed[0], distribution(narrow, CHARACTERISTICS, PARAMETERS)[0])
    torch.testing.assert_close(
        mixed[1], distribution(wide, CHARACTERISTICS, sets["deathmatch_wide"])[0]
    )


def _at_plane_speed(parameters: ImugiParameters, factor: float) -> ImugiParameters:
    """A real-scale set at the planes' speed: game v13 plays the set `factor` times faster
    (docs/rewrite/sections/lethality/scripted_x2.py scaled its time parameters: the memory
    fade's centre / factor, every switch's rate x factor), and the search's bounds are the
    planes' seconds."""
    fade = parameters.memory_fade_in_seconds
    changed = {"memory_fade_in_seconds": fade._replace(centre=fade.centre * factor)}
    for field in dataclasses.fields(parameters):
        value = getattr(parameters, field.name)
        if isinstance(value, Switch):
            changed[field.name] = value._replace(rate=value.rate / factor)
    return dataclasses.replace(parameters, **changed)


def test_the_mode_defaults_lie_within_the_searchs_bounds_and_the_populations_load():
    from bnb.controllers.imugi_defaults import MODE_DEFAULTS, populations
    from bnb.game.real_scale import EQUIVALENT_FACTOR

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
    from bench.imugi_search import clamp

    assert set(MODE_DEFAULTS.sets) == set(MODES)
    for mode, parameters in MODE_DEFAULTS.sets.items():
        if mode in REAL_SCALE_MODES:  # at x2 native speed
            parameters = _at_plane_speed(parameters, EQUIVALENT_FACTOR)
        assert clamp(parameters) == parameters, mode
    stored = populations()
    assert set(stored) == set(STANDARD_MODES)
    assert all(len(stored[mode]) == 32 for mode in STANDARD_MODES)


def test_a_hidden_committed_target_is_pursued_until_lost():
    vision = value("vision_range", 0)
    agent = _agent()
    away = (150.0, 0.0)  # its prediction runs on out of view
    both = _view([_enemy(1.2 * vision, velocity=away), _enemy(1.5 * vision)], views=1)
    carries = _run(agent, [both])
    assert int(carries[0].target[0, 0]) == 0

    def hidden(since: float) -> Obs:
        row = _enemy(0.0, visible=False, since=since)
        return _view([row, _enemy(1.5 * vision)], views=1)

    kept = _run(agent, [hidden(1.0)], carries[0])
    assert int(kept[0].target[0, 0]) == 0 and bool(kept[0].known[0, 0])
    lost = _run(agent, [hidden(TARGET_LOST_SECONDS + 0.5)], kept[0])
    assert int(lost[0].target[0, 0]) == 1


def _extension(enemy_type: int, enemy_heading: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(the carry's extend phase, P(boost)) after one decision: own type 0 at the origin
    (nose +x, a full store), the target half a gun range behind with an empty store."""
    gun = value("gun_range", 0)
    target = ship(
        x=-0.5 * gun, heading=enemy_heading, ship_type=enemy_type, team_relative=1, energy=0.0
    )
    view = _view([target, absent()], views=1)
    agent = _agent()
    carry = agent.init_carry(1, CPU)
    parts, carry = agent.decide(carry, view, torch.ones(1, dtype=torch.bool), torch.arange(1))
    return carry.extend_phase[0, 0], parts.power[0, 0, BOOST]


def test_an_outturned_ship_with_more_energy_extends_after_a_pass_only():
    # type 0 turns slower than type 1 (the standard card); the target flying away: extend
    assert value("cruise_speed", 0) > 0 and float(CHARACTERISTICS.turn_rate[0, 2]) < float(
        CHARACTERISTICS.turn_rate[1, 2]
    )
    phase, boost = _extension(1, math.pi)
    assert float(phase) > 0.0 and float(boost) > 0.9
    # the target on the tail (flying toward the ship): no extension
    assert float(_extension(1, 0.0)[0]) == 0.0
    # a mirror (the same turn rate): it stays in the turn
    assert float(_extension(0, math.pi)[0]) == 0.0
