"""Yemong with the modes (modes spec 7): zones are a map-memory token kind, the mode's
global columns reach the global token, and unroll still equals a loop of step."""

import dataclasses

import pytest
import torch
import yemong_cases as cases

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, ZONE
from bnb.game import api
from bnb.models.yemong.features import respawn_features
from bnb.models.yemong.inputs import POSITION_CODE, TIMER_INPUTS, ZONE_INPUTS, zone_inputs
from bnb.models.yemong.state_code import POSITION_PERIOD

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


OWN = 5


def _frontline_views(ticks: int = 20, environments: int = 4):
    """Team views of a mid-game Frontline match (chaser play), float64."""
    config = cases.SHORT_FRONTLINE
    net = cases.model(config)
    _, trajectory = cases.rollout(net, config=config, environments=environments, length=ticks)
    return cases.double(net), cases.cast(trajectory.obs, torch.float64), trajectory


def test_zones_are_memory_tokens_after_the_fields():
    net, obs, _ = _frontline_views()
    view = Obs(
        ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1], zones=obs.zones[-1]
    )
    carry = net.initial_carry(view.ships.shape[0], OWN, 10)
    replay = net.step(carry, view, torch.ones(view.ships.shape[0], dtype=torch.bool)).replay
    tokens = net.tokens(view, replay)
    assert tokens.zone.shape == view.zones.shape[:-1] + (ZONE_INPUTS,)
    assert tokens.memory_position.shape[-2] == view.fields.shape[-2] + 5
    _, memory, _, _ = net._encode(tokens)
    assert memory.shape[-2] == view.fields.shape[-2] + 5
    # a team deathmatch view has the same five slots, every one absent: the learned token
    rules = api.rules_of(cases.SQUAD)
    _, squad, _ = api.perceive(rules, api.reset(rules, api.ship_types(), api.uniforms(rules, 1)))
    squad = cases.cast(
        Obs(
            ships=squad.ships[:, 0],
            fields=squad.fields[:, 0],
            globals=squad.globals[:, 0],
            zones=squad.zones[:, 0],
        ),
        torch.float64,
    )
    assert squad.zones.shape[-2] == 5 and not squad.zones.any()
    carry = net.initial_carry(1, OWN, 10)
    with torch.no_grad():
        replay = net.step(carry, squad, torch.zeros(1, dtype=torch.bool)).replay
        _, memory, _, _ = net._encode(net.tokens(squad, replay))
    fields = squad.fields.shape[-2]
    assert torch.equal(memory[0, fields:], net.encoder.absent_zone.expand(5, -1))


def _decide(net, view):
    carry = net.initial_carry(view.ships.shape[0], OWN, 10)
    with torch.no_grad():
        return net.step(carry, view, torch.ones(view.ships.shape[0], dtype=torch.bool))


def test_zone_rows_and_mode_globals_reach_the_decision():
    net, obs, _ = _frontline_views()
    view = Obs(
        ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1], zones=obs.zones[-1]
    )
    plain = _decide(net, view)
    for column in ("progress", "role", "offensive_distance"):
        zones = view.zones.clone()
        zones[..., ZONE[column]] += 1.0
        moved = _decide(net, dataclasses.replace(view, zones=zones))
        assert not torch.allclose(moved.policy_logits, plain.policy_logits), column
    for column in ("front", "score_own", "mode_frontline"):
        globals_ = view.globals.clone()
        globals_[..., GLOBAL[column]] += 1.0
        if column == "score_own":
            globals_[..., GLOBAL["score_target"]] = 25.0
        moved = _decide(net, dataclasses.replace(view, globals=globals_))
        assert not torch.allclose(moved.outcome_probabilities, plain.outcome_probabilities), column


def test_frontline_unroll_equals_steps_float64():
    net, obs, trajectory = _frontline_views(ticks=48)
    assert obs.zones.shape[-2] == 5 and cases.resets(obs)[1:].any()
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    for name in ("policy_logits", "values"):
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= 1e-9, (name, error)


def test_the_offset_reaches_zones_only_through_their_position_code():
    """As for ships and fields: with the position-code columns zeroed, shifting the frame
    (the offset) changes nothing; zones sit in the shifted frame like fields."""
    net, obs, _ = _frontline_views()
    with torch.no_grad():
        for kind in (
            net.encoder.ship,
            net.encoder.field,
            net.encoder.global_token,
            net.encoder.zone,
        ):
            kind.first.weight[:, :POSITION_CODE] = 0.0
    view = Obs(
        ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1], zones=obs.zones[-1]
    )
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    delta = torch.rand(views, 2, dtype=torch.float64) * POSITION_PERIOD
    shifted = dataclasses.replace(carry, offset=carry.offset + delta)
    reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        plain, moved = net.step(carry, view, reset), net.step(shifted, view, reset)
    torch.testing.assert_close(moved.policy_logits, plain.policy_logits, rtol=1e-8, atol=1e-8)


def test_frontline_timer_inputs_code_the_locks_and_the_own_wave_as_the_respawn_wait():
    """With `frontline_timer_inputs` the global token ends with the own team's next wave and
    each zone token with its two capture locks, each as `respawn_features` codes the ship's
    respawn wait; without, the tokens are as before."""
    config = cases.SHORT_FRONTLINE
    plain = cases.model(config)
    timed = cases.model(config, frontline_timer_inputs=True)
    rules = api.rules_of(config)
    state = api.reset(rules, api.ship_types(), api.uniforms(rules, 2))
    lock = torch.zeros_like(state.mode.lock)
    lock[:, 0, 1], lock[:, 1, 3] = 300, 45
    state = dataclasses.replace(state, mode=dataclasses.replace(state.mode, lock=lock))
    _, seen, _ = api.perceive(rules, state)
    view = Obs(
        ships=seen.ships[:, 0], fields=seen.fields[:, 0], globals=seen.globals[:, 0],
        zones=seen.zones[:, 0],
    )  # fmt: skip
    tokens = []
    for net in (plain, timed):
        torch.manual_seed(1)  # the same position offsets for both builds
        carry = net.initial_carry(2, config.ships_per_team, config.num_ships)
        with torch.no_grad():
            replay = net.step(carry, view, torch.ones(2, dtype=torch.bool)).replay
        tokens.append(net.tokens(view, replay))
    width = TIMER_INPUTS
    assert tokens[1].zone.shape[-1] == zone_inputs(True) == ZONE_INPUTS + 2 * width
    assert torch.equal(tokens[1].zone[..., :ZONE_INPUTS], tokens[0].zone)
    assert torch.equal(tokens[1].global_token[..., :-width], tokens[0].global_token)
    torch.testing.assert_close(
        tokens[1].zone[..., ZONE_INPUTS : ZONE_INPUTS + width],
        respawn_features(view.zones[..., ZONE["lock_own"]]),
    )
    torch.testing.assert_close(
        tokens[1].zone[..., ZONE_INPUTS + width :],
        respawn_features(view.zones[..., ZONE["lock_enemy"]]),
    )
    torch.testing.assert_close(
        tokens[1].global_token[..., 0, -width:],
        respawn_features(view.globals[..., GLOBAL["next_wave_own"]]),
    )
    assert (
        view.zones[..., ZONE["lock_own"]].any() and view.globals[..., GLOBAL["next_wave_own"]].all()
    )
