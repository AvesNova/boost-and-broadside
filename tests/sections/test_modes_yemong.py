"""Yemong with the modes (modes spec 7): zones are a map-memory token kind, the mode's
global columns reach the global token, and unroll still equals a loop of step."""

import dataclasses

import torch
import yemong_cases as cases

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, ZONE
from bnb.game import api
from bnb.models.yemong.inputs import POSITION_CODE, ZONE_INPUTS
from bnb.models.yemong.state_code import POSITION_PERIOD

OWN = 5


def _frontline_views(ticks: int = 20, envs: int = 4):
    """Team views of a mid-game Frontline match (chaser play), float64."""
    config = cases.SHORT_FRONTLINE
    net = cases.model(config)
    _, trajectory = cases.rollout(net, config=config, envs=envs, length=ticks)
    return cases.double(net), cases.cast(trajectory.obs, torch.float64), trajectory


def test_zones_are_memory_tokens_after_the_fields():
    net, obs, _ = _frontline_views()
    view = Obs(ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1],
               zones=obs.zones[-1])
    carry = net.initial_carry(view.ships.shape[0], OWN, 10)
    replay = net.step(carry, view, torch.ones(view.ships.shape[0], dtype=torch.bool)).replay
    tokens = net.tokens(view, replay)
    assert tokens.zone.shape == view.zones.shape[:-1] + (ZONE_INPUTS,)
    assert tokens.memory_position.shape[-2] == view.fields.shape[-2] + 5
    _, memory, _, _ = net._encode(tokens)
    assert memory.shape[-2] == view.fields.shape[-2] + 5
    # a team deathmatch view has no zones: None and an empty part are the same
    rules = api.rules_of(cases.SQUAD)
    _, squad, _ = api.perceive(rules, api.reset(rules, api.ship_types(), api.uniforms(rules, 1)))
    squad = cases.cast(Obs(ships=squad.ships[:, 0], fields=squad.fields[:, 0],
                           globals=squad.globals[:, 0], zones=squad.zones[:, 0]), torch.float64)
    plain = cases.model(dtype=torch.float64)
    carry = plain.initial_carry(1, OWN, 10)
    reset = torch.zeros(1, dtype=torch.bool)  # (a reset would draw a fresh offset)
    with torch.no_grad():
        given = plain.step(carry, squad, reset)
        left_out = plain.step(carry, dataclasses.replace(squad, zones=None), reset)
    torch.testing.assert_close(given.policy_logits, left_out.policy_logits)


def _decide(net, view):
    carry = net.initial_carry(view.ships.shape[0], OWN, 10)
    with torch.no_grad():
        return net.step(carry, view, torch.ones(view.ships.shape[0], dtype=torch.bool))


def test_zone_rows_and_mode_globals_reach_the_decision():
    net, obs, _ = _frontline_views()
    view = Obs(ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1],
               zones=obs.zones[-1])
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
        for kind in (net.encoder.ship, net.encoder.field, net.encoder.global_token,
                     net.encoder.zone):
            kind.first.weight[:, :POSITION_CODE] = 0.0
    view = Obs(ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1],
               zones=obs.zones[-1])
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    delta = torch.rand(views, 2, dtype=torch.float64) * POSITION_PERIOD
    shifted = dataclasses.replace(carry, offset=carry.offset + delta)
    reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        plain, moved = net.step(carry, view, reset), net.step(shifted, view, reset)
    torch.testing.assert_close(moved.policy_logits, plain.policy_logits, rtol=1e-8, atol=1e-8)
