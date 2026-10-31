"""Yemong's acceptance items 2-4 (spec 11): unroll equals a loop of step, the belief, and
symmetry and size."""

import dataclasses

import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs
from bnb.contracts.obs import SHIP, SHIP_HIDDEN_ZERO
from bnb.game import api
from bnb.models.yemong.inputs import truth

COMPARED = ("policy_logits", "enemy_action_logits", "next_state_logits", "values")


@pytest.fixture(scope="module")
def segment():
    """A real squad-sized rollout, half the environments fogged, with several resets."""
    net = cases.model()
    _, trajectory = cases.rollout(net)
    return net, trajectory


def test_segment_has_resets_fog_and_hidden_ships(segment):
    _, trajectory = segment
    reset = cases.resets(trajectory.obs)
    assert reset[1:].sum(0).min() >= 2  # every view starts at least two episodes
    fogged = trajectory.obs.globals[..., 2] > 0.5
    assert fogged.any() and (~fogged).any()
    assert (trajectory.obs.ships[..., SHIP["visible"]] < 0.5).any()


def _compare(unrolled, decisions, tolerance):
    for name in COMPARED:
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= tolerance, (name, error)
    outcome = torch.softmax(unrolled.outcome_logits, -1)
    assert cases.scale_error(outcome, decisions.outcome) <= tolerance


def test_unroll_equals_steps_float64(segment):
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, torch.float64)
    views = obs.ships.shape[1]
    carry = net.initial_carry(views, 5, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    _compare(unrolled, decisions, 1e-9)


def test_unroll_reproduces_the_stored_rollout(segment):
    """What the runtime stored (values, outcome, replay, the initial carry) is what unroll
    needs: it gives the rollout's own values, outcome and log-probabilities back."""
    net, trajectory = segment
    from bnb.models.yemong import policy

    with torch.no_grad():
        unrolled = net.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay,
                              cases.resets(trajectory.obs))
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-4
    assert cases.scale_error(torch.softmax(unrolled.outcome_logits, -1), trajectory.outcome) <= 1e-4
    assert cases.scale_error(unrolled.values.sum(-1), trajectory.value) <= 1e-4
    logp, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(trajectory))
    assert cases.scale_error(logp, trajectory.logp) <= 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_unroll_equals_steps_float32(segment, compiled):
    net, trajectory = segment
    device = torch.device("cuda") if torch.cuda.is_available() else cases.CPU
    net = cases.double(net).float().to(device)
    obs = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), trajectory.obs)
    joint = cases.joint_actions(trajectory).to(device)
    step, unroll = net.step, net.unroll
    if compiled:
        step = torch.compile(step, fullgraph=True, dynamic=False)
        unroll = torch.compile(unroll, fullgraph=True, dynamic=False)
    carry = net.initial_carry(obs.ships.shape[1], 5, 10, device)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, step, carry, obs, joint)
        unrolled = unroll(carry, obs, decisions.replay, cases.resets(obs))
    _compare(unrolled, decisions, 1e-4)


# ------------------------------------------------------------------ belief


def _views(config: GameConfig, fogged: bool, ticks: int = 0, seed: int = 0):
    """Views (2B, ...) of a game after `ticks` ticks of chaser play."""
    from bnb.contracts import Action
    from bnb.controllers.scripted import Chaser

    generator = torch.Generator().manual_seed(seed)
    rules, types = api.rules_of(config), api.ship_types()
    batch = 2
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator),
                      torch.full((batch,), fogged))
    state, obs, _ = api.perceive(rules, state)
    chaser = Chaser(config.ships_per_team)
    for _ in range(ticks):
        view = Obs(ships=obs.ships.flatten(0, 1), fields=None, globals=None)
        action, _, _ = chaser.step(None, view, None)
        packed = action.packed().reshape(batch, config.num_ships, 3)
        state, _ = api.step(rules, types, state, Action.unpack(packed),
                            api.spread(rules, batch, generator=generator),
                            api.uniforms(rules, batch, generator=generator))
        state, obs, _ = api.perceive(rules, state)
    return Obs(ships=obs.ships.flatten(0, 1), fields=obs.fields.flatten(0, 1),
               globals=obs.globals.flatten(0, 1))


def test_visible_ship_code_is_the_sharp_code_of_its_observation():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True, ticks=1), torch.float64)
    carry = net.initial_carry(view.ships.shape[0], 5, 10)
    with torch.no_grad():
        decision = net.step(carry, view, torch.ones(view.ships.shape[0], dtype=torch.bool))
        tokens = net.tokens(view, decision.replay)
    visible = view.ships[..., SHIP["visible"]] > 0.5
    ship_type = view.ships[..., SHIP["ship_type"]].long()
    sharp = net.state_code.sharp(truth(view.ships), ship_type)
    assert visible.any() and (~visible).any()
    assert torch.equal(tokens.code[visible], sharp[visible])
    assert torch.equal(decision.replay.spreads[visible], torch.zeros_like(sharp[visible][..., :15]))


def test_spawning_voids_a_belief():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True), torch.float64)
    views = view.ships.shape[0]
    carry = net.initial_carry(views, 5, 10)
    believed = dataclasses.replace(carry.belief, valid=torch.ones_like(carry.belief.valid),
                                   means=torch.full_like(carry.belief.means, 7.0),
                                   spreads=torch.full_like(carry.belief.spreads, 3.0),
                                   action=torch.softmax(torch.randn_like(carry.belief.action), -1))
    carry = dataclasses.replace(carry, belief=believed)
    ships = view.ships.clone()
    enemy = slice(5, 10)
    ships[:, enemy, :SHIP_HIDDEN_ZERO] = 0.0
    ships[:, enemy, SHIP["visible"]] = 0.0
    ships[:, enemy, SHIP["spawned"]] = 0.0
    ships[:, 5, SHIP["spawned"]] = 1.0  # spawned out of sight: the old belief is void
    no_reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        replay = net.step(carry, dataclasses.replace(view, ships=ships), no_reset).replay
    assert not replay.valid[:, 5].any() and replay.valid[:, 6:10].all()
    assert (replay.means[:, 5] == 0).all() and (replay.spreads[:, 5] == 0).all()
    uniform = torch.full_like(replay.enemy_action[:, 0], 1 / 42)
    assert torch.allclose(replay.enemy_action[:, 0], uniform)  # row 5 is enemy row 0
    assert (replay.means[:, 6:10] == 7.0).all() and (replay.spreads[:, 6:10] == 3.0).all()


def test_without_fog_every_living_ship_is_certain():
    """Fog off shows every living ship, so its belief is truth with no spread. (A dead
    enemy is not shown even without fog; its row keeps a forecast.)"""
    net = cases.model()
    _, trajectory = cases.rollout(net, fog_share=0.0, length=64)
    alive = trajectory.obs.ships[..., SHIP["alive"]] > 0.5
    assert (trajectory.replay.spreads[alive] == 0).all()
    assert trajectory.replay.valid[alive].all()


def test_hidden_columns_of_hidden_ships_are_not_read():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True, ticks=20), torch.float64)
    views = view.ships.shape[0]
    hidden = view.ships[..., SHIP["visible"]] < 0.5
    assert hidden.any()
    noise = torch.randn(view.ships.shape[:-1] + (SHIP_HIDDEN_ZERO,), dtype=torch.float64) * 100
    ships = view.ships.clone()
    ships[..., :SHIP_HIDDEN_ZERO] = torch.where(hidden[..., None], noise,
                                                ships[..., :SHIP_HIDDEN_ZERO])
    carry = net.initial_carry(views, 5, 10)
    reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        first = net.step(carry, view, reset)
        second = net.step(carry, dataclasses.replace(view, ships=ships), reset)
    for a, b in zip(_pytree.tree_leaves(first), _pytree.tree_leaves(second), strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------ symmetry and size


def test_permuting_ships_within_a_team_permutes_every_output():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True, ticks=30, seed=1), torch.float64)
    views = view.ships.shape[0]
    own = torch.tensor([3, 0, 4, 1, 2])
    rows = torch.cat([own, 5 + torch.tensor([2, 4, 0, 3, 1])])
    permuted = dataclasses.replace(view, ships=view.ships[:, rows])
    carry = net.initial_carry(views, 5, 10)
    reset = torch.ones(views, dtype=torch.bool)
    with torch.no_grad():
        plain, swapped = net.step(carry, view, reset), net.step(carry, permuted, reset)
        # and a second step from each carry, the previous actions permuted alike
        joint = torch.randint(0, 42, (views, 5), dtype=torch.int32)
        plain_carry = dataclasses.replace(plain.carry, previous_action=joint)
        swapped_carry = dataclasses.replace(swapped.carry, previous_action=joint[:, own])
        reset = torch.zeros_like(reset)
        plain_next = net.step(plain_carry, view, reset)
        swapped_next = net.step(swapped_carry, permuted, reset)
    for first, second in ((plain, swapped), (plain_next, swapped_next)):
        for name, index in (("policy_logits", own), ("values", own), ("enemy_action_logits", rows),
                            ("next_state_logits", rows)):
            torch.testing.assert_close(getattr(first, name)[:, index], getattr(second, name),
                                       rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(first.outcome, second.outcome, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("config", [GameConfig(1, 4, 1200.0), cases.SQUAD,
                                    GameConfig(50, 40, 8200.0)], ids=["duel", "squad", "battle"])
def test_one_set_of_weights_plays_every_size(config):
    net = cases.model(cases.SQUAD)
    view = _views(config, fogged=True)
    views, own, ships = view.ships.shape[0], config.ships_per_team, config.num_ships
    carry = net.initial_carry(views, own, ships)
    with torch.no_grad():
        decision = net.step(carry, view, torch.ones(views, dtype=torch.bool))
    assert decision.policy_logits.shape == (views, own, 42)
    assert decision.values.shape == (views, own, 2)
    assert decision.next_state_logits.shape == (views, ships, 490)
    assert all(torch.isfinite(x).all() for x in _pytree.tree_leaves(decision)
               if x.is_floating_point())
