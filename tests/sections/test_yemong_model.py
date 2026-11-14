"""Yemong's acceptance items 2-4 (spec 11): unroll equals a loop of step, the belief, the
position offset, and symmetry and size."""

import dataclasses

import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, GameConfig, Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import GLOBAL, SHIP, SHIP_HIDDEN_ZERO
from bnb.game import api
from bnb.models.yemong import policy
from bnb.models.yemong.inputs import ACTION_SCALE, POSITION_CODE, store_action, truth
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import POSITION_PERIOD

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


OWN = 5  # squad: own ships per view


@pytest.fixture(scope="module")
def segment(module_global_generator):
    """A real squad-sized rollout, half the environments fogged, with several resets."""
    net = cases.model()
    _, trajectory = cases.rollout(net)
    return net, trajectory


def test_segment_has_resets_fog_and_hidden_ships(segment):
    _, trajectory = segment
    reset = cases.resets(trajectory.obs)
    assert reset[1:].sum(0).min() >= 2  # every view starts at least two episodes
    fogged = trajectory.obs.globals[..., GLOBAL["fogged"]] > 0.5
    assert fogged.any() and (~fogged).any()
    assert (trajectory.obs.ships[..., SHIP["visible"]] < 0.5).any()


def _compare(unrolled, decisions, tolerance):
    """step computes the enemy-action and next-state heads for the enemy rows only."""
    pairs = (
        ("policy_logits", unrolled.policy_logits),
        ("enemy_action_logits", unrolled.enemy_action_logits[..., OWN:, :]),
        ("next_state_logits", unrolled.next_state_logits[..., OWN:, :]),
        ("values", unrolled.values),
        ("outcome_probabilities", torch.softmax(unrolled.outcome_logits, -1)),
    )
    for name, actual in pairs:
        error = cases.scale_error(actual, getattr(decisions, name))
        assert error <= tolerance, (name, error)


def _float64(segment):
    net, trajectory = segment
    return cases.double(net), cases.cast(trajectory.obs, torch.float64)


def test_unroll_equals_steps_float64(segment):
    net, obs = _float64(segment)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment[1]))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    _compare(unrolled, decisions, 1e-9)


def test_unroll_continues_from_a_carry_with_history(segment):
    """A segment split in two: unroll of the second half from the step loop's carry at the
    split equals the steps. (A zeroed carry would not: the trunk state matters.)"""
    net, obs = _float64(segment)
    joint = cases.joint_actions(segment[1])
    split, end = 32, 64
    obs = _pytree.tree_map_only(torch.Tensor, lambda x: x[:end], obs)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        head = _pytree.tree_map_only(torch.Tensor, lambda x: x[:split], obs)
        _, middle = cases.step_loop(net, net.step, carry, head, joint)
        decisions, _ = cases.step_loop(net, net.step, middle, obs, joint, start=split)
        tail = _pytree.tree_map_only(torch.Tensor, lambda x: x[split:], obs)
        reset = cases.resets(tail)
        unrolled = net.unroll(middle, tail, decisions.replay, reset)
        zeroed = dataclasses.replace(
            middle,
            state=torch.zeros_like(middle.state),
            convolution_buffer=torch.zeros_like(middle.convolution_buffer),
        )
        cold = net.unroll(zeroed, tail, decisions.replay, reset)
    _compare(unrolled, decisions, 1e-9)
    assert cases.scale_error(cold.policy_logits, decisions.policy_logits) > 1e-3


def test_a_one_step_unroll_equals_the_step(segment):
    """A segment of one step (the play shell's prediction overlays unroll one decision)."""
    net, obs = _float64(segment)
    joint = cases.joint_actions(segment[1])
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    split = 40
    with torch.no_grad():
        head = _pytree.tree_map_only(torch.Tensor, lambda x: x[:split], obs)
        _, middle = cases.step_loop(net, net.step, carry, head, joint)
        one = _pytree.tree_map_only(torch.Tensor, lambda x: x[split : split + 1], obs)
        decisions, _ = cases.step_loop(net, net.step, middle, one, joint, start=0)
        unrolled = net.unroll(middle, one, decisions.replay, cases.resets(one))
    _compare(unrolled, decisions, 1e-9)


def test_inspect_is_the_step_with_every_head_as_unroll_gives_it(segment):
    """`inspect` decides exactly as `step` does, and its outputs are the one-step unroll's
    (every row of the ship heads, the density and the shot heads)."""
    net, obs = _float64(segment)
    joint = cases.joint_actions(segment[1])
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        head = _pytree.tree_map_only(torch.Tensor, lambda x: x[:40], obs)
        _, middle = cases.step_loop(net, net.step, carry, head, joint)
        view = _pytree.tree_map_only(torch.Tensor, lambda x: x[40], obs)
        reset = cases.resets(obs)[40]
        stepped = net.step(middle, view, reset, torch.arange(view.ships.shape[0]))
        decision, inspected = net.inspect(middle, view, reset, torch.arange(view.ships.shape[0]))
        leading = _pytree.tree_map_only(torch.Tensor, lambda x: x[None], (view, decision.replay))
        unrolled = net.unroll(middle, *leading, reset[None])
    for (path, expected), (_, actual) in zip(
        _pytree.tree_flatten_with_path(stepped)[0],
        _pytree.tree_flatten_with_path(decision)[0],
        strict=True,
    ):
        assert torch.equal(actual, expected), path
    for field in dataclasses.fields(unrolled):
        actual, expected = getattr(inspected, field.name), getattr(unrolled, field.name)
        if expected is None:  # a head the model does not have (the rotary read-out)
            assert actual is None, field.name
            continue
        assert actual.shape == expected.shape, field.name
        assert cases.scale_error(actual, expected) <= 1e-9, field.name


def test_unroll_reproduces_the_stored_rollout():
    """What the runtime stored (values, outcome, replay, the initial carry) is what unroll
    needs. The second run's initial carry holds the first run's history."""
    net = cases.model()
    _, trajectory = cases.rollout(net, length=64, runs=2)
    assert trajectory.initial_carry.state.abs().max() > 0
    assert trajectory.initial_carry.belief.valid.any()
    with torch.no_grad():
        unrolled = net.unroll(
            trajectory.initial_carry,
            trajectory.obs,
            trajectory.replay,
            cases.resets(trajectory.obs),
        )
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-4
    assert cases.scale_error(torch.softmax(unrolled.outcome_logits, -1), trajectory.outcome) <= 1e-4
    assert cases.scale_error(unrolled.values.sum(-1), trajectory.value) <= 1e-4
    log_probability, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(trajectory))
    assert cases.scale_error(log_probability, trajectory.log_probability) <= 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_unroll_equals_steps_float32(segment, compiled, device):
    net, trajectory = segment
    device = torch.device(device)
    net = cases.double(net).float().to(device)
    obs = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), trajectory.obs)
    joint = cases.joint_actions(trajectory).to(device)
    step, unroll = net.step, net.unroll
    if compiled:
        step = torch.compile(step, fullgraph=True, dynamic=False)
        unroll = torch.compile(unroll, fullgraph=True, dynamic=False)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10, device)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, step, carry, obs, joint)
        unrolled = unroll(carry, obs, decisions.replay, cases.resets(obs))
    _compare(unrolled, decisions, 1e-4)


# ------------------------------------------------------------------ belief


def _view_sequence(config: GameConfig, fogged: bool, ticks: int, seed: int = 0) -> list[Obs]:
    """Views (2B, ...) of a game, one per tick of chaser play (ticks + 1 of them)."""
    from bnb.contracts import Action
    from bnb.controllers.scripted import Chaser
    from bnb.game.handling.characteristics import characteristics

    generator = torch.Generator().manual_seed(seed)
    rules, types = api.rules_of(config), api.ship_types()
    batch = 2
    state = api.reset(
        rules, types, api.uniforms(rules, batch, generator=generator), torch.full((batch,), fogged)
    )
    state, obs, _ = api.perceive(rules, state)
    chaser = Chaser(characteristics(types, rules), config.ships_per_team)
    views = []
    for tick in range(ticks + 1):
        views.append(
            Obs(
                ships=obs.ships.flatten(0, 1),
                fields=obs.fields.flatten(0, 1),
                globals=obs.globals.flatten(0, 1),
                zones=obs.zones.flatten(0, 1),
            )
        )
        if tick == ticks:
            break
        action, _, _ = chaser.step(
            None,
            Obs(ships=views[-1].ships, fields=None, globals=None),
            None,
            torch.arange(Obs(ships=views[-1].ships, fields=None, globals=None).ships.shape[0]),
        )
        packed = action.packed().reshape(batch, config.num_ships, 3)
        state, _ = api.step(
            rules,
            types,
            state,
            Action.unpack(packed),
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        state, obs, _ = api.perceive(rules, state)
    return views


def _views(config: GameConfig, fogged: bool, ticks: int = 0, seed: int = 0) -> Obs:
    return _view_sequence(config, fogged, ticks, seed)[-1]


def _run(net, views: list[Obs], carry):
    """Steps over views (the first a reset); returns the carry after them."""
    reset = torch.ones(views[0].ships.shape[0], dtype=torch.bool)
    for view in views:
        decision = net.step(carry, view, reset, torch.arange(view.ships.shape[0]))
        carry = dataclasses.replace(
            decision.carry, previous_action=decision.policy_logits.argmax(-1).int()
        )
        reset = torch.zeros_like(reset)
    return carry


def test_visible_ship_code_is_the_sharp_code_of_its_observation():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True, ticks=1), torch.float64)
    carry = net.initial_carry(view.ships.shape[0], OWN, 10)
    no_reset = torch.zeros(view.ships.shape[0], dtype=torch.bool)
    with torch.no_grad():
        decision = net.step(carry, view, no_reset, torch.arange(view.ships.shape[0]))
        tokens = net.tokens(view, decision.replay)
    visible = view.ships[..., SHIP["visible"]] > 0.5
    ship_type = view.ships[..., SHIP["ship_type"]].long()
    sharp = net.state_code.sharp(truth(view.ships, decision.replay.offset), ship_type)
    assert visible.any() and (~visible).any()
    assert torch.equal(tokens.code[visible], sharp[visible])
    assert (decision.replay.spreads[visible[:, OWN:]] == 0).all()  # the replay's enemy rows


def test_spawning_voids_a_belief():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True), torch.float64)
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    believed = dataclasses.replace(
        carry.belief,
        valid=torch.ones_like(carry.belief.valid),
        means=torch.full_like(carry.belief.means, 7.0),
        spreads=torch.full_like(carry.belief.spreads, 3.0),
        action=torch.softmax(torch.randn_like(carry.belief.action), -1),
    )
    carry = dataclasses.replace(carry, belief=believed)
    ships = view.ships.clone()
    enemy = slice(5, 10)
    ships[:, enemy, :SHIP_HIDDEN_ZERO] = 0.0
    ships[:, enemy, SHIP["visible"]] = 0.0
    ships[:, enemy, SHIP["spawned"]] = 0.0
    ships[:, 5, SHIP["spawned"]] = 1.0  # spawned out of sight: the old belief is void
    no_reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        replay = net.step(
            carry,
            dataclasses.replace(view, ships=ships),
            no_reset,
            torch.arange(dataclasses.replace(view, ships=ships).ships.shape[0]),
        ).replay
    # The replay holds the enemy rows: row 5 is its row 0.
    assert not replay.valid[:, 0].any() and replay.valid[:, 1:5].all()
    assert (replay.means[:, 0] == 0).all() and (replay.spreads[:, 0] == 0).all()
    forecast = replay.enemy_action[:, 0].double() / ACTION_SCALE
    assert torch.allclose(forecast, torch.full_like(forecast, 1 / 42), atol=1e-4)
    assert (replay.means[:, 1:5] == 7.0).all() and (replay.spreads[:, 1:5] == 3.0).all()


def test_without_fog_every_living_ship_is_certain():
    """Fog off shows every living ship, so its belief is truth with no spread. (A dead
    enemy is not shown even without fog; its row keeps a forecast.)"""
    net = cases.model()
    _, trajectory = cases.rollout(net, fog_share=0.0, length=64)
    alive = trajectory.obs.ships[..., OWN:, SHIP["alive"]] > 0.5  # the replay's enemy rows
    assert (trajectory.replay.spreads[alive] == 0).all()
    assert trajectory.replay.valid[alive].all()


def test_hidden_columns_of_hidden_ships_are_not_read():
    """After some ticks of history, so hidden ships hold beliefs (on a blank carry valid is
    visible)."""
    net = cases.model(dtype=torch.float64)
    sequence = [
        cases.cast(view, torch.float64)
        for view in _view_sequence(cases.SQUAD, fogged=True, ticks=40)
    ]
    with torch.no_grad():
        carry = _run(net, sequence[:-1], net.initial_carry(sequence[0].ships.shape[0], OWN, 10))
    view = sequence[-1]
    hidden = view.ships[..., SHIP["visible"]] < 0.5
    assert (hidden & carry.belief.valid).any(), "a hidden ship should hold a belief"
    noise = torch.randn(view.ships.shape[:-1] + (SHIP_HIDDEN_ZERO,), dtype=torch.float64) * 100
    ships = view.ships.clone()
    ships[..., :SHIP_HIDDEN_ZERO] = torch.where(
        hidden[..., None], noise, ships[..., :SHIP_HIDDEN_ZERO]
    )
    reset = torch.zeros(view.ships.shape[0], dtype=torch.bool)
    with torch.no_grad():
        first = net.step(carry, view, reset, torch.arange(view.ships.shape[0]))
        second = net.step(
            carry,
            dataclasses.replace(view, ships=ships),
            reset,
            torch.arange(dataclasses.replace(view, ships=ships).ships.shape[0]),
        )
    for a, b in zip(_pytree.tree_leaves(first), _pytree.tree_leaves(second), strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------ forecast feedback


def _feedback_off(segment):
    """The segment's model with forecast_feedback off (the same weights), in float64."""
    net, obs = _float64(segment)
    off = cases.double(cases.model(forecast_feedback=False))
    off.load_state_dict(net.state_dict())
    return off, obs


def test_forecast_feedback_on_is_the_default_and_feeds_the_forecast_back(segment):
    """On, the next tick reads the enemy-action head's probabilities as the pending enemy
    action and a hidden ship's row carries the decoded forecast's spread. (That on is
    today's model bit for bit was checked against the commit before the switch: a rollout,
    its step loop, carry and unroll were identical.)"""
    assert YemongConfig().forecast_feedback
    net, obs = _float64(segment)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment[1]))
    replay = decisions.replay
    spawned = obs.ships[1:, :, OWN:, SHIP["spawned"]] > 0.5
    kept = ~spawned & ~cases.resets(obs)[1:, :, None]
    forecast = store_action(torch.softmax(decisions.enemy_action_logits[:-1], -1))
    assert kept.any()
    assert torch.equal(replay.enemy_action[1:][kept], forecast[kept])
    hidden = (obs.ships[..., OWN:, SHIP["visible"]] < 0.5) & replay.valid
    assert (replay.spreads[hidden] > 0).any()


def test_forecast_feedback_off_holds_the_last_sighting(segment):
    """Off, a hidden ship's row is its last sighting at every later tick until it is seen
    again (state as seen, no spread), its age counts the ticks since, every enemy action is
    uniform, and nothing non-finite is counted."""
    net, obs = _feedback_off(segment)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        joint = cases.joint_actions(segment[1])
        decisions, final = cases.step_loop(net, net.step, carry, obs, joint)
    replay = decisions.replay
    enemy = obs.ships[..., OWN:, :]
    visible = enemy[..., SHIP["visible"]] > 0.5
    sighting = torch.zeros_like(replay.means)
    last_seen = torch.zeros(replay.age.shape[1:], dtype=torch.int64)
    held = 0
    for t in range(obs.ships.shape[0]):
        seen = truth(enemy[t], replay.offset[t])
        sighting[t] = torch.where(visible[t][..., None], seen, sighting[t - 1] if t else 0.0)
        last_seen = torch.where(visible[t], t, last_seen)
        hidden = replay.valid[t] & ~visible[t]
        held += int(hidden.sum())
        assert torch.equal(replay.means[t][replay.valid[t]], sighting[t][replay.valid[t]])
        assert torch.equal(replay.age[t][hidden], (t - last_seen[hidden]).int())
        tokens = net.tokens(
            _pytree.tree_map_only(torch.Tensor, lambda x: x[t], obs),
            _pytree.tree_map_only(torch.Tensor, lambda x: x[t], replay),
        )
        ship_type = enemy[t][..., SHIP["ship_type"]].long()
        sharp = net.state_code.sharp(sighting[t], ship_type)
        assert torch.equal(tokens.code[:, OWN:][hidden], sharp[hidden])
        position = tokens.query_position[:, OWN:-1]
        assert torch.equal(position[hidden], sighting[t][..., 0:2][hidden])
    assert held > 0, "the segment should hide ships the team has seen"
    assert (replay.spreads == 0).all()
    assert (replay.enemy_action == store_action(torch.tensor(1.0 / NUM_JOINT))).all()
    assert (final.nonfinite == 0).all()


def test_unroll_equals_steps_with_forecast_feedback_off(segment):
    net, obs = _feedback_off(segment)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment[1]))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    _compare(unrolled, decisions, 1e-9)


# ------------------------------------------------------------------ the position offset


def test_resets_draw_fresh_offsets_by_the_decision_key_and_keep_them_within_a_match():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True), torch.float64)
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    reset = torch.ones(views, dtype=torch.bool)
    keys = torch.arange(views)
    with torch.no_grad():
        first = net.step(carry, view, reset, keys)
        again = net.step(carry, view, reset, keys)
        second = net.step(carry, view, reset, keys + views)
        kept = net.step(first.carry, view, ~reset, keys + 2 * views)
        unreset = net.step(carry, view, ~reset, keys)  # a fresh carry has none: drawn
    offsets = first.replay.offset
    assert ((offsets >= 0) & (offsets < POSITION_PERIOD)).all()
    assert torch.equal(again.replay.offset, offsets)  # the same key, the same offset
    assert (offsets != second.replay.offset).all()
    assert torch.equal(kept.replay.offset, offsets)
    assert torch.equal(unreset.replay.offset, offsets)


def test_the_offset_reaches_the_outputs_only_through_the_position_codes():
    """With the encoder's position-code columns zeroed, a decision does not depend on the
    offset: the same history in a frame shifted by delta (belief positions and offset moved
    alike) gives the same outputs. Rotary attention and presence read only differences."""
    net = cases.model(dtype=torch.float64)
    with torch.no_grad():
        encoder = net.encoder
        encoder.ship.first.weight[:, :POSITION_CODE] = 0.0
        encoder.field.first.weight[:, :POSITION_CODE] = 0.0
        encoder.global_token.first.weight[:, :POSITION_CODE] = 0.0
    sequence = [
        cases.cast(view, torch.float64)
        for view in _view_sequence(cases.SQUAD, fogged=True, ticks=24)
    ]
    views = sequence[0].ships.shape[0]
    with torch.no_grad():
        carry = _run(net, sequence[:-1], net.initial_carry(views, OWN, 10))
        hidden = sequence[-1].ships[..., SHIP["visible"]] < 0.5
        assert (hidden & carry.belief.valid).any()  # forecasts in the shifted frame
        delta = torch.rand(views, 2, dtype=torch.float64) * POSITION_PERIOD
        means = carry.belief.means.clone()
        means[..., 0:2] += delta[:, None, :]
        shifted = dataclasses.replace(
            carry,
            offset=carry.offset + delta,
            belief=dataclasses.replace(carry.belief, means=means),
        )
        reset = torch.zeros(views, dtype=torch.bool)
        plain, moved = (
            net.step(carry, sequence[-1], reset, torch.arange(sequence[-1].ships.shape[0])),
            net.step(shifted, sequence[-1], reset, torch.arange(sequence[-1].ships.shape[0])),
        )
    for name in ("policy_logits", "enemy_action_logits", "values", "outcome_probabilities"):
        torch.testing.assert_close(getattr(moved, name), getattr(plain, name), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        moved.next_state_logits[..., POSITION_CODE:],
        plain.next_state_logits[..., POSITION_CODE:],
        rtol=1e-8,
        atol=1e-8,
    )


# ------------------------------------------------------------------ symmetry and size


def test_permuting_ships_within_a_team_permutes_every_output():
    net = cases.model(dtype=torch.float64)
    view = cases.cast(_views(cases.SQUAD, fogged=True, ticks=30, seed=1), torch.float64)
    views = view.ships.shape[0]
    own = torch.tensor([3, 0, 4, 1, 2])
    enemy = torch.tensor([2, 4, 0, 3, 1])
    rows = torch.cat([own, OWN + enemy])
    permuted = dataclasses.replace(view, ships=view.ships[:, rows])
    carry = net.initial_carry(views, OWN, 10)  # one offset for both orders
    reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        plain, swapped = (
            net.step(carry, view, reset, torch.arange(view.ships.shape[0])),
            net.step(carry, permuted, reset, torch.arange(permuted.ships.shape[0])),
        )
        # and a second step from each carry, the previous actions permuted alike
        joint = torch.randint(0, 42, (views, OWN), dtype=torch.int32)
        plain_carry = dataclasses.replace(plain.carry, previous_action=joint)
        swapped_carry = dataclasses.replace(swapped.carry, previous_action=joint[:, own])
        plain_next = net.step(plain_carry, view, reset, torch.arange(view.ships.shape[0]))
        swapped_next = net.step(
            swapped_carry, permuted, reset, torch.arange(permuted.ships.shape[0])
        )
    for first, second in ((plain, swapped), (plain_next, swapped_next)):
        for name, index in (
            ("policy_logits", own),
            ("values", own),
            ("enemy_action_logits", enemy),
            ("next_state_logits", enemy),
        ):
            torch.testing.assert_close(
                getattr(first, name)[:, index], getattr(second, name), rtol=1e-10, atol=1e-10
            )
        torch.testing.assert_close(
            first.outcome_probabilities, second.outcome_probabilities, rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize(
    "config",
    [GameConfig(1, 4, 1200.0), cases.SQUAD, GameConfig(50, 40, 8200.0)],
    ids=["duel", "squad", "battle"],
)
def test_one_set_of_weights_plays_every_size(config):
    net = cases.model(cases.SQUAD)
    view = _views(config, fogged=True)
    views, own, ships = view.ships.shape[0], config.ships_per_team, config.num_ships
    carry = net.initial_carry(views, own, ships)
    with torch.no_grad():
        decision = net.step(
            carry, view, torch.ones(views, dtype=torch.bool), torch.arange(view.ships.shape[0])
        )
    assert decision.policy_logits.shape == (views, own, 42)
    assert decision.values.shape == (views, own, len(REWARD_LEVELS))
    assert decision.next_state_logits.shape == (views, ships - own, 544)
    assert all(
        torch.isfinite(x).all() for x in _pytree.tree_leaves(decision) if x.is_floating_point()
    )


# ------------------------------------------------------------------ keyed initialisation


def _initial(seed: int = 0, **settings) -> dict[str, torch.Tensor]:
    build = yemong_build(YemongConfig(**settings), api.rules_of(cases.SQUAD), api.ship_types())
    return {name: value.clone() for name, value in Yemong(build, seed).state_dict().items()}


@pytest.mark.parametrize(
    "settings",
    [
        {"gate": "elementwise"},
        {"geometry_head": False},
        {"blocks": YemongConfig().blocks + 1},
        {"delta_split": False, "physics_hidden_decay": None},
    ],
    ids=["gate", "geometry_head", "blocks", "delta_split"],
)
def test_an_unrelated_option_leaves_every_shared_initial_weight_bit_identical(settings):
    """Each parameter draws from its own generator keyed by the seed and its path: a module
    added or removed shifts no other module's initial values, whatever the default
    generator holds."""
    torch.manual_seed(1)
    reference = _initial()
    torch.manual_seed(2)
    other = _initial(**settings)
    shared = [
        name for name in reference if name in other and other[name].shape == reference[name].shape
    ]
    assert len(shared) >= 0.9 * min(len(reference), len(other))
    for name in shared:
        assert torch.equal(reference[name], other[name]), name


def test_the_seed_changes_every_drawn_initial_weight():
    first, second = _initial(0), _initial(1)
    drawn = [name for name, value in first.items() if value.dim() == 2 and value.std() > 0]
    assert drawn and all(not torch.equal(first[name], second[name]) for name in drawn)
