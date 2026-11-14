"""Yemong's inputs and carry rules the reference fixtures and the unroll tests did not reach
(yemong spec 4-6): a fresh spawn's pending action, presence from valid beliefs only, the
time since seen in seconds, the forecast's nearest position image, and unroll around
resets. Mutation-sweep items 23 to 28.

A tiny Yemong (one block, width 8) with every weight redrawn from a local generator, on
real team-0 views of short games the chaser plays.
"""

import contextlib
import dataclasses

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs
from bnb.contracts.action import NUM_JOINT, to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.models.yemong.critic import symlog
from bnb.models.yemong.decode import nearest_representative
from bnb.models.yemong.inputs import NULL_ACTION, compose, store_action
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import POSITION_PERIOD

TYPES = api.ship_types()
# Half-second episodes (15 ticks), so a short sequence crosses episode ends.
GAME = GameConfig(2, 3, 2000.0, episode_seconds=0.5, mode="sandbox")
S, N = GAME.ships_per_team, GAME.num_ships
# The ship token's tail: ... time since seen (1), respawn features (9), the pending action
# (NUM_JOINT), the gradient (2), presence (2).
PRESENCE = slice(-2, None)
PENDING = slice(-(NUM_JOINT + 4), -4)
SINCE_SEEN = -(NUM_JOINT + 4 + 9 + 1)


@pytest.fixture(autouse=True)
def _default_generator_untouched():
    """The rollout and the model draw from the default generator by design (seats, offsets,
    the random controller): restore it, so no other test sees these draws."""
    with torch.random.fork_rng(devices=[]):
        yield


@contextlib.contextmanager
def _default_generator(seed: int):
    """The default generator at a known state inside the block, restored after it (the
    model's constructor and its offset draws use it)."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.set_state(torch.Generator().manual_seed(seed).get_state())
        yield


def _model(seed: int = 0) -> Yemong:
    config = YemongConfig(
        width=8,
        blocks=1,
        spatial_per_block=1,
        heads=1,
        map_width=4,
        value_bins=11,
        hex_rings=1,
        rotary_octaves=1,
        heading_harmonics=(1,),
    )
    with _default_generator(seed):
        net = Yemong(yemong_build(config, api.rules_of(GAME), TYPES))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    return net.eval()


def _game(environments: int, ticks: int, fogged: bool = True, seed: int = 2):
    """Team 0's views of `ticks` chaser ticks, every clock from zero: (views per tick,
    resets (ticks, B), team 0's joint actions (ticks, B, S))."""
    rules = api.rules_of(GAME)
    generator = torch.Generator().manual_seed(seed)
    fog = torch.full((environments,), fogged)
    state = api.reset(rules, TYPES, api.uniforms(rules, environments, generator=generator), fog)
    state, observation, _ = api.perceive(rules, state)
    chaser = Chaser(characteristics(TYPES, rules), S)
    views, resets, joints = [], [], []
    for _ in range(ticks):
        team0, team1 = (
            Obs(
                **{
                    part: getattr(observation, part)[:, team]
                    for part in ("ships", "fields", "globals", "zones")
                }
            )
            for team in (0, 1)
        )
        views.append(team0)
        resets.append(team0.globals[:, GLOBAL["reset"]] > 0.5)
        actions = [
            chaser.step(None, view, None, torch.arange(view.ships.shape[0]))[0]
            for view in (team0, team1)
        ]
        joints.append(to_joint(actions[0]).to(torch.int32))
        packed = torch.cat([action.packed() for action in actions], 1)
        state, _ = api.step(
            rules,
            TYPES,
            state,
            Action.unpack(packed),
            api.spread(rules, environments, generator=generator),
            api.uniforms(rules, environments, generator=generator),
        )
        state, observation, _ = api.perceive(rules, state)
    return views, torch.stack(resets), torch.stack(joints)


def _loop(net, carry, views, resets, joints, start, stop):
    """Steps start..stop-1, each fed its stored joint action back: (decisions stacked over
    time, the carry after)."""
    decisions = []
    for t in range(start, stop):
        decision = net.step(carry, views[t], resets[t], torch.arange(views[t].ships.shape[0]))
        carry = dataclasses.replace(decision.carry, previous_action=joints[t])
        decisions.append(decision)
    return _pytree.tree_map(lambda *x: torch.stack(x), *decisions), carry


def _stacked(views, start, stop) -> Obs:
    return _pytree.tree_map(lambda *x: torch.stack(x), *views[start:stop])


def _decision(net, view, environments):
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(environments, S, N)
        return net.step(
            carry,
            view,
            torch.ones(environments, dtype=torch.bool),
            torch.arange(view.ships.shape[0]),
        )


# ------------------------------------------------------------------ items 23 to 25: compose


def test_a_ship_that_spawned_this_tick_enters_with_the_null_pending_action():
    net = _model()
    views, _, _ = _game(2, 4)
    view = views[-1]
    replay = _decision(net, view, 2).replay
    forecast = torch.zeros(2, N - S, NUM_JOINT)
    forecast[..., NUM_JOINT - 1] = 1.0  # a stale forecast: boost, hard turn, fire
    replay = dataclasses.replace(replay, enemy_action=store_action(forecast))
    ships = view.ships.clone()
    ships[:, S, SHIP["spawned"]] = 1.0  # enemy row S spawned this tick; row S + 1 did not
    ships[:, S + 1, SHIP["spawned"]] = 0.0
    tokens = compose(
        dataclasses.replace(view, ships=ships),
        replay,
        net.state_code,
        net.ship_types,
        net.config.presence_radius,
        net.tick_length,
    )
    pending = tokens.ship[..., PENDING]  # (R, N, NUM_JOINT)
    null = torch.nn.functional.one_hot(torch.tensor(NULL_ACTION), NUM_JOINT).float()
    torch.testing.assert_close(pending[:, S], null.expand(2, -1), rtol=0, atol=0)
    torch.testing.assert_close(pending[:, S + 1], forecast[:, 1], rtol=0, atol=1e-4)


def test_presence_counts_only_ships_with_a_valid_belief():
    """Every enemy belief invalid (so each sits at the map centre) and own ship 0 at the map
    centre: no enemy presence for anyone."""
    net = _model()
    views, _, _ = _game(2, 4)
    view = views[-1]
    replay = _decision(net, view, 2).replay
    replay = dataclasses.replace(replay, valid=torch.zeros_like(replay.valid))
    ships = view.ships.clone()
    ships[:, 0, SHIP["x"]] = 0.0  # the world origin: the map centre in the shifted frame
    ships[:, 0, SHIP["y"]] = 0.0
    tokens = compose(
        dataclasses.replace(view, ships=ships),
        replay,
        net.state_code,
        net.ship_types,
        net.config.presence_radius,
        net.tick_length,
    )
    enemy_presence = tokens.ship[:, :S, PRESENCE][..., 1]
    assert torch.all(enemy_presence == 0.0)


def test_the_time_since_seen_input_is_in_seconds():
    net = _model()
    views, _, _ = _game(2, 4)
    view = views[-1]
    replay = _decision(net, view, 2).replay
    age = 90  # ticks: three seconds
    replay = dataclasses.replace(replay, age=torch.full_like(replay.age, age))
    tokens = compose(
        view,
        replay,
        net.state_code,
        net.ship_types,
        net.config.presence_radius,
        net.tick_length,
    )
    want = symlog(torch.tensor(age * net.tick_length))
    torch.testing.assert_close(
        tokens.ship[:, S:, SINCE_SEEN], want.expand(2, N - S), rtol=0, atol=1e-6
    )
    assert torch.all(tokens.ship[:, :S, SINCE_SEEN] == 0.0)  # own rows: seen now


def test_the_ship_token_reads_the_benched_flag_and_the_model_steps_with_it():
    """Benching an own and an enemy row moves one ship-token input (the flag's, 1 on exactly
    those rows) and nothing else; a step reads it and stays finite."""
    net = _model()
    views, _, _ = _game(2, 4)
    view = views[-1]
    replay = _decision(net, view, 2).replay

    def tokens(ships):
        return compose(
            dataclasses.replace(view, ships=ships),
            replay,
            net.state_code,
            net.ship_types,
            net.config.presence_radius,
            net.tick_length,
        ).ship

    ships = view.ships.clone()
    assert torch.all(ships[..., SHIP["benched"]] == 0.0)  # every ship active
    benched = ships.clone()
    benched[:, [S - 1, N - 1], SHIP["benched"]] = 1.0
    change = tokens(benched) - tokens(ships)
    moved = (change != 0).any(0).any(0).nonzero()[:, 0]
    assert moved.numel() == 1, moved
    expected = torch.zeros(2, N)
    expected[:, [S - 1, N - 1]] = 1.0
    torch.testing.assert_close(change[..., moved[0]], expected, rtol=0, atol=0)
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(2, S, N)
        decision = net.step(
            carry,
            dataclasses.replace(view, ships=benched),
            torch.ones(2, dtype=torch.bool),
            torch.arange(dataclasses.replace(view, ships=benched).ships.shape[0]),
        )
    assert all(
        bool(torch.isfinite(leaf).all())
        for leaf in _pytree.tree_leaves(decision)
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point()
    )


def _benched_views(ticks: int):
    """Team 0's views of a fogged 5-a-side chaser game, environment 0 at 4 against 5 (own
    slot 4 benched) and environment 1 at 5 against 4 (enemy slot 4, row 9, benched): (the
    views, the benched rows (2, N) bool)."""
    game = dataclasses.replace(GAME, ships_per_team=5)
    rules, S5 = api.rules_of(game), game.ships_per_team
    active = torch.tensor([[4, 5], [5, 4]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(5)
    state = api.reset(rules, TYPES, api.uniforms(rules, 2, generator=generator), active=active)
    state, observation, _ = api.perceive(rules, state, active)
    chaser = Chaser(characteristics(TYPES, rules), S5)
    views = []
    for _ in range(ticks):
        team0, team1 = (
            Obs(
                **{
                    part: getattr(observation, part)[:, team]
                    for part in ("ships", "fields", "globals", "zones")
                }
            )
            for team in (0, 1)
        )
        views.append(team0)
        actions = [
            chaser.step(None, view, None, torch.arange(view.ships.shape[0]))[0]
            for view in (team0, team1)
        ]
        packed = torch.cat([action.packed() for action in actions], 1)
        state, _ = api.step(
            rules,
            TYPES,
            state,
            Action.unpack(packed),
            api.spread(rules, 2, generator=generator),
            api.uniforms(rules, 2, generator=generator),
            active,
        )
        state, observation, _ = api.perceive(rules, state, active)
    benched = views[0].ships[..., SHIP["benched"]] > 0.5
    return game, views, benched


def _benched_model(game: GameConfig, seed: int = 0) -> Yemong:
    config = YemongConfig(
        width=8,
        blocks=1,
        spatial_per_block=1,
        heads=1,
        map_width=4,
        value_bins=11,
        hex_rings=1,
        rotary_octaves=1,
        heading_harmonics=(1,),
    )
    with _default_generator(seed):
        net = Yemong(yemong_build(config, api.rules_of(game), TYPES))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    return net.eval()


def _masked_run(net, game, views, benched, noise: torch.Generator | None):
    """Decisions over the views with each step's benched rows overwritten by noise (every
    column but the flag; and the benched own ship's previous action) when noise is given."""
    S5, N5 = game.ships_per_team, game.num_ships
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(2, S5, N5)
    decisions = []
    for t, view in enumerate(views):
        ships = view.ships
        if noise is not None:
            junk = 1e3 * torch.randn(ships.shape, generator=noise)
            junk[..., SHIP["benched"]] = ships[..., SHIP["benched"]]
            ships = torch.where(benched[..., None], junk, ships)
        with torch.no_grad(), _default_generator(11 + t):  # the reset's offset draw
            decision = net.step(
                carry,
                dataclasses.replace(view, ships=ships),
                torch.full((2,), t == 0),
                torch.arange(dataclasses.replace(view, ships=ships).ships.shape[0]),
            )
        joint = decision.policy_logits.argmax(-1).to(torch.int32)
        if noise is not None:
            other = torch.randint(0, NUM_JOINT, joint.shape, generator=noise, dtype=torch.int32)
            joint = torch.where(benched[:, :S5], other, joint)
        carry = dataclasses.replace(decision.carry, previous_action=joint)
        decisions.append(decision)
    return decisions


def test_with_the_mask_on_the_active_ships_read_nothing_of_the_benched_rows():
    """A 4v5 and a 5v4 game: anything written into the benched rows (own and enemy) leaves
    every active ship's policy, values, enemy forecasts and belief, and the outcome, as
    they were; with the mask off it does not (the test can tell)."""
    game, views, benched = _benched_views(6)
    S5 = game.ships_per_team
    assert benched[0, S5 - 1] and benched[1, -1] and int(benched.sum()) == 2
    own_active, enemy_active = ~benched[:, :S5], ~benched[:, S5:]
    net = _benched_model(game)
    net.mask_benched = True
    clean = _masked_run(net, game, views, benched, None)
    dirty = _masked_run(net, game, views, benched, torch.Generator().manual_seed(9))
    for one, two in zip(clean, dirty, strict=True):
        torch.testing.assert_close(one.policy_logits[own_active], two.policy_logits[own_active])
        torch.testing.assert_close(one.values[own_active], two.values[own_active])
        torch.testing.assert_close(
            one.enemy_action_logits[enemy_active], two.enemy_action_logits[enemy_active]
        )
        torch.testing.assert_close(
            one.next_state_logits[enemy_active], two.next_state_logits[enemy_active]
        )
        torch.testing.assert_close(one.outcome_probabilities, two.outcome_probabilities)
        playing = ~benched
        torch.testing.assert_close(one.carry.belief.means[playing], two.carry.belief.means[playing])
    net.mask_benched = False
    clean = _masked_run(net, game, views, benched, None)
    dirty = _masked_run(net, game, views, benched, torch.Generator().manual_seed(9))
    assert not torch.allclose(
        clean[-1].policy_logits[own_active], dirty[-1].policy_logits[own_active]
    )


def test_the_mask_on_a_full_team_computes_what_it_does_off():
    """No benched row: the masked attention (the plain path, every key shown) and the
    unmasked one agree within rounding, and the switch reads only the flag."""
    net = _model()
    views, _, _ = _game(2, 4)
    off = _decision(net, views[-1], 2)
    net.mask_benched = True
    on = _decision(net, views[-1], 2)
    torch.testing.assert_close(on.policy_logits, off.policy_logits)
    torch.testing.assert_close(on.values, off.values)
    torch.testing.assert_close(on.enemy_action_logits, off.enemy_action_logits)


# ------------------------------------------------------------------ item 26: the image


def test_the_nearest_image_crosses_the_period_boundary():
    period = POSITION_PERIOD
    position = torch.tensor([3.0, period - 3.0, 100.0])
    reference = torch.tensor([period - 5.0, 5.0, 100.0 + 2.0 * period])
    got = nearest_representative(position, reference, period)
    torch.testing.assert_close(got, torch.tensor([period + 3.0, -3.0, 100.0 + 2.0 * period]))


def test_the_forecast_position_of_a_hidden_enemy_is_the_image_nearest_its_belief():
    """A held belief two periods out (the shifted frame is continuous): the decoded
    forecast, which comes back in [0, P), is moved to within half a period of it."""
    net = _model()
    views, _, _ = _game(2, 30, fogged=True)
    view = views[-1]
    hidden = (view.ships[:, S:, SHIP["visible"]] < 0.5) & (view.ships[:, S:, SHIP["spawned"]] < 0.5)
    assert bool(hidden.all()), "every enemy row should be hidden in this view"
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(2, S, N)
    belief = carry.belief
    reference = torch.tensor([2.0 * POSITION_PERIOD + 50.0, -1.5 * POSITION_PERIOD])
    means = belief.means.clone()
    means[:, S:, 0:2] = reference
    belief = dataclasses.replace(
        belief,
        valid=torch.ones_like(belief.valid),
        age=torch.ones_like(belief.age),
        means=means,
    )
    carry = dataclasses.replace(carry, belief=belief)
    with torch.no_grad():
        decision = net.step(
            carry, view, torch.zeros(2, dtype=torch.bool), torch.arange(view.ships.shape[0])
        )
    forecast = decision.carry.belief.means[:, S:, 0:2]
    assert bool(((forecast - reference).abs() <= 0.5 * POSITION_PERIOD).all()), forecast


# ------------------------------------------------------------------ items 27 and 28: unroll


def _unrolled(net, carry, views, resets, decisions, start, stop):
    with torch.no_grad():
        return net.unroll(carry, _stacked(views, start, stop), decisions.replay, resets[start:stop])


def test_unroll_from_a_carry_with_history_whose_first_step_is_a_reset_equals_a_fresh_start():
    """Episode 1 is ticks 0-14; the segment is episode 2 (a reset at its first step)."""
    net = _model()
    views, resets, joints = _game(4, 30)
    assert bool(resets[15].all()) and not bool(resets[1:15].any())
    with torch.no_grad(), _default_generator(7):
        fresh = net.initial_carry(4, S, N)
        _, history = _loop(net, fresh, views, resets, joints, 0, 15)
        decisions, _ = _loop(net, history, views, resets, joints, 15, 30)
    assert float(history.state.abs().max()) > 0.0
    from_history = _unrolled(net, history, views, resets, decisions, 15, 30)
    from_zero = _unrolled(net, fresh, views, resets, decisions, 15, 30)
    for name in ("policy_logits", "values", "outcome_logits"):
        torch.testing.assert_close(getattr(from_history, name), getattr(from_zero, name))
    torch.testing.assert_close(from_history.policy_logits, decisions.policy_logits)


def test_unroll_equals_the_step_loop_around_a_reset_mid_segment():
    """A segment from tick 5 (a carry with history) across the episode ends at ticks 15 and
    30."""
    net = _model()
    views, resets, joints = _game(4, 40)
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(4, S, N)
        _, carry = _loop(net, carry, views, resets, joints, 0, 5)
        decisions, _ = _loop(net, carry, views, resets, joints, 5, 40)
    assert bool(resets[15].all()) and bool(resets[30].all())
    unrolled = _unrolled(net, carry, views, resets, decisions, 5, 40)
    torch.testing.assert_close(unrolled.policy_logits, decisions.policy_logits)
    torch.testing.assert_close(unrolled.values, decisions.values)
