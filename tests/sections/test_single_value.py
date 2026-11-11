"""The single value arm (`YemongConfig` levels (SUMMED_LEVEL,), outcome_level None): one
per-ship categorical value over the sum of every weighted reward level, no outcome head;
and the default layout unchanged."""

import copy

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, SUMMED_LEVEL
from bnb.learn import returns
from bnb.models.yemong.model import YemongConfig
from bnb.runtime import metrics

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")

OWN = lc.OWN
ARM = dict(levels=(SUMMED_LEVEL,), outcome_level=None)
# one discount, trace decay and team spirit for the one value level
SINGLE = dict(discount=(0.999,), trace_decay=(0.97,), team_spirit=(0.0,))


@pytest.fixture(scope="module")
def arm(module_global_generator):
    """The arm's model, a rollout with it as the learner, and a learner over it."""
    net = cases.model(**ARM)
    runner = lc.rollout(net)
    return net, runner


def test_the_default_layout_is_one_value_per_reward_level_and_the_outcome_head():
    config = YemongConfig()
    assert config.levels == REWARD_LEVELS and config.outcome_level == "outcome"
    net = cases.model()
    assert net.outcome is not None
    assert net.own_heads.lasts[1].out_features == (len(REWARD_LEVELS) - 1) * config.value_bins
    rewards = torch.randn(3, 2, OWN, len(REWARD_LEVELS))
    assert returns.value_rewards(rewards, REWARD_LEVELS) is rewards  # untouched, bit for bit


def test_the_arm_has_one_per_ship_value_head_and_no_outcome_head(arm):
    net, _ = arm
    assert net.config.local_levels == (0,)
    assert net.outcome is None
    assert not any(name.startswith("outcome.") for name, _ in net.named_parameters())
    assert net.own_heads.lasts[1].out_features == net.config.value_bins


def test_an_outcome_level_must_be_a_level_and_levels_distinct():
    with pytest.raises(ValueError, match="outcome level"):
        YemongConfig(levels=(SUMMED_LEVEL,), outcome_level="outcome")
    with pytest.raises(ValueError, match="distinct"):
        YemongConfig(levels=(), outcome_level=None)
    with pytest.raises(ValueError, match="distinct"):
        YemongConfig(levels=("damage", "damage"), outcome_level=None)


def test_the_arm_s_values_are_one_column_and_its_step_equals_its_unroll(arm):
    """Values (..., S, 1), no outcome; a loop of step equals unroll in float64."""
    net, runner = arm
    trajectory = lc.whole(runner)
    assert trajectory.values.shape == trajectory.value.shape + (1,)
    assert trajectory.outcome is None
    torch.testing.assert_close(trajectory.value, trajectory.values[..., 0], rtol=0, atol=0)
    double = cases.double(net)
    obs = cases.cast(trajectory.obs, torch.float64)
    carry = double.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(
            double, double.step, carry, obs, cases.joint_actions(trajectory)
        )
        unrolled = double.unroll(carry, obs, decisions.replay, cases.resets(obs))
    assert decisions.outcome_probabilities is None and unrolled.outcome_logits is None
    assert unrolled.values.shape == unrolled.policy_logits.shape[:-1] + (1,)
    assert unrolled.value_logits.shape[-2:] == (1, net.config.value_bins)
    for name in ("policy_logits", "values"):
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= 1e-9, (name, error)


def test_the_summed_reward_s_gae_is_the_hand_computed_one():
    """Two ticks of one ship, each level's weighted reward summed into one stream; the
    episode ends on tick 1 of view 0, view 1 bootstraps."""
    rewards = torch.tensor(
        [
            [[[0.1, 0.0, 0.2, 0.0, 0.0]], [[0.0, 0.5, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.3, 1.0]], [[0.2, 0.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=torch.float64,
    )  # (L 2, R 2, S 1, levels)
    summed = returns.value_rewards(rewards, (SUMMED_LEVEL,))
    assert summed.shape == (2, 2, 1, 1)
    values = torch.tensor([[[[0.4]], [[0.1]]], [[[0.6]], [[-0.2]]]], dtype=torch.float64)
    terminated = torch.tensor([[False, False], [True, False]])
    next_value = torch.tensor([[[9.0]], [[0.5]]], dtype=torch.float64)
    gamma, lam = 0.9, 0.8
    advantages, level_returns = returns.per_level_gae(
        summed,
        values,
        terminated,
        next_value,
        torch.tensor([False, False]),
        torch.tensor([gamma], dtype=torch.float64),
        torch.tensor([lam], dtype=torch.float64),
    )
    # view 0: tick 1 ends the episode (no bootstrap); tick 0 reads tick 1's value
    delta_1 = 1.3 - 0.6
    delta_0 = 0.3 + gamma * 0.6 - 0.4
    # view 1: tick 1 bootstraps from 0.5
    delta_1b = 0.2 + gamma * 0.5 - (-0.2)
    delta_0b = 0.5 + gamma * (-0.2) - 0.1
    expected = torch.tensor(
        [
            [[[delta_0 + gamma * lam * delta_1]], [[delta_0b + gamma * lam * delta_1b]]],
            [[[delta_1]], [[delta_1b]]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(advantages, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(level_returns, expected + values, rtol=0, atol=1e-12)


def test_the_returns_reject_a_layout_they_do_not_know():
    with pytest.raises(ValueError, match="not one of"):
        returns.value_rewards(torch.zeros(1, 5), ("damage",))


def test_the_learner_needs_one_discount_per_value_level(arm):
    net, runner = arm
    with pytest.raises(ValueError, match="discount"):
        lc.learner(net, runner)  # the defaults: five levels
    with pytest.raises(ValueError, match="team_spirit"):
        lc.learner(cases.model(), runner, team_spirit=(0.0,))


def test_the_arm_s_returns_sum_the_reward_levels_and_its_critic_trains(arm):
    """prepare: one value level, its rewards the level sum, no outcome targets; one
    update steps the value head and reports the level's series."""
    net, runner = arm
    learner = lc.learner(copy.deepcopy(net), runner, **SINGLE)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    derived = prepared["derived"]
    assert "outcome_targets" not in derived
    assert derived["level_returns"].shape[-1] == 1
    assert prepared["explained_variance"].shape == (1,)
    trajectory = lc.whole(runner)
    chain = dict(
        reward=trajectory.reward,
        values=trajectory.values,
        done=trajectory.done,
        result=trajectory.result,
    )
    bootstrap = runner.bootstrap()
    chained = learner.chained_returns(chain, trajectory.alive, bootstrap)
    advantages, level_returns = returns.per_level_gae(
        trajectory.reward.sum(-1, keepdim=True),
        trajectory.values,
        trajectory.done,
        bootstrap.values,
        trajectory.done[-1],
        torch.tensor(SINGLE["discount"]),
        torch.tensor(SINGLE["trace_decay"]),
    )
    torch.testing.assert_close(chained["level_returns"], level_returns, rtol=0, atol=0)
    torch.testing.assert_close(chained["advantage"], advantages[..., 0], rtol=0, atol=0)
    before = [parameter.detach().clone() for parameter in learner.model.parameters()]
    learner = lc.learner(learner.model, runner, epochs=1, **SINGLE)
    out = metrics.read(learner.update(runner.store, runner.bootstrap()))
    for name in ("value_loss", "explained_variance", "value_kl", "explained_variance_fit"):
        assert f"critic/{name}/{SUMMED_LEVEL}" in out, name
    assert not any(key.startswith("critic/") and key.endswith("/outcome") for key in out)
    assert torch.isfinite(torch.tensor(out[f"critic/value_loss/{SUMMED_LEVEL}"]))
    value_head = learner.model.own_heads.lasts[1].weight
    index = [p is value_head for p in learner.model.parameters()].index(True)
    assert not torch.equal(before[index], value_head.detach())


def test_the_arm_s_build_record_round_trips(arm, tmp_path):
    """A policy file of the arm records its layout and loads as the arm."""
    from bnb.runs import checkpoint

    net, _ = arm
    checkpoint.save_policy(tmp_path / "policy", net)
    loaded = checkpoint.load_policy(tmp_path / "policy", torch.device("cpu"))
    assert loaded.config.levels == (SUMMED_LEVEL,) and loaded.config.outcome_level is None
    assert loaded.outcome is None
    for (name, expected), (_, actual) in zip(
        _pytree.tree_flatten_with_path(net.state_dict())[0],
        _pytree.tree_flatten_with_path(loaded.state_dict())[0],
        strict=True,
    ):
        assert torch.equal(actual, expected), name


def test_the_critic_figures_of_the_arm_draw_its_values_alone(tmp_path):
    """No outcome head: the games that ended, and the trace of the one value level."""
    pytest.importorskip("matplotlib")
    from bnb.figures import critic

    length, views, ships = 6, 2, 2 * OWN
    done = torch.zeros((length, views), dtype=torch.bool)
    done[3, 0] = True
    result = torch.zeros((length, views), dtype=torch.int32)
    result[3, 0] = 1
    artifact = dict(
        meta=dict(tick_length=1.0, ships_per_team=OWN, levels=(SUMMED_LEVEL,)),
        games=dict(
            done=done,
            result=result,
            values=torch.randn(length, views, OWN, 1),
            alive=torch.ones((length, views, ships), dtype=torch.bool),
        ),
    )
    written, numbers = critic.render(artifact, tmp_path)
    assert numbers == dict(games_ended=1)
    assert [path.stem for path in written] == ["critic_trace"]


def test_the_arm_s_prediction_overlay_shows_its_value_and_no_outcome():
    from play_cases import SMALL, tiny_model

    from bnb.game import api
    from bnb.play import predictions
    from bnb.play.loop import MatchLoop, MatchSetup
    from bnb.play.sides import Agent, Scripted

    model = tiny_model("yemong", SMALL, 3, **ARM)
    agent = Agent(model, "yemong", "tiny", SMALL.ships_per_team, SMALL.num_ships, seed=3)
    agent.watched = True
    sides = [Scripted(api.rules_of(SMALL), api.ship_types()), agent]
    loop = MatchLoop(MatchSetup(SMALL, seed=5), sides=sides, compiled=False)
    for _ in range(30):
        loop.step()
    critic = predictions.side_predictions(agent, 1, loop.previous, loop.current).value
    assert critic.levels == (SUMMED_LEVEL,)
    assert critic.values.shape == (SMALL.ships_per_team, 1)
    assert critic.win is None and critic.value is None
