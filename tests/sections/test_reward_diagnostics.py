"""The reward levels' diagnostics (`bnb.learn.advantages`, the combat series of
`bnb.runtime.analytics`): each level's part of the advantage and of the policy gradient
against hand-computed cases, the training games' combat per mode, and the learner's series
on a real segment, with no recompile of a compiled update."""

import copy
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import REWARD_LEVELS
from bnb.learn import advantages, returns
from bnb.learn.rewards import RewardConfig
from bnb.runtime import analytics, metrics

LEVELS = ("dense", "sparse", "silent")


def _series(levels: torch.Tensor, mask: torch.Tensor, member=None, names=(None,)) -> dict:
    member = torch.ones((levels.shape[1], 1)) if member is None else member
    per_level, whole = advantages.level_sums(levels, levels.sum(-1), mask, member)
    return metrics.read(advantages.level_series(per_level, whole, LEVELS, names))


def test_known_level_sizes_give_their_shares_spreads_and_correlations():
    """Level 0 is +-3, level 1 +-1 and independent of it, level 2 zero: shares 3/4, 1/4, 0;
    standard deviations 3 and 1; correlations with the sum 3 / sqrt(10) and 1 / sqrt(10)."""
    signs_a = torch.tensor([1.0, -1.0, 1.0, -1.0])
    signs_b = torch.tensor([1.0, 1.0, -1.0, -1.0])
    levels = torch.stack([3 * signs_a, signs_b, torch.zeros(4)], -1).reshape(4, 1, 1, 3)
    out = _series(levels, torch.ones((4, 1, 1), dtype=torch.bool))
    assert out["advantage/mean_absolute/dense"] == pytest.approx(3.0)
    assert [out[f"advantage/share/{name}"] for name in LEVELS] == pytest.approx([0.75, 0.25, 0])
    assert out["advantage/standard_deviation/dense"] == pytest.approx(3.0)
    assert out["advantage/standard_deviation/sparse"] == pytest.approx(1.0)
    assert out["advantage/correlation/dense"] == pytest.approx(3 / math.sqrt(10))
    assert out["advantage/correlation/sparse"] == pytest.approx(1 / math.sqrt(10))
    assert out["advantage/correlation/silent"] == 0.0  # no spread: no correlation


def test_masked_tokens_count_for_nothing_and_slices_split_the_views():
    """Two views, each its own slice: view 0's tokens are all level 0, view 1's all level 1;
    a dead token with a huge advantage is left out."""
    levels = torch.zeros((2, 2, 1, 3))
    levels[:, 0, 0, 0] = torch.tensor([2.0, -2.0])
    levels[:, 1, 0, 1] = torch.tensor([1.0, 1.0])
    levels[1, 1, 0, 2] = 1e6
    mask = torch.ones((2, 2, 1), dtype=torch.bool)
    mask[1, 1, 0] = False
    member = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    out = _series(levels, mask, member, (None, "first", "second"))
    assert out["advantage/share/dense/first"] == pytest.approx(1.0)
    assert out["advantage/share/sparse/second"] == pytest.approx(1.0)
    assert out["advantage/share/silent/second"] == 0.0
    assert out["advantage/mean_absolute/dense"] == pytest.approx(4 / 3)  # three live tokens
    assert out["advantage/share/dense"] == pytest.approx(4 / 5)


def test_the_raw_spread_divides_out_the_reward_weight():
    level = torch.tensor([10.0, -10.0, 10.0, -10.0]).reshape(4, 1, 1, 1)
    sums = advantages.raw_sums(level, torch.ones((4, 1, 1), dtype=torch.bool))
    out = metrics.read(advantages.raw_series(sums, (5.0,), ("dense",)))
    assert out["advantage/raw_standard_deviation/dense"] == pytest.approx(2.0)


def test_the_team_mixed_levels_sum_to_the_aggregate():
    generator = torch.Generator().manual_seed(0)
    team = torch.tensor([[0, 0, 1]])
    alive = torch.tensor([[True, True, True]])
    per_level = torch.randn((1, 3, 5), generator=generator)
    spirit = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.2])
    mixed = returns.team_mixed_levels(team, alive, per_level, spirit)
    total, _ = returns.team_aggregate(team, alive, per_level, per_level, spirit)
    torch.testing.assert_close(mixed.sum(-1), total)
    torch.testing.assert_close(mixed[0, 0, 1], per_level[0, 0, 1] + 0.5 * per_level[0, 1, 1])


def test_level_gradients_give_their_norms_shares_and_cosines():
    """Level gradients (3, 0), (0, 4) and zero, the whole policy gradient their sum."""
    levels = torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    out = metrics.read(advantages.policy_level_series(levels, levels.sum(0), LEVELS))
    assert out["gradient/policy_level_norm/dense"] == pytest.approx(3.0)
    assert out["gradient/policy_level_norm/silent"] == 0.0
    assert out["gradient/policy_level_share/dense"] == pytest.approx(9 / 25)
    assert out["gradient/policy_level_share/sparse"] == pytest.approx(16 / 25)
    assert out["gradient/policy_level_share/silent"] == 0.0
    assert out["gradient/policy_level_cosine/dense/sparse"] == pytest.approx(0.0)
    assert out["gradient/policy_level_cosine/dense/policy"] == pytest.approx(3 / 5)
    # a zero gradient has no direction: its cosines are absent, not 0/0
    assert "gradient/policy_level_cosine/dense/silent" not in out


# ------------------------------------------------------------------ the combat series


def test_the_training_games_combat_is_per_mode_and_leaves_out_the_lanes():
    """Three pools: frontline self-play, deathmatch league and a lane (no learner view)
    playing deathmatch; one environment each."""
    rates = analytics.Analytics(
        ("frontline/self_play", "deathmatch/league", "lane"),
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 2]),
        tick_length=0.5,
        mode="frontline",
        pool_modes=("frontline", "team_deathmatch", "team_deathmatch"),
    )
    assert rates.training_modes == ("frontline", "team_deathmatch")
    sums = analytics.blank_game_sums(3, "cpu")
    resolve = sums["resolve"]
    column = analytics.RESOLVE_COLUMNS.index
    resolve[:, column("alive_ship_ticks")] = torch.tensor(
        [240.0, 120.0, 120.0]
    )  # ship-minutes 2, 1, 1
    resolve[:, column("combat_deaths")] = torch.tensor([1.0, 3.0, 50.0])
    resolve[:, column("hits")] = torch.tensor([10.0, 30.0, 500.0])
    out = metrics.read(rates.games(sums, None))
    assert out["combat/deaths_per_ship_minute/frontline"] == pytest.approx(0.5)
    assert out["combat/deaths_per_ship_minute/team_deathmatch"] == pytest.approx(3.0)
    assert out["combat/hits_per_ship_minute/team_deathmatch"] == pytest.approx(30.0)


# ------------------------------------------------------------------ the learner


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    return net, lc.rollout(net)


def _learner(net, runner, **config):
    learner = lc.with_micro_batch(lc.learner(net, runner, epochs=1, minibatches=2, **config), 2)
    learner.reward_weights = RewardConfig().weights
    views = runner.store.views
    learner.view_modes = ("frontline", "team_deathmatch")
    learner.view_mode = torch.arange(views) % 2
    return learner


@pytest.mark.slow
def test_the_learners_levels_add_up_to_its_advantage(segment, global_generator):
    net, runner = segment
    learner = _learner(copy.deepcopy(net), runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    out = metrics.read(prepared["series"])
    shares = [out[f"advantage/share/{level}"] for level in REWARD_LEVELS]
    assert sum(shares) == pytest.approx(1.0)
    for mode in ("frontline", "team_deathmatch"):
        assert sum(out[f"advantage/share/{level}/{mode}"] for level in REWARD_LEVELS) == (
            pytest.approx(1.0)
        )
    for level in REWARD_LEVELS:
        assert -1.0 - 1e-5 <= out[f"advantage/correlation/{level}"] <= 1.0 + 1e-5
        assert out[f"advantage/raw_standard_deviation/{level}"] >= 0.0
    # the cadence's views' levels, over the return scale, sum to the policy's advantage
    kept = prepared["level_advantage"]
    whole = prepared["derived"]["advantage"][:, : kept.shape[1]] / prepared["scale"]
    torch.testing.assert_close(kept.sum(-1), whole, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_the_cadence_measures_each_levels_policy_gradient(segment, global_generator):
    """A level whose advantage is zero has no policy gradient; the levels' gradients add up
    to the policy's where nothing is clipped (the first step: the ratio is 1)."""
    net, runner = segment
    learner = _learner(copy.deepcopy(net), runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views).tensor_split(learner.config.minibatches)[0]
    out = metrics.read(learner._gradient_decomposition(runner.store, prepared, views))
    # before any step the ratio is 1, inside the clip, where the surrogate is linear in A:
    # the levels' gradients sum to the policy's
    total = sum(out[f"gradient/policy_level_share/{level}"] for level in REWARD_LEVELS)
    assert total == pytest.approx(1.0, abs=1e-3)
    for level in REWARD_LEVELS:
        assert out[f"gradient/policy_level_norm/{level}"] > 0.0
        assert -1.0 - 1e-5 <= out[f"gradient/policy_level_cosine/{level}/policy"] <= 1.0 + 1e-5
    prepared["level_advantage"][..., REWARD_LEVELS.index("objective")] = 0.0
    silenced = metrics.read(learner._gradient_decomposition(runner.store, prepared, views))
    assert silenced["gradient/policy_level_norm/objective"] == 0.0
    assert silenced["gradient/policy_level_share/objective"] == 0.0
    assert silenced["gradient/policy_level_norm/damage"] == pytest.approx(
        out["gradient/policy_level_norm/damage"]
    )


@pytest.mark.slow
def test_a_compiled_update_with_the_diagnostics_compiles_nothing_after_its_first(
    segment, global_generator
):
    """The learner's stages under the compiler (the eager backend: the graphs, not the
    kernels): a second update, with the cadence and the levels' diagnostics, compiles no
    new graph."""
    from torch._dynamo.utils import counters

    net, runner = segment
    torch._dynamo.reset()
    stage = lambda function: torch.compile(  # noqa: E731
        function, backend="eager", fullgraph=True, dynamic=False
    )
    learner = lc.learner(copy.deepcopy(net), runner, epochs=1, minibatches=2)
    compiled = type(learner)(
        learner.model,
        learner.config,
        learner.rules,
        learner.types,
        learner.outcome_weight,
        learner.device,
        stage=stage,
        reward_weights=RewardConfig().weights,
        view_modes=("frontline", "team_deathmatch"),
        view_mode=torch.arange(runner.store.views) % 2,
    )
    first = metrics.read(compiled.update(runner.store, runner.bootstrap(), cadence=True))
    graphs = counters["stats"]["unique_graphs"]
    second = metrics.read(compiled.update(runner.store, runner.bootstrap(), cadence=True))
    assert counters["stats"]["unique_graphs"] == graphs
    for out in (first, second):
        assert "advantage/share/damage/team_deathmatch" in out
        assert "gradient/policy_level_norm/damage" in out
