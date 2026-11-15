"""The split critic: each reward level's offence and defence as their own reward streams and
values (`bnb.contracts.VALUE_LEVELS`), the advantage weighing each part, and the offence
premium's schedule (`bnb.learn.ppo.LearnerConfig.offence_premium`)."""

import copy

import learner_cases as lc
import modes_cases as cases
import pytest
import torch
import yemong_cases

from bnb.contracts import REWARD_LEVELS, REWARD_PARTS, VALUE_LEVELS
from bnb.learn import returns, rewards
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import REWARD_EVENTS, RewardConfig
from bnb.learn.schedules import Keypoints
from bnb.models.yemong.critic import ReturnScale

CONFIG = RewardConfig()
PART = {name: index for index, name in enumerate(VALUE_LEVELS)}


def test_the_value_levels_are_each_reward_levels_offence_and_defence():
    assert REWARD_PARTS == ("offence", "defence")
    assert VALUE_LEVELS[:4] == (
        "damage_offence",
        "damage_defence",
        "kill_death_offence",
        "kill_death_defence",
    )
    assert len(VALUE_LEVELS) == 2 * len(REWARD_LEVELS)
    assert {(level, part) for level, part, _ in REWARD_EVENTS} == {
        (level, part) for level in REWARD_LEVELS for part in REWARD_PARTS
    }


# ------------------------------------------------------------------ the reward streams


def _played(mode: str):
    """Real ticks of random play: (events, part rewards, event rewards) per tick."""
    config = cases.MODES[mode]
    batch = 16
    ledger = rewards.blank_ledger(batch, config.num_ships)
    team = torch.arange(config.num_ships) // config.ships_per_team
    nearer = dict(spawn_offset=0.2) if mode == "elimination" else {}
    for _, events, _ in cases.play(config, batch=batch, ticks=300, seed=5, **nearer):
        parts, ledger, events_rewarded = rewards.reward_step_events(CONFIG, ledger, events, team)
        yield config, team, events, parts, events_rewarded


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_each_levels_parts_sum_to_its_zero_sum_reward(mode):
    """A level's offence plus its defence is its reward as it was before the split: the sum
    of its events (v1's charge-back the enemies' charge-back plus the allies' refund), the
    objective levels the weighted objective amounts, the outcome the weighted score of the
    ship's team. Each part is the sum of its own events."""
    seen = torch.zeros(len(VALUE_LEVELS))
    for config, team, events, parts, events_rewarded in _played(mode):
        levels = rewards.level_rewards(parts)
        weights = dict(zip(REWARD_LEVELS, CONFIG.weights, strict=True))
        for index, level in enumerate(REWARD_LEVELS):
            columns = [i for i, (name, _, _) in enumerate(REWARD_EVENTS) if name == level]
            torch.testing.assert_close(levels[..., index], events_rewarded[..., columns].sum(-1))
        torch.testing.assert_close(
            levels[..., REWARD_LEVELS.index("objective_progress")],
            weights["objective_progress"] * events.objective_progress,
        )
        torch.testing.assert_close(
            levels[..., REWARD_LEVELS.index("objective")], weights["objective"] * events.objective
        )
        score = events.win_loss[:, 0] - events.win_loss[:, 1]  # team 0's
        sign = torch.where(team == 0, 1.0, -1.0)
        torch.testing.assert_close(
            levels[..., REWARD_LEVELS.index("outcome")],
            weights["outcome"] * sign * torch.where(events.done, score, 0.0)[:, None],
        )
        for name in VALUE_LEVELS:
            columns = [
                i for i, (level, part, _) in enumerate(REWARD_EVENTS) if f"{level}_{part}" == name
            ]
            torch.testing.assert_close(
                parts[..., PART[name]], events_rewarded[..., columns].sum(-1)
            )
        seen += (parts != 0).float().sum((0, 1))
    seen = {name: seen[PART[name]] for name in VALUE_LEVELS}
    # every combat part; the outcome's where a match ends, the meters' where they move (no
    # Frontline match of the case's ends within its ticks)
    required = ("damage", "kill_death") + {
        "team_deathmatch": ("outcome",),
        "frontline": ("objective_progress",),
    }[mode]
    assert all(seen[f"{level}_{part}"] > 0 for level in required for part in REWARD_PARTS), seen


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_a_teams_offence_is_the_other_teams_defence(mode):
    """Every combat and outcome event pairs with one of the other part on the other team:
    per environment and level, team 0's offence is team 1's defence negated (and the other
    way round)."""
    for config, team, _, parts, _ in _played(mode):
        per_team = torch.stack([parts[:, team == side].sum(1) for side in (0, 1)], 1)  # (B, 2, K)
        largest = parts.abs().amax().clamp_min(1e-6)
        for level in ("damage", "kill_death", "outcome"):
            offence, defence = PART[f"{level}_offence"], PART[f"{level}_defence"]
            for side in (0, 1):
                gap = per_team[:, side, offence] + per_team[:, 1 - side, defence]
                assert (gap.abs() <= 1e-5 * largest).all(), (level, gap.abs().max())


def test_the_outcome_parts_are_the_win_and_the_loss_weights():
    team = torch.tensor([0, 0, 1, 1])
    win_loss = torch.tensor([[1.0, 0.0], [0.3, 0.7], [0.0, 0.0]])  # a win, a kill share, a draw
    done = torch.tensor([True, True, True])
    parts = rewards.outcome_events(team, win_loss, done)
    torch.testing.assert_close(parts[0], torch.tensor([[1.0, 0.0]] * 2 + [[0.0, -1.0]] * 2))
    torch.testing.assert_close(parts[1], torch.tensor([[0.3, -0.7]] * 2 + [[0.7, -0.3]] * 2))
    assert (parts[2] == 0).all()
    assert (rewards.outcome_events(team, win_loss, ~done) == 0).all()


# ------------------------------------------------------------------ the advantage


def _gae_case(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    L, R, S, K = 12, 3, 4, len(VALUE_LEVELS)
    rewards_ = torch.randn((L, R, S, K), generator=generator, dtype=torch.float64)
    values = torch.randn((L, R, S, K), generator=generator, dtype=torch.float64)
    following = torch.randn((R, S, K), generator=generator, dtype=torch.float64)
    done = torch.rand((L, R), generator=generator) < 0.1
    next_done = torch.tensor([False, True, False])
    alive = torch.rand((L, R, S), generator=generator) < 0.8
    return rewards_, values, following, done, next_done, alive


def _advantage(rewards_, values, following, done, next_done, alive, config, weights):
    """The learner's advantage and returns from per-part (or per-level) streams."""
    split = rewards_.shape[-1] == len(VALUE_LEVELS)
    per = returns.per_part if split else (lambda values_: values_)
    discount = torch.tensor(per(config.discount), dtype=torch.float64)
    trace = torch.tensor(per(config.trace_decay), dtype=torch.float64)
    spirit = torch.tensor(per((0.0, 0.5, 0.0, 0.2, 0.0)), dtype=torch.float64)
    advantages, level_returns = returns.per_level_gae(
        rewards_, values, done, following, next_done, discount, trace
    )
    team = torch.zeros_like(alive, dtype=torch.int64)
    advantage, total = returns.team_aggregate(
        team, alive, advantages, level_returns, spirit, weights.to(torch.float64)
    )
    return advantage, total, level_returns


def test_one_to_one_is_the_zero_sum_advantage():
    """With every weight 1, the split advantage is the zero-sum levels' advantage from the
    same values (each level's offence value plus its defence value): GAE is linear."""
    rewards_, values, following, done, next_done, alive = _gae_case()
    config = LearnerConfig()
    split, split_total, _ = _advantage(
        rewards_, values, following, done, next_done, alive, config,
        torch.tensor(returns.part_weights(1.0)),
    )  # fmt: skip
    summed = [rewards.level_rewards(x) for x in (rewards_, values, following)]
    whole, whole_total, _ = _advantage(
        *summed, done, next_done, alive, config, torch.ones(len(REWARD_LEVELS))
    )
    torch.testing.assert_close(split, whole)
    torch.testing.assert_close(split_total, whole_total)


def test_a_premium_weighs_only_the_damage_and_kill_death_offence():
    assert returns.part_weights(2.0) == (2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rewards_, values, following, done, next_done, alive = _gae_case(1)
    config = LearnerConfig()
    even, even_total, even_returns = _advantage(
        rewards_, values, following, done, next_done, alive, config,
        torch.tensor(returns.part_weights(1.0)),
    )  # fmt: skip
    premium, premium_total, premium_returns = _advantage(
        rewards_, values, following, done, next_done, alive, config,
        torch.tensor(returns.part_weights(2.0)),
    )  # fmt: skip
    # the difference is the premium levels' offence once more
    offence = [PART["damage_offence"], PART["kill_death_offence"]]
    only, _, _ = _advantage(
        rewards_, values, following, done, next_done, alive, config,
        torch.zeros(len(VALUE_LEVELS)).index_fill(0, torch.tensor(offence), 1.0),
    )  # fmt: skip
    torch.testing.assert_close(premium - even, only)
    # the targets are each part's own at any premium; the return scale's sum moves as the
    # advantage does, by the premium levels' offence returns once more
    torch.testing.assert_close(premium_returns, even_returns)
    _, only_total, _ = _advantage(
        rewards_, values, following, done, next_done, alive, config,
        torch.zeros(len(VALUE_LEVELS)).index_fill(0, torch.tensor(offence), 1.0),
    )  # fmt: skip
    torch.testing.assert_close(premium_total - even_total, only_total)


def test_the_return_scale_follows_the_premium():
    """Where the premium levels' offence is the only reward, a premium p multiplies the
    advantage and the return scale's sum by p alike, so the advantage over the return scale
    is unchanged: the premium sets the balance of offence and defence, not the step."""
    rewards_, values, following, done, next_done, alive = _gae_case(2)
    offence = torch.tensor([PART["damage_offence"], PART["kill_death_offence"]])
    only = torch.zeros(len(VALUE_LEVELS)).index_fill(0, offence, 1.0).to(torch.float64)
    rewards_, values, following = (x * only for x in (rewards_, values, following))
    config = LearnerConfig()
    scaled = {}
    for premium in (1.0, 2.0, 3.5):
        advantage, total, _ = _advantage(
            rewards_, values, following, done, next_done, alive, config,
            torch.tensor(returns.part_weights(premium)),
        )  # fmt: skip
        scale = ReturnScale(decay=0.0, floor=0.0, device=torch.device("cpu")).update(
            total.float(), alive
        )
        scaled[premium] = advantage.float() / scale
    torch.testing.assert_close(scaled[2.0], scaled[1.0])
    torch.testing.assert_close(scaled[3.5], scaled[1.0])


# ------------------------------------------------------------------ the schedule


def test_the_premium_schedule():
    assert LearnerConfig().offence_premium == Keypoints((0,), (1.0,))  # 1:1, zero-sum
    constant = Keypoints((0,), (2.0,))
    assert constant.value(0) == constant.value(10**9) == 2.0
    anneal = Keypoints((0, 20_000_000, 60_000_000), (2.0, 2.0, 1.0))
    assert anneal.value(0) == anneal.value(20_000_000) == 2.0
    assert anneal.value(40_000_000) == pytest.approx(1.5)
    assert anneal.value(50_000_000) == pytest.approx(1.25)
    assert anneal.value(60_000_000) == anneal.value(500_000_000) == 1.0
    late = Keypoints((10, 20), (3.0, 1.0))
    assert late.value(0) == 3.0 and late.value(15) == pytest.approx(2.0)
    for steps, values in (((), ()), ((0, 0), (1.0, 1.0)), ((0,), (1.0, 2.0)), ((-1,), (1.0,))):
        with pytest.raises(ValueError):
            Keypoints(steps, values)
    with pytest.raises(ValueError, match="offence_premium"):
        LearnerConfig(offence_premium=Keypoints((0,), (0.0,)))
    with pytest.raises(ValueError, match="one value per reward level"):
        LearnerConfig(discount=(0.99,) * len(VALUE_LEVELS))


# ------------------------------------------------------------------ the learner


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = yemong_cases.model()
    return net, lc.rollout(net)


def test_the_learner_weighs_the_advantage_and_never_the_targets(segment, global_generator):
    """Through the learner's returns over a real segment: a 2:1 premium (at the update's
    steps, by its schedule) moves the advantage of the damage and kill/death offence only,
    and the return scale's sum by the same weights; the critic's targets (each part's
    returns) and the outcome's stay as they are."""
    net, runner = segment
    anneal = Keypoints((0, 1000), (2.0, 1.0))
    prepared = {}
    for name, premium in (("even", Keypoints()), ("premium", anneal)):
        learner = lc.learner(copy.deepcopy(net), runner, offence_premium=premium)
        assert learner.levels == VALUE_LEVELS
        chain = {name: runner.store.shard((name,), 0)[name] for name in ("reward", "values")}
        chain.update(runner.store.shard(("outcome", "done", "win_loss"), 0))
        alive = runner.store.shard(("alive",), 0)["alive"]
        prepared[name] = learner.chained_returns(chain, alive, runner.bootstrap())
        assert learner.offence_premium == (2.0 if name == "premium" else 1.0)
        learner.environment_steps = 500
        learner.part_weights()
        assert learner.offence_premium == (1.5 if name == "premium" else 1.0)
    even, premium = prepared["even"], prepared["premium"]
    for name in ("level_returns", "outcome_targets", "monte_carlo_returns"):
        torch.testing.assert_close(premium[name], even[name])
    team = torch.zeros_like(alive[..., : even["level_returns"].shape[-2]], dtype=torch.int64)
    own = alive[..., : even["level_returns"].shape[-2]]
    mixed = returns.team_mixed_levels(team, own, even["level_returns"], learner.team_spirit)
    for name, premium_value in (("even", 1.0), ("premium", 2.0)):
        weights = torch.tensor(returns.part_weights(premium_value))
        torch.testing.assert_close(prepared[name]["total"], (mixed * weights).sum(-1))
    moved = premium["level_advantage"] - even["level_advantage"]
    offence = [PART["damage_offence"], PART["kill_death_offence"]]
    torch.testing.assert_close(moved[..., offence], even["level_advantage"][..., offence])
    others = [index for index in range(len(VALUE_LEVELS)) if index not in offence]
    assert (moved[..., others] == 0).all()
    assert even["level_advantage"][..., offence].abs().sum() > 0  # the premium had a part
    torch.testing.assert_close(premium["advantage"], premium["level_advantage"].sum(-1))


def test_the_learner_refuses_a_critic_from_before_the_split(segment, global_generator):
    net, runner = segment
    unsplit = yemong_cases.model(levels=REWARD_LEVELS)
    with pytest.raises(ValueError, match="split"):
        Learner(
            unsplit,
            LearnerConfig(),
            runner.rules,
            runner.carry["types"],
            CONFIG.outcome,
            lc.CPU,
        )
