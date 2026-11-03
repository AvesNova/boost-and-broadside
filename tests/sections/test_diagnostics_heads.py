"""The auxiliary heads' series (diagnostics audit, groups 7 to 11) against hand-computed
cases."""

import copy
import math
from types import SimpleNamespace

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import SHIP, SHIP_COLUMNS
from bnb.learn import head_statistics
from bnb.models.yemong.state_code import MEANS, SPREADS
from bnb.runtime import metrics


def _series(total: dict) -> dict:
    return metrics.read(head_statistics.series(total, None))


def _enemy_case():
    """Two steps, one view, two enemy rows: uniform logits; actions (0, 0) then (0, 5);
    row 0 visible, row 1 hidden; every row labelled."""
    L, R, E = 2, 1, 2
    unrolled = SimpleNamespace(enemy_action_logits=torch.zeros((L, R, E, NUM_JOINT)))
    label = dict(
        enemy_action=torch.tensor([[[0, 0]], [[0, 5]]]),
        enemy_mask=torch.ones((L, R, E), dtype=torch.bool),
    )
    trajectory = SimpleNamespace(done=torch.zeros((L, R), dtype=torch.bool))
    visible = torch.tensor([True, False]).expand(L, R, E)
    return unrolled, trajectory, label, visible


def test_the_enemy_action_series_match_a_hand_computed_case():
    unrolled, trajectory, label, visible = _enemy_case()
    total = head_statistics._enemy_action(unrolled, trajectory, label, 0, visible)
    out = metrics.read(
        {
            name: value
            for name, value in head_statistics.series(_padded(total), None).items()
            if name.startswith("enemy_action/")
        }
    )
    uniform = math.log(NUM_JOINT)
    for sight in ("visible", "hidden"):
        assert out[f"enemy_action/cross_entropy/{sight}"] == pytest.approx(uniform)
        assert out[f"enemy_action/entropy/{sight}"] == pytest.approx(uniform)
        assert out[f"enemy_action/factor_cross_entropy/power/{sight}"] == pytest.approx(math.log(3))
        assert out[f"enemy_action/factor_cross_entropy/stick/{sight}"] == pytest.approx(math.log(7))
        assert out[f"enemy_action/factor_cross_entropy/shoot/{sight}"] == pytest.approx(math.log(2))
    # visible: actions 0, 0 (one pair, repeated): the marginal is certain, so is the repeat
    assert out["enemy_action/gain/marginal/visible"] == pytest.approx(-uniform)
    assert out["enemy_action/repeat_rate/visible"] == 1.0
    assert out["enemy_action/gain/repeat/visible"] == pytest.approx(-uniform)
    # hidden: actions 0, 5 (one pair, switched): marginal entropy ln 2; repeat rate 0, so
    # the best repeat predictor spreads uniformly over the other 41
    assert out["enemy_action/gain/marginal/hidden"] == pytest.approx(math.log(2) - uniform)
    assert out["enemy_action/repeat_rate/hidden"] == 0.0
    assert out["enemy_action/gain/repeat/hidden"] == pytest.approx(
        math.log(NUM_JOINT - 1) - uniform, abs=1e-6
    )
    # argmax of uniform logits is action 0: right at three of the four tokens
    assert out["enemy_action/accuracy/visible"] == 1.0
    assert out["enemy_action/accuracy/hidden"] == 0.5
    assert out["enemy_action/supervised_share"] == 1.0


def _padded(total: dict) -> dict:
    """The totals of every head, with zero counts for those a case leaves out (absent)."""
    zero = torch.zeros(())
    padding = {}
    for name in head_statistics.POPULATIONS:
        padding[f"next_state/{name}/count"] = zero
        padding[f"next_state/{name}/cross_entropy"] = torch.zeros(len(head_statistics.STATE_PARTS))
        padding[f"next_state/{name}/entropy"] = zero
        if name in head_statistics.GAIN_POPULATIONS:
            padding[f"next_state/{name}/baseline"] = torch.zeros(len(head_statistics.STATE_PARTS))
    padding.update({"next_state/alive": zero, "next_state/supervised": zero})
    for sight in head_statistics.SIGHT:
        prefix = f"enemy_action/{sight}"
        for name in ("count", "cross_entropy", "correct", "entropy", "pairs", "repeats"):
            padding[f"{prefix}/{name}"] = zero
        padding[f"{prefix}/pair_cross_entropy"] = zero
        padding[f"{prefix}/factor_cross_entropy"] = torch.zeros(3)
        padding[f"{prefix}/histogram"] = torch.zeros(NUM_JOINT)
    padding["enemy_action/rows"] = zero
    padding["density/steps"] = torch.ones(())
    for team in head_statistics.TEAMS:
        padding[f"density/{team}/excess"] = zero
        padding[f"density/{team}/counts"] = torch.zeros(1)
        padding[f"density/{team}/count_log_count"] = torch.zeros(1)
    padding["shot/age_count"] = zero
    padding["shot/trajectory_kl"] = torch.zeros(len(head_statistics.SHOT_PARTS))
    padding["shot/own_count"] = zero
    padding["shot/realised"] = torch.zeros(3)
    padding["shot/predicted"] = torch.zeros(3)
    padding["shot/own_alive"] = zero
    return {**padding, **total}


def test_the_density_null_is_the_batch_mean_per_cell():
    """A head that predicts each cell's batch mean explains none of the deviance; one that
    predicts every count exactly explains all of it."""
    counts = torch.tensor([[1.0, 3.0, 0.0, 2.0], [3.0, 1.0, 2.0, 2.0]])  # (steps, 2 cells x 2)
    mean = counts.mean(0, keepdim=True).expand_as(counts)
    for log_rates, explained in ((mean.log(), 0.0), (counts.clamp_min(1e-30).log(), 1.0)):
        unrolled = SimpleNamespace(density_log_rates=log_rates.clamp_min(-10.0))
        total = head_statistics._density(unrolled, dict(density=counts))
        out = _series(_padded(total))
        for team in head_statistics.TEAMS:
            assert out[f"density/explained_deviance/{team}"] == pytest.approx(explained, abs=1e-3)
        if explained == 1.0:
            assert out["density/deviance/ally"] == pytest.approx(0.0, abs=1e-3)


def test_the_shot_rates_and_trajectory_kl_match_a_hand_computed_case():
    """One own ship and one enemy, one step: the own ship's labelled shot hits an enemy
    (realised enemy-hit rate 1) and the head gives the hit a probability of 1/2."""
    width = 19
    logits = torch.full((1, 1, 2, width), -1e4)
    logits[..., 0] = 0.0
    logits[..., -1] = 0.0  # half on the hit, half on the clear miss
    outcome = torch.zeros((1, 1, 2, width))
    outcome[..., 0] = 1.0
    runs = ((1, 2), (1, 2), (1, 2), (1, 2))
    target = torch.tensor([0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).expand(1, 1, 2, 1, 8)
    parts = torch.tensor([math.log(2), 1.0, 2.0, 3.0]).expand(1, 1, 2, 1, 4)  # the CE
    label = dict(
        shot_mask=torch.tensor([True, False]).expand(1, 1, 2),
        shot_ages=torch.ones((1, 1, 2, 1), dtype=torch.bool),
        shot_trajectory=target,
        shot_outcome=outcome,
    )
    alive = torch.ones((1, 1, 2), dtype=torch.bool)
    unrolled = SimpleNamespace(shot_outcome_logits=logits)
    total = head_statistics._shots(unrolled, label, parts, runs, 1, alive)
    out = _series(_padded(total))
    assert out["shot/enemy_hit_rate/realised"] == 1.0
    assert out["shot/enemy_hit_rate/predicted"] == pytest.approx(0.5)
    assert out["shot/clear_miss_rate/predicted"] == pytest.approx(0.5)
    assert out["shot/ally_hit_rate/realised"] == 0.0
    assert out["shot/supervised_share"] == 1.0  # the one own ship
    # KL = CE - the target's entropy: ln 2 - ln 2, then 1, 2, 3 against sharp targets
    expected = (0.0, 1.0, 2.0, 3.0)
    for part, value in zip(head_statistics.SHOT_PARTS, expected, strict=True):
        assert out[f"shot/trajectory_kl/{part}"] == pytest.approx(value, abs=1e-6)


def test_the_belief_errors_match_a_hand_computed_case():
    """One hidden enemy at age 3 ticks (0.1 s at 30 Hz: the 0.1-0.5 s bucket), believed 3 m
    east and 4 m north of the truth across the torus seam, with a 5 m sigma."""
    L, R, E = 1, 1, 1
    truth = torch.zeros((L, R, E, len(SHIP_COLUMNS)))
    truth[..., SHIP["alive"]] = 1.0
    truth[..., SHIP["heading_cos"]] = 1.0
    truth[..., SHIP["x"]] = 1.0
    means = torch.zeros((L, R, E, len(MEANS)))
    means[..., 0] = 1.0 + 3.0 - 26244.0  # across the seam: still 3 m by the minimum image
    means[..., 1] = 4.0
    spreads = torch.zeros((L, R, E, len(SPREADS)))
    spreads[..., 0] = 5.0
    spreads[..., 1:] = 1.0
    obs_ships = torch.zeros((L, R, 1 + E, len(SHIP_COLUMNS)))  # one own row, then the enemy
    trajectory = SimpleNamespace(
        log_probability=torch.zeros((L, R, 1)),
        obs=SimpleNamespace(ships=obs_ships),
        truth=truth,
        replay=SimpleNamespace(
            valid=torch.ones((L, R, E), dtype=torch.bool),
            age=torch.full((L, R, E), 3, dtype=torch.int32),
            means=means,
            spreads=spreads,
            offset=torch.zeros((L, R, 2)),
        ),
    )
    total = head_statistics.belief_sums(trajectory, 1.0 / 30.0)
    out = metrics.read(head_statistics.belief_series(total))
    bucket = "age_0.1_0.5_seconds"
    assert out[f"belief/position_metres/{bucket}"] == pytest.approx(5.0)
    assert out[f"belief/heading_radians/{bucket}"] == pytest.approx(0.0, abs=1e-6)
    # (3^2 + 4^2) / 5^2 over two axes
    assert out[f"belief/normalised_square_error/position_metres/{bucket}"] == pytest.approx(0.5)
    assert out[f"belief/within_one_sigma/position_metres/{bucket}"] == 1.0
    assert "belief/position_metres/age_0_0.1_seconds" not in out  # an empty bucket is absent


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    return net, lc.rollout(net)


@pytest.mark.slow
def test_a_head_that_changes_nothing_gains_nothing(segment, global_generator):
    """With the next-state head's last layer zero the forecast is "nothing changes": its gain
    is 0 for every population and part, and every KL is at least 0."""
    net, runner = segment
    model = copy.deepcopy(net)
    with torch.no_grad():
        model.ship_heads.lasts[1].weight.zero_()
        model.ship_heads.lasts[1].bias.zero_()
    learner = lc.learner(model, runner, epochs=1, learning_rate=0.0)
    out = metrics.read(learner.update(runner.store, [runner.bootstrap()]))
    gains = [name for name in out if name.startswith("next_state/gain/")]
    assert len(gains) == 2 * (len(head_statistics.STATE_PARTS) + 1)
    for name in gains:
        assert out[name] == pytest.approx(0.0, abs=1e-5), name
    for name in head_statistics.POPULATIONS:
        assert out[f"next_state/kl/{name}"] >= -1e-4
    shares = sum(out[f"next_state/token_share/{name}"] for name in head_statistics.POPULATIONS)
    assert shares == pytest.approx(1.0)
    for name in ("density/explained_deviance/enemy", "enemy_action/gain/repeat/hidden"):
        assert math.isfinite(out[name]), name
    assert any(name.startswith("belief/position_metres/") for name in out)


def test_the_input_series_count_envelope_breaks_and_non_finite_entries():
    """Four visible living ships: one faster than the velocity code reaches, one with more
    shield than its type holds; a hidden ship's zeros and a dead ship's count nowhere; one
    NaN in the globals."""
    from bnb.models.yemong.state_code import velocity_reach

    ships = torch.zeros((1, 1, 6, len(SHIP_COLUMNS)))
    ships[..., :4, SHIP["visible"]] = 1.0
    ships[..., :5, SHIP["alive"]] = 1.0
    ships[..., 0, SHIP["velocity_x"]] = velocity_reach() + 1.0
    ships[..., 1, SHIP["shield"]] = 101.0
    ships[..., 5, SHIP["visible"]] = 1.0  # dead
    ships[..., 5, SHIP["velocity_x"]] = 1e6
    globals_ = torch.zeros((1, 1, 4))
    globals_[..., 2] = math.nan
    state_code = SimpleNamespace(
        max_shield=torch.tensor([100.0]),
        max_energy=torch.tensor([100.0]),
        cooldown_bins=4,
        tick_length=0.1,
        angle_of_attack_limit=1.0,
        shield_delay=2.0,
    )
    trajectory = SimpleNamespace(
        obs=SimpleNamespace(ships=ships, fields=None, globals=globals_, zones=None)
    )
    out = metrics.read(
        head_statistics.input_series(head_statistics.input_sums(state_code, trajectory))
    )
    assert out["input/nonfinite_count"] == 1
    assert out["input/envelope_violation_share/speed"] == 0.25
    assert out["input/envelope_violation_share/shield"] == 0.25
    assert out["input/envelope_violation_share/energy"] == 0.0


def test_the_fit_explained_variance_comes_from_the_moments():
    """Returns 1, 2, 3 against values 1, 1, 1 (residuals 0, 1, 2: the same variance) on the
    first level, against themselves on the others."""
    from bnb.contracts import REWARD_LEVELS
    from bnb.learn.ppo import fit_series

    target = torch.tensor([1.0, 2.0, 3.0])
    residual = torch.tensor([0.0, 1.0, 2.0])
    levels = len(REWARD_LEVELS)
    moments = torch.zeros((5, levels))
    moments[0] = 3.0
    moments[1] = target.sum()
    moments[2] = (target**2).sum()
    moments[3, 0], moments[4, 0] = residual.sum(), (residual**2).sum()
    out = metrics.read(fit_series(moments.flatten()))
    assert out[f"critic/explained_variance_fit/{REWARD_LEVELS[0]}"] == pytest.approx(0.0)
    assert out[f"critic/explained_variance_fit/{REWARD_LEVELS[1]}"] == pytest.approx(1.0)
