"""The error head (arms 3-4 brief, `yemong.delta_split`): the projection axes and the
positive semi-definite clamp, the capped step and the unapplied variance, the corrected
advance of hidden rows, the split masks, the stop-gradient, the switches (arms 1 and 2 bit
for bit as recorded, the arm's unroll a loop of steps), the term in the learners and a
policy-file round trip."""

import json
import math
import pathlib
from types import SimpleNamespace

import arm_goldens
import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.learn import bc, head_statistics, labels, ppo
from bnb.learn.ppo import COUNTS, TERMS
from bnb.models.yemong import codes, correction, delta
from bnb.models.yemong.inputs import believed, truth
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.state_code import MEANS, SPREADS, state_code_ranges, unpack_covariance
from bnb.runs.checkpoint import FilledDefaults, load_policy, save_policy
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
F64 = torch.float64
HERE = pathlib.Path(__file__).parent
# the arm's --set line
SETTINGS = dict(
    order="temporal_first",
    spread_fit="coarsest",
    acceleration_inputs=True,
    next_state="delta",
    delta_baseline="persistence",
    spread_inputs="ellipse",
    delta_split=True,
)
ARM_SETS = (
    "yemong.order=temporal_first",
    "yemong.acceleration_inputs=true",
    "yemong.next_state=delta",
    "yemong.delta_baseline=persistence",
    "yemong.spread_inputs=ellipse",
    "yemong.delta_split=true",
    "learner.next_state=0.0033",
    "learner.correction=0.0033",
    # a new run's defaults train the physics head's hidden rows (the champion's): arm 3 does not
    "yemong.physics_hidden_decay=null",
)


def _model(**settings):
    return cases.model(**{**SETTINGS, **settings})


def _code() -> correction.CorrectionCode:
    rules, types = api.rules_of(cases.SQUAD), api.ship_types()
    return correction.CorrectionCode(state_code_ranges(rules, types)).double()


# ------------------------------------------------------------------ the projections


def test_a_vector_and_a_covariance_round_trip_through_the_three_axes():
    unit = correction.axes()
    vector = torch.tensor([[3.0, -4.0], [0.0, 0.0], [-1e3, 2.5e2]], dtype=F64)
    projected = correction.project(vector, unit)
    assert projected.shape == (3, 3)
    torch.testing.assert_close(projected[:, 0], vector[:, 0])  # the 0-degree axis is x
    back = correction.mean_of(projected, correction.least_squares())
    torch.testing.assert_close(back, vector, rtol=0, atol=1e-10)
    # a tilted ellipse: its variances on the axes recover it exactly
    angle, major, minor = 0.7, 9.0, 0.25
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=F64
    )
    matrix = rotation @ torch.diag(torch.tensor([major, minor], dtype=F64)) @ rotation.T
    variances = torch.einsum("ai,ij,aj->a", unit, matrix, unit)
    packed = correction.covariance_of(variances, correction.variance_solve())
    expected = torch.stack([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
    torch.testing.assert_close(packed, expected, rtol=0, atol=1e-10)


def test_the_clamp_makes_the_covariance_positive_semidefinite_and_keeps_one_that_is():
    kept = torch.tensor([4.0, 1.0, 2.0], dtype=F64)
    torch.testing.assert_close(correction.positive_semidefinite(kept), kept)
    # three axis variances no ellipse has (0 on two axes, large on the third)
    variances = torch.tensor([0.0, 0.0, 5.0], dtype=F64)
    raw = variances @ correction.variance_solve().T
    xx, xy, yy = raw.unbind(-1)
    assert float(xx * yy - xy * xy) < 0  # indefinite before the clamp
    clamped = correction.covariance_of(variances, correction.variance_solve())
    xx, xy, yy = clamped.unbind(-1)
    eigenvalues = torch.linalg.eigvalsh(torch.stack([torch.stack([xx, xy]), torch.stack([xy, yy])]))
    assert float(eigenvalues.min()) >= -1e-12
    assert float(eigenvalues.max()) > 0  # the positive part is kept
    # a negative-definite triple clamps to zero
    zero = correction.positive_semidefinite(torch.tensor([-1.0, 0.0, -2.0], dtype=F64))
    assert zero.abs().max() == 0


# ------------------------------------------------------------------ the spread


def _component(name: str):
    return next(c for c in _code().components if c.name == name)


def _spread(component, probabilities: torch.Tensor) -> torch.Tensor:
    return codes.symlog_spread(
        probabilities, component.knee, component.bound, component.bins, below=component.below
    )


def _normal(count: int = 20_000) -> torch.Tensor:
    """A standard normal's quantile points (k + 1/2) / count."""
    levels = (torch.arange(count, dtype=F64) + 0.5) / count
    return math.sqrt(2.0) * torch.erfinv(2.0 * levels - 1.0)


@pytest.mark.parametrize("name", ["position_0", "velocity_60", "heading", "energy"])
def test_an_encoded_value_has_no_spread(name):
    """The probit-interpolated quantiles of an HL-Gauss are its own, so less the smoothing
    nothing is left, at any offset inside a bin and out in the wide bins."""
    component = _component(name)
    values = component.knee * torch.tensor([0.0, 0.13, 0.5, 1.0, -2.7, 9.0, 40.0], dtype=F64)
    values = values[values.abs() < component.bound]
    encoded, _ = component.encode(values)
    assert float(_spread(component, encoded).abs().max()) < 1e-6 * component.knee


@pytest.mark.parametrize("centre", [0.0, 2.0])
@pytest.mark.parametrize("width", [1.0, 3.0, 10.0, 50.0])
def test_an_encoded_gaussian_decodes_to_its_sigma(centre, width):
    """A Gaussian of sigma (in knees: across the knee and out into the wide bins), as the
    mean code of its quantile points: sigma to 6% (measured -5.1% to +1.5%, low where the
    symlog fold near 0 makes it two-humped in the bin coordinate)."""
    component = _component("position_0")
    sigma = width * component.knee
    encoded, _ = component.encode(centre * component.knee + sigma * _normal())
    assert math.isclose(float(_spread(component, encoded.mean(0))), sigma, rel_tol=0.06)


def test_the_spread_reads_each_side_of_an_asymmetric_code():
    """The position along the track (asymmetric sides): a Gaussian of 0.3 m at 2.6 m."""
    component = delta.POSITION_ALONG
    encoded, _ = component.encode(2.6 + 0.3 * _normal())
    assert math.isclose(float(_spread(component, encoded.mean(0))), 0.3, rel_tol=0.02)


def test_a_far_sliver_moves_the_spread_by_its_mass_not_its_distance():
    """The bug's case: 1e-3 of the mass in a bin 10 km out. The second moment's sigma grows
    past 250 m; the quantile spread of a sharp code stays about 0, and of a 20 m Gaussian
    at 20 m within 1% of its own."""
    component = _component("position_0")
    sharp, _ = component.encode(torch.tensor(7.0, dtype=F64))
    far, _ = component.encode(torch.tensor(10_000.0, dtype=F64))
    sliver = 0.999 * sharp + 0.001 * far
    assert float(_spread(component, sliver)) < 0.05
    _, second = codes.symlog_decode(sliver, component.knee, component.bound, component.bins)
    assert float(second) ** 0.5 > 250.0
    broad, _ = component.encode(20.0 + 20.0 * _normal())
    broad = broad.mean(0)
    clean = float(_spread(component, broad))
    assert math.isclose(float(_spread(component, 0.999 * broad + 0.001 * far)), clean, rel_tol=0.01)


def test_the_spread_has_a_finite_gradient_where_it_is_zero():
    component = _component("position_0")
    encoded, _ = component.encode(torch.tensor([0.0, 7.0], dtype=F64))
    logits = torch.log(encoded.clamp_min(1e-30)).float().requires_grad_()
    spread = _spread(component, torch.softmax(logits, -1))
    spread.sum().backward()
    assert torch.isfinite(spread).all() and torch.isfinite(logits.grad).all()


# ------------------------------------------------------------------ the code


@pytest.mark.parametrize("index", range(len(correction.COMPONENT_NAMES)))
def test_the_code_round_trips_with_an_exact_zero_and_odd_bins(index):
    code = _code()
    component = code.components[index]
    assert component.bins % 2 == 1 and component.below is None
    assert component.bins == correction.odd_bins(component.knee, component.bound)
    low, high = component.low, component.width - 1 - component.low
    coordinate = torch.linspace(-low, high, 2001, dtype=F64)
    values = codes.symlog_value(coordinate, component.knee, component.spacing)
    encoded, _ = component.encode(values)
    inverse = code.inverse(component.name)
    mean, _ = codes.symlog_decode(
        encoded, component.knee, component.bound, component.bins, inverse=inverse
    )
    back = codes.symlog_coordinate(mean, component.knee, component.spacing)
    assert (back - coordinate).abs().max() < 1e-3
    zero, _ = component.encode(torch.zeros(1, dtype=F64))
    assert torch.argmax(zero[0]) == component.low


def test_the_bounds_are_the_ranges_a_hidden_ships_error_can_reach():
    code = _code()
    bound = {c.name: c.bound for c in code.components}
    assert bound["position_0"] == 26244.0 and bound["heading"] == math.pi
    assert math.isclose(bound["velocity_60"], 2.0 * 650.6602, rel_tol=1e-4)
    assert bound["shield"] == bound["energy"] == 1.0
    assert math.isclose(bound["gun_cooldown"], 3.0 / 30.0)


def test_errors_are_the_truth_less_the_forecast_in_the_codes_units():
    code = _code()
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    truth_after = forecast.clone()
    truth_after[0, 0:2] = torch.tensor([10.0, -5.0])
    truth_after[0, 2:4] = torch.tensor([1.0, 2.0])
    truth_after[0, MEANS.index("heading")] = 3.0
    forecast[0, MEANS.index("heading")] = -3.0
    truth_after[0, MEANS.index("shield")] = 50.0
    errors = code.errors(forecast, truth_after, torch.zeros(1, dtype=torch.long))
    column = {name: i for i, name in enumerate(correction.COMPONENT_NAMES)}
    torch.testing.assert_close(
        errors[0, 0:3], correction.project(truth_after[0, 0:2], correction.axes())
    )
    assert math.isclose(float(errors[0, column["heading"]]), 6.0 - 2.0 * math.pi, rel_tol=1e-12)
    assert math.isclose(float(errors[0, column["shield"]]), 50.0 / float(code.max_shield[0]))


# ------------------------------------------------------------------ the advance


def test_the_step_is_the_whole_mean_below_r_and_r_far_above_it():
    small = torch.tensor([[0.03, -0.04]], dtype=F64)
    step = correction.capped_step(small, 8.0)
    torch.testing.assert_close(step, small, rtol=1e-4, atol=0)
    large = torch.tensor([[3000.0, 4000.0]], dtype=F64)
    step = correction.capped_step(large, 8.0)
    assert math.isclose(float(torch.linalg.vector_norm(step)), 8.0, rel_tol=1e-9)
    torch.testing.assert_close(step / 8.0, large / 5000.0)  # toward the mean
    assert correction.capped_step(torch.zeros(1, 2, dtype=F64), 8.0).abs().max() == 0


def _sharp_logits(code: correction.CorrectionCode, errors: torch.Tensor) -> torch.Tensor:
    encoded, _ = code.encode(errors)
    return torch.log(encoded.clamp_min(1e-300))


def test_a_known_error_steps_the_forecast_and_the_unapplied_part_widens_it():
    """Position 1 km off (capped at r), velocity 0.5 m/s off (taken whole), heading 0.01
    rad: the corrected mean moves by the step, and the variance holds (E e - step)^2. At a
    rate of 1, so the confident median (a sharp code: gain 1) is applied whole."""
    net = cases.double(_model(correction_rate=1.0))
    code, state_code = net.correction_code, net.state_code
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    forecast[0, MEANS.index("shield")] = 50.0
    forecast[0, MEANS.index("energy")] = 5000.0
    offset = torch.tensor([600.0, -800.0], dtype=F64)
    velocity = torch.tensor([0.3, 0.4], dtype=F64)
    errors = torch.zeros(1, len(correction.COMPONENT_NAMES), dtype=F64)
    errors[0, 0:3] = correction.project(offset, correction.axes())
    errors[0, 3:6] = correction.project(velocity, correction.axes())
    errors[0, correction.COMPONENT_NAMES.index("heading")] = 0.01
    means, spreads = code.correct(
        forecast, _sharp_logits(code, errors), torch.zeros(1, dtype=torch.long), state_code
    )
    limit = correction.STEP_LIMITS["position"]
    direction = offset / 1000.0
    step = limit * math.tanh(1000.0 / limit) * direction
    torch.testing.assert_close(means[0, 0:2], step, rtol=1e-3, atol=1e-3)
    # the windowed median of a coarse code resolves a value to about 1%
    torch.testing.assert_close(means[0, 2:4], velocity, rtol=2e-2, atol=1e-3)
    assert math.isclose(float(means[0, MEANS.index("heading")]), 0.01, rel_tol=2e-2)
    covariance = unpack_covariance(spreads[0, state_code.covariance_slice])
    rest = offset - step
    # the sharp code's own variance is ~0: the covariance is the unapplied part's outer product
    expected = torch.outer(rest, rest)
    torch.testing.assert_close(covariance[0:2, 0:2], expected, rtol=2e-2, atol=50.0)
    assert covariance[0:2, 2:4].abs().max() == 0  # block diagonal
    assert float(spreads[0, SPREADS.index("position")]) > 600.0


def test_the_step_is_the_rate_times_the_confident_median():
    """A sharp error moves by eta of itself; a broad one (sigma >> |m|) hardly at all."""
    net = cases.double(_model())
    code = net.correction_code
    assert code.rate == 0.1 == YemongConfig().correction_rate
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    velocity = torch.tensor([0.3, 0.4], dtype=F64)
    errors[0, 3:6] = correction.project(velocity, correction.axes())
    encoded, _ = code.encode(errors)
    moments = code.moments(torch.log(encoded[:1].clamp_min(1e-300)))
    torch.testing.assert_close(moments.step["velocity"][0], 0.1 * velocity, rtol=2e-2, atol=1e-3)
    # the same median under a broad mixture: the gain collapses the step
    errors[1, 3:6] = correction.project(torch.tensor([300.0, -400.0], dtype=F64), correction.axes())
    encoded, _ = code.encode(errors)
    broad = torch.log((0.6 * encoded[0] + 0.4 * encoded[1]).clamp_min(1e-300))[None]
    moments = code.moments(broad)
    length = float(torch.linalg.vector_norm(moments.step["velocity"][0]))
    assert length < 0.1 * 0.5 * 0.1
    two, four, zero = (torch.tensor(value) for value in (2.0, 4.0, 0.0))
    assert correction.confident(two, four, zero) == (2.0, 1.0)
    with pytest.raises(ValueError, match="rate"):
        correction.CorrectionCode(code_ranges(), 0.0)


def code_ranges():
    return state_code_ranges(api.rules_of(cases.SQUAD), api.ship_types())


def test_the_applied_variance_adds_the_unapplied_median():
    median = {"position": torch.tensor([[3.0, 4.0]], dtype=F64), "heading": torch.tensor([0.5])}
    step = {"position": torch.tensor([[1.0, 1.0]], dtype=F64), "heading": torch.tensor([0.2])}
    variance = {
        "position": torch.tensor([[1.0, 0.0, 2.0]], dtype=F64),
        "heading": torch.tensor([0.1]),
    }
    moments = correction.Moments(median=median, variance=variance, gain={}, target=step, step=step)
    torch.testing.assert_close(
        moments.applied_variance("position"), torch.tensor([[5.0, 6.0, 11.0]], dtype=F64)
    )
    torch.testing.assert_close(moments.applied_variance("heading"), torch.tensor([0.19]))


@pytest.mark.parametrize("jump, low, high", [(0.2, 1.2, 2.2), (0.1, 0.0, 0.15)])
def test_the_rare_jumps_step_by_the_median(jump, low, high):
    """The wait's error: 0, or +4 s at p `jump`: no step. A jump past the 84th percentile's
    reach (p > 0.16) widens the spread by about half its size (the mixture's sigma is 1.6 s
    at 0.2); a rarer one leaves it near the zero's (0.07 s at 0.1: the quantile spread)."""
    net = cases.double(_model())
    code = net.correction_code
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    column = correction.COMPONENT_NAMES.index("shield_recharge_wait")
    errors[1, column] = 4.0
    encoded, _ = code.encode(errors)
    mixture = (1.0 - jump) * encoded[0] + jump * encoded[1]
    forecast = torch.zeros(1, len(MEANS), dtype=F64)
    forecast[0, MEANS.index("shield_recharge_wait")] = 2.0
    means, spreads = code.correct(
        forecast,
        torch.log(mixture.clamp_min(1e-300))[None],
        torch.zeros(1, dtype=torch.long),
        net.state_code,
    )
    assert abs(float(means[0, MEANS.index("shield_recharge_wait")]) - 2.0) < 0.1 / 30.0
    assert low <= float(spreads[0, SPREADS.index("shield_recharge_wait")]) < high


def test_a_far_sliver_of_mass_keeps_the_sharp_spread():
    """The bug's case (2026-11-08): a sharp position error with 1e-3 of the mass 10 km out
    on every axis. The second moment put sigma near 300 m; the quantile spread keeps the
    sharp code's, and the gain stays near 1."""
    net = cases.double(_model())
    code = net.correction_code
    errors = torch.zeros(2, len(correction.COMPONENT_NAMES), dtype=F64)
    errors[0, 0:3] = correction.project(torch.tensor([20.0, -15.0], dtype=F64), code.axes.double())
    errors[1, 0:3] = 10_000.0
    encoded, _ = code.encode(errors)
    sliver = torch.log((0.999 * encoded[0] + 0.001 * encoded[1]).clamp_min(1e-300))[None]
    moments = code.moments(sliver)
    sharp = code.moments(torch.log(encoded[:1].clamp_min(1e-300)))
    assert float(sharp.variance["position"].abs().max()) < 1e-6
    assert float(moments.variance["position"].abs().max()) < 1.0  # m^2, against ~1e5
    assert float(moments.gain["position"][0]) > 0.99


def test_hidden_rows_take_the_correction_and_seen_rows_the_physics():
    net = cases.double(_model())
    rows = 4
    means = torch.zeros(rows, len(MEANS), dtype=F64)
    means[:, 2] = 50.0
    spreads = torch.zeros(rows, net.state_code.spread_count, dtype=F64)
    ship_type = torch.zeros(rows, dtype=torch.long)
    logits = torch.randn(rows, net.next_state_width, dtype=F64)
    correction_logits = torch.randn(rows, net.correction_code.width, dtype=F64)
    hidden = torch.tensor([False, True, False, True])
    physics = net.forecast(logits, means, spreads, ship_type)
    split = net.forecast(logits, means, spreads, ship_type, correction_logits, hidden)
    corrected = net.correction_code.correct(
        physics[0], correction_logits, ship_type, net.state_code
    )
    for got, seen, fixed in zip(split, physics, corrected, strict=True):
        torch.testing.assert_close(got[~hidden], seen[~hidden])
        torch.testing.assert_close(got[hidden], fixed[hidden])
    with pytest.raises(ValueError, match="hidden"):
        net.forecast(logits, means, spreads, ship_type, correction_logits)


# ------------------------------------------------------------------ the switches


def test_split_is_off_by_default_and_needs_the_delta_head():
    assert not YemongConfig().delta_split
    net = cases.model(**{**SETTINGS, "delta_split": False})
    assert net.correction_code is None and net.correction_head is None
    with pytest.raises(ValueError, match="delta_split"):
        YemongConfig(delta_split=True)


def test_arms_1_and_2_step_and_unroll_as_recorded():
    """Recorded before arms 3 and 4 existed (`arm_goldens`), to float32 rounding."""
    for arm in ("arm1", "arm2"):
        arm_goldens.reproduces(arm)


def test_the_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, ARM_SETS)
    )
    build = model_build(config)
    assert build.config.delta_split and build.config.delta_baseline == "persistence"
    assert build.config.next_state == "delta" and build.config.spread_inputs == "ellipse"
    assert build.config.physics_hidden_decay is None
    assert config.learner.next_state == config.learner.correction == 0.0033


def test_a_new_runs_defaults_are_the_champions_model():
    """The 2026-11-09-*-deadreckon-split-hiddenphysics run's model settings are a new run's
    defaults (`TrainConfig.yemong`, `LearnerConfig.next_state`): the deadreckon-split line
    with the hidden rows' physics builds the default model."""
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import TrainConfig

    default = TrainConfig(game=GameConfig(5, 10, 2600.0))
    assert default.yemong.next_state == "delta"
    assert default.yemong.delta_baseline == "extrapolation"
    assert default.yemong.delta_split and default.yemong.physics_hidden_decay == 0.5
    assert LearnerConfig().next_state == default.learner.next_state == 0.0033
    champion = (
        "yemong.next_state=delta",
        "yemong.delta_baseline=extrapolation",
        "yemong.delta_split=true",
        "yemong.physics_hidden_decay=0.5",
        "learner.next_state=0.0033",
    )
    assert with_settings(default, map(parse_setting, champion)) == default
    # the record defaults stay what a build record from before the switches reads
    assert YemongConfig().next_state == "absolute" and not YemongConfig().delta_split


FOURIER_SETS = (
    "yemong.order=temporal_first",
    "yemong.next_state=delta",
    "yemong.delta_baseline=extrapolation",
    "yemong.spread_inputs=scalar",
    "learner.next_state=0.0033",
    "yemong.delta_split=true",
    "learner.correction=0.0033",
    "yemong.position_inputs=fourier",
)


def test_fourier_inputs_are_off_by_default_and_need_the_delta_head():
    assert YemongConfig().position_inputs == "colour"
    with pytest.raises(ValueError, match="position_inputs 'fourier' needs next_state 'delta'"):
        YemongConfig(position_inputs="fourier")
    with pytest.raises(ValueError, match="position_inputs"):
        YemongConfig(next_state="delta", position_inputs="polar")


def test_the_fourier_arms_set_line_builds_it_and_old_records_read_as_colour():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.models.yemong.model import YemongBuild
    from bnb.runs.serialise import from_json, to_json
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, FOURIER_SETS)
    )
    build = model_build(config)
    assert build.config.position_inputs == "fourier" and build.config.delta_split
    assert build.config.delta_baseline == "extrapolation"
    assert build.config.spread_inputs == "scalar" and not build.config.acceleration_inputs
    record = to_json(build)
    assert record["config"]["position_inputs"] == "fourier"
    assert from_json(YemongBuild, record)[0] == build
    del record["config"]["position_inputs"]  # a policy from before the switch
    old, filled = from_json(YemongBuild, record)
    assert old.config.position_inputs == "colour" and "config.position_inputs" in filled


def test_the_fourier_ship_token_reads_fourier_and_the_code_stays_colour():
    """Only the encoder's ship input changes (437 state columns for 490, the token 501 for
    554, the benched flag included): the tokens' code, the next-state baseline, is the
    colour code."""
    colour = _model(**ARMS["arm4d"])
    fourier = _model(**ARMS["arm4d_fourier"])
    assert colour.state_code.input_width == colour.state_code.width == 490
    assert fourier.state_code.input_width == 437
    width = colour.encoder.ship.first.in_features
    assert fourier.encoder.ship.first.in_features == width - 490 + 437 == 501
    _, trajectory = cases.rollout(fourier, length=4)
    obs, replay = trajectory.obs, trajectory.replay
    reset = cases.resets(obs)
    with torch.no_grad():
        previous = fourier.preceding_motion(trajectory.initial_carry, obs, replay, reset)
        tokens = fourier.tokens(obs, replay, previous)
        plain = colour.tokens(obs, replay, previous)
    # float32 and the same maths in the same order, so equal within rounding: 1e-6 is about
    # ten ulps, and a wrong or missing column is of order one
    close = dict(rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tokens.code, plain.code, **close)
    assert tokens.ship.shape[-1] == 501 and plain.ship.shape[-1] == 554
    torch.testing.assert_close(tokens.ship[..., 437:], plain.ship[..., 490:], **close)
    valid, _, means, spreads = believed(obs.ships, replay)
    position = torch.where(valid[..., None], means[..., 0:2], replay.offset[..., None, :])
    means = torch.cat([position, means[..., 2:]], -1)
    expected = fourier.state_code.inputs(tokens.code, means, spreads)
    torch.testing.assert_close(tokens.ship[..., :437], expected, **close)


def test_the_error_head_starts_at_its_marginal_with_a_zero_last_layer():
    from bnb.models.yemong.model import Yemong, yemong_build

    net = Yemong(
        yemong_build(YemongConfig(**SETTINGS), api.rules_of(cases.SQUAD), api.ship_types())
    )
    last = net.correction_head.lasts[0]
    assert last.weight.abs().max() == 0
    torch.testing.assert_close(last.bias, net.correction_code.marginal_logits())
    parts = net.correction_code.split(torch.exp(last.bias.detach().double()))
    for name, part in parts.items():
        assert math.isclose(float(part.sum()), 1.0, rel_tol=1e-5), name


# arm 3 (persistence), arm 4 (extrapolation) and arm 4c (extrapolation without the
# acceleration inputs): every test on a segment runs on each
ARMS = dict(
    arm3=dict(),
    arm4=dict(delta_baseline="extrapolation"),
    arm4c=dict(delta_baseline="extrapolation", acceleration_inputs=False),
    # the split and extrapolation on C-fixed's inputs with the scalar spreads (arms 3d, 4d)
    arm3d=dict(acceleration_inputs=False, spread_inputs="scalar"),
    arm4d=dict(delta_baseline="extrapolation", acceleration_inputs=False, spread_inputs="scalar"),
    # arm 4d with the Fourier position and heading inputs
    arm4d_fourier=dict(
        delta_baseline="extrapolation",
        acceleration_inputs=False,
        spread_inputs="scalar",
        position_inputs="fourier",
    ),
    # arm 4d with the physics head also on the hidden rows (`physics_hidden_decay`)
    arm4dh=dict(
        delta_baseline="extrapolation",
        acceleration_inputs=False,
        spread_inputs="scalar",
        physics_hidden_decay=0.5,
    ),
    # and a hidden row's forecast only the trained share of the head's change (the fade)
    arm4dhf=dict(
        delta_baseline="extrapolation",
        acceleration_inputs=False,
        spread_inputs="scalar",
        physics_hidden_decay=0.5,
        physics_hidden_fade=True,
    ),
)


@pytest.fixture(scope="module", params=tuple(ARMS))
def segment(request, module_global_generator):
    net = _model(**ARMS[request.param])
    _, trajectory = cases.rollout(net, length=32)
    return net, trajectory


def test_the_arms_unroll_equals_a_loop_of_steps(segment):
    net, trajectory = segment
    net = cases.double(net)
    obs = cases.cast(trajectory.obs, F64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            net, net.step, carry, obs, cases.joint_actions(trajectory)
        )
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    pairs = (
        ("policy_logits", unrolled.policy_logits),
        ("next_state_logits", unrolled.next_state_logits[..., OWN:, :]),
        ("correction_logits", unrolled.correction_logits[..., OWN:, :]),
        ("values", unrolled.values),
    )
    for name, actual in pairs:
        assert cases.scale_error(actual, getattr(decisions, name)) <= 1e-9, name
    assert int(final.nonfinite.sum()) == 0
    hidden = decisions.replay.valid & (decisions.replay.age > 0)
    assert hidden.any()
    visible = decisions.replay.valid & (decisions.replay.age == 0)
    assert (decisions.replay.spreads[visible] == 0).all()


def test_the_stored_belief_of_a_hidden_row_is_the_corrected_forecast(segment):
    """Each hidden row's belief at t + 1 is the corrected forecast of its belief at t."""
    net, trajectory = segment
    obs, replay = trajectory.obs, trajectory.replay
    with torch.no_grad():
        unrolled = net.unroll(trajectory.initial_carry, obs, replay, cases.resets(obs))
    _, age, means, spreads = believed(obs.ships, replay)
    ship_type = labels.privileged_rows(trajectory)[..., SHIP["ship_type"]].long()
    hidden = age > 0
    forecast, _ = net.forecast(
        unrolled.next_state_logits,
        means,
        spreads,
        ship_type,
        unrolled.correction_logits,
        hidden,
        labels.rates(net, trajectory),
        age,
    )
    # still hidden at t + 1, no reset between: the stored belief is the forecast
    still = (replay.age[1:] > 1) & replay.valid[1:] & replay.valid[:-1]
    still = still & ~cases.resets(obs)[1:, :, None]
    assert still.any()
    torch.testing.assert_close(
        replay.means[1:][still], forecast[:-1, :, OWN:][still], rtol=1e-5, atol=1e-3
    )


# ------------------------------------------------------------------ the learners


def _true_rates(net, trajectory) -> delta.Rates:
    """Every row's rates as a row seen at t - 1 and t would read them, by hand from the truth:
    its true accelerations and their flag, its change from the truth the tick before (the
    carry's rows at the segment's first tick), its true gradient and shot flag, no rule, a
    full turn."""
    rows = labels.privileged_rows(trajectory)
    offset = trajectory.replay.offset
    means = truth(rows, offset)
    previous = torch.cat([trajectory.initial_carry.previous[None], truth(rows[:-1], offset[:-1])])
    column = lambda name: rows[..., SHIP[name]]  # noqa: E731
    valid = column("acceleration_valid") > 0.5
    return delta.Rates(
        valid=valid,
        acceleration=torch.stack([column("acceleration_x"), column("acceleration_y")], -1),
        nose_acceleration=column("nose_acceleration"),
        change=means - previous,
        gradient=torch.stack([column("log_index_gradient_x"), column("log_index_gradient_y")], -1),
        gradient_known=torch.ones_like(valid),
        fired=column("fired") > 0.5,
        rule=torch.zeros_like(valid),
        turn=torch.ones_like(column("fired")),
    )


def _merged_rates(rows: torch.Tensor, first: delta.Rates, second: delta.Rates) -> delta.Rates:
    """first's rates on rows (L, R, N), second's elsewhere."""
    out = {}
    for name in delta.RATE_FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        out[name] = torch.where(rows[..., None] if a.dim() > rows.dim() else rows, a, b.to(a.dtype))
    return delta.Rates(**out)


def _faded_by_hand(net, forecast, means, ship_type, rates, age) -> torch.Tensor:
    """forecast with each hidden row's (age > 0) change from the baseline times
    exp(-age dt / tau), on the faded means (the heading's change wrapped)."""
    baseline = net.delta_code.baseline(means, rates, ship_type)
    weight = torch.exp(-age.to(forecast.dtype) * net.tick_length / net.config.physics_hidden_decay)
    change = forecast - baseline
    heading = MEANS.index("heading")
    change[..., heading] = delta.wrap(change[..., heading])
    faded = baseline + weight[..., None] * change
    faded[..., heading] = delta.wrap(faded[..., heading])
    jumps = torch.tensor([name in delta.MEDIAN for name in MEANS])
    faded = torch.where(jumps, forecast, faded)
    return torch.where((age > 0)[..., None], faded, forecast)


def _check_terms(net, learner, runner):
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    terms, _, _, out, _ = learner._terms(
        trajectory, label, prepared["derived"], denominator, prepared["scale"]
    )
    mask = label["next_state_mask"]
    hidden = torch.cat(
        [torch.zeros_like(trajectory.replay.age[..., :OWN]), trajectory.replay.age], -1
    )
    hidden = hidden > 0
    assert torch.equal(label["physics_mask"], mask & ~hidden)
    assert torch.equal(label["correction_mask"], mask & hidden)
    assert label["correction_mask"].any() and label["physics_mask"].any()
    count = float(mask.sum())
    # the physics target: the true change from the baseline (extrapolation: the rates read)
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    read = labels.rates(net, trajectory)
    decay = net.config.physics_hidden_decay
    if decay is not None:  # a hidden row's rates are the truth's (`_true_rates`)
        read = _merged_rates(hidden, _true_rates(net, trajectory), read)
    expected, _, _ = net.delta_code.targets(
        truth(rows, offset), truth(following, offset), rows[..., SHIP["ship_type"]].long(), read
    )
    # the hand rates' acceleration skips `Yemong.rates`' frame round trip: float rounding
    tolerance = {} if decay is None else dict(rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(label["next_state"], expected, **tolerance)
    physics = codes.grouped_cross_entropy(
        out.next_state_logits, label["next_state"], net.delta_code.runs
    ).sum(-1)
    expected = float(physics[label["physics_mask"]].sum())
    if decay is not None:  # the hidden rows past the first tick, exp(-age / tau) each
        age = trajectory.replay.age.double() * net.tick_length
        later = torch.arange(age.shape[0])[:, None, None] > 0
        weighted = label["correction_mask"][..., OWN:] & later
        decayed = torch.exp(-age / decay)
        expected += float((physics[..., OWN:].double() * decayed)[weighted].sum())
    assert math.isclose(float(terms["next_state"]), expected / count, rel_tol=1e-5)
    # the error head's target: the truth at t + 1 less the physics forecast of the belief
    # (with the fade, a hidden row's by hand: exp(-age / tau) of the change from the baseline)
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    _, age, means, spreads = believed(trajectory.obs.ships, trajectory.replay)
    ship_type = rows[..., SHIP["ship_type"]].long()
    read = labels.rates(net, trajectory)
    forecast, _ = net.delta_code.advance(
        means, spreads, out.next_state_logits.detach(), ship_type, net.state_code, read
    )
    if net.config.physics_hidden_fade:
        forecast = _faded_by_hand(net, forecast, means, ship_type, read, age)
    errors = net.correction_code.errors(
        forecast, truth(following, trajectory.replay.offset), ship_type
    )
    target, _ = net.correction_code.encode(errors)
    error_entropy = codes.grouped_cross_entropy(
        out.correction_logits, target, net.correction_code.runs
    ).sum(-1)
    expected = float(error_entropy[label["correction_mask"]].sum()) / count
    assert math.isclose(float(terms["correction"]), expected, rel_tol=1e-5)
    # the stop-gradient: the error head's loss gives the physics head nothing
    physics_head = net.ship_heads.lasts[1].weight
    gradient = torch.autograd.grad(
        terms["correction"],
        [physics_head, net.correction_head.lasts[0].weight, net.encoder.ship.first.weight],
        allow_unused=True,
    )
    assert gradient[0] is None or gradient[0].abs().max() == 0
    assert gradient[1].abs().max() > 0 and gradient[2].abs().max() > 0
    return prepared, trajectory, label, denominator


def test_the_ppo_terms_and_their_series(segment):
    net, _ = segment
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared, trajectory, label, denominator = _check_terms(net, learner, runner)
    with torch.no_grad():
        heads_read = learner._loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"]
        )[-1]
        total = learner._head_sums(heads_read, trajectory, label)
    series = metrics.read(head_statistics.series(total, net))
    bucket = head_statistics.AGE_BUCKETS[0][0]
    for name in correction.QUANTITIES:
        for kind in ("realised_error", "predicted_error", "within_one_sigma"):
            assert f"correction/{kind}/{name}/{bucket}" in series
    for name in correction.STEP_LIMITS:
        assert 0.0 <= series[f"correction/capped_share/{name}"] <= 1.0
        assert 0.0 <= series[f"correction/step_over_limit/{name}"] <= 1.0
        assert 0.0 <= series[f"correction/gain/{name}"] <= 1.0
    for name in correction.COMPONENT_NAMES:
        assert 0.0 <= series[f"correction/support_clip/{name}"] <= 1.0
    assert math.isfinite(series[f"next_state/hidden/error/position_metres/{bucket}"])
    assert math.isfinite(series[f"next_state/hidden/cross_entropy/{bucket}"])
    assert "correction" in TERMS


def _bc(net):
    """A BC learner (the scripted teacher) of net, and its rollout, run twice."""
    from test_bc import CHARACTERISTICS, PARAMETERS, POOLS, TEACHER

    controllers = {
        "learner": cases.controller(net, cases.SHORT_SQUAD),
        "scripted": Scripted(CHARACTERISTICS, OWN, PARAMETERS),
    }
    runner = Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        POOLS,
        controllers,
        "learner",
        8,
        32,
        lc.CPU,
        compiled=False,
        extra_reads=Scripted.reads,
    )
    for _ in range(2):
        runner.run()
    views = runner.store.views
    learner = bc.BehaviourCloning(
        net,
        bc.config(minibatches=1, micro_batch_rows=views * runner.rules.num_ships),
        runner.rules,
        runner.carry["types"],
        lc.RewardConfig().outcome,
        lc.CPU,
        TEACHER,
        bc.TeacherConfig(),
        tuple(pool.name for pool in POOLS),
        runner.lineup.pool_of["learner"],
    )
    return learner, runner


def test_the_bc_terms(segment):
    net, _ = segment
    _check_terms(net, *_bc(net))


def test_an_update_with_the_cadence_reports_both_heads_trunk_shares(segment):
    net, _ = segment
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    views = torch.arange(runner.store.views)
    series = metrics.read(learner._gradient_decomposition(runner.store, prepared, views))
    for term in ("next_state", "correction"):
        assert series[f"gradient/norm/trunk/{term}"] > 0
    hidden_share = series["gradient/norm/trunk/next_state_enemy_hidden"]
    if net.config.physics_hidden_decay is None:
        assert hidden_share == 0  # the physics mask
    else:
        assert hidden_share > 0  # the hidden rows, weighted


def test_the_arms_policy_file_round_trips(tmp_path, segment):
    net, trajectory = segment
    save_policy(tmp_path / "policy", net)
    loaded = load_policy(tmp_path / "policy", lc.CPU).eval()
    assert loaded.build == net.build and loaded.config.delta_split
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    joint = cases.joint_actions(trajectory)
    with torch.no_grad():  # each loop draws the same offsets at its resets
        torch.manual_seed(9)
        a, _ = cases.step_loop(net, net.step, carry, obs, joint)
        torch.manual_seed(9)
        b, _ = cases.step_loop(loaded, loaded.step, carry, obs, joint)
    assert torch.equal(a.correction_logits, b.correction_logits)
    assert torch.equal(a.carry.belief.means, b.carry.belief.means)
    assert torch.equal(a.carry.belief.spreads, b.carry.belief.spreads)


def test_the_baseline_rates_trace_whole_under_torch_compile(segment):
    """The learner compiles its labels (`Learner._codes`, fullgraph): the extrapolation
    baseline's segment rates must trace without a graph break. dataclasses.fields on Rates
    broke it (arm 4d's first GPU update, 2026-11-08); the eager backend runs dynamo's tracing
    alone."""
    net, trajectory = segment
    eager = labels.rates(net, trajectory)
    if eager is None:
        return  # persistence: no rates
    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(labels.rates, fullgraph=True, backend="eager")(net, trajectory)
        eager = labels.rates(net, trajectory)
    for name in delta.RATE_FIELDS:
        assert torch.equal(getattr(compiled, name), getattr(eager, name)), name
    if net.config.physics_hidden_decay is None:
        return
    # the hidden rows' weights and truth-to-truth targets trace whole too

    def hidden_labels(net, trajectory):
        rows = labels.privileged_rows(trajectory)
        mask = labels.next_state_mask(trajectory, net)
        weight = labels.split_masks(trajectory, mask, net)["physics_weight"]
        return weight, labels.next_state_targets(net, trajectory, rows)["next_state"]

    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(hidden_labels, fullgraph=True, backend="eager")(net, trajectory)
        eager = hidden_labels(net, trajectory)
    for a, b in zip(compiled, eager, strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------ the hidden rows' physics
# (`yemong.physics_hidden_decay`, on arm 4d: the deadreckon-split arm)

HIDDEN_ARM = ARMS["arm4dh"]
HIDDEN_SETS = (
    "yemong.order=temporal_first",
    "yemong.next_state=delta",
    "yemong.delta_baseline=extrapolation",
    "yemong.spread_inputs=scalar",
    "learner.next_state=0.0033",
    "yemong.delta_split=true",
    "learner.correction=0.0033",
    "yemong.physics_hidden_decay=0.5",
)


@pytest.fixture(scope="module")
def deadreckon(module_global_generator):
    """The arm's model, a learner over a stored segment, and that segment's labels."""
    net = _model(**HIDDEN_ARM)
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    learner.prepare(runner.store, runner.bootstrap())  # stores the shot labels
    trajectory, shot_labels = next(learner.staged(runner.store, [torch.arange(runner.store.views)]))
    return net, trajectory, learner.labels(trajectory, shot_labels)


def _hidden_rows(trajectory) -> torch.Tensor:
    age = trajectory.replay.age
    return torch.cat([torch.zeros_like(age[..., :OWN]), age], -1) > 0


def _decay(net, tau: float) -> SimpleNamespace:
    """What `labels.split_masks` reads of a model, with physics_hidden_decay tau."""
    return SimpleNamespace(
        config=SimpleNamespace(physics_hidden_decay=tau), tick_length=net.tick_length
    )


def test_the_hidden_decay_is_off_by_default_and_needs_the_split():
    assert YemongConfig().physics_hidden_decay is None
    with pytest.raises(ValueError, match="physics_hidden_decay needs delta_split"):
        YemongConfig(next_state="delta", physics_hidden_decay=0.5)
    with pytest.raises(ValueError, match="must be positive"):
        YemongConfig(next_state="delta", delta_split=True, physics_hidden_decay=0.0)


def test_the_hidden_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, HIDDEN_SETS)
    )
    build = model_build(config)
    assert build.config.physics_hidden_decay == 0.5 and build.config.delta_split
    assert build.config.delta_baseline == "extrapolation"


def test_a_hidden_row_weighs_exp_of_minus_its_age_over_tau(deadreckon):
    """A seen row (hidden 0 s) weighs 1; a row hidden tau weighs 1/e; a hidden row of the
    segment's first tick, a row off the next-state mask, 0."""
    net, trajectory, label = deadreckon
    age = torch.cat([torch.zeros_like(trajectory.replay.age[..., :OWN]), trajectory.replay.age], -1)
    hidden, mask = _hidden_rows(trajectory), label["next_state_mask"]
    later = torch.arange(age.shape[0])[:, None, None] > 0
    ticks = 3  # tau = 3 ticks: the rows hidden 3 ticks weigh 1/e
    chosen = mask & hidden & later & (age == ticks)
    assert chosen.any()
    weight = labels.split_masks(trajectory, mask, _decay(net, ticks * net.tick_length))
    weight = weight["physics_weight"]
    assert (weight[mask & ~hidden] == 1).all() and (weight[mask & ~hidden & (age == 0)] == 1).all()
    torch.testing.assert_close(
        weight[chosen], torch.full_like(weight[chosen], math.exp(-1.0)), rtol=1e-6, atol=0
    )
    assert (weight[mask & hidden & ~later] == 0).all() and (weight[~mask] == 0).all()
    # the arm's own: exp(-age dt / 0.5) on every hidden row past the first tick
    seconds = age.double() * net.tick_length
    expected = torch.where(mask & ~hidden, 1.0, torch.exp(-seconds / 0.5))
    expected = torch.where(mask & (~hidden | later), expected, 0.0).float()
    torch.testing.assert_close(label["physics_weight"], expected)


def test_a_hidden_rows_target_is_truth_to_truth_like_a_seen_rows(deadreckon):
    """The rates of a hidden row's target are the truth's, as a row seen at t - 1 and t
    reads them: for such a row the rates the model read give the same target as the
    truth's; the belief's held rates would give another."""
    net, trajectory, label = deadreckon
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    offset = trajectory.replay.offset
    ship_type = rows[..., SHIP["ship_type"]].long()

    def target(rates):
        code, _, _ = net.delta_code.targets(
            truth(rows, offset), truth(following, offset), ship_type, rates
        )
        return code

    true, read = target(_true_rates(net, trajectory)), target(labels.rates(net, trajectory))
    hidden, mask = _hidden_rows(trajectory), label["physics_weight"] > 0
    torch.testing.assert_close(  # less the rounding of `Yemong.rates`' frame round trip
        label["next_state"][mask & hidden], true[mask & hidden], rtol=1e-5, atol=1e-3
    )
    torch.testing.assert_close(label["next_state"][~hidden], read[~hidden])
    # a row seen at both ticks: the read rates are the truth's, so its target is the same
    seen = mask & ~hidden & (trajectory.obs.ships[..., SHIP["acceleration_valid"]] > 0.5)
    assert seen.any()
    torch.testing.assert_close(read[seen], true[seen], rtol=1e-5, atol=1e-3)
    # the belief's held rates differ from the truth's on hidden rows
    assert not torch.allclose(read[mask & hidden], true[mask & hidden])


def test_a_vanishing_tau_leaves_the_physics_term_as_without_the_switch(deadreckon):
    net, trajectory, label = deadreckon
    cross_entropy = torch.rand(label["next_state_mask"].shape, dtype=F64)
    mask = label["next_state_mask"]
    off = {key: value for key, value in label.items() if key != "physics_weight"}
    expected = float(ppo.physics_sum(cross_entropy, off))
    for tau in (1e-9, 1e-30):
        vanishing = dict(label, **labels.split_masks(trajectory, mask, _decay(net, tau)))
        # float64 sums: equal to rounding (1e-16), so 1e-12 relative still fails a real term
        assert float(ppo.physics_sum(cross_entropy, vanishing)) == pytest.approx(
            expected, rel=1e-12
        )
    # and the switch's own weights only add
    assert float(ppo.physics_sum(cross_entropy, label)) > expected


def test_the_hidden_arms_bc_learner_steps():
    net = _model(**HIDDEN_ARM)
    learner, runner = _bc(net)
    before = net.ship_heads.lasts[1].weight.detach().clone()
    series = metrics.read(learner.update(runner.store, runner.bootstrap()))
    assert math.isfinite(series["next_state/loss"]) and series["next_state/loss"] > 0
    assert not torch.equal(net.ship_heads.lasts[1].weight, before)


def test_a_policy_file_without_the_hidden_decay_loads_it_as_none(tmp_path):
    net = _model(**HIDDEN_ARM)
    save_policy(tmp_path / "policy", net)
    assert load_policy(tmp_path / "policy", lc.CPU).config.physics_hidden_decay == 0.5
    record_path = tmp_path / "policy" / "checkpoint.json"
    record = json.loads(record_path.read_text())
    config = record["model"]["build"]["config"]
    assert config.pop("physics_hidden_decay") == 0.5
    config["delta_split"] = True
    record_path.write_text(json.dumps(record))
    with pytest.warns(FilledDefaults, match="physics_hidden_decay"):
        loaded = load_policy(tmp_path / "policy", lc.CPU)
    assert loaded.config.physics_hidden_decay is None and loaded.config.delta_split


# ------------------------------------------------------------------ the fade
# (`yemong.physics_hidden_fade`: a hidden row's forecast takes exp(-age / tau) of the change)

FADE_ARM = ARMS["arm4dhf"]
FADE_SETS = HIDDEN_SETS + ("yemong.physics_hidden_fade=true",)
FADE_TICKS = 3  # the constructed nets' tau: 3 ticks


def _fade_pair(tau_ticks: float = FADE_TICKS):
    """(faded, unfaded): the fade arm in float64 with tau = tau_ticks ticks, and the same
    weights without the switch."""
    tick = _model(**FADE_ARM).tick_length
    settings = dict(FADE_ARM, physics_hidden_decay=tau_ticks * tick)
    faded = cases.double(_model(**settings))
    unfaded = cases.double(_model(**dict(settings, physics_hidden_fade=False)))
    unfaded.load_state_dict(faded.state_dict())
    return faded, unfaded


def _fade_case(net, rows: int = 4):
    """A constructed batch of rows moving with last tick's rates (valid, hidden: the rule),
    its physics logits and its error head's logits."""
    generator = torch.Generator().manual_seed(3)
    randn = lambda *shape: torch.randn(*shape, dtype=F64, generator=generator)  # noqa: E731
    means = torch.zeros(rows, len(MEANS), dtype=F64)
    means[:, 0:2] = 100.0 * randn(rows, 2)
    means[:, 2:4] = 40.0 + 5.0 * randn(rows, 2)
    means[:, MEANS.index("heading")] = 3.0  # near pi: the heading's change wraps
    means[:, MEANS.index("nose_rate")] = 0.5 * randn(rows)
    means[:, MEANS.index("shield")] = 0.5
    means[:, MEANS.index("energy")] = 0.5
    change = torch.zeros(rows, len(MEANS), dtype=F64)
    change[:, MEANS.index("angle_of_attack")] = 0.01
    change[:, MEANS.index("energy")] = -0.01
    true = torch.ones(rows, dtype=torch.bool)
    rates = delta.Rates(
        valid=true,
        acceleration=5.0 * randn(rows, 2),
        nose_acceleration=randn(rows),
        change=change,
        gradient=torch.zeros(rows, 2, dtype=F64),
        gradient_known=~true,
        fired=~true,
        rule=true,
        turn=torch.full((rows,), 0.5, dtype=F64),
    )
    spreads = torch.zeros(rows, net.state_code.spread_count, dtype=F64)
    ship_type = torch.zeros(rows, dtype=torch.long)
    logits = randn(rows, net.next_state_width)
    correction_logits = randn(rows, net.correction_code.width)
    return means, spreads, ship_type, logits, correction_logits, rates


def test_the_fade_is_off_by_default_and_needs_the_split_and_the_hidden_decay():
    assert not YemongConfig().physics_hidden_fade
    with pytest.raises(ValueError, match="physics_hidden_fade needs"):
        YemongConfig(next_state="delta", delta_split=True, physics_hidden_fade=True)
    with pytest.raises(ValueError, match="physics_hidden_fade needs"):
        YemongConfig(next_state="delta", physics_hidden_fade=True)
    net = _model(**FADE_ARM)
    means, spreads, ship_type, logits, _, rates = _fade_case(net)
    with pytest.raises(ValueError, match="age"):
        cases.double(net).forecast(logits, means, spreads, ship_type, rates=rates)


def test_seen_rows_forecast_as_without_the_fade():
    # float64 models: equal to rounding (1e-12 is far above 1e-16 and far below any real term)
    faded, unfaded = _fade_pair()
    means, spreads, ship_type, logits, correction_logits, rates = _fade_case(faded)
    age = torch.tensor([0, 2, 0, 7], dtype=torch.int32)
    hidden = age > 0
    with_fade = faded.forecast(
        logits, means, spreads, ship_type, correction_logits, hidden, rates, age
    )
    without = unfaded.forecast(logits, means, spreads, ship_type, correction_logits, hidden, rates)
    torch.testing.assert_close(with_fade[0][~hidden], without[0][~hidden], rtol=1e-12, atol=1e-12)
    # the spreads: hidden rows' the error head's
    torch.testing.assert_close(with_fade[1], without[1], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(with_fade[0][hidden], without[0][hidden])


def test_a_hidden_rows_change_fades_to_one_over_e_at_age_tau():
    """Age 0: the unfaded forecast; age tau: 1/e of the change from the baseline on the
    faded means, the jumps and every spread the unfaded forecast's."""
    faded, unfaded = _fade_pair()
    means, spreads, ship_type, logits, _, rates = _fade_case(faded)
    without, without_spreads = unfaded.forecast(logits, means, spreads, ship_type, rates=rates)
    zero = torch.zeros(len(means), dtype=torch.int32)
    at_zero = faded.forecast(logits, means, spreads, ship_type, rates=rates, age=zero)
    close = dict(rtol=1e-12, atol=1e-12)  # float64: rounding is 1e-16
    torch.testing.assert_close(at_zero[0], without, **close)
    torch.testing.assert_close(at_zero[1], without_spreads, **close)
    tau = torch.full_like(zero, FADE_TICKS)
    got, got_spreads = faded.forecast(logits, means, spreads, ship_type, rates=rates, age=tau)
    torch.testing.assert_close(got_spreads, without_spreads, **close)
    baseline = faded.delta_code.baseline(means, rates, ship_type)
    for i, name in enumerate(MEANS):
        if name in delta.MEDIAN:
            torch.testing.assert_close(got[:, i], without[:, i], **close, msg=name)
            continue
        full, part = without[:, i] - baseline[:, i], got[:, i] - baseline[:, i]
        if name == "heading":
            full, part = delta.wrap(full), delta.wrap(part)
        assert full.abs().min() > 1e-9, name  # the head moves every faded quantity
        torch.testing.assert_close(part, full * math.exp(-1.0), rtol=1e-9, atol=1e-12)
    # the heading near pi: its change wraps, the faded heading stays in range
    assert (got[:, MEANS.index("heading")].abs() <= math.pi).all()


def test_a_huge_tau_forecasts_as_without_the_fade():
    tick = _model(**FADE_ARM).tick_length
    faded, unfaded = _fade_pair(1e9 / tick)
    means, spreads, ship_type, logits, correction_logits, rates = _fade_case(faded)
    age = torch.tensor([0, 1, 30, 600], dtype=torch.int32)
    hidden = age > 0
    with_fade = faded.forecast(
        logits, means, spreads, ship_type, correction_logits, hidden, rates, age
    )
    without = unfaded.forecast(logits, means, spreads, ship_type, correction_logits, hidden, rates)
    for got, expected in zip(with_fade, without, strict=True):  # 1 - w is age dt / 1e9 s
        torch.testing.assert_close(got, expected, rtol=1e-7, atol=1e-7)


def test_the_error_heads_label_is_the_error_of_the_faded_forecast(segment):
    net, trajectory = segment
    if not net.config.physics_hidden_fade:
        return
    with torch.no_grad():
        unrolled = net.unroll(
            trajectory.initial_carry,
            trajectory.obs,
            trajectory.replay,
            cases.resets(trajectory.obs),
        )
    _, _, errors = labels.correction_targets(net, trajectory, unrolled.next_state_logits)
    rows = labels.privileged_rows(trajectory)
    following = labels.next_rows(trajectory, rows)
    _, age, means, spreads = believed(trajectory.obs.ships, trajectory.replay)
    ship_type = rows[..., SHIP["ship_type"]].long()
    read = labels.rates(net, trajectory)
    logits = unrolled.next_state_logits.float()
    physics, _ = net.delta_code.advance(means, spreads, logits, ship_type, net.state_code, read)
    faded = _faded_by_hand(net, physics, means, ship_type, read, age)
    after = truth(following, trajectory.replay.offset)
    hidden = age > 0
    expected = net.correction_code.errors(faded, after, ship_type)
    torch.testing.assert_close(errors[hidden], expected[hidden], rtol=1e-5, atol=1e-3)
    unfaded = net.correction_code.errors(physics, after, ship_type)
    torch.testing.assert_close(errors[~hidden], unfaded[~hidden], rtol=1e-5, atol=1e-3)
    assert not torch.allclose(errors[hidden], unfaded[hidden], rtol=1e-5, atol=1e-3)


def test_the_fade_arms_set_line_builds_the_arm():
    from bnb.cli import parse_setting, with_settings
    from bnb.contracts import GameConfig
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), map(parse_setting, FADE_SETS)
    )
    build = model_build(config)
    assert build.config.physics_hidden_fade and build.config.physics_hidden_decay == 0.5
    assert build.config.delta_split and build.config.delta_baseline == "extrapolation"


def test_a_policy_file_without_the_fade_loads_it_as_off(tmp_path):
    net = _model(**FADE_ARM)
    save_policy(tmp_path / "policy", net)
    assert load_policy(tmp_path / "policy", lc.CPU).config.physics_hidden_fade
    record_path = tmp_path / "policy" / "checkpoint.json"
    record = json.loads(record_path.read_text())
    assert record["model"]["build"]["config"].pop("physics_hidden_fade") is True
    record_path.write_text(json.dumps(record))
    with pytest.warns(FilledDefaults, match="physics_hidden_fade"):
        loaded = load_policy(tmp_path / "policy", lc.CPU)
    assert not loaded.config.physics_hidden_fade and loaded.config.physics_hidden_decay == 0.5
